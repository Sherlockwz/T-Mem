"""Stage 5 (LoCoMo-Plus): per-sample top-K scene builder via 3-way RRF.

Operates in **per-sample memory-library mode**: every plus sample owns its own
conversation (built by stage0_locomo_plus_stitch + stage1) and therefore its
own scene library `scene_list_conv_{sidx}.json` and trigger pack
`triggers_conv_{sidx}.json`. Sample i is scored against library i alone, so
cue insertion + stitch context are end-to-end honoured.

Shares prepare_conv / score_one_query / rrf_fuse_topk with stage5_retrieval_locomo.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# CRITICAL: bootstrap providers BEFORE importing stage modules — otherwise
# EmbeddingProvider stays as the placeholder class and fails below.
from T_mem.bootstrap import patch_providers  # noqa: E402

patch_providers()

from T_mem.config import ExperimentConfig  # noqa: E402
from T_mem.main.stage5_retrieval_locomo import (  # noqa: E402
    DEFAULT_TOPK,
    DEFAULT_RRF_K,
    N_TURNS_SKIP,
    prepare_conv,
    score_one_query,
    rrf_fuse_topk,
    _embed_batch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("T_mem.bgem3_embedding").setLevel(logging.WARNING)
log = logging.getLogger("stage5.locomo_plus")


def _scene_to_dialogue(sc: dict) -> str:
    """Line-by-line ``speaker_name: content`` (matches qa_judge reference)."""
    lines = []
    for m in sc.get("original_data", []) or []:
        spk = (m.get("speaker_name") or m.get("speaker") or "?").strip()
        cnt = (m.get("content") or "").strip()
        if cnt:
            lines.append(f"{spk}: {cnt}")
    return "\n".join(lines)


def _load_scenes_full(scenes_path: Path) -> dict[str, dict]:
    """scene_id -> {summary, scene_description, original_dialogue}."""
    scs = json.load(scenes_path.open("r", encoding="utf-8"))
    out: dict[str, dict] = {}
    for sc in scs:
        sid = sc["scene_id"]
        out[sid] = {
            "summary": (sc.get("summary") or "").strip(),
            "scene_description": (sc.get("scene_description") or "").strip(),
            "original_dialogue": _scene_to_dialogue(sc),
        }
    return out


def _format_sample_id(idx: int) -> str:
    return f"sample_{idx:03d}"


def build_locomo_plus_topk(
    scenes_dir: Path,
    triggers_dir: Path,
    locomo_plus_file: Path,
    out_file: Path,
    topk: int = DEFAULT_TOPK,
    rrf_k: int = DEFAULT_RRF_K,
) -> dict[str, Any]:
    """Build per-sample top-K JSON for LoCoMo-Plus.

    Sample-to-library binding is **identity**: sample idx i uses
    `scene_list_conv_{i}.json` and `triggers_conv_{i}.json`. Those libraries
    were produced by stage1/stage4 over the 401-conv stitched dataset emitted
    by stage0_locomo_plus_stitch, where conv i was built from plus_item[i].
    """
    scenes_dir = Path(scenes_dir).resolve()
    triggers_dir = Path(triggers_dir).resolve()
    locomo_plus_file = Path(locomo_plus_file).resolve()
    out_file = Path(out_file).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not scenes_dir.exists():
        raise SystemExit(f"scenes dir missing: {scenes_dir}")
    if not triggers_dir.exists():
        raise SystemExit(f"triggers dir missing: {triggers_dir}")
    if not locomo_plus_file.exists():
        raise SystemExit(f"locomo_plus file missing: {locomo_plus_file}")

    from T_mem.llm.bgem3_provider import EmbeddingProvider
    emb = EmbeddingProvider(timeout=60, max_retries=5)

    samples = json.load(locomo_plus_file.open("r", encoding="utf-8"))
    n_samples = len(samples)

    # n_libs is bounded by what stage1 actually produced (e.g. --limit 25
    # smoke runs ⇒ only 25 stitched convs ⇒ 25 scene_list_conv_*.json).
    n_libs = sum(
        1 for _ in scenes_dir.glob("scene_list_conv_*.json")
    )
    n_eff = min(n_samples, n_libs)
    log.info("n_samples=%d  n_libs=%d  n_eff=%d", n_samples, n_libs, n_eff)
    if n_eff < n_samples:
        log.warning(
            "[stage5/locomo_plus] only %d libraries found under %s; "
            "running first %d samples only (smoke / partial mode)",
            n_libs, scenes_dir, n_eff,
        )

    # Embed all trigger queries once, in a single batch (much faster than
    # 401 separate one-shot embeddings).
    trigger_texts = [
        (samples[i].get("trigger_query") or "").strip()
        for i in range(n_eff)
    ]
    log.info("embedding %d trigger_queries in one batch ...", n_eff)
    qvecs = _embed_batch(emb, trigger_texts)

    samples_out: dict[str, list[dict]] = {}
    t0 = time.perf_counter()
    for sidx in range(n_eff):
        sc_path = scenes_dir / f"scene_list_conv_{sidx}.json"
        trig_path = triggers_dir / f"triggers_conv_{sidx}.json"
        if not sc_path.exists():
            raise SystemExit(f"missing {sc_path}")
        if not trig_path.exists():
            raise SystemExit(f"missing {trig_path}")

        prep = prepare_conv(sidx, sc_path, trig_path, emb)
        sc_payload = _load_scenes_full(sc_path)

        per_sc = score_one_query(qvecs[sidx], prep)
        topk_pairs = rrf_fuse_topk(per_sc, topk=topk, k=rrf_k)

        rows: list[dict] = []
        for rank, (sid, score) in enumerate(topk_pairs, 1):
            sc_stats = prep["scenes"].get(sid, {})
            n_turns = int(sc_stats.get("n_turns", 0))
            # n_turns > N_TURNS_SKIP → L2/L3 zeroed in prepare_conv.
            skipped = n_turns > N_TURNS_SKIP
            payload = sc_payload.get(sid, {})
            rows.append({
                "rank": rank,
                "scene_id": sid,
                "score": round(float(score), 6),
                "n_turns": n_turns,
                "skipped": skipped,
                "summary": payload.get("summary", ""),
                "scene_description": payload.get("scene_description", ""),
                "original_dialogue": payload.get("original_dialogue", ""),
            })
        samples_out[_format_sample_id(sidx)] = rows

        if (sidx + 1) % 25 == 0 or (sidx + 1) == n_eff:
            log.info("scored %d/%d samples (elapsed %.1fs)",
                     sidx + 1, n_eff, time.perf_counter() - t0)

    log.info("per-sample scoring done in %.1fs | n=%d",
             time.perf_counter() - t0, len(samples_out))

    payload = {
        "method": "RRF_k30_3way",
        "mode": "per_sample_library",  # tag: distinguishes from old i % n_conv mode
        "topk_max": topk,
        "rrf_k": rrf_k,
        "n_turns_skip_threshold": N_TURNS_SKIP,
        "n_samples": len(samples_out),
        "n_libs": n_libs,
        "scenes_dir": str(scenes_dir),
        "triggers_dir": str(triggers_dir),
        "locomo_plus_file": str(locomo_plus_file),
        "samples": samples_out,
    }
    out_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    log.info("wrote %s", out_file)
    return payload


def main() -> None:
    """Default entry: paths from ExperimentConfig + benchmark_eval.
    Env overrides: T_MEM_L2L3_TOPK / T_MEM_L2L3_RRF_K / T_MEM_LOCOMO_PLUS_FILE."""
    config = ExperimentConfig()
    scenes_dir = config.scenes_dir()
    triggers_dir = config.experiment_dir() / "l2l3_triggers"

    project_root = _PROJECT_ROOT
    locomo_plus_file = Path(
        os.environ.get(
            "T_MEM_LOCOMO_PLUS_FILE",
            str(project_root / "benchmark_eval" / "locomo_plus" / "data" / "locomo_plus.json"),
        )
    )
    out_file = config.experiment_dir() / "locomo_plus_topk_per_sample.json"

    topk = int(os.environ.get("T_MEM_L2L3_TOPK", str(DEFAULT_TOPK)))
    rrf_k = int(os.environ.get("T_MEM_L2L3_RRF_K", str(DEFAULT_RRF_K)))

    log.info("[stage5/locomo_plus] experiment_dir=%s", config.experiment_dir())
    log.info("[stage5/locomo_plus] scenes_dir=%s", scenes_dir)
    log.info("[stage5/locomo_plus] triggers_dir=%s", triggers_dir)
    log.info("[stage5/locomo_plus] locomo_plus_file=%s", locomo_plus_file)
    log.info("[stage5/locomo_plus] out_file=%s", out_file)
    log.info("[stage5/locomo_plus] topk=%d rrf_k=%d", topk, rrf_k)

    build_locomo_plus_topk(
        scenes_dir=scenes_dir,
        triggers_dir=triggers_dir,
        locomo_plus_file=locomo_plus_file,
        out_file=out_file,
        topk=topk,
        rrf_k=rrf_k,
    )


if __name__ == "__main__":
    main()
