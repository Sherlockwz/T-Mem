"""Stage 4: per-conv Scene/Horizon trigger extraction.
Reads scenes/scene_list_conv_*.json, writes scene_horizon_triggers/triggers_conv_{i}.json
(consumed by stage5_retrieval_locomo)."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

try:
    import json_repair  # type: ignore
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from T_mem.llm.llm_provider import LLMProvider  # noqa: E402
from T_mem.prompts.trigger_prompts import build_prompt, SCENE_TRIGGER_KEYS, HORIZON_TRIGGER_KEYS, N_TURNS_SKIP  # type: ignore  # noqa: E402
from T_mem.config import ExperimentConfig, MODELS  # noqa: E402

DEFAULT_CONCURRENCY = 14
DEFAULT_TIMEOUT = 240

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("root").setLevel(logging.WARNING)
logging.getLogger("T_mem.llm").setLevel(logging.WARNING)
log = logging.getLogger("stage4.extract")

TERMINAL_STATUSES = {
    "ok", "skipped_by_n_turns", "skipped_empty", "parse_failed", "llm_failed",
}


def _parse_json_safe(text: str) -> Optional[dict]:
    if text is None:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        nl = t.find("\n")
        if nl >= 0 and t[:nl].strip().lower() in ("json", ""):
            t = t[nl + 1:]
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()
    try:
        return json.loads(t)
    except Exception:
        if HAS_JSON_REPAIR:
            try:
                obj = json_repair.loads(t)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return None
        return None

def _validate_group_a(obj: dict) -> Optional[str]:
    if not isinstance(obj, dict):
        return "not a dict"
    if "scene_attributes" not in obj or "horizon_channels" not in obj:
        return "missing scene_attributes or horizon_channels"
    scene_attrs = obj["scene_attributes"]
    if not isinstance(scene_attrs, dict):
        return "scene_attributes not a dict"
    for k in SCENE_TRIGGER_KEYS:
        if k not in scene_attrs or not isinstance(scene_attrs[k], str) or not scene_attrs[k].strip():
            return f"scene.{k} missing or empty"
    horizon_chs = obj["horizon_channels"]
    if not isinstance(horizon_chs, dict):
        return "horizon_channels not a dict"
    for k in HORIZON_TRIGGER_KEYS:
        if k not in horizon_chs or not isinstance(horizon_chs[k], dict):
            return f"horizon.{k} missing or not a dict"
        if "sent" not in horizon_chs[k] or "confidence" not in horizon_chs[k]:
            return f"horizon.{k} missing sent/confidence"
    return None

def scene_to_dialogue(sc: dict) -> str:
    lines = []
    for m in sc.get("original_data", []):
        spk = (m.get("speaker_name") or m.get("speaker") or "?").strip()
        cnt = (m.get("content") or "").strip()
        if cnt:
            lines.append(f"{spk}: {cnt}")
    return "\n".join(lines)

async def extract_one_scene(
    llm: LLMProvider,
    sem: asyncio.Semaphore,
    dialogue: str,
) -> dict:
    prompt = build_prompt("A", dialogue)
    rec: dict[str, Any] = {
        "status": "pending",
        "latency_s": None,
        "scene_attributes": None,
        "horizon_channels": None,
        "error": None,
    }
    start = time.perf_counter()
    try:
        async with sem:
            raw = await llm.generate(
                prompt,
                response_format={"type": "json_object"},
                call_site="stage4.scene_trigger",
            )
    except Exception as e:
        rec["status"] = "llm_failed"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["latency_s"] = round(time.perf_counter() - start, 3)
        return rec

    rec["latency_s"] = round(time.perf_counter() - start, 3)
    obj = _parse_json_safe(raw)
    if obj is None:
        rec["status"] = "parse_failed"
        rec["error"] = "json parse failed"
        rec["raw_response"] = (raw if isinstance(raw, str) else str(raw))[:1500]
        return rec

    err = _validate_group_a(obj)
    if err is not None:
        rec["status"] = "parse_failed"
        rec["error"] = f"schema invalid: {err}"
        rec["scene_attributes"] = obj.get("scene_attributes")
        rec["horizon_channels"] = obj.get("horizon_channels")
        rec["raw_response"] = (raw if isinstance(raw, str) else str(raw))[:1500]
        return rec

    rec["status"] = "ok"
    rec["scene_attributes"] = obj["scene_attributes"]
    rec["horizon_channels"] = obj["horizon_channels"]
    return rec


async def process_conv(
    llm: LLMProvider,
    conv_id: int,
    scenes_path: Path,
    out_file: Path,
    concurrency: int,
    resume: bool = True,
) -> dict:
    conv_tag = f"conv_{conv_id}"
    from T_mem.utils.cost_ledger import set_conv as _cost_set_conv
    _cost_set_conv(conv_id)
    scenes = json.load(scenes_path.open("r", encoding="utf-8"))

    existing: dict[str, dict] = {}
    if resume and out_file.exists():
        try:
            prev = json.load(out_file.open("r", encoding="utf-8"))
            existing = prev.get("scenes", {}) or {}
        except Exception:
            existing = {}

    to_run: list[tuple[str, str]] = []
    n_skip_n_turns_new = 0
    for sc in scenes:
        sid = sc["scene_id"]
        prev = existing.get(sid)
        if prev and prev.get("status") in TERMINAL_STATUSES and prev.get("status") != "llm_failed":
            continue
        n_turns = len(sc.get("original_data", []) or [])
        if n_turns > N_TURNS_SKIP:
            existing[sid] = {
                "status": "skipped_by_n_turns",
                "latency_s": 0.0,
                "scene_attributes": None,
                "horizon_channels": None,
                "error": None,
                "n_turns": n_turns,
            }
            n_skip_n_turns_new += 1
            continue
        dialogue = scene_to_dialogue(sc)
        if not dialogue.strip():
            existing[sid] = {
                "status": "skipped_empty",
                "latency_s": 0.0,
                "scene_attributes": None,
                "horizon_channels": None,
                "error": "empty dialogue",
                "n_turns": n_turns,
            }
            continue
        to_run.append((sid, dialogue))

    n_skip_total = sum(1 for r in existing.values() if r.get("status") == "skipped_by_n_turns")
    log.info(
        "[%s] %d scenes total | to_run=%d | skipped_by_n_turns(total)=%d (+%d new) | resume=%s",
        conv_tag, len(scenes), len(to_run), n_skip_total, n_skip_n_turns_new, resume,
    )

    if to_run:
        sem = asyncio.Semaphore(concurrency)
        tasks = [
            extract_one_scene(llm, sem, dlg)
            for _sid, dlg in to_run
        ]
        t0 = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=False)
        dt = time.perf_counter() - t0
        for (sid, _), rec in zip(to_run, results):
            existing[sid] = rec
        n_ok = sum(1 for r in existing.values() if r.get("status") == "ok")
        n_fail = sum(
            1 for r in existing.values()
            if r.get("status") in ("parse_failed", "llm_failed")
        )
        log.info(
            "[%s] wall=%.1fs | ok=%d / fail=%d / total=%d",
            conv_tag, dt, n_ok, n_fail, len(existing),
        )

    out_file.parent.mkdir(exist_ok=True, parents=True)
    payload = {
        "conv_id": conv_id,
        "meta": {
            "n_scenes": len(scenes),
            "n_turns_skip_threshold": N_TURNS_SKIP,
        },
        "scenes": {
            sc["scene_id"]: existing.get(sc["scene_id"], {"status": "missing"})
            for sc in scenes
        },
    }
    out_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


async def run_extract(
    scenes_dir: Path,
    out_dir: Path,
    model: str = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: int = DEFAULT_TIMEOUT,
    resume: bool = True,
) -> dict:
    """Drive extraction over every `scene_list_conv_*.json` under scenes_dir.

    Returns the aggregated stats dict (also written to `out_dir/extract_stats.json`).
    """
    scenes_dir = Path(scenes_dir).resolve()
    out_dir = Path(out_dir).resolve()
    if not scenes_dir.exists():
        raise SystemExit(f"scenes dir not found: {scenes_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if model is None:
        model = MODELS["memory_build"]

    conv_files: list[tuple[int, Path]] = []
    for p in sorted(scenes_dir.glob("scene_list_conv_*.json")):
        try:
            cid = int(p.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        conv_files.append((cid, p))
    conv_files.sort(key=lambda x: x[0])
    if not conv_files:
        raise SystemExit(f"no scene_list_conv_*.json found under {scenes_dir}")

    log.info("Triggers output dir: %s", out_dir)

    llm = LLMProvider(
        model=model,
        timeout=timeout,
        retries=3,
        json_max_retries=3,
        max_workers=max(14, concurrency + 2),
    )
    log.info("LLM provider: %r", llm)

    total_ok, total_fail, total_scenes, total_skip_nturns = 0, 0, 0, 0
    t_all = time.perf_counter()

    for i, (cid, sc_path) in enumerate(conv_files, 1):
        log.info(">>> (%d/%d) conv_%d  scenes=%s",
                 i, len(conv_files), cid, sc_path.name)
        out_file = out_dir / f"triggers_conv_{cid}.json"
        payload = await process_conv(
            llm=llm,
            conv_id=cid,
            scenes_path=sc_path,
            out_file=out_file,
            concurrency=concurrency,
            resume=resume,
        )
        scs = payload["scenes"]
        total_scenes += len(scs)
        total_ok += sum(1 for v in scs.values() if v.get("status") == "ok")
        total_fail += sum(
            1 for v in scs.values()
            if v.get("status") in ("parse_failed", "llm_failed")
        )
        total_skip_nturns += sum(
            1 for v in scs.values() if v.get("status") == "skipped_by_n_turns"
        )

    log.info("==== ALL DONE in %.1fs ====", time.perf_counter() - t_all)
    log.info(
        "scenes total=%d | ok=%d | skipped_by_n_turns=%d | fail=%d",
        total_scenes, total_ok, total_skip_nturns, total_fail,
    )

    stats = {
        "n_conv": len(conv_files),
        "n_scenes": total_scenes,
        "n_ok": total_ok,
        "n_fail": total_fail,
        "n_skipped_by_n_turns": total_skip_nturns,
        "ok_rate_over_nonskip": round(
            total_ok / max(1, total_scenes - total_skip_nturns), 4
        ),
        "wall_clock_s": round(time.perf_counter() - t_all, 2),
        "n_turns_skip_threshold": N_TURNS_SKIP,
        "model": model,
        "concurrency_per_conv": concurrency,
        "scenes_dir": str(scenes_dir),
        "out_dir": str(out_dir),
    }
    (out_dir / "extract_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return stats

async def main() -> None:
    """Default entry: read paths from ExperimentConfig.

    Inputs  : <experiment_dir>/scenes/scene_list_conv_*.json
    Outputs : <experiment_dir>/scene_horizon_triggers/triggers_conv_*.json
              <experiment_dir>/scene_horizon_triggers/extract_stats.json

    Model id is read from ``T_mem.config.MODELS['memory_build']`` —
    the single source of truth. Other knobs:
      T_MEM_SCENE_HORIZON_EXTRACT_CONCURRENCY default 14
      T_MEM_SCENE_HORIZON_EXTRACT_TIMEOUT     default 240
      T_MEM_SCENE_HORIZON_NO_RESUME=1         force re-run all scenes
    """
    import os
    config = ExperimentConfig()
    scenes_dir = config.scenes_dir()
    out_dir = config.experiment_dir() / "scene_horizon_triggers"

    model = MODELS["memory_build"]
    concurrency = int(os.environ.get(
        "T_MEM_SCENE_HORIZON_EXTRACT_CONCURRENCY", str(DEFAULT_CONCURRENCY)
    ))
    timeout = int(os.environ.get(
        "T_MEM_SCENE_HORIZON_EXTRACT_TIMEOUT", str(DEFAULT_TIMEOUT)
    ))
    resume = os.environ.get("T_MEM_SCENE_HORIZON_NO_RESUME", "").strip() not in ("1", "true", "yes", "on")

    log.info("[stage4/extract] experiment_dir=%s", config.experiment_dir())
    log.info("[stage4/extract] scenes_dir=%s", scenes_dir)
    log.info("[stage4/extract] out_dir=%s", out_dir)
    log.info("[stage4/extract] model=%s concurrency=%d timeout=%d resume=%s",
             model, concurrency, timeout, resume)

    await run_extract(
        scenes_dir=scenes_dir,
        out_dir=out_dir,
        model=model,
        concurrency=concurrency,
        timeout=timeout,
        resume=resume,
    )

if __name__ == "__main__":
    asyncio.run(main())
