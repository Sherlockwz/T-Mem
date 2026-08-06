"""Stage 5: per-QA top-K scene builder via three-channel RRF fusion.

For each (conv_id, question) pair, fuses three cosine-similarity rankings
(dialogue / scene attributes / horizon channels) with RRF and writes the top-K scene_ids per QA.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from T_mem.config import ExperimentConfig  # noqa: E402
from T_mem.prompts.trigger_prompts import SCENE_TRIGGER_KEYS, HORIZON_TRIGGER_KEYS, N_TURNS_SKIP  # noqa: E402

DEFAULT_TOPK = 10
DEFAULT_RRF_K = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("T_mem.bgem3_embedding").setLevel(logging.WARNING)
log = logging.getLogger("stage5.build_topk")


def _scene_dialogue(sc: dict) -> str:
    lines = []
    for m in sc.get("original_data", []) or []:
        spk = (m.get("speaker_name") or m.get("speaker") or "?").strip()
        cnt = (m.get("content") or "").strip()
        if cnt:
            lines.append(f"{spk}: {cnt}")
    return "\n".join(lines)

def _collect_scene_horizon(trig_rec: dict) -> tuple[list[str], list[str]]:
    scene_list: list[str] = []
    horizon_list: list[str] = []
    if not trig_rec or trig_rec.get("status") != "ok":
        return scene_list, horizon_list
    sa = trig_rec.get("scene_attributes") or {}
    for k in SCENE_TRIGGER_KEYS:
        v = sa.get(k)
        if isinstance(v, str) and v.strip():
            scene_list.append(v.strip())
    hc = trig_rec.get("horizon_channels") or {}
    for k in HORIZON_TRIGGER_KEYS:
        sub = hc.get(k) or {}
        sent = sub.get("sent")
        if isinstance(sent, str) and sent.strip():
            horizon_list.append(sent.strip())
    return scene_list, horizon_list

def _cos(q: np.ndarray, mat: np.ndarray) -> np.ndarray:
    if mat.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    num = mat @ q
    den = np.linalg.norm(mat, axis=1) * (np.linalg.norm(q) + 1e-9)
    den = np.where(den == 0, 1e-9, den)
    return num / den

def _embed_batch(emb, texts: list[str], batch: int = 64) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        out.extend(emb.embed(texts[i : i + batch]))
    return np.array(out, dtype=np.float32)


def prepare_conv(
    conv_id: int,
    scenes_path: Path,
    triggers_path: Path,
    emb,
) -> dict:
    """Returns:
      {
        "scenes_order": [sid, ...],
        "scenes": {
           sid: {
              "n_turns": int,
              "trigger_status": str,
              "dlg_vec": np.ndarray(dim,) or None,
              "l2_mat":  np.ndarray(n_l2, dim),
              "l3_mat":  np.ndarray(n_l3, dim),
           },
           ...
        }
      }
    """
    scs_raw = json.load(scenes_path.open("r", encoding="utf-8"))
    trig_payload = json.load(triggers_path.open("r", encoding="utf-8"))
    trig_map = trig_payload.get("scenes", {}) or {}

    all_texts: list[str] = []
    entries: list[dict] = []
    for sc in scs_raw:
        sid = sc["scene_id"]
        dlg = _scene_dialogue(sc)
        n_turns = len(sc.get("original_data", []) or [])
        trig_rec = trig_map.get(sid, {}) or {}
        status = trig_rec.get("status")
        # Hard filter: n_turns>N_TURNS_SKIP or non-ok trigger → drop Scene/Horizon.
        if n_turns > N_TURNS_SKIP or status != "ok":
            scene_list, horizon_list = [], []
        else:
            scene_list, horizon_list = _collect_scene_horizon(trig_rec)

        entry = {
            "scene_id": sid,
            "n_turns": n_turns,
            "trigger_status": status,
            "dlg_idx": None,
            "scene_attr_idxs": [],
            "horizon_channel_idxs": [],
        }
        if dlg.strip():
            entry["dlg_idx"] = len(all_texts)
            all_texts.append(dlg)
        for v in scene_list:
            entry["scene_attr_idxs"].append(len(all_texts))
            all_texts.append(v)
        for v in horizon_list:
            entry["horizon_channel_idxs"].append(len(all_texts))
            all_texts.append(v)
        entries.append(entry)

    log.info("[conv_%d] embedding %d texts over %d scenes ...",
             conv_id, len(all_texts), len(scs_raw))
    t0 = time.perf_counter()
    all_vecs = _embed_batch(emb, all_texts)
    log.info("[conv_%d] embed done in %.1fs (%d x %d)",
             conv_id, time.perf_counter() - t0,
             all_vecs.shape[0] if all_vecs.ndim == 2 else 0,
             all_vecs.shape[1] if all_vecs.ndim == 2 else 0)

    sc_vecs: dict[str, dict] = {}
    order: list[str] = []
    for e in entries:
        sid = e["scene_id"]
        order.append(sid)
        dlg_vec = all_vecs[e["dlg_idx"]] if e["dlg_idx"] is not None else None
        scene_attr_mat = (
            np.stack([all_vecs[i] for i in e["scene_attr_idxs"]], axis=0)
            if e["scene_attr_idxs"] else np.zeros((0, 0), dtype=np.float32)
        )
        horizon_channel_mat = (
            np.stack([all_vecs[i] for i in e["horizon_channel_idxs"]], axis=0)
            if e["horizon_channel_idxs"] else np.zeros((0, 0), dtype=np.float32)
        )
        sc_vecs[sid] = {
            "n_turns": e["n_turns"],
            "trigger_status": e["trigger_status"],
            "dlg_vec": dlg_vec,
            "scene_attr_mat": scene_attr_mat,
            "horizon_channel_mat": horizon_channel_mat,
        }
    return {"scenes_order": order, "scenes": sc_vecs}


def score_one_query(
    q_vec: np.ndarray,
    prep: dict,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sid in prep["scenes_order"]:
        sc = prep["scenes"][sid]
        b = 0.0
        if sc["dlg_vec"] is not None:
            b = float(_cos(q_vec, sc["dlg_vec"][np.newaxis, :])[0])
        scene = 0.0
        if sc["scene_attr_mat"].shape[0] > 0:
            scene = float(_cos(q_vec, sc["scene_attr_mat"]).max())
        horizon = 0.0
        if sc["horizon_channel_mat"].shape[0] > 0:
            horizon = float(_cos(q_vec, sc["horizon_channel_mat"]).max())
        out[sid] = {"b": b, "scene": scene, "horizon": horizon}
    return out

def rrf_fuse_topk(
    per_scene_scores: dict[str, dict],
    topk: int,
    k: int = DEFAULT_RRF_K,
) -> list[tuple[str, float]]:
    """RRF fusion of three ranklists (b, scene, horizon).

    score(sc) = Σ_c 1/(k + rank_c(sc)), ties broken by scene_id asc.
    """
    sids = list(per_scene_scores.keys())
    if not sids:
        return []

    def _rank(channel: str) -> dict[str, int]:
        ordered = sorted(sids, key=lambda s: (-per_scene_scores[s][channel], s))
        return {s: i + 1 for i, s in enumerate(ordered)}

    rk_b = _rank("b")
    rk_scene = _rank("scene")
    rk_horizon = _rank("horizon")

    fused: list[tuple[str, float]] = []
    for sid in sids:
        s = 1.0 / (k + rk_b[sid]) + 1.0 / (k + rk_scene[sid]) + 1.0 / (k + rk_horizon[sid])
        fused.append((sid, s))
    fused.sort(key=lambda x: (-x[1], x[0]))
    return fused[:topk]


def build_topk(
    scenes_dir: Path,
    triggers_dir: Path,
    locomo_file: Path,
    out_file: Path,
    topk: int = DEFAULT_TOPK,
    rrf_k: int = DEFAULT_RRF_K,
    exclude_cat5: bool = True,
) -> dict[str, Any]:
    """Build per-QA top-K JSON. Returns the in-memory payload (also written to disk)."""
    scenes_dir = Path(scenes_dir).resolve()
    triggers_dir = Path(triggers_dir).resolve()
    locomo_file = Path(locomo_file).resolve()
    out_file = Path(out_file).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not scenes_dir.exists():
        raise SystemExit(f"scenes dir missing: {scenes_dir}")
    if not triggers_dir.exists():
        raise SystemExit(f"triggers dir missing: {triggers_dir}")
    if not locomo_file.exists():
        raise SystemExit(f"locomo file missing: {locomo_file}")

    from T_mem.llm.bgem3_provider import EmbeddingProvider
    emb = EmbeddingProvider(timeout=60, max_retries=5)

    locomo = json.load(locomo_file.open("r", encoding="utf-8"))
        # Conv iteration order follows locomo10.json list order, matching stage6
    # `for i, conversation_data in enumerate(dataset)` exactly.
    n_conv = len(locomo)
    log.info("locomo convs=%d", n_conv)

    preps: dict[int, dict] = {}
    for cid in range(n_conv):
        sc_path = scenes_dir / f"scene_list_conv_{cid}.json"
        trig_path = triggers_dir / f"triggers_conv_{cid}.json"
        if not sc_path.exists():
            raise SystemExit(f"missing {sc_path}")
        if not trig_path.exists():
            raise SystemExit(f"missing {trig_path}")
        preps[cid] = prepare_conv(cid, sc_path, trig_path, emb)

    per_qa_out: list[dict] = []
    total_qa = 0
    skipped_cat5 = 0
    t_qa = time.perf_counter()
    for cid, conv_data in enumerate(locomo):
        qas = conv_data.get("qa", []) or []
        queries: list[tuple[int, str]] = []
        for qidx, qa in enumerate(qas):
            q_text = qa.get("question")
            if not q_text:
                continue
            if exclude_cat5 and qa.get("category") == 5:
                skipped_cat5 += 1
                continue
            queries.append((qidx, q_text))
        if not queries:
            continue
        qtexts = [q for _, q in queries]
        log.info("[conv_%d] embedding %d queries ...", cid, len(qtexts))
        qvecs = _embed_batch(emb, qtexts)

        for (qidx, qtext), qv in zip(queries, qvecs):
            per_sc = score_one_query(qv, preps[cid])
            topk_pairs = rrf_fuse_topk(per_sc, topk=topk, k=rrf_k)
            per_qa_out.append({
                "conv_id": cid,
                "qa_index": qidx,
                "question": qtext,
                "category": conv_data["qa"][qidx].get("category"),
                "topk": [sid for sid, _ in topk_pairs],
            })
            total_qa += 1

    log.info("per-qa scoring done in %.1fs | total=%d  skipped_cat5=%d",
             time.perf_counter() - t_qa, total_qa, skipped_cat5)

    payload = {
        "meta": {
            "scenes_dir": str(scenes_dir),
            "triggers_dir": str(triggers_dir),
            "locomo_file": str(locomo_file),
            "topk": topk,
            "rrf_k": rrf_k,
            "n_turns_skip_threshold": N_TURNS_SKIP,
            "exclude_cat5": bool(exclude_cat5),
            "n_conv": n_conv,
            "n_qa": total_qa,
            "n_skipped_cat5": skipped_cat5,
        },
        "per_qa": per_qa_out,
    }
    out_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    log.info("wrote %s", out_file)
    return payload

def main() -> None:
    """Default entry: read paths from ExperimentConfig.

    Inputs  : <experiment_dir>/scenes/scene_list_conv_*.json
              <experiment_dir>/scene_horizon_triggers/triggers_conv_*.json
              ExperimentConfig.dataset_path (T_MEM_DATA_FILE)
    Output  : <experiment_dir>/scene_horizon_topk_per_qa.json

    Env overrides (optional):
      T_MEM_SCENE_HORIZON_TOPK    default 10
      T_MEM_SCENE_HORIZON_RRF_K   default 30
    """
    config = ExperimentConfig()
    scenes_dir = config.scenes_dir()
    triggers_dir = config.experiment_dir() / "scene_horizon_triggers"
    locomo_file = Path(config.dataset_path)
    out_file = config.experiment_dir() / "scene_horizon_topk_per_qa.json"

    topk = int(os.environ.get("T_MEM_SCENE_HORIZON_TOPK", str(DEFAULT_TOPK)))
    rrf_k = int(os.environ.get("T_MEM_SCENE_HORIZON_RRF_K", str(DEFAULT_RRF_K)))

    log.info("[stage5/topk] experiment_dir=%s", config.experiment_dir())
    log.info("[stage5/topk] scenes_dir=%s", scenes_dir)
    log.info("[stage5/topk] triggers_dir=%s", triggers_dir)
    log.info("[stage5/topk] locomo_file=%s", locomo_file)
    log.info("[stage5/topk] out_file=%s", out_file)
    log.info("[stage5/topk] topk=%d rrf_k=%d", topk, rrf_k)

    build_topk(
        scenes_dir=scenes_dir,
        triggers_dir=triggers_dir,
        locomo_file=locomo_file,
        out_file=out_file,
        topk=topk,
        rrf_k=rrf_k,
    )

if __name__ == "__main__":
    main()
