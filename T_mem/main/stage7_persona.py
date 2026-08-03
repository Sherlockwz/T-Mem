# -*- coding: utf-8 -*-
"""CLI entry point for scene-driven persona extraction."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from T_mem.persona.scheduler import build_personas_for_conv

logger = logging.getLogger(__name__)

# Guard concurrent mutation of the shared summary dict from worker threads.
_summary_lock = threading.Lock()


def _discover_conv_ids(scenes_root: str) -> List[int]:
    out: List[int] = []
    try:
        names = os.listdir(scenes_root)
    except Exception as e:
        raise SystemExit(f"[persona-cli] failed to list {scenes_root}: {e}")
    for n in names:
        if not (n.startswith("scene_list_conv_") and n.endswith(".json")):
            continue
        try:
            cid = int(n[len("scene_list_conv_"):-len(".json")])
        except ValueError:
            continue
        out.append(cid)
    return sorted(out)


def _load_scenes(scenes_root: str, conv_id: int) -> List[dict]:
    path = os.path.join(scenes_root, f"scene_list_conv_{conv_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        scenes = json.load(f)
    if not isinstance(scenes, list):
        raise ValueError(f"{path}: expected list at top level, got {type(scenes).__name__}")
    return scenes


def _detect_speakers(scenes: List[dict]) -> Tuple[str, str]:
    """Return (speaker_a, speaker_b) based on first-seen order in utterances."""
    ordered: List[str] = []
    for sc in scenes:
        for u in sc.get("original_data") or []:
            name = (
                u.get("speaker_name")
                or u.get("user_name")
                or u.get("speaker")
                or ""
            )
            if name and name not in ordered:
                ordered.append(name)
            if len(ordered) >= 2:
                break
        if len(ordered) >= 2:
            break
    if len(ordered) < 2:
        raise ValueError(
            f"Could not detect two distinct speakers "
            f"(found: {ordered!r})."
        )
    return ordered[0], ordered[1]


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes-root", required=True,
                    help="Directory containing scene_list_conv_*.json")
    ap.add_argument("--store-root", required=True,
                    help="Output root; qa_conv{n}/<speaker>.json written here")
    ap.add_argument("--convs", default="",
                    help="Comma-separated conv ids (default: all discovered)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Number of conversations processed in parallel (each "
                         "conv stays SERIAL internally). Default 1 = fully sequential.")
    ap.add_argument("--conv-label-prefix", default="",
                    help="Prefix prepended to the `qa_conv{N+1}` directory name, "
                         "defaults to current timestamp. Use '' to auto-generate.")
    ap.add_argument("--conv-id-base", type=int, default=1, choices=[0, 1],
                    help="Index base used in the output directory name: "
                         "1 (default, yields `qa_conv{conv_id+1}`) "
                         "or 0 (`qa_conv{conv_id}`, internal use only).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip LLM calls and file writes; only print the chunk plan.")
    args = ap.parse_args(argv)

    if args.concurrency < 1:
        raise SystemExit(f"[persona-cli] --concurrency must be >= 1 (got {args.concurrency})")

    if not os.path.isdir(args.scenes_root):
        raise SystemExit(f"[persona-cli] scenes-root not a directory: {args.scenes_root}")

    if args.convs.strip():
        try:
            conv_ids = [int(x) for x in args.convs.split(",") if x.strip()]
        except ValueError:
            raise SystemExit(f"[persona-cli] invalid --convs: {args.convs!r}")
    else:
        conv_ids = _discover_conv_ids(args.scenes_root)
        if not conv_ids:
            raise SystemExit(
                f"[persona-cli] no scene_list_conv_*.json under {args.scenes_root}"
            )

    os.makedirs(args.store_root, exist_ok=True)

    # Output layout: <store-root>/<ts>_qa_conv{conv_id+1}/<Speaker>.json
    ts_prefix = args.conv_label_prefix.strip() or time.strftime("%Y%m%d_%H%M%S")

    summary: Dict[int, dict] = {}
    t0 = time.time()

    def _process_one(cid: int) -> None:
        """Handle a single conv end-to-end; safe from worker threads (per-conv state is local)."""
        try:
            from T_mem.utils.cost_ledger import set_conv as _cost_set_conv
            _cost_set_conv(cid)
        except Exception:
            pass
        try:
            scenes = _load_scenes(args.scenes_root, cid)
        except Exception as e:
            logger.warning("[persona-cli] conv_%d load failed: %s", cid, e)
            return

        real_scenes = [sc for sc in scenes if (sc.get("original_data") or [])]
        if not real_scenes:
            logger.warning("[persona-cli] conv_%d has no non-empty scenes, skip", cid)
            return

        try:
            speaker_a, speaker_b = _detect_speakers(real_scenes)
        except Exception as e:
            logger.warning("[persona-cli] conv_%d speaker detection failed: %s", cid, e)
            return

        logger.info(
            "[persona-cli] conv_%d START: %d scenes (%d non-empty), speakers=(%s, %s)",
            cid, len(scenes), len(real_scenes), speaker_a, speaker_b,
        )

        t_conv = time.time()
        idx_for_label = cid + 1 if args.conv_id_base == 1 else cid
        conv_label = f"{ts_prefix}_qa_conv{idx_for_label}"
        try:
            pm, chunks = build_personas_for_conv(
                conv_id=cid,
                scenes=real_scenes,
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                store_root=args.store_root,
                conv_label=conv_label,
                dry_run=args.dry_run,
            )
        except Exception as e:
            logger.warning("[persona-cli] conv_%d scheduler failed: %s", cid, e)
            return

        dt = time.time() - t_conv
        chunk_summary = [
            {
                "kind": c.kind,
                "turns": c.turns,
                "scene_count": len(c.scenes),
                "current_time": c.current_time,
            }
            for c in chunks
        ]
        with _summary_lock:
            summary[cid] = {
                "speakers": [speaker_a, speaker_b],
                "conv_label": conv_label,
                "scene_count": len(real_scenes),
                "chunks": chunk_summary,
                "elapsed_sec": round(dt, 2),
            }
        logger.info(
            "[persona-cli] conv_%d DONE in %.1fs: %d chunks (%s)",
            cid, dt, len(chunks),
            ",".join(f"{c.kind}:{c.turns}t" for c in chunks),
        )

    if args.concurrency == 1 or len(conv_ids) == 1:
        for cid in conv_ids:
            _process_one(cid)
    else:
        workers = min(args.concurrency, len(conv_ids))
        logger.info(
            "[persona-cli] processing %d convs with concurrency=%d (per-conv parallel, in-conv serial)",
            len(conv_ids), workers,
        )
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="persona-conv") as ex:
            futures = {ex.submit(_process_one, cid): cid for cid in conv_ids}
            for fut in as_completed(futures):
                cid = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    logger.warning("[persona-cli] conv_%d worker raised: %s", cid, e)

    total = time.time() - t0
    logger.info(
        "[persona-cli] ALL DONE in %.1fs over %d convs (dry_run=%s)",
        total, len(summary), args.dry_run,
    )

    try:
        os.makedirs(args.store_root, exist_ok=True)
        plan_path = os.path.join(
            args.store_root,
            "chunk_plan_dry_run.json" if args.dry_run else "chunk_plan.json",
        )
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("[persona-cli] chunk plan written to %s", plan_path)
    except Exception as e:
        logger.warning("[persona-cli] failed to write chunk plan: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
