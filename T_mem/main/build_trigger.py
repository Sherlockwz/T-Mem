"""Build Entity/Bridge trigger graph and embeddings from an existing T_mem experiment dir."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _ensure_paths():
    here = Path(__file__).resolve()
    main_dir = here.parent                  # .../T_mem/main
    t_mem_root = main_dir.parent            # .../T_mem
    project_root = t_mem_root.parent        # .../T-Mem
    for p in (str(t_mem_root), str(project_root)):
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_paths()


# Bootstrap T_mem providers (env overrides + failure logger) BEFORE importing them
from T_mem.bootstrap import patch_providers  # noqa: E402

patch_providers()


from T_mem.llm.llm_provider import LLMProvider  # noqa: E402
from T_mem.llm.embedding_provider import (  # noqa: E402
    EmbeddingProvider,
)

from T_mem.extractors.trigger_extractor import TriggerExtractor, ExtractorConfig  # noqa: E402
from T_mem.index.trigger_embedder import embed_trigger_graph_triview  # noqa: E402
from T_mem.config import MODELS  # noqa: E402


def load_items_from_memory_graph(mg_path: Path) -> List[Dict[str, Any]]:
    """Read a memory_graph_conv_*.json and return a flat list of item dicts."""
    with open(mg_path, "r", encoding="utf-8") as f:
        mg = json.load(f)
    items_blob = mg.get("items") or {}
    out: List[Dict[str, Any]] = []
    for iid, it in items_blob.items():
        if not isinstance(it, dict):
            continue
        content = (it.get("content") or "").strip()
        if not content:
            continue
        out.append(
            {
                "item_id": iid,
                "content": content,
                "temporal": (it.get("temporal") or "").strip(),
                "keywords": it.get("keywords") or [],
            }
        )
    return out


def parse_conv_list(only_convs: Optional[str], all_paths: List[Path]) -> List[int]:
    """Parse --only-convs (e.g. '0', '0,3,7', '0-4') against actual files."""
    available = {
        int(p.stem.replace("memory_graph_conv_", "")): p for p in all_paths
    }
    if not only_convs:
        return sorted(available.keys())

    wanted: List[int] = []
    for tok in only_convs.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            wanted.extend(range(int(a), int(b) + 1))
        else:
            wanted.append(int(tok))
    return sorted(set(wanted) & set(available.keys()))


async def build_for_conv(
    conv_id: int,
    mg_path: Path,
    output_dir: Path,
    llm_provider: LLMProvider,
    embed_provider: EmbeddingProvider,
    extractor_cfg: ExtractorConfig,
) -> Dict[str, Any]:
    """Run Phase 1~4 for a single conv. Returns build stats dict."""

    def log(msg: str):
        print(f"[conv {conv_id}] {msg}", flush=True)

    t_start = time.time()
    from T_mem.utils.cost_ledger import set_conv as _cost_set_conv
    _cost_set_conv(conv_id)
    log(f"Loading items from {mg_path.name}")
    items = load_items_from_memory_graph(mg_path)
    log(f"  {len(items)} items loaded")

    if not items:
        log("  no items, skipping")
        return {"conv_id": conv_id, "skipped": True, "reason": "no items"}

    # ---- Phase 1-3: extraction ----
    extractor = TriggerExtractor(
        llm_provider=llm_provider,
        config=extractor_cfg,
        logger=log,
    )
    t_extract = time.time()
    graph, stats = await extractor.extract(items, conv_id=conv_id)
    extract_elapsed = time.time() - t_extract
    log(f"Extraction done in {extract_elapsed:.1f}s")

    # ---- Phase 4: tri-view embedding ----
    t_embed = time.time()
    emb_concept, emb_bridge, emb_joint = embed_trigger_graph_triview(
        graph=graph,
        embedding_provider=embed_provider,
        batch_size=32,
        logger=log,
    )
    embed_elapsed = time.time() - t_embed
    log(f"Embedding done in {embed_elapsed:.1f}s")

    graph_path = output_dir / f"entity_bridge_graph_conv_{conv_id}.json"
    emb_path = output_dir / f"entity_bridge_embeddings_conv_{conv_id}.npz"
    stats_path = output_dir / f"build_stats_conv_{conv_id}.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    graph.save(graph_path)
    graph.save_embeddings_triview(emb_path, emb_concept, emb_bridge, emb_joint)

    total_elapsed = time.time() - t_start
    build_stats = {
        "conv_id": conv_id,
        "n_items_input": len(items),
        "extraction_stats": stats.as_dict(),
        "graph_stats": graph.stats(),
        "n_embeddings": {
            "concept": len(emb_concept),
            "bridge": len(emb_bridge),
            "joint": len(emb_joint),
        },
        "elapsed_sec": {
            "extract": round(extract_elapsed, 1),
            "embed": round(embed_elapsed, 1),
            "total": round(total_elapsed, 1),
        },
        "output_paths": {
            "graph": str(graph_path),
            "embeddings": str(emb_path),
        },
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(build_stats, f, ensure_ascii=False, indent=2)

    log(f"Saved graph → {graph_path.name}")
    log(f"Saved embeddings → {emb_path.name}")
    log(f"Saved stats → {stats_path.name}")
    log(f"Conv {conv_id} total: {total_elapsed:.1f}s")
    return build_stats


async def _amain(args):
    memory_dir = Path(args.memory_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    mg_dir = memory_dir / "memory_graphs"
    if not mg_dir.is_dir():
        raise FileNotFoundError(f"memory_graphs dir not found: {mg_dir}")
    all_paths = sorted(mg_dir.glob("memory_graph_conv_*.json"))
    if not all_paths:
        raise FileNotFoundError(f"no memory_graph_conv_*.json under {mg_dir}")

    conv_ids = parse_conv_list(args.only_convs, all_paths)
    if not conv_ids:
        raise ValueError(f"no matching convs for --only-convs={args.only_convs}")

    print(f"[build] memory_dir = {memory_dir}")
    print(f"[build] output_dir = {output_dir}")
    print(f"[build] convs      = {conv_ids}")

    llm_provider = LLMProvider(
        model=args.llm_model,
        json_max_retries=args.json_retries,
        temperature=0.0,
    )
    embed_provider = EmbeddingProvider(model_name="bge-m3")

    cfg = ExtractorConfig(
        entity_bridge_count_per_item=args.entity_bridge_per_item,
        item_conf_threshold=args.item_conf_threshold,
        entity_bridge_dedup_ratio=args.entity_bridge_dedup_ratio,
        max_concurrent=args.max_concurrent,
        extract_retries=args.extract_retries,
    )
    print(f"[build] extractor cfg = {cfg}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, Any]] = []

    for conv_id in conv_ids:
        mg_path = mg_dir / f"memory_graph_conv_{conv_id}.json"
        try:
            stats = await build_for_conv(
                conv_id=conv_id,
                mg_path=mg_path,
                output_dir=output_dir,
                llm_provider=llm_provider,
                embed_provider=embed_provider,
                extractor_cfg=cfg,
            )
            summaries.append(stats)
        except Exception as e:  # noqa: BLE001
            print(f"[conv {conv_id}] FAILED: {type(e).__name__}: {e}", flush=True)
            summaries.append(
                {"conv_id": conv_id, "failed": True, "error": f"{type(e).__name__}: {e}"}
            )

    summary_path = output_dir / "build_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "memory_dir": str(memory_dir),
                "conv_ids": conv_ids,
                "extractor_cfg": cfg.__dict__,
                "conv_summaries": summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n[build] ALL DONE. Summary → {summary_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build trigger graph + embeddings")
    ap.add_argument(
        "--memory-dir",
        required=True,
        help="T_mem experiment directory containing a memory_graphs/ subdir",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Where to write entity_bridge_graph_conv_*.json and entity_bridge_embeddings_conv_*.npz",
    )
    ap.add_argument(
        "--only-convs",
        default=None,
        help="Comma-separated conv ids or ranges, e.g. '0' or '0,3' or '0-4'",
    )

    # NOTE: --llm-model default comes from T_mem.config.MODELS (single source of truth);
    # production shell pipeline does NOT pass it.
    ap.add_argument("--llm-model", default=MODELS["memory_build"])
    ap.add_argument("--json-retries", type=int, default=5)

    ap.add_argument("--entity-bridge-per-item", type=int, default=5,
                    help="Target number of entity/bridge triggers per memory item")
    ap.add_argument("--item-conf-threshold", type=float, default=0.70,
                    help="Min LLM-rated confidence for item-trigger edge")
    ap.add_argument("--entity-bridge-dedup-ratio", type=float, default=0.90,
                    help="rapidfuzz ratio threshold for entity/bridge dedup")
    ap.add_argument("--max-concurrent", type=int, default=14,
                    help="Concurrent LLM calls (matches the default pool cap of 14)")
    ap.add_argument("--extract-retries", type=int, default=3,
                    help="Per-item LLM retry count")
    args = ap.parse_args()
    asyncio.run(_amain(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
