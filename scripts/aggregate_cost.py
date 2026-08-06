#!/usr/bin/env python3
"""Aggregate the per-call cost ledger (T_MEM_COST_LOG JSONL) into cost tables.

Reads one JSON row per successful LLM completion and emits:
  * cost_by_substage.csv   - (stage, call_site) -> calls / prompt / completion / total / mean
  * cost_by_stage.csv      - stage -> calls / prompt / completion / total
  * cost_by_conv.csv       - conv_id -> calls / prompt / completion / total
  * cost_by_conv_substage.csv - (conv_id, stage, call_site) fine grid
  * summary.json           - grand total + per-conversation mean/std/min/max
  * cost_report.md         - a ready-to-paste Markdown summary for the rebuttal

Usage:
  python3 scripts/aggregate_cost.py --ledger <path/llm_calls.jsonl> [--out-dir <dir>]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path


def _load(ledger_path: Path):
    rows = []
    with open(ledger_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                continue
    return rows


def _agg(rows, keyfn):
    out = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "total": 0})
    for r in rows:
        k = keyfn(r)
        b = out[k]
        b["calls"] += 1
        b["prompt"] += int(r.get("prompt_tokens") or 0)
        b["completion"] += int(r.get("completion_tokens") or 0)
        b["total"] += int(r.get("total_tokens") or 0)
    return out


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# Human-readable ordering of the construction sub-stages.
_SUBSTAGE_ORDER = [
    "stage1.boundary",
    "stage1.scene_memory",
    "stage2.topic.new",
    "stage2.topic.match",
    "stage2.topic.update",
    "stage2.item",
    "stage4.scene_trigger",
    "entity_bridge_trigger.item",
    "stage7.persona",
]


def _order_key(call_site):
    try:
        return _SUBSTAGE_ORDER.index(call_site)
    except ValueError:
        return len(_SUBSTAGE_ORDER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True, help="Path to llm_calls.jsonl")
    ap.add_argument("--out-dir", default=None, help="Output dir (default: ledger's dir)")
    args = ap.parse_args()

    ledger_path = Path(args.ledger).resolve()
    if not ledger_path.exists():
        raise SystemExit(f"[aggregate_cost] ledger not found: {ledger_path}")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else ledger_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load(ledger_path)
    if not rows:
        raise SystemExit(f"[aggregate_cost] ledger is empty: {ledger_path}")

    n_calls = len(rows)
    conv_ids = sorted({str(r.get("conv_id")) for r in rows if r.get("conv_id") is not None})
    n_conv = len(conv_ids)
    n_conv_null = sum(1 for r in rows if r.get("conv_id") is None)
    token_sources = defaultdict(int)
    for r in rows:
        token_sources[r.get("token_source", "?")] += 1

    # ---- by (stage, call_site) ----
    by_sub = _agg(rows, lambda r: (r.get("stage", "unknown"), r.get("call_site", "unknown")))
    sub_rows = []
    for (stage, cs), b in by_sub.items():
        mean_total = b["total"] / b["calls"] if b["calls"] else 0
        sub_rows.append([stage, cs, b["calls"], b["prompt"], b["completion"], b["total"], round(mean_total, 1)])
    sub_rows.sort(key=lambda x: (_order_key(x[1]), x[0]))
    _write_csv(out_dir / "cost_by_substage.csv",
               ["stage", "call_site", "calls", "prompt_tokens", "completion_tokens", "total_tokens", "mean_total_per_call"],
               sub_rows)

    # ---- by stage ----
    by_stage = _agg(rows, lambda r: r.get("stage", "unknown"))
    stage_rows = [[s, b["calls"], b["prompt"], b["completion"], b["total"]] for s, b in by_stage.items()]
    stage_rows.sort(key=lambda x: x[0])
    _write_csv(out_dir / "cost_by_stage.csv",
               ["stage", "calls", "prompt_tokens", "completion_tokens", "total_tokens"], stage_rows)

    # ---- by conv ----
    by_conv = _agg(rows, lambda r: str(r.get("conv_id")))
    conv_rows = [[c, b["calls"], b["prompt"], b["completion"], b["total"]] for c, b in by_conv.items()]
    conv_rows.sort(key=lambda x: (len(x[0]), x[0]))
    _write_csv(out_dir / "cost_by_conv.csv",
               ["conv_id", "calls", "prompt_tokens", "completion_tokens", "total_tokens"], conv_rows)

    # ---- by (conv, stage, call_site) fine grid ----
    by_cs = _agg(rows, lambda r: (str(r.get("conv_id")), r.get("stage", "unknown"), r.get("call_site", "unknown")))
    csrows = [[c, s, cs, b["calls"], b["prompt"], b["completion"], b["total"]] for (c, s, cs), b in by_cs.items()]
    csrows.sort(key=lambda x: (len(x[0]), x[0], _order_key(x[2])))
    _write_csv(out_dir / "cost_by_conv_substage.csv",
               ["conv_id", "stage", "call_site", "calls", "prompt_tokens", "completion_tokens", "total_tokens"], csrows)

    # ---- per-conversation distribution (over real conv ids only) ----
    per_conv_total = [by_conv[c]["total"] for c in conv_ids]
    per_conv_calls = [by_conv[c]["calls"] for c in conv_ids]
    per_conv_prompt = [by_conv[c]["prompt"] for c in conv_ids]
    per_conv_completion = [by_conv[c]["completion"] for c in conv_ids]

    def _stats(xs):
        if not xs:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "sum": 0}
        return {
            "mean": round(statistics.mean(xs), 1),
            "std": round(statistics.pstdev(xs), 1) if len(xs) > 1 else 0.0,
            "min": min(xs), "max": max(xs), "sum": sum(xs),
        }

    grand = {
        "prompt": sum(int(r.get("prompt_tokens") or 0) for r in rows),
        "completion": sum(int(r.get("completion_tokens") or 0) for r in rows),
        "total": sum(int(r.get("total_tokens") or 0) for r in rows),
        "calls": n_calls,
    }

    summary = {
        "ledger": str(ledger_path),
        "n_calls": n_calls,
        "n_conversations": n_conv,
        "n_calls_with_null_conv": n_conv_null,
        "token_source_counts": dict(token_sources),
        "model": rows[0].get("model"),
        "encoder": rows[0].get("enc"),
        "grand_total": grand,
        "per_conversation": {
            "total_tokens": _stats(per_conv_total),
            "calls": _stats(per_conv_calls),
            "prompt_tokens": _stats(per_conv_prompt),
            "completion_tokens": _stats(per_conv_completion),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- Markdown report ----
    md = []
    md.append("# T-Mem Construction Cost (write-time LLM)\n")
    md.append(f"- Model (memory_build): **{rows[0].get('model')}**  ")
    md.append(f"- Conversations: **{n_conv}**  |  Total LLM calls: **{n_calls}**  |  "
              f"token source: {dict(token_sources)} (enc={rows[0].get('enc')})\n")
    md.append(f"- Rows with unattributed conv_id: {n_conv_null}\n")

    md.append("\n## Per sub-stage (summed over all conversations)\n")
    md.append("| Stage | Sub-stage (call_site) | LLM calls | Input tokens | Output tokens | Total tokens | Mean tot/call |")
    md.append("|---|---|--:|--:|--:|--:|--:|")
    for r in sub_rows:
        md.append(f"| {r[0]} | {r[1]} | {r[2]:,} | {r[3]:,} | {r[4]:,} | {r[5]:,} | {r[6]:,.1f} |")
    md.append(f"| **TOTAL** |  | **{grand['calls']:,}** | **{grand['prompt']:,}** | "
              f"**{grand['completion']:,}** | **{grand['total']:,}** |  |")

    md.append("\n## Per stage\n")
    md.append("| Stage | LLM calls | Input tokens | Output tokens | Total tokens |")
    md.append("|---|--:|--:|--:|--:|")
    for r in stage_rows:
        md.append(f"| {r[0]} | {r[1]:,} | {r[2]:,} | {r[3]:,} | {r[4]:,} |")

    pc = summary["per_conversation"]
    md.append("\n## Per conversation (mean +/- std over %d conversations)\n" % n_conv)
    md.append("| Metric | Mean | Std | Min | Max |")
    md.append("|---|--:|--:|--:|--:|")
    md.append(f"| LLM calls / conv | {pc['calls']['mean']:,} | {pc['calls']['std']:,} | {pc['calls']['min']:,} | {pc['calls']['max']:,} |")
    md.append(f"| Input tokens / conv | {pc['prompt_tokens']['mean']:,} | {pc['prompt_tokens']['std']:,} | {pc['prompt_tokens']['min']:,} | {pc['prompt_tokens']['max']:,} |")
    md.append(f"| Output tokens / conv | {pc['completion_tokens']['mean']:,} | {pc['completion_tokens']['std']:,} | {pc['completion_tokens']['min']:,} | {pc['completion_tokens']['max']:,} |")
    md.append(f"| Total tokens / conv | {pc['total_tokens']['mean']:,} | {pc['total_tokens']['std']:,} | {pc['total_tokens']['min']:,} | {pc['total_tokens']['max']:,} |")

    with open(out_dir / "cost_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"\n[aggregate_cost] wrote CSVs + summary.json + cost_report.md to {out_dir}")


if __name__ == "__main__":
    main()
