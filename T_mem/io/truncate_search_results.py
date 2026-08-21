"""Truncate an existing stage6 search_results.json to top-K_scene / top-K_item
and write a new experiment directory that is drop-in compatible with
scripts/eval_locomo.sh --resume <new_dir>.

Rules (matching stage6 format_hierarchical_results):
  - scenes block is written as "[Scene 1] ...\n\n[Scene 2] ...\n\n..." and is
    ordered by stage6 rerank score (desc). Truncating to the first K is
    equivalent to re-running stage6 with scene_top_k=K, item_top_k=K.
  - items block is analogous.

Only two things are rewritten per record:
  (a) record["context"]             : the Scenes / Items sections are shortened
                                       to the first K_scene / K_item entries
  (b) record["hierarchical_results"]: scenes_count / items_count are updated
                                       so the struct stays consistent.

Everything else (retrieval_log, query, ...) is copied as-is.

Besides search_results.json we also materialise a personas/ symlink pointing
to the source run's personas directory so the downstream stage8 persona loader
still works.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# ------------------------------------------------------------
# Regexes (mirror stage6 format_hierarchical_results output)
# ------------------------------------------------------------
# "## Relevant Scenes:\n" ... up to next "## " or EOS.
_SCENES_HEADER_RE = re.compile(r"^##\s*Relevant\s*Scenes\s*:\s*\n", re.M)
_ITEMS_HEADER_RE  = re.compile(r"^##\s*Relevant\s*Items\s*:\s*\n",  re.M)
_NEXT_SECTION_RE  = re.compile(r"^##\s*", re.M)

# A single "[Scene N] ..." / "[Item N] ..." entry starts at a line
# beginning with "[Scene " / "[Item ". Entries are separated by a blank line.
# Inside a single entry we may have "\n  Time: ..." / "\n  Location: ..." lines.


def _slice_entries(block: str, tag: str, keep_k: int) -> str:
    """Given the body of a "## Relevant Scenes:" / "## Relevant Items:"
    section (without the header), keep only the first `keep_k` entries and
    re-number them 1..keep_k.

    The body format produced by stage6 is:
        "[<Tag> 1] ...\n  Time: ...\n\n[<Tag> 2] ...\n\n..."
    with "\n\n" as entry separator. We split on the entry-start marker at
    line start, not on blank lines, so multi-line entries stay intact.
    """
    entry_start_re = re.compile(rf"^\[{re.escape(tag)} \d+\]", re.M)
    # find all start offsets
    starts = [m.start() for m in entry_start_re.finditer(block)]
    if not starts:
        return block.rstrip()
    if keep_k <= 0:
        return f"No relevant {tag.lower()}s found."
    if keep_k >= len(starts):
        kept = starts
    else:
        kept = starts[:keep_k]
    # slice each entry
    entries = []
    for i, s in enumerate(kept):
        e = starts[i + 1] if i + 1 < len(kept) else (
            starts[len(kept)] if len(kept) < len(starts) else len(block)
        )
        chunk = block[s:e].rstrip()
        # re-number
        chunk = re.sub(rf"^\[{re.escape(tag)} \d+\]", f"[{tag} {i+1}]", chunk, count=1)
        entries.append(chunk)
    return "\n\n".join(entries)


def _replace_section(ctx: str, header_re: re.Pattern, tag: str, keep_k: int) -> str:
    """Replace the body between ``header`` and the next ``## `` (or EOS) with
    only the first ``keep_k`` entries, preserving the original surrounding
    whitespace exactly:
      - the header line itself (``## Relevant Xxx:\n``) is kept verbatim;
      - whatever came after the last kept entry in the original (newlines
        before the next section header, or trailing whitespace at EOS) is
        re-attached verbatim.
    This keeps the byte-level shape of the context identical to what stage6
    would emit if it had been run with top_k=keep_k.
    """
    m = header_re.search(ctx)
    if m is None:
        return ctx
    header_end = m.end()
    tail = ctx[header_end:]
    nm = _NEXT_SECTION_RE.search(tail)
    body_end = header_end + nm.start() if nm else len(ctx)
    body = ctx[header_end:body_end]

    # Locate the entries and the trailing whitespace tail (between last entry
    # and the body_end boundary) so we can keep the exact inter-section glue.
    entry_start_re = re.compile(rf"^\[{re.escape(tag)} \d+\]", re.M)
    starts = [mm.start() for mm in entry_start_re.finditer(body)]
    if not starts:
        # nothing to truncate; return ctx unchanged
        return ctx

    # Trailing glue = whatever comes after the LAST original entry up to body_end.
    # stage6 writes entries as "...\n\n...\n\n<last entry>" then the section ends.
    # Find the end of the last entry in the original body.
    entry_ranges = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(body)
        entry_ranges.append((s, e))
    # The trailing glue is body[last_entry_end : ].  In stage6 output this is
    # exactly the bytes that glue the section to the next "## " header (or EOS).
    # Examples observed in src: last item ends without trailing \n; last scene
    # ends with "\n  Time: ...\n" then immediately "## Relevant Items:".
    last_s, last_e = entry_ranges[-1]
    # For entries kept by slicing, use the same per-entry boundary style as
    # stage6: "\n\n" between entries, and whatever trailing glue the ORIGINAL
    # had after its last entry (so "Scene K -> ## Relevant Items" inter-section
    # glue is preserved byte-for-byte).
    if keep_k <= 0:
        new_body = f"No relevant {tag.lower()}s found."
        # Preserve the inter-section glue that followed the last entry so the
        # next "## " header (or EOS) keeps its exact byte layout and still
        # matches at line start.
        last_start = starts[-1]
        tail_after_last = body[last_start:]
        content = tail_after_last.rstrip("\n")
        trailing_glue = tail_after_last[len(content):]
    else:
        keep_k_actual = min(keep_k, len(starts))
        # Build the kept-prefix text by taking the exact byte range from the
        # FIRST kept entry start to the end of the K-th kept entry, where
        # "end of the K-th entry" is defined as:
        #   - if there is a (K+1)-th entry in the original body: the start of
        #     that next entry MINUS the inter-entry "\n\n" separator glue, i.e.
        #     the body offset right after the K-th entry's own trailing content;
        #   - otherwise: the body offset of the K-th entry's end (body[last_e]).
        # stage6's format is "<entry>\n\n<entry>\n\n...<entry>" so the separator
        # is exactly "\n\n". We recover the K-th entry's content end by
        # finding the last "\n\n" before starts[K] (if K < total entries) and
        # ending there; then renumber the first K.
        first_start = starts[0]
        # Determine whether the current section is followed by another "## "
        # header in the ORIGINAL ctx. If it's the LAST section (e.g. items),
        # the native stage6 style has NO trailing "\n" after the final entry;
        # otherwise (e.g. scenes followed by items) we must keep exactly ONE
        # "\n" so "...Time: ...\n## Relevant Items:" holds byte-for-byte.
        is_last_section = (nm is None)
        if keep_k_actual < len(starts):
            next_start = starts[keep_k_actual]
            glue_idx = body.rfind("\n\n", first_start, next_start)
            if glue_idx == -1:
                kept_range_end = next_start
                while kept_range_end > first_start and body[kept_range_end - 1] == "\n":
                    kept_range_end -= 1
                if not is_last_section:
                    kept_range_end += 1  # keep one "\n" as inter-section glue
            else:
                # glue_idx points at the first char of "\n\n". Keep one "\n"
                # for non-last sections, none for the last section.
                kept_range_end = glue_idx + (1 if not is_last_section else 0)
        else:
            kept_range_end = entry_ranges[-1][1]
        kept_raw = body[first_start:kept_range_end]
        # Renumber entries 1..K in-place.
        def _mk_renumber():
            counter = {"n": 0}
            def _sub(_m):
                counter["n"] += 1
                return f"[{tag} {counter['n']}]"
            return _sub
        new_body = re.sub(rf"^\[{re.escape(tag)} \d+\]", _mk_renumber(), kept_raw, flags=re.M)
        # Use the ORIGINAL body's trailing glue (bytes after the last original
        # entry ended) so the boundary with the next "## " header is byte-exact.
        trailing_glue = body[last_e:]

    return ctx[:header_end] + new_body + trailing_glue + ctx[body_end:]


def truncate_context(ctx: str, *, keep_scene: int, keep_item: int) -> str:
    ctx = _replace_section(ctx, _SCENES_HEADER_RE, "Scene", keep_scene)
    ctx = _replace_section(ctx, _ITEMS_HEADER_RE,  "Item",  keep_item)
    return ctx


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True,
                    help="Source experiment dir (must contain search_results.json).")
    ap.add_argument("--dst-dir", required=True,
                    help="New experiment dir to create.")
    ap.add_argument("--keep-scene", type=int, default=None,
                    help="Final scenes kept for QA. Defaults to "
                         "T_mem.config.ExperimentConfig.retrieval_config['final_keep_scene'].")
    ap.add_argument("--keep-item",  type=int, default=None,
                    help="Final items kept for QA. Defaults to "
                         "T_mem.config.ExperimentConfig.retrieval_config['final_keep_item'].")
    args = ap.parse_args()

    if args.keep_scene is None or args.keep_item is None:
        try:
            from T_mem.config import ExperimentConfig as _EC
            rc = _EC.retrieval_config
        except Exception as e:  # pragma: no cover -- fail loud, never silently fall back to magic numbers
            print(
                f"[truncate] FATAL: --keep-scene/--keep-item not given and "
                f"T_mem.config import failed: {e!r}", file=sys.stderr,
            )
            return 2
        if args.keep_scene is None:
            args.keep_scene = int(rc["final_keep_scene"])
        if args.keep_item is None:
            args.keep_item = int(rc["final_keep_item"])
        print(
            f"[truncate] using config defaults: "
            f"keep_scene={args.keep_scene} keep_item={args.keep_item}",
            flush=True,
        )

    src = Path(args.src_dir).resolve()
    dst = Path(args.dst_dir).resolve()
    src_sr = src / "search_results.json"
    if not src_sr.exists():
        print(f"[truncate] FATAL: {src_sr} missing", file=sys.stderr)
        return 2

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "logs").mkdir(parents=True, exist_ok=True)

    # Symlink personas so stage8 persona loader still resolves.
    src_persona = src / "personas"
    dst_persona = dst / "personas"
    if src_persona.exists() and not dst_persona.exists():
        os.symlink(str(src_persona), str(dst_persona))

    # Load + truncate
    print(f"[truncate] loading {src_sr}", flush=True)
    sr = json.loads(src_sr.read_text(encoding="utf-8"))

    n_rec = 0
    n_scene_before = 0
    n_item_before  = 0
    n_scene_after  = 0
    n_item_after   = 0
    SCN_RE = re.compile(r"^\[Scene \d+\]", re.M)
    ITM_RE = re.compile(r"^\[Item \d+\]",  re.M)

    for ukey, recs in sr.items():
        if not isinstance(recs, list):
            continue
        for r in recs:
            if not isinstance(r, dict):
                continue
            ctx = r.get("context") or ""
            n_scene_before += len(SCN_RE.findall(ctx))
            n_item_before  += len(ITM_RE.findall(ctx))
            new_ctx = truncate_context(ctx, keep_scene=args.keep_scene, keep_item=args.keep_item)
            r["context"] = new_ctx
            n_scene_after += len(SCN_RE.findall(new_ctx))
            n_item_after  += len(ITM_RE.findall(new_ctx))

            hr = r.get("hierarchical_results")
            if isinstance(hr, dict):
                hr["scenes_count"] = min(hr.get("scenes_count", args.keep_scene), args.keep_scene)
                hr["items_count"]  = min(hr.get("items_count",  args.keep_item),  args.keep_item)
            n_rec += 1

    out_path = dst / "search_results.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sr, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_path)

    stats = {
        "src_dir": str(src),
        "dst_dir": str(dst),
        "keep_scene": args.keep_scene,
        "keep_item":  args.keep_item,
        "records":    n_rec,
        "avg_scenes_before": (n_scene_before / n_rec) if n_rec else 0.0,
        "avg_scenes_after":  (n_scene_after  / n_rec) if n_rec else 0.0,
        "avg_items_before":  (n_item_before  / n_rec) if n_rec else 0.0,
        "avg_items_after":   (n_item_after   / n_rec) if n_rec else 0.0,
    }
    (dst / "truncate_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[truncate] DONE  records={n_rec}", flush=True)
    print(f"[truncate] scenes: {stats['avg_scenes_before']:.2f} -> {stats['avg_scenes_after']:.2f}", flush=True)
    print(f"[truncate] items : {stats['avg_items_before']:.2f} -> {stats['avg_items_after']:.2f}", flush=True)
    print(f"[truncate] wrote  {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
