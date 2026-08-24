# Source: https://github.com/xjtuleeyf/Locomo-Plus (file: data/unified_input.py)
# Copyright (c) Locomo-Plus authors (xjtuleeyf). Vendored as-is for
# LoCoMo-Plus conversation stitching in T-Mem's stage0_locomo_plus_stitch.py.
"""
Build a unified input dataset for visualization and downstream input.

Merges:
- Locomo: original 5 categories — multi-hop (1), temporal (2), common-sense (3), single-hop (4), adversarial (5).
- Locomo_plus: sixth category — Cognitive (perceptual/memory; no reference answer).

input_prompt build logic (different by source):
- Locomo (5 categories): conversation as-is + question (no insertion).
- Locomo_plus (Cognitive): same as build_conv.py — (1) map A/B to the conversation’s speaker names (map_speaker), (2) parse time_gap and insert cue_dialogue and trigger_query at the correct positions (compute_insertion, events sorted by time); build_context returns the stitched dialogue used as input_prompt.

Output fields per sample:
- input_prompt: full dialogue (+ question for Locomo) text (for inspection only; not saved at final model input).
- trigger: Locomo = question; Cognitive (plus) = trigger_query.
- answer: optional; absent for adversarial and for Cognitive (sixth).
- evidence: human-readable "Speaker: text" (Locomo Dn:k rewritten; Cognitive: cue_dialogue only, A/B→speaker names for judge input).
- category: multi-hop, temporal, common-sense, single-hop, adversarial, or Cognitive.
- time_gap: only for Cognitive; from plus item (e.g. "two weeks later").

Inspection: set SAVE_JSON_FOR_INSPECTION = True to write a JSON file for manual check. After verification, set to False so the script only produces the in-memory "ready-to-input" form (no file save).
"""

import json
from pathlib import Path

# Set to False after you have inspected the JSON; then no file is written, only in-memory samples.
SAVE_JSON_FOR_INSPECTION = True
INSPECTION_JSON_NAME = "unified_input_samples_v2.json"


def _parse_evidence_list(raw_evidence):
    """Normalize evidence to list of 'Dn:k' tokens. Handles 'D8:6; D9:17'."""
    if isinstance(raw_evidence, str):
        raw_evidence = [raw_evidence]
    out = []
    for ev in raw_evidence:
        for part in str(ev).split(";"):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _evidence_to_text(conversation: dict, evidence_list: list) -> str:
    """Convert list of 'Dn:k' to 'Speaker: text' lines (no Dn:k prefix)."""
    lines = []
    for evid in evidence_list:
        try:
            session_id, turn_id = evid.split(":")
            session_idx = int(session_id.replace("D", ""))
            turn_idx = int(turn_id)
            session_key = f"session_{session_idx}"
            turns = conversation.get(session_key, [])
            if 0 <= turn_idx - 1 < len(turns):
                turn = turns[turn_idx - 1]
                speaker = turn.get("speaker", "Unknown")
                text = turn.get("text", "")
                lines.append(f"{speaker}：{text}")
            else:
                lines.append(f"[{evid}] [Missing turn]")
        except Exception:
            lines.append(f"[{evid}] [Parse error]")
    return "\n".join(lines)


def _build_conversation_context(conversation: dict) -> str:
    """Single conversation as DATE + CONVERSATION lines (same style as task_eval)."""
    sessions = sorted(
        [k for k in conversation.keys() if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.split("_")[-1]),
    )
    context = ""
    for s in sessions:
        date = conversation.get(f"{s}_date_time", "")
        turns = conversation.get(s, [])
        context += f"DATE: {date}\nCONVERSATION:\n"
        for d in turns:
            speaker = d.get("speaker", "?")
            text = (d.get("text") or "").strip()
            if "blip_caption" in d:
                context += f'{speaker} said, "{text}" and shared {d["blip_caption"]}.\n'
            else:
                context += f'{speaker} said, "{text}"\n'
        context += "\n"
    return context


# Locomo original 5 categories (by id in data).
LOCOMO_CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "common-sense",
    4: "single-hop",
    5: "adversarial",
}

# Sixth category: Locomo_plus data (perceptual/memory).
CATEGORY_SIXTH = "Cognitive"


def _process_locomo(locomo_path: str) -> list:
    """Locomo: input_prompt = conversation (no insertion) + question."""
    with open(locomo_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out = []
    for item in raw:
        conv = item.get("conversation") or {}
        qa_list = item.get("qa") or []
        ctx_text = _build_conversation_context(conv)

        for qa in qa_list:
            question = qa.get("question", "")
            cat_id = qa.get("category")
            category = LOCOMO_CATEGORY_NAMES.get(cat_id, f"category_{cat_id}")

            evidence_raw = qa.get("evidence") or []
            evidence_list = _parse_evidence_list(evidence_raw)
            evidence_text = _evidence_to_text(conv, evidence_list)

            # Build: dialogue only + question (no insertion).
            input_prompt = ctx_text.rstrip() + "\n\nQuestion: " + question

            sample = {
                "input_prompt": input_prompt,
                "trigger": question,
                "evidence": evidence_text,
                "category": category,
            }

            if "answer" in qa and qa["answer"] is not None:
                sample["answer"] = qa["answer"]

            out.append(sample)

    return out


def _cue_dialogue_to_evidence(cue_dialogue: str, locomo_item: dict) -> str:
    """
    Cognitive evidence: cue_dialogue only (no trigger_query), with A/B replaced by
    the conversation's speaker names (same as build_conv.map_speaker) for LLM-as-a-judge input.
    """
    from build_conv import parse_ab_dialogue, map_speaker

    conv = locomo_item.get("conversation") or {}
    speaker_a = conv.get("speaker_a", "A")
    speaker_b = conv.get("speaker_b", "B")
    turns = map_speaker(parse_ab_dialogue(cue_dialogue or ""), speaker_a, speaker_b)
    return "\n".join(f'{t["speaker"]}：{t["text"].strip()}' for t in turns if t.get("text"))


def _stitch_dialogue_for_plus(plus_item: dict, locomo_item: dict) -> str:
    """
    Cognitive: input_prompt = stitched dialogue from build_conv.build_context.
    Follows build_conv.py: (1) A/B replaced with conversation’s speaker_a/speaker_b (map_speaker);
    (2) time_gap parsed and cue/query inserted at the right positions (compute_insertion, sort by time).
    We only format the returned 'dialogue' as text (Speaker said, "...").
    """
    from build_conv import build_context

    ctx = build_context(plus_item, locomo_item)
    dialogue = ctx.get("dialogue") or []
    lines = []
    for turn in dialogue:
        speaker = turn.get("speaker", "?")
        text = (turn.get("text") or "").strip()
        lines.append(f'{speaker} said, "{text}"')
    return "\n".join(lines)


def _process_locomo_plus(locomo_plus_path: str, locomo_path: str, plus_sample_size=None) -> list:
    """Cognitive (sixth): stitching via build_conv.build_context; no answer."""
    with open(locomo_plus_path, "r", encoding="utf-8") as f:
        plus_list = json.load(f)
    with open(locomo_path, "r", encoding="utf-8") as f:
        locomo_list = json.load(f)

    if plus_sample_size is not None:
        plus_list = plus_list[:plus_sample_size]

    out = []
    for i, plus in enumerate(plus_list):
        locomo_item = locomo_list[i % len(locomo_list)]
        try:
            input_prompt = _stitch_dialogue_for_plus(plus, locomo_item)
        except Exception:
            input_prompt = ""

        evidence_text = _cue_dialogue_to_evidence(
            plus.get("cue_dialogue", ""),
            locomo_item,
        )
        # Sixth category: Cognitive (Locomo_plus); no answer; include time_gap in output.
        sample = {
            "input_prompt": input_prompt,
            "trigger": plus.get("trigger_query", ""),
            "evidence": evidence_text,
            "category": CATEGORY_SIXTH,
            "time_gap": plus.get("time_gap", ""),
        }
        out.append(sample)

    return out


def build_unified_samples(data_dir=None):
    """Build the unified sample list (ready-to-input form). No file write unless caller does it."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent
    data_dir = Path(data_dir)
    locomo_path = data_dir / "locomo10.json"
    locomo_plus_path = data_dir / "locomo_plus.json"

    samples = []
    if locomo_path.exists():
        samples.extend(_process_locomo(str(locomo_path)))
    if locomo_plus_path.exists():
        samples.extend(_process_locomo_plus(str(locomo_plus_path), str(locomo_path)))
    return samples


def main():
    data_dir = Path(__file__).resolve().parent
    samples = build_unified_samples(data_dir)
    out_path = data_dir / INSPECTION_JSON_NAME

    if SAVE_JSON_FOR_INSPECTION:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(samples)} samples to {out_path} (inspection)")
    else:
        print(f"Built {len(samples)} samples (no file saved; ready-to-input form only)")


if __name__ == "__main__":
    main()
