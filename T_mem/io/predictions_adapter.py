"""Build benchmark-format ``predictions.json`` from pipeline ``responses.json`` + raw LoCoMo."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from T_mem.config import MODELS

_logger = logging.getLogger("T_mem.predictions_adapter")


LOCOMO_CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "common-sense",
    4: "single-hop",
    5: "adversarial",
}


def _parse_evidence_list(raw_evidence):
    if raw_evidence is None:
        return []
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
    lines = []
    for evid in evidence_list:
        try:
            session_id, turn_id = evid.split(":")
            session_idx = int(session_id.replace("D", ""))
            turn_idx = int(turn_id)
            session_key = f"session_{session_idx}"
            turns = conversation.get(session_key, []) or []
            if 0 <= turn_idx - 1 < len(turns):
                turn = turns[turn_idx - 1]
                speaker = turn.get("speaker", "Unknown")
                text = turn.get("text", "")
                blip_caption = (turn.get("blip_caption") or "").strip()
                if blip_caption:
                    lines.append(f"{speaker}: {text} [shared image: {blip_caption}]")
                else:
                    lines.append(f"{speaker}: {text}")
            else:
                lines.append(f"[{evid}] [Missing turn]")
        except Exception:
            lines.append(f"[{evid}] [Parse error]")
    return "\n".join(lines)


def build_predictions(
    locomo_path,
    responses_path,
    output_path,
    qa_model: str = None,
) -> int:
    locomo_path = Path(locomo_path)
    responses_path = Path(responses_path)
    output_path = Path(output_path)

    if qa_model is None:
        qa_model = MODELS["locomo_qa"]

    with open(locomo_path, "r", encoding="utf-8") as f:
        locomo_data = json.load(f)

    responses_by_conv = {}
    if responses_path.exists():
        with open(responses_path, "r", encoding="utf-8") as f:
            responses_by_conv = json.load(f)
    else:
        _logger.warning("responses.json not found at %s", responses_path)

    records = []
    missing_answers = 0

    for conv_idx, item in enumerate(locomo_data):
        conversation = item.get("conversation") or {}
        qa_list = item.get("qa") or []
        group_id = f"locomo_exp_user_{conv_idx}"
        conv_responses = responses_by_conv.get(group_id, []) or []
        by_question = {}
        for r in conv_responses:
            q = r.get("question")
            if q is not None:
                by_question[q.strip()] = r

        for qa in qa_list:
            question = (qa.get("question") or "").strip()
            cat_id = qa.get("category")
            category_str = LOCOMO_CATEGORY_NAMES.get(cat_id, f"category_{cat_id}")

            if cat_id == 5 or category_str == "adversarial":
                continue

            evidence_list = _parse_evidence_list(qa.get("evidence"))
            evidence_text = _evidence_to_text(conversation, evidence_list)

            matched = by_question.get(question)
            if matched is not None:
                prediction = (matched.get("answer") or "").strip()
            else:
                prediction = ""
                missing_answers += 1

            records.append({
                "question_input": question,
                "evidence": evidence_text,
                "category": category_str,
                "ground_truth": qa.get("answer", "") or "",
                "prediction": prediction,
                "model": qa_model,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    _logger.info(
        "Wrote %d predictions to %s (%d without pipeline answer)",
        len(records), output_path, missing_answers,
    )
    return len(records)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo-file", required=True)
    ap.add_argument("--responses-file", required=True)
    ap.add_argument("--output-file", required=True)
    ap.add_argument("--qa-model", default=MODELS["locomo_qa"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    n = build_predictions(args.locomo_file, args.responses_file, args.output_file,
                          qa_model=args.qa_model)
    print(f"[predictions_adapter] wrote {n} records -> {args.output_file}")
