"""Convert pipeline predictions.json -> MemOS locomo_responses.json (grouped by user, int categories)."""

import argparse
import json
import os
import sys

CATEGORY_STR_TO_INT = {
    "single-hop": 4,
    "multi-hop": 1,
    "temporal": 2,
    "common-sense": 3,
    "adversarial": 5,
}

def main():
    parser = argparse.ArgumentParser(description="Convert our predictions to MemOS locomo_responses.json format")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to our memory_predictions.json")
    parser.add_argument("--locomo", type=str, required=True,
                        help="Path to locomo10.json (for grouping by conv)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for MemOS-format locomo_responses.json")
    parser.add_argument("--keep-adversarial", action="store_true",
                        help="Keep adversarial (category=5) samples (default: skip, matching MemOS protocol)")
    args = parser.parse_args()

    with open(args.predictions, "r", encoding="utf-8") as f:
        preds = json.load(f)
    with open(args.locomo, "r", encoding="utf-8") as f:
        locomo = json.load(f)

    print(f"Loaded predictions: {len(preds)} items")
    print(f"Loaded locomo10: {len(locomo)} conversations")

    # CRITICAL: index as multimap, not unique-key dict — locomo has 11 literally
    # duplicated questions; a plain dict would drop those duplicate predictions.
    from collections import defaultdict
    pred_by_question = defaultdict(list)
    for p in preds:
        q = p.get("question_input", "").strip()
        pred_by_question[q].append(p)

    n_unique = len(pred_by_question)
    n_dup_keys = sum(1 for v in pred_by_question.values() if len(v) > 1)
    n_total_pred = sum(len(v) for v in pred_by_question.values())
    if n_dup_keys > 0:
        print(f"[info] {n_dup_keys} questions have multiple predictions "
              f"(unique={n_unique}, total={n_total_pred}); consumed in order")

    total_locomo_non_adv = sum(
        1 for item in locomo for q in item.get("qa", []) if q.get("category") != 5
    )
    print(f"locomo non-adversarial QA: {total_locomo_non_adv}")
    print(f"predictions unique questions: {n_unique} (total records {n_total_pred})")

    all_responses = {}
    total_converted = 0
    total_skipped_adv = 0
    total_not_found = 0

    for conv_idx, item in enumerate(locomo):
        group_id = f"locomo_exp_user_{conv_idx}"
        group_responses = []

        for qa in item.get("qa", []):
            question = qa.get("question", "").strip()
            cat_int = qa.get("category", 0)

            if cat_int == 5 and not args.keep_adversarial:
                total_skipped_adv += 1
                continue

            bucket = pred_by_question.get(question)
            if not bucket:
                total_not_found += 1
                if total_not_found <= 5:
                    print(f"  [warn] conv{conv_idx}: no prediction for '{question[:60]}'")
                continue
            pred = bucket.pop(0)

            cat_str = pred.get("category", "")
            if cat_str in CATEGORY_STR_TO_INT:
                cat_int = CATEGORY_STR_TO_INT[cat_str]

            response = {
                "question": question,
                "answer": pred.get("prediction", ""),
                "category": cat_int,
                "golden_answer": pred.get("ground_truth", qa.get("answer", "")),
                "search_context": "",
                "response_duration_ms": 0.0,
                "search_duration_ms": 0.0,
            }
            group_responses.append(response)
            total_converted += 1

        all_responses[group_id] = group_responses
        print(f"  {group_id}: {len(group_responses)} QA")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_responses, f, indent=2, ensure_ascii=False)

    print(f"\n[done] conversion complete:")
    print(f"  converted: {total_converted}")
    print(f"  skipped adversarial: {total_skipped_adv}")
    print(f"  predictions not found: {total_not_found}")
    print(f"  output: {args.output}")

    n_leftover = sum(len(v) for v in pred_by_question.values())
    if n_leftover > 0:
        leftover_qs = [q for q, v in pred_by_question.items() if v][:5]
        print(f"  [warn] {n_leftover} predictions were not consumed (sample questions):")
        for q in leftover_qs:
            print(f"     - {q[:80]}")

    if total_not_found > 0:
        print(f"\n[warn] {total_not_found} locomo questions had no matching prediction!")

if __name__ == "__main__":
    main()
