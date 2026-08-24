"""Patch F1 + BLEU-{1..4} (MemOS locomo_eval formulation) onto an existing locomo_judged.json."""

import argparse
import json

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass


def calculate_f1_score(gold_tokens, response_tokens):
    try:
        gold_set = set(gold_tokens)
        response_set = set(response_tokens)

        if len(gold_set) == 0 or len(response_set) == 0:
            return 0.0

        precision = len(gold_set.intersection(response_set)) / len(response_set)
        recall = len(gold_set.intersection(response_set)) / len(gold_set)

        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0
    except Exception as e:
        print(f"Failed to calculate F1 score: {e}")
        return 0.0


def calculate_bleu_scores(gold_tokens, response_tokens):
    metrics = {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    try:
        smoothing = SmoothingFunction().method1
        weights = [
            (1, 0, 0, 0),
            (0.5, 0.5, 0, 0),
            (0.33, 0.33, 0.33, 0),
            (0.25, 0.25, 0.25, 0.25),
        ]

        for i, weight in enumerate(weights, 1):
            metrics[f"bleu{i}"] = sentence_bleu(
                [gold_tokens], response_tokens, weights=weight, smoothing_function=smoothing
            )
    except ZeroDivisionError:
        pass
    except Exception as e:
        print(f"Failed to calculate BLEU scores: {e}")

    return metrics


def patch_judged_file(input_path, output_path=None):
    if output_path is None:
        output_path = input_path

    print(f"Reading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_patched = 0
    for group_id, questions in data.items():
        for question in questions:
            answer = str(question.get("answer", "") or "")
            golden_answer = str(question.get("golden_answer", "") or "")

            gold_tokens = nltk.word_tokenize(golden_answer.lower())
            response_tokens = nltk.word_tokenize(answer.lower())

            f1 = calculate_f1_score(gold_tokens, response_tokens)
            bleu_scores = calculate_bleu_scores(gold_tokens, response_tokens)

            question["nlp_metrics"] = {
                "lexical": {
                    "f1": f1,
                    **bleu_scores,
                },
            }
            total_patched += 1

    print(f"Patched NLP metrics for {total_patched} records")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Patch F1 and BLEU scores onto an existing judged.json.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to locomo_judged.json.")
    parser.add_argument("--output", type=str, default="",
                        help="Output path (defaults to overwriting the input file).")
    args = parser.parse_args()

    output = args.output if args.output else args.input
    patch_judged_file(args.input, output)


if __name__ == "__main__":
    main()
