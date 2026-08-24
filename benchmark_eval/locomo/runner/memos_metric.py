"""Aggregate MemOS-style LoCoMo metrics from judged JSON.
Computes overall / per-category / per-user LLM-judge scores + lexical F1/BLEU1-4."""

import argparse
import json

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed, Excel output will be skipped")


CATEGORY_MAPPING = {
    "4": "single hop",
    "1": "multi hop",
    "2": "temporal reasoning",
    "3": "open domain",
    "5": "adversarial",
}


LEXICAL_METRICS = ["f1", "bleu1", "bleu2", "bleu3", "bleu4"]


def calculate_scores(data):
    """Reproduce MemOS locomo_metric.calculate_scores (with lexical metrics)."""
    category_scores = {}
    total_questions = 0

    all_judgment_keys = set()
    for _user, questions in data.items():
        for question in questions:
            if "llm_judgments" in question:
                all_judgment_keys.update(question["llm_judgments"].keys())

    judgment_run_scores = {key: [] for key in all_judgment_keys}

    overall_lexical = {m: [] for m in LEXICAL_METRICS}

    user_metrics = {}

    for user, questions in data.items():
        user_total = 0
        user_metrics[user] = {
            "total": 0,
            "llm_judge_score": 0,
            "llm_judge_std": 0,
            "judgment_run_scores": {key: [] for key in all_judgment_keys},
            "lexical": {m: [] for m in LEXICAL_METRICS},
        }

        for question in questions:
            total_questions += 1
            user_total += 1

            category = question.get("category", 0)
            cat_key = str(category)

            if cat_key not in category_scores:
                category_scores[cat_key] = {
                    "total": 0,
                    "category_name": CATEGORY_MAPPING.get(cat_key, f"category_{cat_key}"),
                    "judgment_run_scores": {key: [] for key in all_judgment_keys},
                    "lexical": {m: [] for m in LEXICAL_METRICS},
                }

            category_scores[cat_key]["total"] += 1

            if "llm_judgments" in question:
                for judgment_key, judgment_value in question["llm_judgments"].items():
                    score = 1 if judgment_value else 0
                    judgment_run_scores[judgment_key].append(score)
                    user_metrics[user]["judgment_run_scores"][judgment_key].append(score)
                    category_scores[cat_key]["judgment_run_scores"][judgment_key].append(score)

            nlp = question.get("nlp_metrics", {})
            for metric in LEXICAL_METRICS:
                v = nlp.get("lexical", {}).get(metric)
                if v is not None:
                    overall_lexical[metric].append(v)
                    category_scores[cat_key]["lexical"][metric].append(v)
                    user_metrics[user]["lexical"][metric].append(v)

        user_metrics[user]["total"] = user_total

        judgment_avgs = []
        for _key, scores in user_metrics[user]["judgment_run_scores"].items():
            if scores:
                judgment_avgs.append(np.mean(scores))
        user_metrics[user]["llm_judge_score"] = np.mean(judgment_avgs) if judgment_avgs else 0.0
        user_metrics[user]["llm_judge_std"] = np.std(judgment_avgs) if len(judgment_avgs) > 1 else 0.0

        for metric in LEXICAL_METRICS:
            values = user_metrics[user]["lexical"][metric]
            user_metrics[user]["lexical"][metric] = float(np.mean(values)) if values else 0.0

    judgment_run_averages = []
    for _key, scores in judgment_run_scores.items():
        if scores:
            judgment_run_averages.append(np.mean(scores))

    llm_judge_score = np.mean(judgment_run_averages) if judgment_run_averages else 0.0
    llm_judge_std = np.std(judgment_run_averages) if len(judgment_run_averages) > 1 else 0.0

    overall_lexical_avg = {}
    for metric in LEXICAL_METRICS:
        values = overall_lexical[metric]
        overall_lexical_avg[metric] = float(np.mean(values)) if values else 0.0

    category_overall_scores = {}
    for cat_key, score_data in category_scores.items():
        cat_judgment_avgs = []
        for _key, scores in score_data["judgment_run_scores"].items():
            if scores:
                cat_judgment_avgs.append(np.mean(scores))

        cat_lexical_avg = {}
        for metric in LEXICAL_METRICS:
            values = score_data["lexical"][metric]
            cat_lexical_avg[metric] = float(np.mean(values)) if values else 0.0

        category_overall_scores[cat_key] = {
            "category_name": score_data["category_name"],
            "llm_judge_score": np.mean(cat_judgment_avgs) if cat_judgment_avgs else 0.0,
            "llm_judge_std": np.std(cat_judgment_avgs) if len(cat_judgment_avgs) > 1 else 0.0,
            "total": score_data["total"],
            "lexical": cat_lexical_avg,
        }

    return {
        "metrics": {
            "llm_judge_score": float(llm_judge_score),
            "llm_judge_std": float(llm_judge_std),
            "lexical": overall_lexical_avg,
        },
        "category_scores": category_overall_scores,
        "user_scores": {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                            for kk, vv in v.items() if kk not in ("judgment_run_scores",)}
                        for k, v in user_metrics.items()},
        "total_questions": total_questions,
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate MemOS-style LoCoMo metrics from judged results")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to judged JSON (output of memos_judge.py)")
    parser.add_argument("--output", type=str, default="",
                        help="Output path for grades JSON (optional)")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = calculate_scores(data)

    print("\n" + "=" * 70)
    print("  MemOS-style LoCoMo Evaluation Metrics")
    print("=" * 70)

    print(f"\n  Overall LLM-as-Judge Score: {results['metrics']['llm_judge_score']:.4f} ± {results['metrics']['llm_judge_std']:.4f}")
    lexical = results['metrics'].get('lexical', {})
    if lexical:
        print(f"  Overall F1:    {lexical.get('f1', 0):.4f}")
        print(f"  Overall BLEU1: {lexical.get('bleu1', 0):.4f}")
    print(f"  Total questions evaluated: {results['total_questions']}")

    print("\n  By Category:")
    print(f"  {'Category':<25} {'Judge':>8} {'F1':>8} {'BLEU1':>8} {'Count':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for cat_key in sorted(results["category_scores"].keys(), key=lambda x: int(x)):
        cat = results["category_scores"][cat_key]
        cat_lex = cat.get('lexical', {})
        print(f"  {cat['category_name']:<25} {cat['llm_judge_score']:>8.4f} {cat_lex.get('f1', 0):>8.4f} {cat_lex.get('bleu1', 0):>8.4f} {cat['total']:>8}")

    print("\n  By User:")
    print(f"  {'User':<30} {'Score':>10} {'Count':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*8}")
    for uid in sorted(results["user_scores"].keys()):
        u = results["user_scores"][uid]
        print(f"  {uid:<30} {u['llm_judge_score']:>10.4f} {u['total']:>8}")

    print("=" * 70)

    if args.output:
        output_path = args.output
    else:
        output_path = args.input.replace("_judged.json", "_grades.json")
        if output_path == args.input:
            output_path = args.input.replace(".json", "_grades.json")

    def convert_numpy(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    results = convert_numpy(results)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Grades saved to: {output_path}")

    if HAS_PANDAS:
        excel_path = output_path.replace(".json", ".xlsx")
        rows = []
        overall_lex = results["metrics"].get("lexical", {})
        overall_row = {
            "category": "overall",
            "llm_judge_score": results["metrics"]["llm_judge_score"],
            "llm_judge_std": results["metrics"]["llm_judge_std"],
            "count": results["total_questions"],
        }
        for m in LEXICAL_METRICS:
            overall_row[m] = overall_lex.get(m, 0.0)
        rows.append(overall_row)
        for cat_key in sorted(results["category_scores"].keys(), key=lambda x: int(x)):
            cat = results["category_scores"][cat_key]
            cat_lex = cat.get("lexical", {})
            cat_row = {
                "category": cat["category_name"],
                "llm_judge_score": cat["llm_judge_score"],
                "llm_judge_std": cat["llm_judge_std"],
                "count": cat["total"],
            }
            for m in LEXICAL_METRICS:
                cat_row[m] = cat_lex.get(m, 0.0)
            rows.append(cat_row)
        df = pd.DataFrame(rows)
        df.to_excel(excel_path, index=False)
        print(f"  Excel saved to: {excel_path}")


if __name__ == "__main__":
    main()
