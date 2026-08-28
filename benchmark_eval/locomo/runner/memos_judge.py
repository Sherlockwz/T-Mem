"""LLM-as-judge scorer for LoCoMo predictions (MemOS protocol; CORRECT/WRONG).
Aggregates mean ± std over --num-runs passes; default judge = T_mem.config.MODELS['locomo_judge'].

Grader prompt and metric definitions follow the upstream MemOS LoCoMo harness
(https://github.com/MemTensor/MemOS, evaluation/scripts/locomo/) so that scores
are directly comparable with published numbers.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from T_mem.config import MODELS  # noqa: E402


def call_llm(prompt: str, model: str = None, temperature: float = 0.0) -> str:
    """Call an OpenAI-compatible chat/completions endpoint (see T_mem.llm.llm_provider)."""
    from T_mem.llm.llm_provider import chat_completion

    if model is None:
        model = MODELS["locomo_judge"]

    # Include the system prompt inline (chat_completion is a single-turn helper).
    system_hint = ("You are an expert grader that determines if answers to questions "
                   "match a gold standard answer.\n\n")
    for attempt in range(3):
        try:
            response = chat_completion(
                system_hint + prompt,
                model=model,
                timeout=240,
                max_retries=3,
                temperature=float(temperature),
                meta={"call_site": "memos_judge"},
            )
            return response.strip() if response else "(empty)"
        except Exception as e:
            print(f"Judge call error (attempt {attempt + 1}/3): {e}")
            time.sleep(5)
            continue
    return "(LLM API error after 3 retries)"


def extract_label_json(text: str):
    """Extract a {"label": "..."} JSON object from the judge output."""
    pattern = r'\{\s*"label"\s*:\s*["\']([^"\']*)["\']\s*\}'
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    return None


def locomo_grader(question: str, gold_answer: str, response: str, model: str = None) -> bool:
    """MemOS LLM judge scorer (verbatim from locomo_eval.py::locomo_grader).

    Returns True for CORRECT, False for WRONG.
    """
    if model is None:
        model = MODELS["locomo_judge"]
    accuracy_prompt = f"""
    Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a 'gold' (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it's time for the real question:
    Question: {question}
    Gold answer: {gold_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Just return the label CORRECT or WRONG in a json format with the key as "label".
    """

    try:
        raw_response = call_llm(accuracy_prompt, model=model, temperature=0.0)
        label_json = extract_label_json(raw_response)
        if label_json:
            label = json.loads(label_json)["label"]
            return label.strip().lower() == "correct"
        else:
            if "CORRECT" in raw_response and "WRONG" not in raw_response:
                return True
            return False
    except Exception as e:
        print(f"Judge error: {e}")
        return False


def judge_single_response(response: dict, num_runs: int, model: str) -> dict:
    """Run num_runs judge passes over a single QA record."""
    question = response.get("question", "")
    answer = response.get("answer", "")
    golden_answer = response.get("golden_answer", "")
    category = response.get("category", 0)

    judgments = {}
    for run_idx in range(1, num_runs + 1):
        try:
            result = locomo_grader(question, golden_answer, answer, model=model)
            judgments[f"judgment_{run_idx}"] = result
        except Exception as e:
            print(f"Judge run {run_idx} error: {e}")
            judgments[f"judgment_{run_idx}"] = False

    return {
        "question": question,
        "answer": answer,
        "golden_answer": golden_answer,
        "category": category,
        "llm_judgments": judgments,
        "nlp_metrics": {},  # LLM judge only — NLP metrics intentionally not computed.
        "response_duration_ms": response.get("response_duration_ms", 0.0),
        "search_duration_ms": response.get("search_duration_ms", 0.0),
        "total_duration_ms": 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="MemOS-style LLM Judge evaluation using an OpenAI-compatible endpoint")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to MemOS-format locomo_responses.json")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for judged results JSON")
    parser.add_argument("--model", type=str, default=MODELS["locomo_judge"],
                        help=f"Judge model (default from T_mem.config.MODELS['locomo_judge']: {MODELS['locomo_judge']})")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of judge runs per question (default: 3)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of concurrent workers (default: 10)")
    args = parser.parse_args()

    print("=== MemOS-style LLM Judge Evaluation ===")
    print(f"  Input: {args.input}")
    print(f"  Model: {args.model}")
    print(f"  Runs per question: {args.num_runs}")
    print(f"  Workers: {args.workers}")

    with open(args.input, "r", encoding="utf-8") as f:
        locomo_responses = json.load(f)

    total_count = sum(len(v) for v in locomo_responses.values())
    print(f"  Total questions: {total_count}")
    print()

    all_grades = {}

    for group_id, group_responses in locomo_responses.items():
        if not group_responses:
            print(f"  {group_id}: 0 questions, skipping")
            continue

        print(f"  Processing {group_id} ({len(group_responses)} questions)...")
        start_time = time.time()

        graded_responses = []

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(judge_single_response, resp, args.num_runs, args.model): idx
                for idx, resp in enumerate(group_responses)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {group_id}"):
                idx = futures[future]
                try:
                    result = future.result()
                    graded_responses.append((idx, result))
                except Exception as e:
                    print(f"  Error processing {group_id}[{idx}]: {e}")

        graded_responses.sort(key=lambda x: x[0])
        all_grades[group_id] = [r for _, r in graded_responses]

        elapsed = time.time() - start_time
        print(f"  {group_id} done in {elapsed:.1f}s")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_grades, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Judged results saved to {args.output}")

    print_scores(all_grades, args.num_runs)


def print_scores(all_grades: dict, num_runs: int):
    """Compute and print scores (mirrors MemOS locomo_metric.py)."""
    category_mapping = {
        4: "single hop",
        1: "multi hop",
        2: "temporal reasoning",
        3: "open domain",
        5: "adversarial",
    }

    run_scores = []
    for run_idx in range(1, num_runs + 1):
        key = f"judgment_{run_idx}"
        correct = 0
        total = 0
        for group in all_grades.values():
            for resp in group:
                if key in resp.get("llm_judgments", {}):
                    total += 1
                    if resp["llm_judgments"][key]:
                        correct += 1
        if total > 0:
            run_scores.append(correct / total)

    category_scores = defaultdict(lambda: {"correct": [], "total": 0})
    for group in all_grades.values():
        for resp in group:
            cat = resp.get("category", 0)
            cat_name = category_mapping.get(cat, f"category_{cat}")
            category_scores[cat_name]["total"] += 1
            # Per-record score = mean over judge runs.
            judgments = resp.get("llm_judgments", {})
            if judgments:
                avg_score = sum(1 for v in judgments.values() if v) / len(judgments)
                category_scores[cat_name]["correct"].append(avg_score)

    user_scores = {}
    for group_id, group in all_grades.items():
        correct_list = []
        for resp in group:
            judgments = resp.get("llm_judgments", {})
            if judgments:
                avg_score = sum(1 for v in judgments.values() if v) / len(judgments)
                correct_list.append(avg_score)
        if correct_list:
            user_scores[group_id] = np.mean(correct_list)

    total_questions = sum(len(g) for g in all_grades.values())

    print("\n" + "=" * 70)
    print("  MemOS-style LLM-as-Judge Evaluation Results")
    print("=" * 70)

    if run_scores:
        mean_score = np.mean(run_scores)
        std_score = np.std(run_scores)
        print(f"\n  Overall LLM-as-Judge Score: {mean_score:.4f} ± {std_score:.4f}")
        print(f"  ({num_runs} runs over {total_questions} questions)")
        print(f"  Individual run scores: {[round(s, 4) for s in run_scores]}")

    print("\n  By Category:")
    print(f"  {'Category':<25} {'Score':>8} {'Count':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8}")
    for cat_name in ["single hop", "multi hop", "temporal reasoning", "open domain", "adversarial"]:
        if cat_name in category_scores:
            data = category_scores[cat_name]
            avg = np.mean(data["correct"]) if data["correct"] else 0.0
            print(f"  {cat_name:<25} {avg:>8.4f} {data['total']:>8}")

    print("\n  By User:")
    print(f"  {'User':<30} {'Score':>8}")
    print(f"  {'-'*30} {'-'*8}")
    for uid, score in sorted(user_scores.items()):
        print(f"  {uid:<30} {score:>8.4f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
