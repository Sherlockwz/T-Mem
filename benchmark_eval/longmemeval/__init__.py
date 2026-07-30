"""LongMemEval (ICLR 2025) benchmark evaluation harness for T-Mem.

100% aligned with the official LongMemEval evaluation protocol:
  - QA reader prompt: byte-for-byte copy of `con` reading method from
    LongMemEval-main/src/generation/run_generation.py.
  - QA reader model: gpt-4o-mini-2024-07-18 (LongMemEval official baseline).
  - Judge prompt: byte-for-byte copy of LongMemEval-main/src/evaluation/evaluate_qa.py
    (5 question-type templates + 1 abstention template).
  - Judge model: gpt-4o (LongMemEval official judge alias).
  - Hypothesis schema: jsonl with {question_id, hypothesis} per line.
  - Metrics: Overall / Task-averaged / Abstention accuracy + 6 question-type
    breakdown, exactly matching LongMemEval-main/src/evaluation/print_qa_metrics.py.

The T-Mem memory pipeline (stages 1-7) is reused unchanged via stage0_lme_stitch
which converts LongMemEval haystack_sessions into 500 locomo10-shape conversations
(one per question instance). Only stage8_qa_lme is bespoke -- it injects
`Current Date: {question_date}` into the reader prompt because LongMemEval has
133/500 temporal-reasoning questions that depend on this signal.
"""
