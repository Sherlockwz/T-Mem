"""LongMemEval judge prompts -- byte-for-byte aligned with the official judge.

Source of truth: `LongMemEval-main/src/evaluation/evaluate_qa.py::get_anscheck_prompt`.
We replicate the 4 question-type templates (with 5 qtypes mapped to 4 templates)
plus 1 abstention template here verbatim, then expose `build_judge_prompt(...)`
as the single API that `run_judge.py` calls.

Mapping (qtype -> template), copied from the upstream `if/elif` chain:
  - single-session-user      -> _GENERIC_TEMPLATE
  - single-session-assistant -> _GENERIC_TEMPLATE
  - multi-session            -> _GENERIC_TEMPLATE
  - temporal-reasoning       -> _TEMPORAL_TEMPLATE   (off-by-one tolerance)
  - knowledge-update         -> _KNOWLEDGE_UPDATE_TEMPLATE
  - single-session-preference-> _PREFERENCE_TEMPLATE (rubric-based)
  - <any qtype, abstention>  -> _ABSTENTION_TEMPLATE  (selected when qid ends `_abs`)

All templates produce yes/no responses; `run_judge.py` parses
`'yes' in response.lower()` exactly like the upstream script.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Upstream-aligned templates. Strings copied byte-for-byte from
# evaluate_qa.py (lines ~22--46) with `{}` left as-is so we can call
# `.format(question, answer, response)` in the same positional order.
# ---------------------------------------------------------------------------

_GENERIC_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. "
    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_TEMPORAL_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. "
    "In addition, do not penalize off-by-one errors for the number of days. "
    "If the question asks for the number of days/weeks/months, etc., and the model makes "
    "off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's "
    "response is still correct. "
    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_KNOWLEDGE_UPDATE_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response contains some previous information along with an updated answer, the "
    "response should be considered as correct as long as the updated answer is the required answer."
    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_PREFERENCE_TEMPLATE = (
    "I will give you a question, a rubric for desired personalized response, and a response "
    "from a model. Please answer yes if the response satisfies the desired response. Otherwise, "
    "answer no. The model does not need to reflect all the points in the rubric. The response "
    "is correct as long as it recalls and utilizes the user's personal information correctly."
    "\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_ABSTENTION_TEMPLATE = (
    "I will give you an unanswerable question, an explanation, and a response from a model. "
    "Please answer yes if the model correctly identifies the question as unanswerable. "
    "The model could say that the information is incomplete, or some other information is "
    "given but the asked information is not."
    "\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
    "Does the model correctly identify the question as unanswerable? Answer yes or no only."
)

# qtype -> (template, "qtype" | "preference" | "temporal" | "knowledge")
_QTYPE_TO_TEMPLATE: dict[str, str] = {
    "single-session-user":       _GENERIC_TEMPLATE,
    "single-session-assistant":  _GENERIC_TEMPLATE,
    "multi-session":             _GENERIC_TEMPLATE,
    "temporal-reasoning":        _TEMPORAL_TEMPLATE,
    "knowledge-update":          _KNOWLEDGE_UPDATE_TEMPLATE,
    "single-session-preference": _PREFERENCE_TEMPLATE,
}

SUPPORTED_QTYPES = tuple(_QTYPE_TO_TEMPLATE.keys())


def build_judge_prompt(
    qtype: str,
    question: str,
    answer: str,
    response: str,
    *,
    abstention: bool,
) -> str:
    """Format the judge prompt for one (qtype, question, answer, response) triple.

    Args:
        qtype:     LongMemEval `question_type` (must be in SUPPORTED_QTYPES
                   when `abstention=False`; ignored when `abstention=True`).
        question:  the question text.
        answer:    for non-abstention: gold answer (preference: rubric);
                   for abstention: the gold "explanation" of WHY the question
                   is unanswerable (LongMemEval stores it in `answer`).
        response:  the model's hypothesis being judged.
        abstention: True iff `question_id` ends with `_abs`.

    Returns:
        The prompt string ready to be sent to the judge LLM (gpt-4o).

    Raises:
        ValueError: when `abstention=False` and `qtype` is not supported.
    """
    if abstention:
        return _ABSTENTION_TEMPLATE.format(question, answer, response)
    template = _QTYPE_TO_TEMPLATE.get(qtype)
    if template is None:
        raise ValueError(
            f"unsupported question_type {qtype!r}; expected one of "
            f"{SUPPORTED_QTYPES} or abstention (qid endswith '_abs')"
        )
    return template.format(question, answer, response)
