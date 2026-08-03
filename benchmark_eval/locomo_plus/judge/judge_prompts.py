"""Locomo-Plus memory-awareness judge prompts (Table 7 minimal version).
Placeholders {question}/{evidence}/{pred}; output JSON {label, reason} with label correct|wrong."""
from __future__ import annotations

import re

A_PAPER_ORIGINAL = """
You are a Memory Awareness Judge. Your task is to determine whether the model's response demonstrates recall of the memory cue described in the Evidence.

Question:
{question}

Memory/Evidence:
{evidence}

Model Prediction:
{pred}

Labels:
- "correct": explicitly acknowledges or adapts to the Memory/Cue (proves recall).
- "wrong": completely ignores the Evidence and gives a generic response.

Return your judgment strictly in JSON format:
{{"label": "correct"|"wrong", "reason": "<short explanation>"}}
"""

PROMPT_REGISTRY: dict[str, str] = {
    "A_paper_original": A_PAPER_ORIGINAL,
}

# CRITICAL: only replace line-start A:/B: prefixes — never touch "plan A" / "option B"
# or other occurrences of A/B inside the dialogue body.
_AB_PATTERN_A = re.compile(r'(^|\n)A:')
_AB_PATTERN_B = re.compile(r'(^|\n)B:')


def replace_ab_with_names(text: str, speaker_a: str, speaker_b: str) -> str:
    """Replace leading A:/B: prefixes with real speaker names; body 'A'/'B' untouched."""
    if not text:
        return text
    text = _AB_PATTERN_A.sub(rf'\1{speaker_a}:', text)
    text = _AB_PATTERN_B.sub(rf'\1{speaker_b}:', text)
    return text


def get_prompt(prompt_key: str) -> str:
    if prompt_key not in PROMPT_REGISTRY:
        raise KeyError(
            f"unknown prompt_key={prompt_key!r}; "
            f"available: {sorted(PROMPT_REGISTRY)}"
        )
    return PROMPT_REGISTRY[prompt_key]
