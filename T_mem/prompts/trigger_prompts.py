"""Trigger-generation prompts (Entity/Bridge (item-level) and Scene/Horizon (scene-level)).
Helpers build_prompt / SCENE_TRIGGER_KEYS / HORIZON_TRIGGER_KEYS are consumed by T_mem.main.stage4_associative_extract."""

# Paper-to-code mapping (Entity/Bridge/Scene/Horizon → code structure):
#   Entity Trigger  -> EntityBridgeTrigger (concept field)
#   Bridge Trigger  -> EntityBridgeTrigger (bridge field)
#   Scene Trigger   -> scene_attributes (scene / object / event / emotion)
#   Horizon Trigger -> horizon_channels (PERSONAL_ARC / AVOIDANCE_HABIT /
#                          GOAL_ARC / BELIEF_IN_ACTION / LEGACY_ANCHOR)

ENTITY_BRIDGE_TRIGGER_PROMPT = """
You are a specialist in generating retrieval triggers that sit one semantic step above a memory item.

Your task: Generate {trigger_count} item-level triggers (Entity + Bridge routes) for the memory item below. Each trigger is a short noun phrase (2-6 words) that, if mentioned later, should reliably pull THIS item to mind.

## MEMORY ITEM

Content: {item_content}
Time: {item_temporal}

---

# CORE PRINCIPLES

## 1. Two Accepted Routes
Every trigger must fall into exactly ONE of these:

- **Route A — SEMANTIC ANCHOR**: a noun phrase that NAMES the item one or two rungs up the is-a ladder. It is the item's own semantics, generalized.
- **Route B — STRONG-ASSOCIATION SCENE**: a concrete scene, prop, or situation where P(this item is relevant | this scene is mentioned) is high. Weak or incidental adjacencies do NOT qualify.

## 2. What Disqualifies a Trigger
- **Restatement**: rewriting the item in 2-6 words adds no retrieval value.
- **Over-general label**: fits almost anything, loses the item's specificity.
- **Weakly-predictive scene**: the scene could belong to many unrelated items; mentioning it does NOT reliably surface THIS item.

## 3. Diversity Across Triggers
The {trigger_count} triggers must cover different angles — do NOT output variants of the same concept. Prefer a mix that spans both routes and different abstraction rungs.

---

# EXAMPLES

**Memory item**: "Tim developed urticaria after eating shrimp at a dinner party"

**Route A (semantic anchor) — good**:
- "seafood sensitivity"       (tight mid-level category)
- "food allergy"              (clear upper-level category)
- "allergic reaction"         (broader superordinate)

**Route B (strong-association scene) — good**:
- "seafood buffet"            (mixed-seafood setting forces the item to matter)
- "ordering at a sushi bar"   (decision point where the item applies)

**Bad — do not produce these**:
- "shrimp allergy"        → Restatement. Rewrites the item in two words.
- "health issue"          → Over-general. Fits any medical item.
- "EpiPen in a purse"     → Weakly-predictive. An EpiPen belongs to any allergy.

---

# BRIDGE FIELD

Each trigger must include a one-line `bridge` (≤12 words) explaining WHY it is a legitimate trigger for this item.

Shape: `<cue lifted from the item> -> <one-step inference>`

- The cue MUST come from the item's own wording.
- The inference MUST be a real semantic step: categorization (x ∈ Y), causal implication, or decision-relevance.
- Do NOT restate the trigger concept and do NOT write a type label like "semantic anchor".

**Examples**:
- concept: "seafood sensitivity"      → bridge: "shrimp is seafood; urticaria after eating = sensitivity"
- concept: "ordering at a sushi bar"  → bridge: "sushi menu = decision point where shrimp-allergy applies"

---

# CONFIDENCE

Score how RELIABLY a listener who knows the person would link the trigger back to this item. Judge each trigger on its own merit; do NOT force a target distribution.

- **[0.8, 1.0]**: tight mid-level anchor, or scene/prop that almost certainly surfaces the item.
- **[0.5, 0.8)**: clear upper-level anchor, or scene that strongly (but not necessarily) surfaces the item.
- **[0.3, 0.5)**: broader superordinate, or scene with moderate association.
- **[0.0, 0.3)**: very broad category OR weakly-predictive scene — avoid unless nothing better fits.

---

# QUALITY CHECKLIST

Before finalizing each trigger, verify:
- [ ] Ask in order: (1) Is it a more abstract NAME for the item? → Route A, KEEP. (2) If I only heard this scene/prop, would it reliably recall THIS specific item? → Route B, KEEP. (3) Otherwise DISCARD.
- [ ] The concept is 2-6 words.
- [ ] The bridge cites a cue from the item and makes a real inference step.
- [ ] The confidence score matches the rung/association strength above.
- [ ] Across all {trigger_count} triggers, angles differ.

---

# OUTPUT FORMAT

Return strict JSON only. No markdown, no commentary.

```json
{{
  "triggers": [
    {{
      "concept": "short phrase, 2-6 words (Route A anchor OR Route B scene)",
      "bridge": "<cue from item> -> <inference> (<=12 words, no type labels)",
      "confidence": 0.0
    }}
  ]
}}
```
"""


_HORIZON_DEF_BLOCK = """\
## Horizon Trigger Channels — 5 semantic-bridge dimensions

Each channel is a **forward-looking bridge** from the cue toward a likely future
query. The trigger sentence should NOT summarize the cue; it should capture the
implicit extension the cue foreshadows.

| ID | Channel          | Guiding question (what the trigger should capture)                          | Main relation_type |
|----|------------------|-----------------------------------------------------------------------------|--------------------|
| D1 | PERSONAL_ARC     | "Who did the speaker become? Capture before_self -> after_self."            | causal             |
| D2 | AVOIDANCE_HABIT  | "What is the speaker now avoiding/vigilant about? How might it generalize?" | state, causal      |
| D3 | GOAL_ARC         | "What commitment is declared? What cost/fatigue/replacement might follow?"  | goal               |
| D4 | BELIEF_IN_ACTION | "Which stated principle is fragile - where might reality push back?"        | value              |
| D5 | LEGACY_ANCHOR    | "Which concrete object/person/ritual in the cue may resurface later?"       | causal, state      |

### Per-channel bad examples (DO NOT produce these)

- D1 bad: "Joanna cut sugary drinks" - restating the cue, not an arc.
- D2 bad: "fear" - too generic; must name what is being avoided and hint at generalization.
- D3 bad: "wants to travel to Japan" - restates goal; missing cost/replacement reflex.
- D4 bad: "likes to ask team for input" - describes behavior, not the belief's fragile edge.
- D5 bad: "home" / "memory" - too vague; must name a specific concrete anchor.

### Empty-channel policy

- If the cue gives NO signal for a channel, set "sent": null and "confidence": 0.0.
- If you would have to fabricate to fill the field, LEAVE IT EMPTY. Hallucination hurts recall more than a missing channel.
- Otherwise confidence must be in [0.3, 1.0] and sent must be one complete English sentence.
"""

_SCENE_DEF_BLOCK = """\
## Scene Trigger Attributes - 4 attribute fields

One-sentence description of the cue along 4 orthogonal axes. These are
**about the cue itself**, not about future queries.

| Field   | Guiding question                                                    | Bad example (DO NOT do this) |
|---------|---------------------------------------------------------------------|-------------------------------|
| scene   | "What situation/setting does this happen in?"                       | "a scene about devices" (collapses into object) |
| object  | "What essential entity / item / person does it involve?"            | "phone battery" (that's an entity-level keyword) |
| event   | "What abstracted action/change happened (abstracted, not literal)?" | "replaced battery" (literal, = entity-level) |
| emotion | "What emotional arc / psychological tone shines through?"           | "happy" (too generic) |

All 4 fields are REQUIRED. Each must be one concise sentence. Do NOT leave null.
"""

_FEW_SHOT_1_CUE = (
    "A: Being rejected from my dream architecture school made me take a gap year working on construction sites.\n"
    "B: That detour clearly shaped the way you approached the field."
)

_FEW_SHOT_1_ANSWER_FULL = """\
{
  "scene_attributes": {
    "scene":   "An early-career identity crisis framed by a rejection and a forced detour into manual work.",
    "object":  "An aspiring architect re-encountering the built environment from the ground up, through a construction-site gap year.",
    "event":   "A painful rejection redirected into hands-on immersion, reshaping how the field is understood.",
    "emotion": "Disappointment transformed into grounded, earned respect for the craft."
  },
  "horizon_channels": {
    "PERSONAL_ARC":     { "sent": "Shifted from a self defined by academic ambition to one whose understanding of the field is grounded in physical, ground-level labor.", "confidence": 0.85 },
    "AVOIDANCE_HABIT":  { "sent": null, "confidence": 0.0 },
    "GOAL_ARC":         { "sent": null, "confidence": 0.0 },
    "BELIEF_IN_ACTION": { "sent": null, "confidence": 0.0 },
    "LEGACY_ANCHOR":    { "sent": "The construction-site era lingers as a physical residue - gear, clothing, or keepsakes from that gap year that later symbolize the pivotal detour.", "confidence": 0.70 }
  }
}"""

_FEW_SHOT_1_ANSWER_HORIZON_ONLY = """\
{
  "horizon_channels": {
    "PERSONAL_ARC":     { "sent": "Shifted from a self defined by academic ambition to one whose understanding of the field is grounded in physical, ground-level labor.", "confidence": 0.85 },
    "AVOIDANCE_HABIT":  { "sent": null, "confidence": 0.0 },
    "GOAL_ARC":         { "sent": null, "confidence": 0.0 },
    "BELIEF_IN_ACTION": { "sent": null, "confidence": 0.0 },
    "LEGACY_ANCHOR":    { "sent": "The construction-site era lingers as a physical residue - gear, clothing, or keepsakes from that gap year that later symbolize the pivotal detour.", "confidence": 0.70 }
  }
}"""

_FEW_SHOT_2_CUE = (
    "A: I'm saving all my vacation days for one big trip because I want to spend a month exploring Japan's countryside.\n"
    "B: That sounds like an unforgettable cultural experience."
)

_FEW_SHOT_2_ANSWER_FULL = """\
{
  "scene_attributes": {
    "scene":   "Long-term planning of a singular high-investment leisure pursuit.",
    "object":  "Accumulated vacation days being channeled toward one ambitious cultural-immersion trip.",
    "event":   "Concentration of personal time and energy into a single, hard-to-reverse commitment.",
    "emotion": "Eager anticipation tied to a somewhat romanticized ideal."
  },
  "horizon_channels": {
    "PERSONAL_ARC":     { "sent": null, "confidence": 0.0 },
    "AVOIDANCE_HABIT":  { "sent": null, "confidence": 0.0 },
    "GOAL_ARC":         { "sent": "A large, singular travel commitment that will invite its own cost and replacement reflex - downscaling to smaller local or everyday fulfillment if circumstances change.", "confidence": 0.80 },
    "BELIEF_IN_ACTION": { "sent": null, "confidence": 0.0 },
    "LEGACY_ANCHOR":    { "sent": null, "confidence": 0.0 }
  }
}"""

_PROMPT_A_PREFIX = (
    "You are a specialist in extracting **cognitive-memory triggers** - compact textual\n"
    "descriptors that help a future associative query find this cue again, even when\n"
    "the two share almost no literal vocabulary.\n"
    "\n"
    "You will be given ONE cue dialogue (2 short turns). Produce two layers of\n"
    "triggers from it, at different abstraction levels:\n"
    "\n"
    "- **Scene Trigger Attributes**: 4 one-sentence attribute descriptors of the cue itself.\n"
    "- **Horizon Trigger Channels**: 5 forward-looking semantic-bridge sentences to likely future queries.\n"
    "\n---\n\n"
    + _SCENE_DEF_BLOCK
    + "\n---\n\n"
    + _HORIZON_DEF_BLOCK
    + "\n---\n\n"
    "## Core rules\n"
    "\n"
    "1. **NEVER peek at a query.** You are given only the cue. Trigger sentences must\n"
    "   be derivable from the cue alone. Do NOT invent details that aren't implied.\n"
    "2. **Scene Trigger is about the cue itself; Horizon Trigger is about what the cue foreshadows.** Do not\n"
    "   mix the two layers.\n"
    "3. Scene attributes are **all required**. Horizon channels are **empty when there is no signal**\n"
    "   (sent=null, confidence=0). Never force-fill a channel.\n"
    "4. Every sentence is in **English**. One cue turn != one sentence - synthesize.\n"
    "5. Keep each sentence under 35 words.\n"
    "\n---\n\n"
    "## Few-shot example 1\n\n"
    "**CUE**\n```\n" + _FEW_SHOT_1_CUE + "\n```\n\n"
    "**Correct output**\n```json\n" + _FEW_SHOT_1_ANSWER_FULL + "\n```\n\n"
    "## Few-shot example 2\n\n"
    "**CUE**\n```\n" + _FEW_SHOT_2_CUE + "\n```\n\n"
    "**Correct output**\n```json\n" + _FEW_SHOT_2_ANSWER_FULL + "\n```\n\n"
    "---\n\n"
    "## Now extract from the following cue\n\n"
    "**CUE**\n```\n"
)

_PROMPT_A_SUFFIX = (
    "\n```\n\n"
    "Return a single JSON object with exactly these top-level keys:\n"
    "- \"scene_attributes\": object with keys scene / object / event / emotion, each a non-null string.\n"
    "- \"horizon_channels\": object with keys PERSONAL_ARC / AVOIDANCE_HABIT / GOAL_ARC / BELIEF_IN_ACTION / LEGACY_ANCHOR, each an object with fields sent (string or null) and confidence (float).\n"
    "\n"
    "Return JSON only. No markdown fences. No commentary.\n"
)

_PROMPT_B_PREFIX = (
    "You are a specialist in extracting **cognitive-memory triggers** - compact textual\n"
    "descriptors that help a future associative query find this cue again, even when\n"
    "the two share almost no literal vocabulary.\n"
    "\n"
    "You will be given ONE cue dialogue (2 short turns). Produce **only** the Horizon Trigger\n"
    "channel layer: 5 forward-looking semantic-bridge sentences from the cue toward\n"
    "a likely future associative query.\n"
    "\n---\n\n"
    + _HORIZON_DEF_BLOCK
    + "\n---\n\n"
    "## Core rules\n"
    "\n"
    "1. **NEVER peek at a query.** You are given only the cue. Trigger sentences must\n"
    "   be derivable from the cue alone. Do NOT invent details that aren't implied.\n"
    "2. **Horizon Trigger is about what the cue foreshadows**, not a summary of the cue.\n"
    "3. Channels are **empty when there is no signal** (sent=null, confidence=0).\n"
    "   Never force-fill a channel.\n"
    "4. Every sentence is in **English**. Keep each sentence under 35 words.\n"
    "\n---\n\n"
    "## Few-shot example\n\n"
    "**CUE**\n```\n" + _FEW_SHOT_1_CUE + "\n```\n\n"
    "**Correct output**\n```json\n" + _FEW_SHOT_1_ANSWER_HORIZON_ONLY + "\n```\n\n"
    "---\n\n"
    "## Now extract from the following cue\n\n"
    "**CUE**\n```\n"
)

_PROMPT_B_SUFFIX = (
    "\n```\n\n"
    "Return a single JSON object with exactly this top-level key:\n"
    "- \"horizon_channels\": object with keys PERSONAL_ARC / AVOIDANCE_HABIT / GOAL_ARC / BELIEF_IN_ACTION / LEGACY_ANCHOR, each an object with fields sent (string or null) and confidence (float).\n"
    "\n"
    "Return JSON only. No markdown fences. No commentary.\n"
)


def build_prompt(group: str, cue_dialogue: str) -> str:
    """group in {'A', 'B'}. A = Scene+Horizon, B = Horizon-only."""
    if group == "A":
        return _PROMPT_A_PREFIX + cue_dialogue + _PROMPT_A_SUFFIX
    if group == "B":
        return _PROMPT_B_PREFIX + cue_dialogue + _PROMPT_B_SUFFIX
    raise ValueError(f"unknown group: {group}")


HORIZON_TRIGGER_KEYS = ["PERSONAL_ARC", "AVOIDANCE_HABIT", "GOAL_ARC", "BELIEF_IN_ACTION", "LEGACY_ANCHOR"]
SCENE_TRIGGER_KEYS = ["scene", "object", "event", "emotion"]

# N_TURNS_SKIP: scene-level skip threshold — scenes with more than N_TURNS_SKIP turns
# are excluded from Scene/Horizon trigger extraction (too long for reliable LLM extraction).
N_TURNS_SKIP = 10
