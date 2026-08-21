"""Memory-item extraction prompt: queryable items pulled from a topic's associated scenes."""

_ITEM_HEADER = """
You are a specialist in pulling out searchable memory items from memory topics.

Your job: Pull out ALL items, details, and facts from the topic that may address future queries. Emphasize COMPLETENESS and ACCURACY.

## TOPIC CONTEXT

Topic ID: {topic_id}
Title: {topic_title}

## ASSOCIATED SCENES

{scenes_content}

## REFERENCE TIME

{reference_time}

Use this as the anchor point for converting relative time expressions.
"""

_ITEM_CORE_PRINCIPLES = """\
# CORE PRINCIPLES

## 1. Completeness
Pull out EVERY searchable item. If uncertain, include it.
- Target 3-5+ items per scene
- Every unique item warrants its own entry

## 2. Specificity
Always favor concrete information over vague categories:
- Names over "someone/something"
- Exact titles over "a book/movie/song"
- Precise numbers over "some/several/many"
- Actual dates over fuzzy time references

## 3. Self-Containment
Each item should be self-explanatory on its own:
- Include WHO, WHAT, WHEN, WHERE when applicable
- A reader should grasp the item without needing other items
"""

_ITEM_EXTRACTION_STRATEGY = """\
# EXTRACTION STRATEGY

**Pass 1 - Individual Items**:
Pull out all items from each scene on its own.
- Each gets its source scene_id
- These form the backbone - never skip them

**Pass 2 - Connected Items (Supplement)**:
When items across scenes are logically linked, generate extra merged items.
- These get multiple scene_ids
- These SUPPLEMENT Pass 1, they don't replace it

**Example**:
```
Scene_A: "Visited the bookstore yesterday"
Scene_B: "Bought a novel titled 'Silent Harbor' for $25"
Scene_C: "The author was named Marcus"

Pass 1: Three separate items [A], [B], [C]
Pass 2: "Bought 'Silent Harbor' by Marcus at the bookstore for $25" [A,B,C]

Output: All 4 items
```
"""

_ITEM_INFORMATION_INTEGRITY = """\
# INFORMATION INTEGRITY

## Preserve Logical Connections
When items have causal, conditional, or purposive relationships, retain them:

- Wrong: "Adopted a cat" + "Felt lonely after moving" (split)
- Right: "Adopted a cat because felt lonely after moving" (connected)

## Strict Time Reasoning
When converting relative time expressions:
1. Pinpoint the exact reference point (the date of the conversation/event)
2. Compute exactly based on the reference
3. Retain both the original phrase AND the calculated date

**Example** (reference: July 14, 2023):
```
Original: "I did this earlier this week"

Wrong: "in July 2023" (too vague)
Wrong: "July 3-9" (that's LAST week, not THIS week)
Right: "around July 10-13, 2023 (earlier that week)"
```

**Time phrase meanings**:
- "yesterday" = reference date minus 1 day
- "this week" = the week containing the reference date
- "last week" = the week before the reference week
- "earlier this week" = days before reference date within same week

## Keep Exact Names and Titles
Proper nouns are crucial for queries - never generalize them:
- Book/movie/song/game titles: keep exact title in quotes
- Person/pet/place/organization/event names: keep exact names
"""

_ITEM_WHAT_TO_EXTRACT = """\
# WHAT TO EXTRACT

**Always pull out**:
- Named facts (people, places, organizations, titles)
- Actions and events with their participants
- Time information (dates, durations, frequencies)
- Quantities and measurements
- Relationships between people
- Items acquired, created, or shared
- Emotional states and reactions
- Reasons and motivations when stated

**Watch closely for**:
- Content of photos/artworks shared (not just "shared a photo")
- Text on signs, labels, or messages
- Specific preferences stated ("favorite X is Y")
- Details that seem minor but are concrete
"""

_ITEM_QUALITY_CHECKLIST = """\
# QUALITY CHECKLIST

Before finalizing, confirm:
- [ ] Every proper noun (name, title, place) is kept exactly
- [ ] Every number and date is captured
- [ ] Time expressions include both relative and absolute forms
- [ ] Causal relationships are retained, not split
- [ ] Each item is self-contained and understandable alone
- [ ] At least 3-5 items per scene
"""

_ITEM_OUTPUT_FORMAT = """\
# OUTPUT FORMAT

Return JSON:
```json
{{
    "items": [
        {{
            "item_id": "1",
            "content": "Complete item with context",
            "scene_ids": ["scene_id_1"],
            "temporal": "yesterday (July 13, 2023)",
            "spatial": "location if applicable",
            "keywords": ["keyword1", "keyword2"],
            "query_patterns": ["Example query this could answer"]
        }}
    ],
    "reasoning": "Brief extraction strategy explanation"
}}
```

**Notes**:
- temporal: Use format "relative_phrase (absolute_date)" when applicable
- spatial: Location/place information, null if not applicable
- Emphasize completeness - more items is better than fewer
"""


_SEP = "\n---\n\n"

ITEM_EXTRACTION_PROMPT = (
    _ITEM_HEADER
    + _SEP + _ITEM_CORE_PRINCIPLES
    + _SEP + _ITEM_EXTRACTION_STRATEGY
    + _SEP + _ITEM_INFORMATION_INTEGRITY
    + _SEP + _ITEM_WHAT_TO_EXTRACT
    + _SEP + _ITEM_QUALITY_CHECKLIST
    + _SEP + _ITEM_OUTPUT_FORMAT
)
