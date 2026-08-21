"""Topic-related prompts: extraction (new topic), update (extend existing), and scene-vs-topic matching."""

_TOPIC_EXTRACTION_BODY = """\
You are a topic extraction specialist focusing on identifying specific situations.

Your task: Extract a topic representing ONE SPECIFIC situation/event/theme.

**IMPORTANT: A topic should represent ONE specific, identifiable situation**
- NOT a broad category (e.g., "work discussions")
- BUT a specific event/journey/ongoing discussion (e.g., "Project Alpha launch preparation")

SCENE:
{scenes}

**Topic Definition Guidelines**:

1. **Specificity** (CRITICAL):
   - Identify the ONE specific situation this scene describes
   - Be precise: "Alice's piano learning journey" not "music learning"
   - A topic should have a clear subject (person, project, event)

2. **Title Guidelines** (3-10 words):
   - Include the SPECIFIC subject; be concrete, not abstract
   - Good: "Jon's Dance Studio Launch Journey" | Bad: "Career Development" (too broad)
   - Good: "Product X Marketing Campaign" | Bad: "Marketing Activities" (too vague)

3. **Keyword Guidelines**:
   - Extract ALL relevant keywords: person names, locations, activities, objects, emotions, time references
   - Include both specific terms (e.g., "piano recital") and broader terms (e.g., "music")
   - More keywords improve retrieval accuracy — aim for 15+ keywords per topic

Return JSON format:
{{
    "title": "Specific, focused topic title",
    "keywords": ["keyword1", "keyword2", "keyword3", "...aim for 15+ keywords"],
    "extend": {{
        "topic_type": "work/social/leisure/personal_development/etc",
        "key_subjects": ["main person/project/entity"],
        "situation_type": "journey/event/project/relationship/etc"
    }}
}}

Focus on creating a topic that represents ONE identifiable, specific situation.
"""

TOPIC_EXTRACTION_PROMPT = _TOPIC_EXTRACTION_BODY


TOPIC_UPDATE_PROMPT = """You are an expert in updating topics while maintaining their specific identity.

Your task: Update the topic by incorporating new developments in THE SAME situation.

EXISTING TOPIC:
{existing_topic}

NEW SCENE (continuing the same situation):
{new_scene}

**Update Principles**:

1. **Maintain Topic Identity and Focus**:
   - The topic represents ONE specific situation
   - New scene adds to/develops this SAME situation
   - Do NOT broaden the scope to different situations or aspects
   - If topic already has 12+ scenes, be VERY strict about adding more
   - If topic has 15+ scenes, consider if it's becoming too broad

2. **Update Keywords**:
   - ADD new keywords from the new scene to the existing keyword list
   - Never remove existing keywords — only add
   - Include person names, locations, activities, objects, emotions, time references from the new scene

3. **Title Stability**:
   - Usually keep the title (it identifies the situation)
   - Only adjust if new scene reveals more specific identity
   - Example: "Jon's Business" → "Jon's Dance Studio Launch"

Return JSON format:
{{
    "title": "Keep or refine to maintain specific identity",
    "keywords": ["all", "existing", "keywords", "plus", "new", "ones"],
    "extend": {{
        "topic_type": "keep or refine",
        "update_note": "What new development was added"
    }}
}}

Keep the topic focused on its ONE specific situation.

**REMINDER**:
- Topics should be rich and contextualized (5-10 scenes ideal, max 15)
- Continuously aggregating related developments of THE SAME situation over time is GOOD
- BUT do NOT broaden the topic to cover DIFFERENT aspects or topics
- The topic should tell ONE complete, focused story with multiple time points
"""


_TOPIC_MATCH_INTRO = """\
You are an expert at determining whether a memory scene belongs to an existing topic.

## What is a "topic"?

A topic is a **specific, identifiable event thread or activity line**. It represents ONE concrete thing that is happening, not a broad life category. Conversations often jump between topics — two people might discuss Topic A, switch to Topic B, then return to Topic A. Scenes from Topic A should be grouped together even if they are not consecutive.

## Key principle: specificity over breadth

The most common mistake is making topics too broad. A topic should be narrow enough that you could give it a **specific, concrete name** — not a vague category.

Good topic names (specific): "Training for the April marathon", "Adopting a rescue dog from the shelter", "Debugging the payment API issue"
Bad topic names (too broad): "Health and fitness activities", "Personal life updates", "Work discussions"

If a topic already has a broad/vague name like "daily catch-up" or "personal life updates", that is a sign the topic was poorly defined. New scenes should NOT match such overly broad topics — they deserve their own specific topic instead.

## Matching criteria

A scene belongs to a topic when it describes:
1. A **direct continuation** of the same event (e.g., preparation → execution → aftermath of ONE event)
2. A **natural follow-up** or update on the same situation (e.g., applied for a job → got an interview → received an offer)
3. A **return to the same thread** after discussing other things (conversations often jump between topics and come back)

A scene does NOT belong to a topic when:
1. They merely share a **broad category** (both about "fitness", both about "work") but are different specific activities
2. The topic name is **too vague** to represent a real event thread
3. They involve the **same people** but are about a **different matter**
"""

_TOPIC_MATCH_EXAMPLES = """\
## Examples

SAME topic (true):
- Topic: "Training for the April marathon"
- Scene: "Bought new running shoes for the marathon"
→ true. Same specific event: preparing for that particular marathon.

SAME topic (true):
- Topic: "Planning the Europe trip"
- Scene: "Sorting through photos from the Europe trip"
→ true. Same specific trip, different phase (aftermath).

DIFFERENT topic (false):
- Topic: "Training for the April marathon"
- Scene: "Started taking yoga classes on weekends"
→ false. Both are fitness activities, but they are different activity lines.

DIFFERENT topic (false):
- Topic: "Work stress and career concerns"
- Scene: "Talked about feeling overwhelmed with childcare"
→ false. The topic name is too broad. Childcare stress is a separate life thread.
"""

_TOPIC_MATCH_TASK = """\
## Your task

For each existing topic below, determine whether the given scene belongs to it.

SCENE:
Subject: {scene_subject}
Summary: {scene_summary}

TOPICS (total {num_topics}):
{topics_text}

Return JSON format:
{{
    "results": [
        {{"topic_id": "topic_1", "match": true/false}},
        ...
    ]
}}

Rules:
- Output true when the scene is clearly part of the SAME specific event thread — including direct continuations, follow-ups, and returns to the same thread.
- If the topic name is vague or overly broad, lean towards false.
- A scene CAN match multiple topics if it genuinely bridges two specific event threads.
- If the scene does not match ANY topic, return all false — a new topic will be created for it.
- When in doubt, output false. It is better to create a new specific topic than to pollute an existing one.
"""

TOPIC_MATCH_PROMPT = _TOPIC_MATCH_INTRO + _TOPIC_MATCH_EXAMPLES + _TOPIC_MATCH_TASK
