"""Scene-related prompts: scene generation, custom instructions, and conversation-boundary detection."""

DEFAULT_CUSTOM_INSTRUCTIONS = """
Follow these principles when generating scene memories:
1. Each scene should be a complete, independent story or event
2. Preserve all important information including names, time, location, emotions, etc.
3. Use declarative language to describe scenes, not dialogue format
4. Highlight key information and emotional changes
5. Ensure scene content is easy to retrieve later
"""

_SCENE_HEADER = """
You are a scene memory generation specialist. Please transform the following conversation into a scene memory.

Conversation start time: {conversation_start_time}
Conversation content:
{conversation}

Custom instructions:
{custom_instructions}

IMPORTANT TIME HANDLING:
- Use the provided "Conversation start time" as the exact time when this conversation/scene began
- When the conversation mentions relative times (e.g., "yesterday", "last week"), preserve both the original relative expression AND calculate the absolute date
- Format time references as: "original relative time (absolute date)" - e.g., "last week (May 7, 2023)"
- This dual format supports both absolute and relative time-based questions
- All absolute time calculations should be based on the provided start time

Please generate a structured scene memory and return only a JSON object containing the following three fields:
{{
    "title": "A compact, descriptive title that precisely captures the theme (10-20 words)",
    "summary": "A short summary (2-4 sentences) that captures the core content and scenario of this scene. It should express WHO did WHAT in WHAT context, and is primarily used for matching this scene to a broader scenario. Focus on the key theme, main participants, and the situational context rather than exhaustive details.",
    "content": "A detailed factual account of the conversation in third-person narrative. It must include all important information: who participated at what time, what was discussed, what decisions were made, what emotions were expressed, and what plans or outcomes were formed. Write it as a chronological narrative focusing on observable actions and direct statements. Use the provided conversation start time as the base time for this scene."
}}

Requirements:
1. The title should be specific and easy to search (including key topics/activities).
2. Convert the dialogue format into a narrative description, using third-person unless explicitly first-person.
3. Include specific details that aid keyword search, especially concrete activities, places, and objects.
4. For time references, use the dual format: "relative time (absolute date)" to support different question types.
5. Use specific names consistently rather than pronouns to avoid ambiguity in retrieval.
6. When describing decisions or actions, naturally incorporate the reasoning or motivation behind them.
7. Keep chronological order and causal relationships throughout the content.
"""

_SCENE_EXAMPLE = """\
Example:
If the conversation start time is "March 14, 2024 (Thursday) at 3:00 PM UTC" and the conversation is about Kai planning a weekend bicycle trip:
{{
    "title": "Kai's Weekend Cycling Plan March 14, 2024: Coastal Route Preparation",
    "summary": "Kai and Yuki discussed plans for a weekend cycling trip along the coast. They covered route and gear preparation, with Kai planning to leave early Saturday to catch the sunrise at the lighthouse.",
    "content": "On March 14, 2024 at 3:00 PM UTC, Kai expressed interest in cycling this weekend (March 16-17, 2024) and sought advice. He wanted to see the sunrise at the coastal lighthouse. When asked about gear by Yuki, Kai received suggestions: spare inner tubes, padded shorts, a windbreaker, water, and high-energy snacks. Kai decided to leave early Saturday morning (March 16, 2024) to catch the sunrise and planned to invite friends. He was excited about the trip."
}}

Return only the JSON object, do not add any other text:
"""

SCENE_GENERATION_PROMPT = _SCENE_HEADER + _SCENE_EXAMPLE


_BOUNDARY_HEADER = """\
You are a scene memory boundary detection specialist. You need to decide whether the newly added dialogue should end the current scene and start a new one.

Current conversation history:
{conversation_history}

Time gap information:
{time_gap_info}

Newly added messages:
{new_messages}

Please carefully assess the following aspects to determine if a new scene should begin:

1. **Substantive Topic Change** (Highest Priority):
   - Do the new messages introduce a completely different substantive topic with meaningful content?
   - Is there a shift from one specific event/experience to another distinct event/experience?
   - Has the conversation moved from one meaningful question to an unrelated new question?

2. **Intent and Purpose Transition**:
   - Has the core purpose of the conversation changed notably?
   - Has the main question or issue of the current topic been fully settled and a new substantial topic begun?

3. **Meaningful Content Assessment**:
   - **IMPORTANT**: Ignore pure greetings, small talk, transition phrases, and social pleasantries
   - Focus only on content that would be memorable and worth recalling later
   - Consider: Would a person remember this as part of the main conversation topic or as a separate discussion?

4. **Structural and Temporal Signals**:
   - Are there explicit topic transition phrases introducing substantial new content?
   - Are there clear concluding statements followed by genuinely new topics?
   - Is there a notable time gap between messages?

5. **Content Relevance and Independence**:
   - How related is the new substantive content to the previous meaningful discussion?
   - Does it involve completely different events, experiences, or substantial topics?
"""

_BOUNDARY_PATTERNS = """\
**Special Rules for Common Patterns**:
- **Greetings + Topic**: "Hey!" followed by actual content should be ONE scene
- **Transition Phrases**: "By the way", "Oh, also", "Speaking of which" usually continue the same scene unless introducing major topic shifts
- **Social Closures and Farewells**: "Thanks!", "Take care!", "Talk to you soon!", "I'm off to go...", "See you later!" should continue the current scene as natural conversation endings
- **Supportive Responses**: Short encouragement or acknowledgment should usually continue the current scene
"""

_BOUNDARY_DECISION = """\
Decision Principles:
- **Prioritize meaningful content**: Each scene should contain substantive, memorable content
- **Ignore social formalities**: Don't split on greetings, pleasantries, brief transitions, or conversation closures
- **Treat closures as scene endings**: Messages that announce departure ("I'm off to go...", "Talk to you soon!") or provide closure ("Thanks!", "Take care!") should stay with the current scene as natural endings
- **Consider time gaps**: Long time gaps (hours or days) strongly suggest new scenes, while short gaps (minutes) usually indicate continuing conversation
- **Scene memory focus**: Think about what a person would naturally group together when recalling this conversation
- **Reasonable scene length**: Aim for scenes with 3-20 meaningful exchanges
- **When in doubt, consider context**: If unsure, keep related content together rather than over-splitting

Please return your judgment in JSON format:
{{
    "reasoning": "One sentence summary of your reasoning process",
    "should_end": true/false,
    "confidence": 0.0-1.0,
    "topic_summary": "If should_end = true, summarize the core meaningful topic of the current scene, otherwise leave it blank"
}}

Note:
- If conversation history is empty, this is the first message, return false
- Each scene should contain substantive content that stands alone as a meaningful memory unit
"""

CONV_BOUNDARY_DETECTION_PROMPT = _BOUNDARY_HEADER + _BOUNDARY_PATTERNS + _BOUNDARY_DECISION
