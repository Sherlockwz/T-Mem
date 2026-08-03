"""Scene-related prompts: scene generation, custom instructions, and conversation-boundary detection."""

DEFAULT_CUSTOM_INSTRUCTIONS = """
Follow these principles when generating scene memories:
1. Each scene should be a complete, independent story or event
2. Preserve all important information including names, time, location, emotions, etc.
3. Use declarative language to describe scenes, not dialogue format
4. Highlight key information and emotional changes
5. Ensure scene content is easy to retrieve later
"""

SCENE_GENERATION_PROMPT = """
You are a scene memory generation expert. Please convert the following conversation into a scene memory.

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
    "title": "A concise, descriptive title that accurately summarizes the theme (10-20 words)",
    "summary": "A brief summary (2-4 sentences) that captures the core content and scenario of this scene. It should convey WHO did WHAT in WHAT context, and is primarily used for matching this scene to a broader scenario. Focus on the key theme, main participants, and the situational context rather than exhaustive details.",
    "content": "A detailed factual record of the conversation in third-person narrative. It must include all important information: who participated at what time, what was discussed, what decisions were made, what emotions were expressed, and what plans or outcomes were formed. Write it as a chronological account focusing on observable actions and direct statements. Use the provided conversation start time as the base time for this scene."
}}

Requirements:
1. The title should be specific and easy to search (including key topics/activities).
2. The content must include all important information from the conversation.
3. Convert the dialogue format into a narrative description.
4. Maintain chronological order and causal relationships.
5. Use third-person unless explicitly first-person.
6. Include specific details that aid keyword search, especially concrete activities, places, and objects.
7. For time references, use the dual format: "relative time (absolute date)" to support different question types.
8. When describing decisions or actions, naturally include the reasoning or motivation behind them.
9. Use specific names consistently rather than pronouns to avoid ambiguity in retrieval.

Example:
If the conversation start time is "March 14, 2024 (Thursday) at 3:00 PM UTC" and the conversation is about Caroline planning to go hiking:
{{
    "title": "Caroline's Mount Rainier Hiking Plan March 14, 2024: Weekend Adventure Planning Session",
    "summary": "Caroline and Melanie discussed plans for a weekend hiking trip to Mount Rainier. They covered gear preparation and logistics, with Caroline planning to leave early Saturday to catch the sunrise.",
    "content": "On March 14, 2024 at 3:00 PM UTC, Caroline expressed interest in hiking this weekend (March 16-17, 2024) and sought advice. She wanted to see the sunrise at Mount Rainier. When asked about gear by Melanie, Caroline received suggestions: hiking boots, warm clothing, flashlight, water, and high-energy food. Caroline decided to leave early Saturday morning (March 16, 2024) to catch the sunrise and planned to invite friends. She was excited about the trip."
}}

Return only the JSON object, do not add any other text:
"""


CONV_BOUNDARY_DETECTION_PROMPT = """
You are a scene memory boundary detection expert. You need to determine if the newly added dialogue should end the current scene and start a new one.

Current conversation history:
{conversation_history}

Time gap information:
{time_gap_info}

Newly added messages:
{new_messages}

Please carefully analyze the following aspects to determine if a new scene should begin:

1. **Substantive Topic Change** (Highest Priority):
   - Do the new messages introduce a completely different substantive topic with meaningful content?
   - Is there a shift from one specific event/experience to another distinct event/experience?
   - Has the conversation moved from one meaningful question to an unrelated new question?

2. **Intent and Purpose Transition**:
   - Has the fundamental purpose of the conversation changed significantly?
   - Has the core question or issue of the current topic been fully resolved and a new substantial topic begun?

3. **Meaningful Content Assessment**:
   - **IMPORTANT**: Ignore pure greetings, small talk, transition phrases, and social pleasantries
   - Focus only on content that would be memorable and worth recalling later
   - Consider: Would a person remember this as part of the main conversation topic or as a separate discussion?

4. **Structural and Temporal Signals**:
   - Are there explicit topic transition phrases introducing substantial new content?
   - Are there clear concluding statements followed by genuinely new topics?
   - Is there a significant time gap between messages?

5. **Content Relevance and Independence**:
   - How related is the new substantive content to the previous meaningful discussion?
   - Does it involve completely different events, experiences, or substantial topics?

**Special Rules for Common Patterns**:
- **Greetings + Topic**: "Hey!" followed by actual content should be ONE scene
- **Transition Phrases**: "By the way", "Oh, also", "Speaking of which" usually continue the same scene unless introducing major topic shifts
- **Social Closures and Farewells**: "Thanks!", "Take care!", "Talk to you soon!", "I'm off to go...", "See you later!" should continue the current scene as natural conversation endings
- **Supportive Responses**: Brief encouragement or acknowledgment should usually continue the current scene

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
- Focus on scene memory principles: what would people naturally remember as distinct experiences?
- Each scene should contain substantive content that stands alone as a meaningful memory unit
"""
