# -*- coding: utf-8 -*-
"""Persona extraction prompt: 6-key delta (identity/preferences/traits/relations/aggregations/timeline).
Consumed by qa_support.load_persona_block via str.format(speaker_name, current_time, current_profile_json, chat_current)."""

PROFILE_EXTRACT_PROMPT = '''You are an expert at building structured speaker profiles from dialogue. Your task: read the conversation window below and output a profile DELTA for ONE specific speaker ({speaker_name}) — only the NEW facts this window reveals, merged against what is already known.

Your extraction will be used for downstream multi-hop QA such as "How many X has {speaker_name} done?", "What are all the Y {speaker_name} likes?", "Who is {speaker_name} close to?" — so COMPLETENESS and SPECIFICITY of the delta directly determine whether those questions can be answered.

---

# CORE PRINCIPLES

## 1. Completeness
Extract EVERY fact about {speaker_name} that this window reveals and is not already in the current profile. When in doubt, extract it. Missing a fact is worse than emitting a slightly redundant one (dedup will clean up).

## 2. Specificity
Always prefer specific over general:
- Exact names / titles / brands over "a book", "someone", "a game"
- Exact numbers over "some", "several", "many"
- Concrete dates over vague "recently"
- Concrete activities ("made clay pottery bowl") over abstract summaries ("made pottery")

## 3. Preserve Granularity — Do NOT Over-Merge
If two items differ in shape, function, brand, or role, keep them as SEPARATE items. This is the single most common failure mode on this task.

- Wrong: `favorite_foods: ["pottery items"]` (abstract roll-up)
- Right: `favorite_foods: ["blueberry muffins", "chicken pot pie", "sushi"]`
- Wrong: one timeline event "made pottery" covering 4 separate sessions
- Right: four separate timeline events — "made clay bowl with flower design", "made pot at workshop", "made plate with flowers", "made colorful pottery bowl"
- Wrong: one relation "friends" covering several people
- Right: one relation entry per named person

The ONLY merge allowed is `traits_and_attitudes` (see that section).

## 4. Weak Signals Matter
People mentioned only 2-3 times, passing hobby mentions, throwaway dislikes, and off-hand counts ("I've written three of those") are exactly the signals downstream QA depends on. Treat them with the same seriousness as main-topic facts. Do NOT drop something because it "feels minor".

---

# EXTRACTION STRATEGY (three passes, in order)

## Pass 1 — Event scan (timeline)
Walk through the window chronologically. For every discrete action {speaker_name} actually performed, create a timeline entry. This is your foundation.

## Pass 2 — Counter roll-up (aggregations)
After Pass 1, look back at the timeline entries you just created PLUS anything the speaker quantifies in the window ("I've done three of those", "my fourth screenplay"). For each repeatable pattern, emit a counter with the DELTA count for this window. Do NOT skip this pass — downstream "how many" questions depend entirely on aggregations being populated. See the dedicated section below for how.

## Pass 3 — Relation harvest (relations)
Re-scan the window for EVERY other person / pet / group {speaker_name} interacts with or mentions by name ≥ 2 times. Create a relation entry for each — even for weak / peripheral relations (writers group, brother, neighbour, mentor). Do NOT merge multiple distinct people into a generic "friends" entry.

Pass 1-3 are cumulative, not replacements — the final JSON contains output from all three.

---

# WHAT TO EXTRACT / WHAT TO SKIP

## Always extract (common misses)
- Quantified statements even when casual: "my fourth screenplay", "I've written three"
- Food / book / game / music NAMES whenever stated, even in passing
- Dislikes and allergies (often dropped — don't)
- Named places the speaker has been to (cities, trails, restaurants, events)
- Any other person mentioned by name ≥ 2 times in this window
- Fine-grained creative output (bowl vs plate vs cup are different items)

## Do NOT extract
- The OTHER speaker's facts (record them only as a `relations` entry of {speaker_name})
- Raw conversation text, greetings, jokes, hypothetical statements, momentary feelings
- Pure plans / intentions / meta-conversation ("plans to X", "last talked with X", "discussed X")
- Anything already present in `{current_profile_json}` with equal or greater detail

## Time rules
- Use absolute dates (YYYY-MM-DD / YYYY-MM / YYYY) when stated or easily derivable from `{current_time}`.
- Otherwise keep the original phrase verbatim ("when she was younger", "last summer"). Do NOT invent dates.
- If the annotation gives a resolved date ("(yesterday->2023-01-19)"), use the resolved date.

---

# INPUT

Reference time: {current_time}
Target speaker: {speaker_name}

Current profile (skip anything already covered here with equal or greater detail):
{current_profile_json}

Conversation window (extract the delta from here):
{chat_current}

---

# OUTPUT SCHEMA

Return ONE JSON object with EXACTLY these 6 top-level keys. Every key must exist, though individual values may be empty.

{{
  "identity": {{}},
  "preferences": {{}},
  "traits_and_attitudes": {{}},
  "relations": [],
  "aggregations": {{}},
  "timeline": []
}}

---

## FIELD 1 — identity (dict[str, str], OVERWRITE)

One string per key. Use ONLY these allowed keys (never invent new ones):
  age, gender, origin, current_location, occupation, relationship_status, financial_status, education, family_role, languages, ethnicity

- Preserve multi-word descriptors whole ("transgender woman", not "transgender").
- If the window reveals a NEW value for an existing key (e.g. occupation changed), emit the NEW value — the framework will overwrite.
- Pet info goes to `relations` (type: "pet"), NOT here.

Always-extract within identity:
- `origin` when the speaker says where they're from / grew up / moved from.
- `current_location` when the speaker names the city / region they now live in.
- `occupation` when the speaker states any job role (even informally: "I write screenplays" → "screenwriter").
- `family_role` when the speaker mentions being a parent / sibling / spouse / child.

---

## FIELD 2 — preferences (dict[str, list[str]], APPEND)

Allowed keys (descriptive plurals): `favorite_books`, `favorite_movies`, `favorite_games`, `favorite_foods`, `favorite_music`, `favorite_sports`, `hobbies`, `dislikes`.

Each item = concrete named title / brand / activity, verbatim from the window. Skip items already present in the current profile.

Rules:
- No roll-ups. Keep distinct items distinct.
  - Wrong: `favorite_foods: ["various dairy-free desserts"]`
  - Right: `favorite_foods: ["dairy-free chocolate cake", "chocolate raspberry tart", "coconut milk ice cream"]`
- `dislikes` is first-class — include allergies, foods they avoid, animals they fear, activities they refuse. These get dropped most often; don't.
  - Wrong: omit "allergic to cockroaches"
  - Right: `dislikes: ["cockroaches", "dairy", "reptiles"]`
- For `hobbies`, prefer the concrete activity form ("playing violin", "pottery", "screenwriting") over abstract nouns.

---

## FIELD 3 — traits_and_attitudes (dict[str, list[str]], APPEND, FIXED 4 SUB-KEYS)

Use EXACTLY these 4 sub-keys (omit if empty; NEVER invent new ones):
  - `personality` — 1-3 word tags. e.g. ["empathetic", "resilient"]
  - `values` — 1-3 word tags. e.g. ["family-oriented", "creativity"]
  - `attitudes` — one declarative sentence per item (≤ 20 words). e.g. ["Supports LGBTQ+ rights."]
  - `beliefs` — one declarative sentence per item (≤ 20 words). e.g. ["Believes art is essential for self-expression."]

This is the ONE field that favours consolidation:
- MAX 6 items per sub-key.
- Consolidate, don't split: multiple attitudes toward the SAME object merge into ONE sentence.
  - Wrong: ["Believes art is healing.", "Believes art is therapeutic."]
  - Right: ["Views art as a healing, therapeutic outlet."]
- No per-object sub-keys (never create `view_on_art`; put it as an `attitudes` sentence).
- Skip items already in `{current_profile_json}.traits_and_attitudes`, including obvious paraphrases.

---

## FIELD 4 — relations (list[dict], MERGE-by-person)

Each item shape: {{"person": "<name>", "type": "<partner|friend|family|colleague|mentor|pet|group|...>", "nickname": "<optional>", "shared_activities": ["<act1>"]}}

### MIN coverage rule
In Pass 3, create one relation entry for EVERY other person / pet / group that {speaker_name} mentions or interacts with named ≥ 2 times in this window. Do NOT skip weak / peripheral relations.
- Wrong (common failure): relations = [{{"person": "Nate", ...}}] only, because Nate is the other speaker.
- Right: relations = [Nate, writers group, brother, Tilly (dog), …] — every named person/group gets their own entry.
- Prefer a named individual or a specifically-labeled group (e.g. "Sam", "grandma", "Connected LGBTQ Activists") as `person`; avoid bare collective nouns like "friends" / "family" / "mentors" unless the window gives no more specific handle.

### shared_activities rules
- MAX 15 items per person; emit only NEW activities for this window.
- Each activity = short concrete verb phrase (2-6 words). Use the real verb+object, not a summary.
  - Wrong: "provided emotional support"
  - Right: "encouraged persistence after rejection letter"
- Skip activities already present (including obvious paraphrases like "giving marketing advice" vs "giving business advice").
- Keep distinct activities distinct. Do NOT collapse 5 camping trips into "camping".

---

## FIELD 5 — aggregations (dict[str, int], INCREMENT — DELTA ONLY)

This is the highest-value, easiest-missed field. Downstream "how many" QA lives entirely in this field — if you under-populate it, those questions are unanswerable.

### How to build it (do this explicitly in Pass 2)
1. Look at the timeline entries you emitted in Pass 1.
2. Group them by repeatable pattern (e.g. "went hiking", "attended concert", "wrote screenplay", "ran charity race", "went camping with family").
3. For each group that has ≥ 1 item in THIS window, emit a snake_case counter with the NEW count (the delta, NOT the running total).
4. Also catch explicit quantifications from the speaker: "I've written three screenplays" → if three haven't been seen before and this is the first mention, emit `screenplays_written: 3`. If the counter already exists in `{current_profile_json}.aggregations`, emit only the delta.

### Counter naming
- Reuse existing keys from `{current_profile_json}.aggregations` whenever a new event fits an existing counter.
- Introduce new snake_case keys when a new repeatable pattern first appears: `tournaments_won`, `blog_posts_written`, `pottery_pieces_made`, `hikes_completed`, `rejection_letters_received`, `concerts_attended`, `road_trips_taken`.
- Granularity: separate counters for separate activity types. Do NOT collapse `bowls_made` + `plates_made` into `pottery_pieces_made` unless the window lacks detail.

### Examples — what SHOULD appear
Given a window with timeline events like [hiking, hiking, camping, baking dairy-free dessert, baking dairy-free dessert, writing screenplay]:
- Wrong: aggregations = {{}} (empty — this is the default failure mode; do NOT do this)
- Wrong: aggregations = {{"events_completed": 6}} (too generic)
- Right: aggregations = {{"hikes_completed": 2, "camping_trips": 1, "dairy_free_desserts_made": 2, "screenplays_written": 1}}

### Delta semantics
- If the speaker says "finished my THIRD screenplay" and the current profile has no screenplay counter, emit `screenplays_written: 3` (initial value). Don't just emit `1`.
- If the current profile already has `screenplays_written: 2` and this window shows finishing one more, emit `screenplays_written: 1` (the delta).
- Omit a key entirely if there are zero new events for it.

---

## FIELD 6 — timeline (list[dict], APPEND)

Each item shape: {{"event": "<verb + object, subject omitted>", "date": "<date or relative phrase>", "date_range": "<optional>", "event_type": "<optional short tag>"}}

### Hard rules
1. **Subject omitted.** The subject is always {speaker_name}; never write their name inside `event`. If the true subject of an action is NOT {speaker_name} (e.g. "dance studio was struggling"), drop the item.
   - Right: "opened online store"
   - Wrong: "{speaker_name} opened her online store"
2. **Actually happened only.** An event is a DISCRETE PAST ACTION THAT OCCURRED, not a state / plan / chat topic.
   - Drop: "is working on X" / "plans to X" / "last talked with X" / "discussed X" / "was preparing X" / "felt motivated"
   - Keep: "launched X" / "attended X" / "won X" / "bought X" / "opened X"
3. **Name a concrete object.** Every event must include at least one specific entity: proper noun (person / place / brand / title), role / job title, quantified modifier (amount / count / distance), or creative work name. Generic nouns alone aren't enough.
   - Right: "bought 'The Midnight Library'"
   - Right: "lost banker job at DoorDash"
   - Wrong: "bought a book" → drop if no title available
   - Wrong: "attended networking event" → upgrade to "attended dance-industry networking event" or drop
4. **Every item needs a date.** If unknown, use `{current_time}`. Keep `event` text to ≤ 25 tokens, typical 6-15.
5. **Dedup before emitting.** Skip any (date, event) pair already in `{current_profile_json}.timeline` (case-insensitive paraphrase check).

### One event = one item (with one narrow merge)
- Different themes on same day → SPLIT ("got tattoo" + "finished book" stay separate).
- Same theme on same day → MERGE into one entry using "and" / "plus": "prepared investor pitch: refined plan and updated deck".
- Do NOT merge across dates. Do NOT merge across themes.

### Preserve granularity in events
If the speaker describes four distinct creative outputs across four dates, emit four events — do NOT collapse to one generic "made art".
   - Wrong: one event "made pottery"
   - Right: "made clay bowl with flower design" (date A), "made pot at workshop" (date B), "made plate with flowers" (date C), "made colorful pottery bowl" (date D)

---

# GENERAL RULES

1. Every specific fact (time / place / number / title / brand / named person) about {speaker_name} in the window should land in SOME field of the delta.
2. Before emitting, compare each candidate against `{current_profile_json}` and skip near-duplicates.
3. Only profile {speaker_name}; the other speaker is only a `relations` entry.
4. **Strict JSON output.** All 6 top-level keys present, no markdown fences, no commentary, no trailing text.

---

# QUALITY CHECKLIST (run mentally before outputting)

- [ ] Did I run Pass 1 (timeline events) through the whole window?
- [ ] Did I run Pass 2 and emit aggregations — is the aggregations dict non-empty if Pass 1 found ≥ 2 events of any repeatable pattern?
- [ ] Did I run Pass 3 and include every other named person / pet / group mentioned ≥ 2 times as a separate relations entry?
- [ ] Did I preserve distinct items as distinct (no "pottery items", no "various desserts", no "events attended")?
- [ ] Did I capture any `dislikes`, allergies, or things-they-avoid stated in this window?
- [ ] Did I capture quantified statements ("my fourth screenplay", "I've run 3 charity races") as aggregations with correct delta values?
- [ ] Did I drop pure plans / states / meta-conversation from timeline?
- [ ] Is the output strict JSON with all 6 top-level keys?

---

# OUTPUT FORMAT

Return ONLY the JSON object. Illustrative example (content not representative):

```json
{{
  "identity": {{"occupation": "freelance illustrator", "gender": "transgender woman", "origin": "Michigan"}},
  "preferences": {{
    "favorite_games": ["Xenoblade Chronicles"],
    "favorite_foods": ["ginger snaps", "dairy-free chocolate cake"],
    "dislikes": ["cockroaches", "dairy"],
    "hobbies": ["screenwriting", "pottery", "hiking"]
  }},
  "traits_and_attitudes": {{
    "personality": ["introverted", "empathetic"],
    "values": ["family-oriented"],
    "attitudes": ["Prefers staying in current city due to family ties."],
    "beliefs": ["Believes art is essential for self-expression."]
  }},
  "relations": [
    {{"person": "Sam", "type": "close friend", "nickname": "", "shared_activities": ["hiked Mount Rainier", "traded book recommendations"]}},
    {{"person": "writers group", "type": "group", "nickname": "", "shared_activities": ["shared screenplay draft", "received feedback on 'Finding Home'"]}},
    {{"person": "Tilly", "type": "pet", "nickname": "", "shared_activities": []}}
  ],
  "aggregations": {{"hikes_completed": 2, "screenplays_written": 1, "rejection_letters_received": 1}},
  "timeline": [
    {{"event": "opened online store", "date": "2023-03-16", "event_type": "milestone"}},
    {{"event": "hiked Mount Rainier with Sam", "date": "2023-03-18", "event_type": "hike"}},
    {{"event": "finished third screenplay about loss, identity, and connection", "date": "2023-05-20", "event_type": "milestone"}}
  ]
}}
```
</Output Format>'''
