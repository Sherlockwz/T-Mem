# Source: https://github.com/xjtuleeyf/Locomo-Plus (file: data/build_conv.py)
# Copyright (c) Locomo-Plus authors (xjtuleeyf). Vendored as-is for
# LoCoMo-Plus conversation stitching in T-Mem's stage0_locomo_plus_stitch.py.
import json
import random
import re
import copy
from datetime import datetime, timedelta


def parse_locomo_session_time(s):
    return datetime.strptime(s, "%I:%M %p on %d %B, %Y")


import re

_NUM_WORD = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12
}

def parse_time_gap(time_gap):
    s = time_gap.lower().strip()

    m = re.search(r"\b(a|an|about|around|approximately|approx\.?)\b", s)
    if m:
        approx = True
    else:
        approx = False

    m = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a|an)\b\s*(week|weeks|month|months|year|years)\b", s)
    if not m:
        return 0

    num, unit = m.groups()

    if num.isdigit():
        count = int(num)
    elif num in {"a", "an"}:
        count = 1
    else:
        count = _NUM_WORD.get(num, 0)

    if unit.startswith("week"):
        return count * 7
    if unit.startswith("month"):
        return count * 30
    if unit.startswith("year"):
        return count * 365
    return 0


def parse_ab_dialogue(text):
    turns = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("A:"):
            turns.append({"speaker": "A", "text": line[2:].strip()})
        elif line.startswith("B:"):
            turns.append({"speaker": "B", "text": line[2:].strip()})
    return turns


def map_speaker(dialogue, speaker_a, speaker_b):
    out = []
    for t in dialogue:
        x = copy.deepcopy(t)
        if x["speaker"] == "A":
            x["speaker"] = speaker_a
        else:
            x["speaker"] = speaker_b
        out.append(x)
    return out


def analyze_conversation(conv):
    speaker_a = conv["speaker_a"]
    speaker_b = conv["speaker_b"]

    sessions = []
    session_times = []

    idx = 1
    while True:
        sk = f"session_{idx}"
        tk = f"session_{idx}_date_time"
        if sk not in conv:
            break
        sessions.append(conv[sk])
        session_times.append(parse_locomo_session_time(conv[tk]))
        idx += 1

    return speaker_a, speaker_b, sessions, session_times


def compute_insertion(session_times, time_gap):
    query_time = session_times[-1] + timedelta(days=7)
    back_days = parse_time_gap(time_gap)
    cue_time = query_time - timedelta(days=back_days)

    cue_idx = None
    for i, t in enumerate(session_times):
        if t <= cue_time:
            cue_idx = i
        else:
            break

    return cue_idx, cue_time, query_time


def build_context(plus_item, locomo_item):
    conv = locomo_item["conversation"]

    speaker_a, speaker_b, sessions, session_times = analyze_conversation(conv)

    cue_idx, cue_time, query_time = compute_insertion(
        session_times,
        plus_item["time_gap"]
    )

    cue_turns = map_speaker(
        parse_ab_dialogue(plus_item["cue_dialogue"]),
        speaker_a,
        speaker_b
    )

    query_turns = map_speaker(
        parse_ab_dialogue(plus_item["trigger_query"]),
        speaker_a,
        speaker_b
    )

    cue_anchor = None
    if cue_idx is not None:
        cue_anchor = sessions[cue_idx][-1]

    query_anchor = sessions[-1][-1]

    events = []

    for sess, t in zip(sessions, session_times):
        events.append((t, sess))

    events.append((cue_time, cue_turns))
    events.append((query_time, query_turns))

    events.sort(key=lambda x: x[0])

    stitched = []
    for _, content in events:
        stitched.extend(content)

    # stitched = []

    # for i, sess in enumerate(sessions):
    #     stitched.extend(sess)
    #     if cue_idx is not None and i == cue_idx:
    #         stitched.extend(cue_turns)

    # stitched.extend(query_turns)
    # import pdb; pdb.set_trace()
    return {
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "time_gap": plus_item["time_gap"],
        "cue_session_idx": cue_idx,
        "cue_time": cue_time.strftime("%Y-%m-%d %H:%M"),
        "query_time": query_time.strftime("%Y-%m-%d %H:%M"),
        "cue_turns": cue_turns,
        "query_turns": query_turns,
        "dialogue": stitched
    }


if __name__ == "__main__":
    with open("../data/locomo_plus.json", "r") as f:
        locomo_plus = json.load(f)

    with open("../data/locomo10.json", "r") as f:
        locomo_raw = json.load(f)

    outputs = []

    for plus in locomo_plus:
        locomo_item = random.choice(locomo_raw)
        ctx = build_context(plus, locomo_item)
        outputs.append(ctx)

    with open("stitched_contexts.json", "w") as f:
        json.dump(outputs, f, indent=2)