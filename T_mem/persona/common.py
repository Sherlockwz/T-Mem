# -*- coding: utf-8 -*-
"""Persona module constants. Scheduler R1/R3 thresholds; see scheduler.py for the algorithm."""

import os

from T_mem.config import MODELS


PROFILE_AGGREGATION_MIN_COUNT = 3
PROFILE_TA_MAX_ITEMS_PER_KEY = 3
PROFILE_TA_ALLOWED_KEYS = ("personality", "values", "attitudes", "beliefs")
PROFILE_SHARED_ACTIVITIES_MAX = 4
PROFILE_TIMELINE_MAX = 6
PROFILE_PREFERENCES_MAX_PER_KEY = 5
PROFILE_IDENTITY_ALLOWED_KEYS = (
    "age", "gender", "origin", "current_location", "occupation",
    "relationship_status", "financial_status", "education",
    "family_role", "languages", "ethnicity",
)

# CRITICAL: model id pulled from T_mem.config.MODELS — do NOT add an env fallback.
PERSONA_EXTRACT_MODEL = MODELS["memory_build"]
PERSONA_EXTRACT_TIMEOUT = int(os.environ.get("T_MEM_PERSONA_TIMEOUT", "300"))

PROFILE_DEBUG = os.environ.get("T_MEM_PROFILE_DEBUG", "0") == "1"

# R1 main trigger: buffer >= THIS utterances -> LLM extract + clear buffer.
PERSONA_BUFFER_THRESHOLD = 16

# R3 big-scene pre-intercept: incoming scene with >= THIS utterances -> flush
# any pending buffer (terminal), then extract this big scene alone (also terminal).
PERSONA_BIG_SCENE_THRESHOLD = 28
