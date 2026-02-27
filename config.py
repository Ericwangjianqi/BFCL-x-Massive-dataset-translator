"""
Central configuration for the translation tool.

TRANSLATE_FIELDS: list of field paths to translate. Supports three notations:
    "title"                    – simple top-level key (string value)
    "metadata.description"     – nested key via dot notation
    "question[*][*].content"   – wildcard [*] iterates over every list element

FILE_FORMAT:
    "json"   – file contains a single JSON value (object or array)
    "jsonl"  – file contains one JSON object per line (JSON Lines)

Leave TRANSLATE_FIELDS empty ([]) to auto-translate every string leaf.
"""

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL = "gpt-4o-mini"          # change to "gpt-4o", "gpt-3.5-turbo", etc.
TEMPERATURE = 0.2              # lower = more consistent/literal translations

# ── File format ────────────────────────────────────────────────────────────────
# "json"  → standard JSON file  (array or single object)
# "jsonl" → JSON Lines, one record per line
FILE_FORMAT = "jsonl"

# ── Fields to translate ────────────────────────────────────────────────────────
# Each entry is a field path.  Use [*] to iterate over list elements.
# Current target: the "content" of every user turn inside "question".
TRANSLATE_FIELDS: list[str] = [
    "question[*][*].content",
]

# ── Processing ─────────────────────────────────────────────────────────────────
# How many text strings to send in a single API call.
# Lower this if you hit token-limit errors.
BATCH_SIZE = 10

# ── Judge (Gemini) ─────────────────────────────────────────────────────────────
# Set USE_JUDGE = True to enable Gemini as a translation quality reviewer.
# After each GPT translation batch, Gemini checks:
#   1. Grammar correctness
#   2. Naturalness (no stiff/literal translations)
#   3. Proper noun preservation (file names, paths, abbreviations, etc.)
# If Gemini flags an issue, GPT retranslates with the feedback.
USE_JUDGE       = True
JUDGE_MODEL     = "gemini-2.5-pro"
MAX_JUDGE_RETRIES = 1   # how many fix attempts per failed translation

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
RESULT_DIR = "result"
