"""
Gemini-based translation quality judge.

For each (original, translation) pair it checks:
  1. Grammar — is the translation grammatically correct?
  2. Naturalness — does it sound like a native speaker, not a literal rendering?
  3. Proper noun preservation — are file names, paths, abbreviations, technical
     terms, etc. kept untranslated?

Returns a verdict per pair:
  {"ok": True}
  {"ok": False, "feedback": "<explanation + suggested fix>"}
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from tenacity.stop import stop_base
from tenacity.wait import wait_base

load_dotenv()

_client: genai.Client | None = None

JUDGE_SYSTEM_PROMPT = """\
You are a strict translation quality reviewer.

You will receive a list of translation pairs — each containing an original English text \
and its translation into a target language — along with the target language name.

For every pair evaluate these three criteria:

1. Grammar
   Is the translation grammatically correct in the target language?

2. Naturalness
   Does it sound natural and fluent, like something a native speaker would write?
   Flag stiff, awkward, or word-for-word literal renderings.

3. Proper noun preservation
   The following must NEVER be translated — they must appear exactly as in the source:
   - File names and extensions (e.g. report.pdf, config.yaml, Annual_Report_2023.docx)
   - Directory / folder names used as proper nouns (e.g. workspace, documents, temp)
     — only the generic word "folder" / "directory" may be translated
   - Technical abbreviations: CWD, CLI, API, CPU, RAM, etc.
   - Shell commands, flags (e.g. grep, sort, diff, --output)
   - Function names, class names, API names (e.g. GorillaFileSystem, post_tweet)
   - Variable names and code identifiers
   - URLs, email addresses, domain names

Output format — return ONLY a valid JSON array, one object per pair, same order:
[
  {"ok": true},
  {"ok": false, "feedback": "Explanation of the issue and guidance on how to fix it. Do NOT provide a full rewritten sentence."},
  ...
]

IMPORTANT: When providing feedback, do NOT provide a specific rewritten sentence.
Instead, explain the issue and give general guidance or suggestions (e.g., "The tone is too formal", "The word X is unnatural here, consider using Y", "Do not translate the variable name Z").

Do NOT output anything outside the JSON array.
"""


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file (see .env.example)."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _is_infinite_retry(exc: BaseException) -> bool:
    """Check if the error should trigger an infinite retry loop (Quota or Server Overloaded)."""
    if isinstance(exc, ClientError) and exc.code == 429:
        return True
    if isinstance(exc, ServerError) and exc.code == 503:
        return True
    return False


def _parse_retry_delay(exc: BaseException, default: float = 30.0) -> float:
    """Extract the suggested retry delay (seconds) from the error message."""
    import re
    msg = str(exc)
    m = re.search(r'retry[^\d]*(\d+(?:\.\d+)?)\s*s', msg, re.IGNORECASE)
    return float(m.group(1)) if m else default


class _SmartWait(wait_base):
    """Suggested-delay wait for quota errors; exponential back-off for everything else."""

    _exp = wait_exponential(multiplier=1, min=2, max=10)

    def __call__(self, retry_state: object) -> float:
        exc = retry_state.outcome.exception()  # type: ignore[union-attr]
        if _is_infinite_retry(exc):
            delay = _parse_retry_delay(exc)
            print(f"\n  [WARN] Gemini unavailable (429/503) — waiting {delay:.0f} s before retry …")
            return delay
        return self._exp(retry_state)


class _SmartStop(stop_base):
    """Never stop on quota/server errors; stop after 3 attempts for everything else."""

    _limit = stop_after_attempt(3)

    def __call__(self, retry_state: object) -> bool:
        exc = retry_state.outcome.exception()  # type: ignore[union-attr]
        if exc is not None and _is_infinite_retry(exc):
            return False
        return self._limit(retry_state)


@retry(
    retry=retry_if_exception(lambda e: isinstance(e, Exception)),
    wait=_SmartWait(),
    stop=_SmartStop(),
    reraise=True,
)
def judge_batch(
    originals: list[str],
    translations: list[str],
    target_language: str,
    judge_model: str,
) -> list[dict]:
    """
    Ask Gemini to review a batch of (original, translation) pairs.

    Args:
        originals:       Source English strings.
        translations:    Translated strings from GPT (same order).
        target_language: e.g. "Chinese", "French".
        judge_model:     Gemini model identifier.

    Returns:
        List of dicts: {"ok": bool} or {"ok": False, "feedback": str}.
        Length always equals len(originals).
    """
    pairs = [
        {"original": o, "translation": t}
        for o, t in zip(originals, translations)
    ]

    user_message = (
        f"Target language: {target_language}\n\n"
        f"Translation pairs to review:\n"
        f"{json.dumps(pairs, ensure_ascii=False, indent=2)}\n\n"
        f"Return only the JSON array."
    )

    client = _get_client()
    response = client.models.generate_content(
        model=judge_model,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=JUDGE_SYSTEM_PROMPT,
        ),
    )
    raw = response.text.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        verdicts: list = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini returned non-JSON output:\n{raw}") from exc

    if not isinstance(verdicts, list):
        raise ValueError(f"Expected a JSON array from Gemini, got: {type(verdicts)}")

    if len(verdicts) != len(originals):
        raise ValueError(
            f"Verdict count mismatch: sent {len(originals)} pairs, "
            f"received {len(verdicts)} verdicts"
        )

    # Normalise: ensure every entry has at least {"ok": bool}
    normalised = []
    for v in verdicts:
        if not isinstance(v, dict) or "ok" not in v:
            normalised.append({"ok": False, "feedback": f"Malformed verdict: {v}"})
        else:
            normalised.append(v)

    return normalised
