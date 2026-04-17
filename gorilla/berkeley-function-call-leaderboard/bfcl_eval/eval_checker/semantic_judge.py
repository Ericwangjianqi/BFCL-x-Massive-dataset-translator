"""
Gemini-based semantic equivalence judge for BFCL multilingual evaluation.

When a translated dataset is evaluated, model outputs may express the same
meaning as the ground truth but in a different language or phrasing.
This module provides helpers that use Gemini to decide whether two values
are semantically equivalent, falling back to exact equality for non-string
types and short-circuiting when values are already identical.

An optional *prompt_context* string (typically the user's question text) can
be supplied to the comparison functions.  When provided, it is forwarded to
Gemini so that context-dependent equivalences (e.g. "New York" ≡ "NYC"
when the prompt is clearly about New York City) are handled
correctly, and so that values for optional parameters (ground-truth = "")
can be validated against the real intent of the request.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from tenacity.stop import stop_base
from tenacity.wait import wait_base

load_dotenv()

_client: genai.Client | None = None

_SYSTEM_PROMPT = """\
You are a semantic equivalence judge for multilingual function-call evaluation.

You will receive two values that originate from function call parameters or \
return values. They may be expressed in different languages (e.g. one in \
English, one in Chinese).

An optional "prompt_context" field may also be provided. It contains the \
user's original question. Use it to resolve ambiguity: for example, if the \
prompt is about flights departing from New York, then "New York" and \
"NYC" can be considered equivalent in that context.

Your task: decide whether the two values convey the same meaning.

Rules:
- Semantically equivalent means they refer to the same real-world concept, \
  object, or action — even if the wording or language differs.
- Ignore minor punctuation, spacing, or capitalisation differences.
- File names, directory paths, and technical identifiers (e.g. API names, \
  variable names, shell commands) must match exactly; if they differ, \
  return false.
- If prompt_context is present, use it to inform your judgment before \
  defaulting to context-free comparison.
- Return ONLY a valid JSON object with a single boolean key:
    {"equivalent": true}   or   {"equivalent": false}
- Do NOT add any explanation or extra text outside the JSON object.
"""

_OPTIONAL_PARAM_SYSTEM_PROMPT = """\
You are a validator for function call parameters in a multilingual evaluation \
system.

A parameter is marked as optional in the ground truth (its expected value is \
empty).  You must decide whether the value provided by the model is a valid \
and reasonable choice for this parameter given the conversation context and \
the parameter description.

Rules:
- If the model's value is consistent with the conversation context and the \
  parameter description, return true.
- If the model's value is clearly wrong, nonsensical, or contradicts the \
  context, return false.
- When in doubt, return true (the parameter is optional anyway; penalising \
  extra-but-reasonable values harms recall unfairly).
- Return ONLY a valid JSON object with a single boolean key:
    {"valid": true}   or   {"valid": false}
- Do NOT add any explanation or extra text outside the JSON object.
"""


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _is_quota_or_server_error(exc: BaseException) -> bool:
    if isinstance(exc, ClientError) and exc.code == 429:
        return True
    if isinstance(exc, ServerError) and exc.code == 503:
        return True
    return False


class _SmartWait(wait_base):
    """Use Gemini's suggested retry delay on quota errors; exponential back-off otherwise."""

    _exp = wait_exponential(multiplier=1, min=2, max=30)

    def __call__(self, retry_state: object) -> float:
        import re
        exc = retry_state.outcome.exception()  # type: ignore[union-attr]
        if _is_quota_or_server_error(exc):
            msg = str(exc)
            m = re.search(r"retry[^\d]*(\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
            delay = float(m.group(1)) if m else 30.0
            print(f"\n  [WARN] Gemini unavailable (429/503) — waiting {delay:.0f}s …")
            return delay
        return self._exp(retry_state)


class _SmartStop(stop_base):
    """Never stop on quota/server errors; stop after 3 attempts for everything else."""

    _limit = stop_after_attempt(3)

    def __call__(self, retry_state: object) -> bool:
        exc = retry_state.outcome.exception()  # type: ignore[union-attr]
        if exc is not None and _is_quota_or_server_error(exc):
            return False
        return self._limit(retry_state)


def _parse_gemini_json(raw: str) -> dict:
    """Strip optional markdown fences and parse JSON from a Gemini response."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


@retry(
    retry=retry_if_exception(lambda e: isinstance(e, Exception)),
    wait=_SmartWait(),
    stop=_SmartStop(),
    reraise=True,
)
def _call_gemini(
    s1: str,
    s2: str,
    prompt_context: str | None = None,
) -> bool:
    """Send a single pair to Gemini and return whether they are equivalent."""
    client = _get_client()
    judge_model = os.getenv("SEMANTIC_JUDGE_MODEL", "gemini-3-flash-preview")
    payload: dict = {"value_1": s1, "value_2": s2}
    if prompt_context:
        payload["prompt_context"] = prompt_context
    user_message = json.dumps(payload, ensure_ascii=False)
    response = client.models.generate_content(
        model=judge_model,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
        ),
    )
    result = _parse_gemini_json(response.text)
    return bool(result.get("equivalent", False))


@lru_cache(maxsize=4096)
def _cached_gemini(
    s1: str,
    s2: str,
    prompt_context: str | None = None,
) -> bool:
    """LRU-cached wrapper around _call_gemini to avoid redundant API calls."""
    return _call_gemini(s1, s2, prompt_context)


def is_semantically_equivalent_str(
    s1: str,
    s2: str,
    prompt_context: str | None = None,
) -> bool:
    """
    Return True if two strings are semantically equivalent, possibly across
    different languages.

    Short-circuits on exact equality or case/whitespace-normalised equality
    before calling Gemini, to minimise API calls.

    An optional *prompt_context* (the user's question text) is forwarded to
    Gemini to resolve context-dependent equivalences.
    """
    if s1 == s2:
        return True
    if s1.strip().lower() == s2.strip().lower():
        return True
    return _cached_gemini(s1, s2, prompt_context)


def are_values_equivalent(
    v1,
    v2,
    prompt_context: str | None = None,
) -> bool:
    """
    Recursively check whether two values are semantically equivalent.

    - str  → semantic string comparison via Gemini (with caching)
    - dict → key sets must match; values compared recursively
    - list → lengths must match; elements compared pairwise (order-sensitive)
    - other → exact equality (int, float, bool, None, …)

    An optional *prompt_context* is threaded through to all string comparisons.
    """
    if v1 == v2:
        return True

    if type(v1) != type(v2):
        return False

    if isinstance(v1, str):
        return is_semantically_equivalent_str(v1, v2, prompt_context)

    if isinstance(v1, dict):
        if set(v1.keys()) != set(v2.keys()):
            return False
        return all(are_values_equivalent(v1[k], v2[k], prompt_context) for k in v1)

    if isinstance(v1, list):
        if len(v1) != len(v2):
            return False
        return all(are_values_equivalent(a, b, prompt_context) for a, b in zip(v1, v2))

    return False


# ---------------------------------------------------------------------------
# Optional-parameter validator
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception(lambda e: isinstance(e, Exception)),
    wait=_SmartWait(),
    stop=_SmartStop(),
    reraise=True,
)
def _call_gemini_optional(
    param_name: str,
    model_value: str,
    param_description: str,
    prompt_context: str,
) -> bool:
    """Ask Gemini whether *model_value* is a valid choice for an optional param."""
    client = _get_client()
    judge_model = os.getenv("SEMANTIC_JUDGE_MODEL", "gemini-3-flash-preview")
    payload = {
        "param_name": param_name,
        "param_description": param_description,
        "model_value": model_value,
        "prompt_context": prompt_context,
    }
    user_message = json.dumps(payload, ensure_ascii=False)
    response = client.models.generate_content(
        model=judge_model,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=_OPTIONAL_PARAM_SYSTEM_PROMPT,
        ),
    )
    result = _parse_gemini_json(response.text)
    return bool(result.get("valid", True))


@lru_cache(maxsize=4096)
def _cached_gemini_optional(
    param_name: str,
    model_value: str,
    param_description: str,
    prompt_context: str,
) -> bool:
    return _call_gemini_optional(param_name, model_value, param_description, prompt_context)


def is_valid_optional_param_value(
    param_name: str,
    model_value: str,
    param_description: str,
    prompt_context: str,
) -> bool:
    """
    Return True if *model_value* is a reasonable value for an optional parameter
    given the conversation context and the parameter's description.

    This is used when the ground truth for a parameter is "" (optional), but the
    model chose to supply a value.  Without context the value would be rejected
    because it doesn't match the empty-string ground truth; with context Gemini
    can judge whether the model's choice is sensible.
    """
    return _cached_gemini_optional(
        param_name, model_value, param_description, prompt_context
    )
