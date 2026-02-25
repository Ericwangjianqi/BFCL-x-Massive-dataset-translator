"""
OpenAI client wrapper with automatic retry and JSON-response parsing.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from translate.prompts import SYSTEM_PROMPT, build_user_prompt

load_dotenv()

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Copy .env.example to .env and fill in your key."
            )
        _client = OpenAI(api_key=api_key)
    return _client


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def translate_batch(
    texts: list[str],
    target_language: str,
    model: str,
    temperature: float,
) -> list[str]:
    """
    Send a batch of English strings to the OpenAI API and return translations.

    Args:
        texts:           List of source strings (English).
        target_language: Target language name, e.g. "Chinese".
        model:           OpenAI model identifier.
        temperature:     Sampling temperature.

    Returns:
        List of translated strings, same length as `texts`.

    Raises:
        ValueError: If the model returns an unparseable or mis-sized response.
    """
    if not texts:
        return []

    # Replace None / non-string values with empty strings before sending
    safe_texts = [t if isinstance(t, str) else "" for t in texts]

    user_message = build_user_prompt(target_language, safe_texts)

    response = _get_client().chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model wraps the array in ```json … ```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        translated: list = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned non-JSON output:\n{raw}") from exc

    if not isinstance(translated, list):
        raise ValueError(f"Expected a JSON array, got: {type(translated)}")

    if len(translated) != len(safe_texts):
        raise ValueError(
            f"Translation count mismatch: sent {len(safe_texts)}, "
            f"received {len(translated)}"
        )

    return [str(t) for t in translated]
