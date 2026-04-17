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
   Examples of issues to check for:
   - Incorrect verb conjugations, tense, or aspect.
   - Mismatched gender, number, or case agreement.
   - Wrong particle/preposition usage.
   - Syntactic errors specific to the target language.

2. Naturalness and Grammar Balance
   Does the translation strike the right balance between target-language grammar, \
   English source structure, and naturalness?
   The translation must be grammatically correct in the target language first, and \
   then be as natural as possible — it should read like something a native speaker \
   would actually say or write, not a mechanical rendering of English syntax.
   Flag anything that sounds stiff, unnatural, or overly literal even if it is \
   technically grammatical.
   Examples of issues to check for:
   - "Translationese": The sentence follows English word order/structure too closely, making it sound foreign.
   - Literal translation of idioms or metaphors (e.g., translating "piece of cake" literally instead of "easy").
   - Contextual mismatch: Using a valid word that doesn't fit the specific context (e.g., "run" a business vs. "run" a race).
   - Stiff phrasing for common operations: e.g., translating "duplicate" as "副本" (formal/legal copy) in file operations instead of "复制" (copy) or "备份" (backup).
   - "Word-for-word" translation that ignores target language sentence structure. Paraphrasing is encouraged to make the translation sound natural, as long as the original meaning and technical details are preserved.
   - Tone inconsistency: Using formal language in a casual context or vice versa.

3. Accuracy
   Does the translation accurately convey the meaning of the original text?
   Check for:
   - Semantic equivalence: Ensure the core meaning is preserved.
   - Nuance preservation: Check if subtle differences in meaning are lost or altered.
   - Vocabulary choice: Ensure words are translated with their correct meaning in context.
   - Appositives: a noun phrase immediately following another noun/name, set off by a \
comma, identifies or describes that noun — it is NOT a second item joined by "and". \
Example: "Julia, our insightful team" means Julia IS the insightful team; translating it \
as "Julia and our team" is a meaning error and must be flagged.
   - Non-restrictive vs restrictive relative clauses: "the file, which is hidden," (commas) \
adds extra info about a specific file; "the file that is hidden" restricts which file. If \
the translation collapses this distinction or changes the scope of modification, flag it.
   - Participial phrase scope: a participial phrase modifies the nearest noun or describes \
the manner of the main action — it is not an independent verb. "Sort the lines, starting \
from the oldest" should be translated as a single sorting action (manner: from oldest), not \
as two separate actions.
   - Coordination scope: "and/or" joins the closest parallel elements. "Find and delete \
files in A and B" means (find and delete) applied to (A and B). If the translation \
incorrectly splits or reorders the scope, flag it.
   - Ellipsis in parallel structures: English often omits a repeated verb. "Sort the first \
file and the second" means "sort … and [sort] the second". The translation must express \
the complete meaning; if the omitted verb is not implied correctly in the target language, \
flag it.
   - Fronted conditional / adverbial clauses: "Once the file is sorted, share it" has a \
dependency — the sharing happens after/because of the sorting. If the translation merges \
these into one simultaneous action or loses the conditionality, flag it.

4. Completeness
   Did the translation miss any information from the original text?
   Check if any sentences, clauses, or meaningful details were dropped.
   Ensure the entire content is represented in the target language.

5. Proper noun preservation
   The following must NEVER be translated — they must appear exactly as in the source:
   - File names and extensions (e.g. report.pdf, config.yaml, Annual_Report_2023.docx)
   - Directory / folder names used as proper nouns (e.g. workspace, documents, temp)
     — only the generic word "folder" / "directory" may be translated
   - Technical abbreviations: CWD, CLI, API, CPU, RAM, etc.
   - Shell commands, flags (e.g. grep, sort, diff, --output)
   - Function names, class names, API names (e.g. GorillaFileSystem, post_tweet)
   - Variable names and code identifiers
   - URLs, email addresses, domain names
   - Content enclosed in quotation marks (both " " and ' ') — the quoted text must appear
     exactly as in the source; only the surrounding sentence structure outside the quotes
     may be translated.
   - Search or filter subjects/topics: when the instruction specifies a topic, subject,
     keyword, or search term as a concrete value (e.g. "find files about X", "search with
     Y as the topic", "filter by theme Z"), the topic/subject value X, Y, or Z must remain
     untranslated.
   - Person names — names of people must appear exactly as in the source; do not translate
     or romanise them.
   - Ensure that non-proper nouns are translated correctly and not left in English unless they are technical terms that are commonly used in English in the target language context.
   - Examples of words that SHOULD be translated (and not left in English):
     - "directory", "folder", "file", "document" (unless part of a specific path like "/usr/bin")
     - "current directory", "working directory"
     - "sort", "move", "copy", "find", "search", "list" (verbs describing actions)
     - "keyword", "budget analysis", "report", "summary"

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
