"""
Entry point for the LLM-based JSON/JSONL translation tool.

Usage:
    python main.py --lang Chinese
    python main.py --lang Japanese --fields "question[*][*].content"
    python main.py --lang French --model gpt-4o --batch-size 5
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

import config
from translate.client import translate_batch


# ── Path parsing & wildcard traversal ─────────────────────────────────────────

def _parse_path(path: str) -> list[str | None]:
    """
    Parse a field path into a list of tokens.

    "question[*][*].content"  →  ["question", None, None, "content"]
    "metadata.description"    →  ["metadata", "description"]
    "title"                   →  ["title"]

    None represents a [*] wildcard (iterate over every list element).
    """
    tokens: list[str | None] = []
    for segment in re.findall(r'\[\*\]|[^\.\[\]]+', path):
        tokens.append(None if segment == "[*]" else segment)
    return tokens


def _extract(obj: Any, tokens: list, address: list, results: list[tuple[list, str]]) -> None:
    """
    Recursively walk `obj` following `tokens`, collecting (address, value) pairs
    for every string leaf that is reached.

    Args:
        obj:     current node in the data structure
        tokens:  remaining path tokens to consume
        address: list of keys/indices taken so far (used to set values back)
        results: accumulator — each entry is ([address], str_value)
    """
    if not tokens:
        if isinstance(obj, str) and obj:
            results.append((list(address), obj))
        return

    token = tokens[0]
    rest  = tokens[1:]

    if token is None:              # [*] — iterate list
        if isinstance(obj, list):
            for i, item in enumerate(obj):
                _extract(item, rest, address + [i], results)
    else:                          # named key — enter dict
        if isinstance(obj, dict) and token in obj:
            _extract(obj[token], rest, address + [token], results)


def _set_by_address(obj: Any, address: list, value: str) -> None:
    """Set a value at a given address (list of keys/indices) inside `obj`."""
    cur = obj
    for key in address[:-1]:
        cur = cur[key]
    cur[address[-1]] = value


def _collect_all_string_paths(obj: dict, prefix: str = "") -> list[str]:
    """Recursively collect dot-notation paths for every string leaf (auto-detect mode)."""
    paths: list[str] = []
    for k, v in obj.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, str):
            paths.append(full)
        elif isinstance(v, dict):
            paths.extend(_collect_all_string_paths(v, full))
    return paths


# ── Core translation logic ─────────────────────────────────────────────────────

def translate_records(
    records: list[dict],
    fields: list[str],
    target_language: str,
    model: str,
    temperature: float,
    batch_size: int,
) -> list[dict]:
    """
    Translate the specified fields across all records.
    Returns new record dicts with targeted fields replaced by their translations.
    """
    auto_detect = len(fields) == 0

    # ── Step 1: collect all (record_idx, address, text) tasks ─────────────────
    tasks: list[tuple[int, list, str]] = []

    for rec_idx, record in enumerate(records):
        if auto_detect:
            # simple string leaves only (no wildcard support in auto mode)
            for path in _collect_all_string_paths(record):
                tokens = _parse_path(path)
                hits: list[tuple[list, str]] = []
                _extract(record, tokens, [], hits)
                for address, text in hits:
                    tasks.append((rec_idx, address, text))
        else:
            for path in fields:
                tokens = _parse_path(path)
                hits: list[tuple[list, str]] = []
                _extract(record, tokens, [], hits)
                for address, text in hits:
                    tasks.append((rec_idx, address, text))

    if not tasks:
        print("  No translatable content found.")
        return records

    # ── Step 2: deep-copy so originals stay untouched ─────────────────────────
    translated_records = copy.deepcopy(records)

    # ── Step 3: batch API calls ───────────────────────────────────────────────
    texts_only = [t[2] for t in tasks]
    all_translations: list[str] = []

    batches = [texts_only[s: s + batch_size] for s in range(0, len(texts_only), batch_size)]

    for batch in tqdm(batches, desc="  Translating batches", unit="batch"):
        translated = translate_batch(
            texts=batch,
            target_language=target_language,
            model=model,
            temperature=temperature,
        )
        all_translations.extend(translated)

    # ── Step 4: write translations back ───────────────────────────────────────
    for (rec_idx, address, _), translation in zip(tasks, all_translations):
        _set_by_address(translated_records[rec_idx], address, translation)

    return translated_records


# ── File I/O helpers ───────────────────────────────────────────────────────────

def read_records(path: Path, file_format: str) -> tuple[list[dict], bool]:
    """
    Load records from a JSON or JSONL file.

    Returns:
        (records, is_jsonl) — is_jsonl is True when the file was read line-by-line.
    """
    with path.open("r", encoding="utf-8") as f:
        if file_format == "jsonl":
            records = []
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {lineno} of {path.name}: {exc}")
            return records, True
        else:
            data = json.load(f)
            if isinstance(data, list):
                return data, False
            return [data], False


def write_records(records: list[dict], path: Path, is_jsonl: bool, was_single: bool) -> None:
    """Write records to a JSON or JSONL file."""
    with path.open("w", encoding="utf-8") as f:
        if is_jsonl:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            payload = records[0] if was_single else records
            json.dump(payload, f, ensure_ascii=False, indent=2)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate JSON/JSONL data files from English to a target language."
    )
    parser.add_argument(
        "--lang", "-l",
        required=True,
        help='Target language, e.g. "Chinese", "French", "Japanese"',
    )
    parser.add_argument(
        "--fields", "-f",
        nargs="*",
        default=None,
        help=(
            "Field paths to translate (dot-notation, supports [*] wildcards). "
            "Overrides TRANSLATE_FIELDS in config.py. "
            "Pass with no value to auto-detect all string fields."
        ),
    )
    parser.add_argument(
        "--model", "-m",
        default=config.MODEL,
        help=f"OpenAI model to use (default: {config.MODEL})",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=config.BATCH_SIZE,
        dest="batch_size",
        help=f"Texts per API call (default: {config.BATCH_SIZE})",
    )
    parser.add_argument(
        "--format",
        default=config.FILE_FORMAT,
        choices=["json", "jsonl"],
        dest="file_format",
        help=f"Input file format (default: {config.FILE_FORMAT})",
    )
    parser.add_argument(
        "--data-dir",
        default=config.DATA_DIR,
        help=f"Source data directory (default: {config.DATA_DIR})",
    )
    parser.add_argument(
        "--result-dir",
        default=config.RESULT_DIR,
        help=f"Output directory (default: {config.RESULT_DIR})",
    )
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    fields: list[str]
    if args.fields is not None:
        fields = args.fields
    else:
        fields = config.TRANSLATE_FIELDS

    data_dir   = Path(args.data_dir)
    result_dir = Path(args.result_dir)

    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        sys.exit(1)

    result_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        print(f"[WARN] No JSON files found in '{data_dir}'.")
        sys.exit(0)

    print(f"Target language : {args.lang}")
    print(f"Model           : {args.model}")
    print(f"File format     : {args.file_format}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Fields          : {fields if fields else '(auto-detect all string fields)'}")
    print(f"Files found     : {len(json_files)}\n")

    for json_file in json_files:
        print(f"Processing: {json_file.name}")

        records, is_jsonl = read_records(json_file, args.file_format)
        was_single = not is_jsonl and len(records) == 1

        print(f"  Records loaded: {len(records)}")

        translated = translate_records(
            records=records,
            fields=fields,
            target_language=args.lang,
            model=args.model,
            temperature=config.TEMPERATURE,
            batch_size=args.batch_size,
        )

        out_path = result_dir / json_file.name
        write_records(translated, out_path, is_jsonl, was_single)

        print(f"  Saved → {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
