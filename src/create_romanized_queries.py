"""Create native, romanized, and mixed-script Kannada query subsets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

try:
    from .preprocess_kn import normalize_text
except ImportError:  # pragma: no cover - supports `python src/create_romanized_queries.py`
    from preprocess_kn import normalize_text

KANNADA_RANGE = range(0x0C80, 0x0D00)
ZERO_WIDTH_CHARS = {"\u200c", "\u200d"}

INDEPENDENT_VOWELS = {
    "ಅ": "a",
    "ಆ": "aa",
    "ಇ": "i",
    "ಈ": "ii",
    "ಉ": "u",
    "ಊ": "uu",
    "ಋ": "ru",
    "ೠ": "ruu",
    "ಎ": "e",
    "ಏ": "ee",
    "ಐ": "ai",
    "ಒ": "o",
    "ಓ": "oo",
    "ಔ": "au",
}

VOWEL_SIGNS = {
    "ಾ": "aa",
    "ಿ": "i",
    "ೀ": "ii",
    "ು": "u",
    "ೂ": "uu",
    "ೃ": "ru",
    "ೄ": "ruu",
    "ೆ": "e",
    "ೇ": "ee",
    "ೈ": "ai",
    "ೊ": "o",
    "ೋ": "oo",
    "ೌ": "au",
}

CONSONANTS = {
    "ಕ": "k",
    "ಖ": "kh",
    "ಗ": "g",
    "ಘ": "gh",
    "ಙ": "ng",
    "ಚ": "ch",
    "ಛ": "chh",
    "ಜ": "j",
    "ಝ": "jh",
    "ಞ": "ny",
    "ಟ": "t",
    "ಠ": "th",
    "ಡ": "d",
    "ಢ": "dh",
    "ಣ": "n",
    "ತ": "t",
    "ಥ": "th",
    "ದ": "d",
    "ಧ": "dh",
    "ನ": "n",
    "ಪ": "p",
    "ಫ": "ph",
    "ಬ": "b",
    "ಭ": "bh",
    "ಮ": "m",
    "ಯ": "y",
    "ರ": "r",
    "ಱ": "r",
    "ಲ": "l",
    "ವ": "v",
    "ಶ": "sh",
    "ಷ": "sh",
    "ಸ": "s",
    "ಹ": "h",
    "ಳ": "l",
    "ೞ": "l",
}

OTHER_MARKS = {
    "ಂ": "m",
    "ಃ": "h",
    "ಽ": "",
    "್": "",
}


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 100-query script-variant retrieval files.")
    parser.add_argument("--queries", default="data/processed/queries.jsonl")
    parser.add_argument("--qrels", default="data/processed/qrels.tsv")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--mapping", default="data/processed/romanized_query_mapping.csv")
    parser.add_argument(
        "--mode",
        choices=("manual", "auto"),
        default="auto",
        help="manual leaves editable blanks; auto pre-fills draft romanized and mixed queries.",
    )
    parser.add_argument(
        "--overwrite-mapping",
        action="store_true",
        help="Regenerate mapping rows instead of preserving existing romanized/mixed edits.",
    )
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_qrels(path: str | Path) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if line_number == 1 and parts[:3] == ["query_id", "doc_id", "relevance"]:
                continue
            if len(parts) < 3:
                raise ValueError(f"Malformed qrels line {line_number}: {line}")
            query_id, doc_id, relevance = parts[:3]
            qrels.setdefault(query_id, {})[doc_id] = float(relevance)
    return qrels


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_qrels(path: str | Path, query_ids: set[str], qrels: dict[str, dict[str, float]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write("query_id\tdoc_id\trelevance\n")
        for query_id in sorted(query_ids):
            for doc_id, relevance in sorted(qrels.get(query_id, {}).items()):
                handle.write(f"{query_id}\t{doc_id}\t{relevance:g}\n")


def has_kannada(text: str) -> bool:
    return any(ord(char) in KANNADA_RANGE for char in text)


def package_transliterate(text: str) -> str | None:
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except Exception:
        return None

    value = transliterate(text, sanscript.KANNADA, sanscript.HK)
    if not value or "?" in value or has_kannada(value) or not value.isascii():
        return None
    return normalize_text(value.lower())


def builtin_romanize(text: str) -> str:
    chars = list(normalize_text(text))
    output: list[str] = []
    idx = 0

    while idx < len(chars):
        char = chars[idx]

        if char in CONSONANTS:
            base = CONSONANTS[char]
            next_char = chars[idx + 1] if idx + 1 < len(chars) else ""
            if next_char == "್":
                output.append(base)
                idx += 2
                continue
            if next_char in VOWEL_SIGNS:
                output.append(base + VOWEL_SIGNS[next_char])
                idx += 2
                continue
            output.append(base + "a")
            idx += 1
            continue

        if char in INDEPENDENT_VOWELS:
            output.append(INDEPENDENT_VOWELS[char])
        elif char in OTHER_MARKS:
            output.append(OTHER_MARKS[char])
        elif char in VOWEL_SIGNS:
            output.append(VOWEL_SIGNS[char])
        elif char in ZERO_WIDTH_CHARS:
            pass
        else:
            output.append(char)
        idx += 1

    return normalize_text("".join(output).lower())


def romanize(text: str, mode: str) -> str:
    if mode == "manual":
        return ""

    packaged = package_transliterate(text)
    if packaged:
        return packaged
    return builtin_romanize(text)


def make_mixed_query(native_query: str, romanized_query: str) -> str:
    native_tokens = normalize_text(native_query).split()
    roman_tokens = normalize_text(romanized_query).split()
    if not roman_tokens:
        return ""

    mixed: list[str] = []
    for idx, native_token in enumerate(native_tokens):
        if idx % 2 == 0 and idx < len(roman_tokens):
            mixed.append(roman_tokens[idx])
        else:
            mixed.append(native_token)
    return normalize_text(" ".join(mixed))


def read_mapping(path: str | Path) -> dict[str, dict[str, str]]:
    mapping_path = Path(path)
    if not mapping_path.exists():
        return {}
    with mapping_path.open("r", encoding="utf-8", newline="") as handle:
        return {row["query_id"]: row for row in csv.DictReader(handle)}


def write_mapping(path: str | Path, rows: list[dict[str, str]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_id", "native_query", "romanized_query", "mixed_query"],
        )
        writer.writeheader()
        writer.writerows(rows)


def select_queries(
    queries: list[dict[str, Any]],
    qrels: dict[str, dict[str, float]],
    limit: int,
) -> list[dict[str, Any]]:
    selected = [query for query in queries if str(query["query_id"]) in qrels]
    return selected[:limit]


def variant_rows(selected: list[dict[str, Any]], mapping_rows: list[dict[str, str]], field: str):
    by_id = {row["query_id"]: row for row in mapping_rows}
    rows: list[dict[str, Any]] = []
    for query in selected:
        query_id = str(query["query_id"])
        text = by_id[query_id][field]
        rows.append(
            {
                "query_id": query_id,
                "text": text,
                "metadata": {
                    **query.get("metadata", {}),
                    "script_variant": field.replace("_query", ""),
                },
            }
        )
    return rows


def main() -> None:
    configure_stdout()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = read_jsonl(args.queries)
    qrels = read_qrels(args.qrels)
    selected = select_queries(queries, qrels, args.limit)
    existing = {} if args.overwrite_mapping else read_mapping(args.mapping)

    mapping_rows: list[dict[str, str]] = []
    for query in selected:
        query_id = str(query["query_id"])
        native_query = normalize_text(query["text"])
        existing_row = existing.get(query_id, {})
        romanized_query = normalize_text(existing_row.get("romanized_query", ""))
        mixed_query = normalize_text(existing_row.get("mixed_query", ""))

        if not romanized_query:
            romanized_query = romanize(native_query, args.mode)
        if not mixed_query:
            mixed_query = make_mixed_query(native_query, romanized_query)

        mapping_rows.append(
            {
                "query_id": query_id,
                "native_query": native_query,
                "romanized_query": romanized_query,
                "mixed_query": mixed_query,
            }
        )

    write_mapping(args.mapping, mapping_rows)
    write_jsonl(output_dir / "queries_native_100.jsonl", variant_rows(selected, mapping_rows, "native_query"))
    write_jsonl(
        output_dir / "queries_romanized_100.jsonl",
        variant_rows(selected, mapping_rows, "romanized_query"),
    )
    write_jsonl(output_dir / "queries_mixed_100.jsonl", variant_rows(selected, mapping_rows, "mixed_query"))
    write_qrels(output_dir / "qrels_100.tsv", {str(query["query_id"]) for query in selected}, qrels)

    empty_romanized = sum(1 for row in mapping_rows if not row["romanized_query"])
    print(
        json.dumps(
            {
                "queries": len(selected),
                "mapping": args.mapping,
                "mode": args.mode,
                "empty_romanized_queries": empty_romanized,
                "outputs": [
                    str(output_dir / "queries_native_100.jsonl"),
                    str(output_dir / "queries_romanized_100.jsonl"),
                    str(output_dir / "queries_mixed_100.jsonl"),
                    str(output_dir / "qrels_100.tsv"),
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.mode == "manual" or empty_romanized:
        print("Review and edit romanized_query_mapping.csv before final reporting.")


if __name__ == "__main__":
    main()
