"""Load Kannada retrieval data from Hugging Face and write evaluation files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - exercised only without dependency
    load_dataset = None

try:
    from .preprocess_kn import normalize_text
except ImportError:  # pragma: no cover - supports `python src/load_data.py`
    from preprocess_kn import normalize_text

DEFAULT_DATASET = "ai4bharat/IndicMSMARCO"
DEFAULT_FALLBACK_DATASETS = ("ai4bharat/INDIC-MARCO", "saifulhaq9/indicmarco")

QUERY_FIELDS = ("query", "question", "query_text", "text_query")
PASSAGE_FIELDS = ("passage", "document", "doc", "context", "content", "text_passage")
QUERY_ID_FIELDS = ("query_id", "qid", "question_id")
PASSAGE_ID_FIELDS = ("passage_id", "doc_id", "document_id", "pid", "id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Kannada IndicMSMARCO-style split to corpus/query/qrels files."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset id.")
    parser.add_argument("--config", default="kn", help="Dataset config/subset, e.g. kn.")
    parser.add_argument("--split", default="train", help="Dataset split to load.")
    parser.add_argument(
        "--fallback-datasets",
        default=",".join(DEFAULT_FALLBACK_DATASETS),
        help="Comma-separated dataset ids to try if --dataset cannot be loaded.",
    )
    parser.add_argument("--output-dir", default="data/processed", help="Output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick runs.")
    parser.add_argument("--query-field", default=None, help="Override query text field.")
    parser.add_argument("--passage-field", default=None, help="Override passage text field.")
    parser.add_argument("--query-id-field", default=None, help="Override query id field.")
    parser.add_argument("--passage-id-field", default=None, help="Override passage id field.")
    parser.add_argument(
        "--min-relevance",
        type=float,
        default=0.0,
        help="Only write qrels with relevance greater than this value.",
    )
    return parser.parse_args()


def load_hf_split(args: argparse.Namespace):
    if load_dataset is None:
        raise ImportError("Install dependencies first: pip install -r requirements.txt")

    fallback_ids = [item.strip() for item in args.fallback_datasets.split(",") if item.strip()]
    dataset_ids = [args.dataset, *[item for item in fallback_ids if item != args.dataset]]
    errors: list[str] = []

    for dataset_id in dataset_ids:
        configs_to_try: list[str | None] = [args.config]
        if dataset_id != args.dataset:
            configs_to_try.append(None)

        for config in configs_to_try:
            try:
                if config:
                    dataset = load_dataset(dataset_id, config, split=args.split)
                else:
                    dataset = load_dataset(dataset_id, split=args.split)
                return dataset, dataset_id, config
            except Exception as exc:  # pragma: no cover - depends on remote availability
                label = f"{dataset_id}/{config or '<default>'}/{args.split}"
                errors.append(f"{label}: {exc}")

    detail = "\n".join(errors)
    raise RuntimeError(f"Could not load any configured dataset candidate:\n{detail}")


def iter_limited(dataset: Iterable[dict[str, Any]], limit: int | None):
    for idx, row in enumerate(dataset):
        if limit is not None and idx >= limit:
            break
        yield idx, row


def first_text(row: dict[str, Any], preferred: str | None, candidates: tuple[str, ...]) -> str:
    fields = (preferred, *candidates) if preferred else candidates
    for field in fields:
        if not field or field not in row:
            continue
        value = row[field]
        text = coerce_text(value)
        if text:
            return normalize_text(text)
    return ""


def first_id(row: dict[str, Any], preferred: str | None, candidates: tuple[str, ...]) -> str:
    fields = (preferred, *candidates) if preferred else candidates
    for field in fields:
        if not field or field not in row:
            continue
        value = normalize_text(row[field])
        if value:
            return value
    return ""


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "passage", "content", "title"):
            if key in value:
                return coerce_text(value[key])
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        parts = [coerce_text(item) for item in value]
        return " ".join(part for part in parts if part)
    return str(value)


def stable_doc_id(text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"doc-{digest}"


def infer_relevance(row: dict[str, Any]) -> float:
    score = row.get("relevance_score")
    selected = row.get("is_selected")

    if score is None:
        return 1.0 if selected is not False else 0.0

    try:
        relevance = float(score)
    except (TypeError, ValueError):
        relevance = 1.0 if selected is not False else 0.0

    if relevance <= 0.0 and selected is True:
        return 1.0
    return relevance


def query_metadata(row: dict[str, Any], row_idx: int) -> dict[str, Any]:
    keys = ("language", "query_type", "dataset", "source")
    return {key: row[key] for key in keys if key in row and row[key] not in (None, "")} | {
        "source_row": row_idx
    }


def corpus_metadata(row: dict[str, Any], row_idx: int) -> dict[str, Any]:
    keys = ("language", "title", "url", "dataset", "source")
    return {key: row[key] for key in keys if key in row and row[key] not in (None, "")} | {
        "source_row": row_idx
    }


def convert_dataset(dataset: Iterable[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    corpus: dict[str, dict[str, Any]] = {}
    queries: dict[str, dict[str, Any]] = {}
    qrels: dict[tuple[str, str], float] = {}
    skipped = 0

    for row_idx, row in iter_limited(dataset, args.limit):
        query_text = first_text(row, args.query_field, QUERY_FIELDS)
        passage_text = first_text(row, args.passage_field, PASSAGE_FIELDS)
        if not query_text or not passage_text:
            skipped += 1
            continue

        query_id = first_id(row, args.query_id_field, QUERY_ID_FIELDS) or f"q-{row_idx}"
        doc_id = first_id(row, args.passage_id_field, PASSAGE_ID_FIELDS) or stable_doc_id(
            passage_text
        )

        queries.setdefault(
            query_id,
            {"query_id": query_id, "text": query_text, "metadata": query_metadata(row, row_idx)},
        )
        corpus.setdefault(
            doc_id,
            {"doc_id": doc_id, "text": passage_text, "metadata": corpus_metadata(row, row_idx)},
        )

        relevance = infer_relevance(row)
        if relevance > args.min_relevance:
            key = (query_id, doc_id)
            qrels[key] = max(qrels.get(key, 0.0), relevance)

    return {"corpus": corpus, "queries": queries, "qrels": qrels, "skipped": skipped}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_qrels(path: Path, qrels: dict[tuple[str, str], float]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("query_id\tdoc_id\trelevance\n")
        for (query_id, doc_id), relevance in sorted(qrels.items()):
            handle.write(f"{query_id}\t{doc_id}\t{relevance:g}\n")


def main() -> None:
    args = parse_args()
    dataset, dataset_id, config = load_hf_split(args)
    converted = convert_dataset(dataset, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "corpus.jsonl", converted["corpus"].values())
    write_jsonl(output_dir / "queries.jsonl", converted["queries"].values())
    write_qrels(output_dir / "qrels.tsv", converted["qrels"])

    summary = {
        "dataset": dataset_id,
        "config": config,
        "split": args.split,
        "documents": len(converted["corpus"]),
        "queries": len(converted["queries"]),
        "qrels": len(converted["qrels"]),
        "skipped_rows": converted["skipped"],
        "output_dir": str(output_dir),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

