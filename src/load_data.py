"""Load Kannada retrieval data and write processed evaluation files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
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

QUERY_FIELDS = ("query", "question", "query_text", "text_query", "text")
PASSAGE_FIELDS = ("passage", "document", "doc", "context", "content", "text_passage")
QUERY_ID_FIELDS = ("query_id", "qid", "question_id")
PASSAGE_ID_FIELDS = ("passage_id", "doc_id", "document_id", "pid", "id")
RELEVANCE_FIELDS = ("relevance", "relevance_score", "score", "label")


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Kannada retrieval data to corpus.jsonl, queries.jsonl, and qrels.tsv."
    )
    parser.add_argument(
        "--source",
        choices=("auto", "hf", "raw"),
        default="auto",
        help="Load from Hugging Face, local raw files, or try Hugging Face then raw.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset id.")
    parser.add_argument("--config", default="kn", help="Dataset config/subset, e.g. kn.")
    parser.add_argument("--split", default="train", help="Dataset split to load.")
    parser.add_argument(
        "--fallback-datasets",
        default=",".join(DEFAULT_FALLBACK_DATASETS),
        help="Comma-separated Hugging Face dataset ids to try after --dataset.",
    )
    parser.add_argument("--raw-dir", default="data/raw", help="Directory for local raw files.")
    parser.add_argument("--raw-file", default=None, help="Specific local JSON/JSONL/TSV raw file.")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional paired-row limit.")
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
        configs_to_try: list[str | None] = []
        if args.config:
            configs_to_try.append(args.config)
        configs_to_try.append(None)

        for config in configs_to_try:
            label = f"{dataset_id}/{config or '<default>'}/{args.split}"
            try:
                if config:
                    dataset = load_dataset(dataset_id, config, split=args.split)
                else:
                    dataset = load_dataset(dataset_id, split=args.split)
                return dataset, {"source": "hf", "dataset": dataset_id, "config": config}
            except Exception as exc:  # pragma: no cover - depends on remote availability
                errors.append(f"{label}: {exc}")

    detail = "\n".join(errors)
    raise RuntimeError(f"Could not load any configured Hugging Face dataset:\n{detail}")


def load_raw(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_dir = Path(args.raw_dir)
    raw_file = Path(args.raw_file) if args.raw_file else None

    if raw_file:
        rows = list(read_raw_rows(raw_file))
        converted = convert_paired_rows(rows, args)
        return converted, {"source": "raw", "raw_file": str(raw_file)}

    triplet = read_prebuilt_triplet(raw_dir)
    if triplet is not None:
        return triplet, {"source": "raw", "raw_dir": str(raw_dir), "format": "triplet"}

    files = sorted(
        path
        for path in raw_dir.glob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".tsv", ".csv"}
    )
    if not files:
        raise FileNotFoundError(
            "No raw files found. Add either corpus.jsonl + queries.jsonl + qrels.tsv "
            "or a paired JSON/JSONL/TSV file under data/raw/."
        )

    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(read_raw_rows(path))
    converted = convert_paired_rows(rows, args)
    return converted, {"source": "raw", "raw_dir": str(raw_dir), "files": [str(path) for path in files]}


def read_prebuilt_triplet(raw_dir: Path) -> dict[str, Any] | None:
    corpus_path = raw_dir / "corpus.jsonl"
    queries_path = raw_dir / "queries.jsonl"
    qrels_path = raw_dir / "qrels.tsv"
    if not (corpus_path.exists() and queries_path.exists() and qrels_path.exists()):
        return None

    corpus = {
        str(row["doc_id"]): {
            "doc_id": str(row["doc_id"]),
            "text": normalize_text(row.get("text", "")),
            "metadata": row.get("metadata", {}),
        }
        for row in read_jsonl(corpus_path)
        if row.get("doc_id") and normalize_text(row.get("text", ""))
    }
    queries = {
        str(row["query_id"]): {
            "query_id": str(row["query_id"]),
            "text": normalize_text(row.get("text", "")),
            "metadata": row.get("metadata", {}),
        }
        for row in read_jsonl(queries_path)
        if row.get("query_id") and normalize_text(row.get("text", ""))
    }
    qrels = read_qrels(qrels_path)
    return {"corpus": corpus, "queries": queries, "qrels": qrels, "skipped": 0}


def read_raw_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".json":
        return read_json(path)
    if suffix in {".tsv", ".csv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
    raise ValueError(f"Unsupported raw file format: {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain JSON objects.")
            rows.append(row)
    return rows


def read_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("data", "rows", "examples", "train"):
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        if any(field in data for field in (*QUERY_FIELDS, *PASSAGE_FIELDS)):
            return [data]
    raise ValueError(f"Could not find a list of row objects in {path}")


def read_qrels(path: Path) -> dict[tuple[str, str], float]:
    qrels: dict[tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if line_number == 1 and parts[:3] == ["query_id", "doc_id", "relevance"]:
                continue
            if len(parts) >= 4 and parts[1].upper() == "Q0":
                query_id, doc_id, relevance_text = parts[0], parts[2], parts[3]
            elif len(parts) >= 3:
                query_id, doc_id, relevance_text = parts[0], parts[1], parts[2]
            else:
                raise ValueError(f"Malformed qrels line {line_number}: {line}")
            qrels[(normalize_text(query_id), normalize_text(doc_id))] = float(relevance_text)
    return qrels


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
        text = coerce_text(row[field])
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
        return " ".join(part for part in (coerce_text(item) for item in value) if part)
    return str(value)


def stable_doc_id(text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"doc-{digest}"


def infer_relevance(row: dict[str, Any]) -> float:
    selected = row.get("is_selected")
    for field in RELEVANCE_FIELDS:
        if field not in row or row[field] in (None, ""):
            continue
        try:
            relevance = float(row[field])
        except (TypeError, ValueError):
            continue
        if relevance <= 0.0 and selected is True:
            return 1.0
        return relevance
    return 1.0 if selected is not False else 0.0


def metadata(row: dict[str, Any], row_idx: int, keys: tuple[str, ...]) -> dict[str, Any]:
    values = {key: row[key] for key in keys if key in row and row[key] not in (None, "")}
    values["source_row"] = row_idx
    return values


def convert_paired_rows(rows: Iterable[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    corpus: dict[str, dict[str, Any]] = {}
    queries: dict[str, dict[str, Any]] = {}
    qrels: dict[tuple[str, str], float] = {}
    skipped = 0

    for row_idx, row in iter_limited(rows, args.limit):
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
            {
                "query_id": query_id,
                "text": query_text,
                "metadata": metadata(row, row_idx, ("language", "query_type", "dataset", "source")),
            },
        )
        corpus.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "text": passage_text,
                "metadata": metadata(row, row_idx, ("language", "title", "url", "dataset", "source")),
            },
        )

        relevance = infer_relevance(row)
        if relevance > args.min_relevance:
            key = (query_id, doc_id)
            qrels[key] = max(qrels.get(key, 0.0), relevance)

    return {"corpus": corpus, "queries": queries, "qrels": qrels, "skipped": skipped}


def load_and_convert(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    hf_error: Exception | None = None
    if args.source in {"auto", "hf"}:
        try:
            dataset, source_info = load_hf_split(args)
            return convert_paired_rows(dataset, args), source_info
        except Exception as exc:
            hf_error = exc
            if args.source == "hf":
                raise

    if args.source in {"auto", "raw"}:
        try:
            return load_raw(args)
        except Exception as raw_error:
            if hf_error:
                raise RuntimeError(
                    f"Hugging Face loading failed, then raw fallback failed.\n"
                    f"Hugging Face error: {hf_error}\nRaw error: {raw_error}"
                ) from raw_error
            raise

    raise ValueError(f"Unsupported source: {args.source}")


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
    configure_stdout()
    args = parse_args()
    converted, source_info = load_and_convert(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "corpus.jsonl", converted["corpus"].values())
    write_jsonl(output_dir / "queries.jsonl", converted["queries"].values())
    write_qrels(output_dir / "qrels.tsv", converted["qrels"])

    summary = {
        **source_info,
        "documents": len(converted["corpus"]),
        "queries": len(converted["queries"]),
        "qrels": len(converted["qrels"]),
        "skipped_rows": converted["skipped"],
        "output_dir": str(output_dir),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
