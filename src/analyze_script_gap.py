"""Analyze retrieval gaps between native and romanized Kannada queries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

try:
    from .evaluate import read_qrels, read_results
except ImportError:  # pragma: no cover - supports `python src/analyze_script_gap.py`
    from evaluate import read_qrels, read_results

METHOD_FILES = {
    "BM25": "bm25_{variant}_100.tsv",
    "Dense": "dense_{variant}_100.tsv",
    "Hybrid RRF": "hybrid_rrf_{variant}_100.tsv",
    "Best weighted fusion": "weighted_best_{variant}_100.tsv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze native-to-romanized retrieval gaps.")
    parser.add_argument("--qrels", default="data/processed/qrels_100.tsv")
    parser.add_argument("--native-queries", default="data/processed/queries_native_100.jsonl")
    parser.add_argument("--romanized-queries", default="data/processed/queries_romanized_100.jsonl")
    parser.add_argument("--mixed-queries", default="data/processed/queries_mixed_100.jsonl")
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--output-dir", default="experiments/results")
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def query_texts(path: str | Path) -> dict[str, str]:
    return {str(row["query_id"]): str(row.get("text", "")) for row in read_jsonl(path)}


def rank_map(path: str | Path) -> dict[tuple[str, str], int]:
    ranks: dict[tuple[str, str], int] = {}
    for query_id, results in read_results(path).items():
        for rank, (doc_id, _) in enumerate(results, start=1):
            ranks[(query_id, doc_id)] = rank
    return ranks


def load_rank_maps(results_dir: str | Path, variant: str) -> dict[str, dict[tuple[str, str], int]]:
    root = Path(results_dir)
    return {
        method: rank_map(root / pattern.format(variant=variant))
        for method, pattern in METHOD_FILES.items()
    }


def rank_value(rank: int | None) -> float:
    return float(rank) if rank is not None else math.inf


def csv_rank(rank: int | None) -> str:
    return "" if rank is None else str(rank)


def best_rank(ranks: dict[str, dict[tuple[str, str], int]], query_id: str, doc_id: str) -> float:
    return min(rank_value(method_ranks.get((query_id, doc_id))) for method_ranks in ranks.values())


def row_for(
    query_id: str,
    doc_id: str,
    native_queries: dict[str, str],
    romanized_queries: dict[str, str],
    mixed_queries: dict[str, str],
    romanized_ranks: dict[str, dict[tuple[str, str], int]],
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "native_query": native_queries.get(query_id, ""),
        "romanized_query": romanized_queries.get(query_id, ""),
        "mixed_query": mixed_queries.get(query_id, ""),
        "relevant_doc_id": doc_id,
        "BM25 rank": csv_rank(romanized_ranks["BM25"].get((query_id, doc_id))),
        "Dense rank": csv_rank(romanized_ranks["Dense"].get((query_id, doc_id))),
        "Hybrid RRF rank": csv_rank(romanized_ranks["Hybrid RRF"].get((query_id, doc_id))),
        "Best weighted fusion rank": csv_rank(
            romanized_ranks["Best weighted fusion"].get((query_id, doc_id))
        ),
    }


def write_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "query_id",
        "native_query",
        "romanized_query",
        "mixed_query",
        "relevant_doc_id",
        "BM25 rank",
        "Dense rank",
        "Hybrid RRF rank",
        "Best weighted fusion rank",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    qrels = read_qrels(args.qrels)
    native_queries = query_texts(args.native_queries)
    romanized_queries = query_texts(args.romanized_queries)
    mixed_queries = query_texts(args.mixed_queries)
    native_ranks = load_rank_maps(args.results_dir, "native")
    romanized_ranks = load_rank_maps(args.results_dir, "romanized")

    native_to_romanized_drop: list[dict[str, object]] = []
    romanized_dense_beats_bm25: list[dict[str, object]] = []
    romanized_hybrid_helped: list[dict[str, object]] = []
    all_methods_failed_romanized: list[dict[str, object]] = []

    for query_id, docs in sorted(qrels.items()):
        for doc_id, relevance in sorted(docs.items()):
            if relevance <= 0.0:
                continue

            row = row_for(
                query_id,
                doc_id,
                native_queries,
                romanized_queries,
                mixed_queries,
                romanized_ranks,
            )

            native_best = best_rank(native_ranks, query_id, doc_id)
            romanized_best = best_rank(romanized_ranks, query_id, doc_id)

            bm25_rank = rank_value(romanized_ranks["BM25"].get((query_id, doc_id)))
            dense_rank = rank_value(romanized_ranks["Dense"].get((query_id, doc_id)))
            hybrid_rank = rank_value(romanized_ranks["Hybrid RRF"].get((query_id, doc_id)))
            weighted_rank = rank_value(romanized_ranks["Best weighted fusion"].get((query_id, doc_id)))

            if native_best <= 10 and romanized_best > 10:
                native_to_romanized_drop.append(row)
            if dense_rank < bm25_rank:
                romanized_dense_beats_bm25.append(row)
            if hybrid_rank < min(bm25_rank, dense_rank):
                romanized_hybrid_helped.append(row)
            if min(bm25_rank, dense_rank, hybrid_rank, weighted_rank) > 100:
                all_methods_failed_romanized.append(row)

    output_dir = Path(args.output_dir)
    outputs = {
        "native_to_romanized_drop.csv": native_to_romanized_drop,
        "romanized_dense_beats_bm25.csv": romanized_dense_beats_bm25,
        "romanized_hybrid_helped.csv": romanized_hybrid_helped,
        "all_methods_failed_romanized.csv": all_methods_failed_romanized,
    }
    for filename, rows in outputs.items():
        write_rows(output_dir / filename, rows)
        print(f"{filename}: {len(rows)} rows")


if __name__ == "__main__":
    main()
