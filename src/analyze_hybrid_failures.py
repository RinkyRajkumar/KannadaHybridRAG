"""Diagnose where dense, BM25, RRF, and weighted fusion differ by query."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

try:
    from .evaluate import read_qrels, read_results
except ImportError:  # pragma: no cover - supports `python src/analyze_hybrid_failures.py`
    from evaluate import read_qrels, read_results

Runs = dict[str, list[tuple[str, float]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze hybrid retrieval rank changes.")
    parser.add_argument("--qrels", default="data/processed/qrels.tsv")
    parser.add_argument("--queries", default="data/processed/queries.jsonl")
    parser.add_argument("--bm25-results", default="experiments/results/bm25_results.tsv")
    parser.add_argument("--dense-results", default="experiments/results/dense_results.tsv")
    parser.add_argument("--hybrid-results", default="experiments/results/hybrid_rrf_results.tsv")
    parser.add_argument("--weighted-dir", default="experiments/results")
    parser.add_argument("--output-dir", default="experiments/results")
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    import json

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rank_map(runs: Runs) -> dict[tuple[str, str], int]:
    ranks: dict[tuple[str, str], int] = {}
    for query_id, results in runs.items():
        for rank, (doc_id, _) in enumerate(results, start=1):
            ranks[(query_id, doc_id)] = rank
    return ranks


def load_weighted_rank_maps(weighted_dir: str | Path) -> list[tuple[str, dict[tuple[str, str], int]]]:
    maps: list[tuple[str, dict[tuple[str, str], int]]] = []
    for path in sorted(Path(weighted_dir).glob("weighted_*_alpha_*.tsv")):
        maps.append((path.stem, rank_map(read_results(path))))
    return maps


def best_weighted_rank(
    query_id: str,
    doc_id: str,
    weighted_maps: list[tuple[str, dict[tuple[str, str], int]]],
) -> tuple[int | None, str]:
    best_rank: int | None = None
    best_method = ""
    for method, ranks in weighted_maps:
        rank = ranks.get((query_id, doc_id))
        if rank is not None and (best_rank is None or rank < best_rank):
            best_rank = rank
            best_method = method
    return best_rank, best_method


def comparable_rank(rank: int | None) -> float:
    return float(rank) if rank is not None else math.inf


def csv_rank(rank: int | None) -> str:
    return "" if rank is None else str(rank)


def write_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "query_text",
        "relevant_doc_id",
        "BM25 rank",
        "Dense rank",
        "Hybrid RRF rank",
        "best weighted fusion rank",
        "best weighted fusion method",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    qrels = read_qrels(args.qrels)
    queries = {str(row["query_id"]): str(row.get("text", "")) for row in read_jsonl(args.queries)}
    bm25_ranks = rank_map(read_results(args.bm25_results))
    dense_ranks = rank_map(read_results(args.dense_results))
    hybrid_ranks = rank_map(read_results(args.hybrid_results))
    weighted_maps = load_weighted_rank_maps(args.weighted_dir)

    dense_beats_hybrid: list[dict[str, object]] = []
    hybrid_beats_dense: list[dict[str, object]] = []
    bm25_helped: list[dict[str, object]] = []
    bm25_hurt: list[dict[str, object]] = []

    for query_id, docs in sorted(qrels.items()):
        for doc_id, relevance in sorted(docs.items()):
            if relevance <= 0.0:
                continue

            bm25_rank = bm25_ranks.get((query_id, doc_id))
            dense_rank = dense_ranks.get((query_id, doc_id))
            hybrid_rank = hybrid_ranks.get((query_id, doc_id))
            best_rank, best_method = best_weighted_rank(query_id, doc_id, weighted_maps)

            row = {
                "query_id": query_id,
                "query_text": queries.get(query_id, ""),
                "relevant_doc_id": doc_id,
                "BM25 rank": csv_rank(bm25_rank),
                "Dense rank": csv_rank(dense_rank),
                "Hybrid RRF rank": csv_rank(hybrid_rank),
                "best weighted fusion rank": csv_rank(best_rank),
                "best weighted fusion method": best_method,
            }

            dense_value = comparable_rank(dense_rank)
            hybrid_value = comparable_rank(hybrid_rank)
            bm25_value = comparable_rank(bm25_rank)

            if dense_value < hybrid_value:
                dense_beats_hybrid.append(row)
                if bm25_value > dense_value:
                    bm25_hurt.append(row)

            if hybrid_value < dense_value:
                hybrid_beats_dense.append(row)
                if bm25_value < dense_value:
                    bm25_helped.append(row)

    output_dir = Path(args.output_dir)
    outputs = {
        "dense_beats_hybrid_queries.csv": dense_beats_hybrid,
        "hybrid_beats_dense_queries.csv": hybrid_beats_dense,
        "bm25_helped_queries.csv": bm25_helped,
        "bm25_hurt_queries.csv": bm25_hurt,
    }
    for filename, rows in outputs.items():
        write_rows(output_dir / filename, rows)

    for filename, rows in outputs.items():
        print(f"{filename}: {len(rows)} rows")


if __name__ == "__main__":
    main()
