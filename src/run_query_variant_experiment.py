"""Run BM25, dense, Hybrid RRF, and best weighted fusion across query scripts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

try:
    from .bm25_retriever import BM25Retriever, write_results_tsv as write_bm25_results
    from .dense_retriever import (
        DEFAULT_MODEL,
        dense_search,
        load_model,
        load_or_create_document_embeddings,
        write_results_tsv as write_dense_results,
    )
    from .evaluate import compute_metrics, read_qrels, read_results
    from .hybrid_retriever import reciprocal_rank_fusion, write_results_tsv as write_hybrid_results
    from .weighted_fusion import weighted_fusion, write_results_tsv as write_weighted_results
except ImportError:  # pragma: no cover - supports `python src/run_query_variant_experiment.py`
    from bm25_retriever import BM25Retriever, write_results_tsv as write_bm25_results
    from dense_retriever import (
        DEFAULT_MODEL,
        dense_search,
        load_model,
        load_or_create_document_embeddings,
        write_results_tsv as write_dense_results,
    )
    from evaluate import compute_metrics, read_qrels, read_results
    from hybrid_retriever import reciprocal_rank_fusion, write_results_tsv as write_hybrid_results
    from weighted_fusion import weighted_fusion, write_results_tsv as write_weighted_results

VARIANTS = {
    "native": "data/processed/queries_native_100.jsonl",
    "romanized": "data/processed/queries_romanized_100.jsonl",
    "mixed": "data/processed/queries_mixed_100.jsonl",
}

METRIC_NAMES = ("MRR@10", "Recall@10", "NDCG@10", "Recall@100")


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval across native/romanized/mixed queries.")
    parser.add_argument("--corpus", default="data/processed/corpus.jsonl")
    parser.add_argument("--qrels", default="data/processed/qrels_100.tsv")
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--summary", default="experiments/results/script_variant_summary.csv")
    parser.add_argument("--weighted-summary", default="experiments/results/weighted_fusion_summary.csv")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", default="data/processed/dense_cache")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def best_weighted_config(path: str | Path) -> tuple[str, float]:
    summary_path = Path(path)
    if not summary_path.exists():
        return "minmax", 0.1
    rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8", newline="")))
    if not rows:
        return "minmax", 0.1
    best = max(rows, key=lambda row: float(row["MRR@10"]))
    return best["normalization"], float(best["alpha"])


def metric_value(metrics: dict[str, float], prefix: str, metric: str) -> float:
    return metrics.get(f"{prefix} {metric}", 0.0)


def summary_row(query_variant: str, method: str, prefix: str, runs, qrels) -> dict[str, object]:
    metrics = compute_metrics(runs, qrels, metric_prefix=prefix)
    return {
        "query_variant": query_variant,
        "method": method,
        **{metric: metric_value(metrics, prefix, metric) for metric in METRIC_NAMES},
    }


def write_summary(path: str | Path, rows: list[dict[str, object]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_variant", "method", *METRIC_NAMES],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict[str, object]]) -> None:
    print("Query Variant | Method | MRR@10 | Recall@10 | NDCG@10 | Recall@100")
    for row in rows:
        print(
            f"{row['query_variant']} | {row['method']} | "
            f"{float(row['MRR@10']):.6f} | {float(row['Recall@10']):.6f} | "
            f"{float(row['NDCG@10']):.6f} | {float(row['Recall@100']):.6f}"
        )


def main() -> None:
    configure_stdout()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = read_jsonl(args.corpus)
    qrels = read_qrels(args.qrels)
    normalization, alpha = best_weighted_config(args.weighted_summary)
    model = load_model(args.model_name, args.device)
    _, index, doc_ids, cache_hit = load_or_create_document_embeddings(
        corpus=corpus,
        model=model,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        force_recompute=False,
    )

    print(
        json.dumps(
            {
                "best_weighted_normalization": normalization,
                "best_weighted_alpha": alpha,
                "dense_cache_hit": cache_hit,
            },
            indent=2,
        )
    )

    rows: list[dict[str, object]] = []
    for variant, query_path in VARIANTS.items():
        queries = read_jsonl(query_path)

        bm25_runs = BM25Retriever(corpus).batch_search(queries, top_k=args.top_k)
        bm25_path = output_dir / f"bm25_{variant}_100.tsv"
        write_bm25_results(bm25_path, bm25_runs)

        dense_runs = dense_search(
            model=model,
            index=index,
            doc_ids=doc_ids,
            queries=queries,
            top_k=args.top_k,
            batch_size=args.batch_size,
        )
        dense_path = output_dir / f"dense_{variant}_100.tsv"
        write_dense_results(dense_path, dense_runs)

        hybrid_runs = reciprocal_rank_fusion(
            bm25_runs=bm25_runs,
            dense_runs=dense_runs,
            rrf_k=args.rrf_k,
            top_k=args.top_k,
        )
        hybrid_path = output_dir / f"hybrid_rrf_{variant}_100.tsv"
        write_hybrid_results(hybrid_path, hybrid_runs)

        weighted_runs = weighted_fusion(
            bm25_runs=bm25_runs,
            dense_runs=dense_runs,
            normalization=normalization,
            alpha=alpha,
            top_k=args.top_k,
        )
        weighted_path = output_dir / f"weighted_best_{variant}_100.tsv"
        write_weighted_results(weighted_path, weighted_runs)

        rows.extend(
            [
                summary_row(variant, "BM25", "BM25", bm25_runs, qrels),
                summary_row(variant, "Dense", "Dense", dense_runs, qrels),
                summary_row(variant, "Hybrid RRF", "Hybrid RRF", hybrid_runs, qrels),
                summary_row(variant, "Best weighted fusion", "Weighted", weighted_runs, qrels),
            ]
        )

    write_summary(args.summary, rows)
    print_table(rows)
    print(f"Wrote script variant summary: {args.summary}")


if __name__ == "__main__":
    main()
