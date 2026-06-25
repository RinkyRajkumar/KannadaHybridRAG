"""Print a comparison table for BM25, dense, and hybrid retrieval runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .evaluate import compute_metrics, read_qrels, read_results
except ImportError:  # pragma: no cover - supports `python src/compare_results.py`
    from evaluate import compute_metrics, read_qrels, read_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare BM25, dense, and Hybrid RRF metrics.")
    parser.add_argument("--qrels", default="data/processed/qrels.tsv")
    parser.add_argument("--bm25-results", default="experiments/results/bm25_results.tsv")
    parser.add_argument("--dense-results", default="experiments/results/dense_results.tsv")
    parser.add_argument("--hybrid-results", default="experiments/results/hybrid_rrf_results.tsv")
    parser.add_argument("--output", default=None, help="Optional JSON metrics summary path.")
    return parser.parse_args()


def metric_value(metrics: dict[str, float], prefix: str, name: str) -> float:
    return metrics.get(f"{prefix} {name}", 0.0)


def print_table(summary: dict[str, dict[str, float]]) -> None:
    rows = [
        ("BM25", summary["BM25"], "BM25"),
        ("Dense", summary["Dense"], "Dense"),
        ("Hybrid RRF", summary["Hybrid RRF"], "Hybrid RRF"),
    ]

    print("Method | MRR@10 | Recall@10 | NDCG@10 | Recall@100")
    for label, metrics, prefix in rows:
        print(
            f"{label} | "
            f"{metric_value(metrics, prefix, 'MRR@10'):.6f} | "
            f"{metric_value(metrics, prefix, 'Recall@10'):.6f} | "
            f"{metric_value(metrics, prefix, 'NDCG@10'):.6f} | "
            f"{metric_value(metrics, prefix, 'Recall@100'):.6f}"
        )


def main() -> None:
    args = parse_args()
    qrels = read_qrels(args.qrels)
    summary = {
        "BM25": compute_metrics(read_results(args.bm25_results), qrels, metric_prefix="BM25"),
        "Dense": compute_metrics(read_results(args.dense_results), qrels, metric_prefix="Dense"),
        "Hybrid RRF": compute_metrics(
            read_results(args.hybrid_results),
            qrels,
            metric_prefix="Hybrid RRF",
        ),
    }

    print_table(summary)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
