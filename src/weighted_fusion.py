"""Weighted score fusion experiments for BM25 and dense retrieval runs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

try:
    from .evaluate import compute_metrics, read_qrels, read_results
except ImportError:  # pragma: no cover - supports `python src/weighted_fusion.py`
    from evaluate import compute_metrics, read_qrels, read_results

SearchResult = tuple[str, float]
Runs = dict[str, list[SearchResult]]

DEFAULT_ALPHAS = tuple(round(value / 10, 1) for value in range(11))
DEFAULT_NORMALIZATIONS = ("minmax", "zscore", "rank")
METRIC_NAMES = ("MRR@10", "Recall@10", "NDCG@10", "Recall@100")


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run weighted BM25+dense fusion sweeps.")
    parser.add_argument("--bm25-results", default="experiments/results/bm25_results.tsv")
    parser.add_argument("--dense-results", default="experiments/results/dense_results.tsv")
    parser.add_argument("--hybrid-rrf-results", default="experiments/results/hybrid_rrf_results.tsv")
    parser.add_argument("--qrels", default="data/processed/qrels.tsv")
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--summary", default="experiments/results/weighted_fusion_summary.csv")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--normalizations",
        default=",".join(DEFAULT_NORMALIZATIONS),
        help="Comma-separated choices from: minmax,zscore,rank.",
    )
    parser.add_argument(
        "--alphas",
        default=",".join(f"{alpha:.1f}" for alpha in DEFAULT_ALPHAS),
        help="Comma-separated alpha values. alpha=0.0 is dense-only; alpha=1.0 is BM25-only.",
    )
    return parser.parse_args()


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_alphas(value: str) -> list[float]:
    alphas = [round(float(item), 3) for item in parse_csv_list(value)]
    for alpha in alphas:
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"alpha must be between 0.0 and 1.0: {alpha}")
    return alphas


def scores_by_doc(results: list[SearchResult]) -> dict[str, float]:
    return {doc_id: score for doc_id, score in results}


def normalize_scores(results: list[SearchResult], method: str) -> dict[str, float]:
    if not results:
        return {}

    if method == "rank":
        return {doc_id: 1.0 / rank for rank, (doc_id, _) in enumerate(results, start=1)}

    raw_scores = scores_by_doc(results)
    values = list(raw_scores.values())

    if method == "minmax":
        min_score = min(values)
        max_score = max(values)
        if max_score == min_score:
            return {doc_id: 1.0 for doc_id in raw_scores}
        return {
            doc_id: (score - min_score) / (max_score - min_score)
            for doc_id, score in raw_scores.items()
        }

    if method == "zscore":
        mean = sum(values) / len(values)
        variance = sum((score - mean) ** 2 for score in values) / len(values)
        std = math.sqrt(variance)
        if std == 0.0:
            return {doc_id: 1.0 for doc_id in raw_scores}
        return {doc_id: (score - mean) / std for doc_id, score in raw_scores.items()}

    raise ValueError(f"Unsupported normalization: {method}")


def fuse_query(
    bm25_results: list[SearchResult],
    dense_results: list[SearchResult],
    normalization: str,
    alpha: float,
    top_k: int,
) -> list[SearchResult]:
    if alpha == 0.0:
        doc_ids = [doc_id for doc_id, _ in dense_results]
    elif alpha == 1.0:
        doc_ids = [doc_id for doc_id, _ in bm25_results]
    else:
        doc_ids = sorted({doc_id for doc_id, _ in bm25_results} | {doc_id for doc_id, _ in dense_results})

    bm25_scores = normalize_scores(bm25_results, normalization)
    dense_scores = normalize_scores(dense_results, normalization)
    fused = [
        (
            doc_id,
            alpha * bm25_scores.get(doc_id, 0.0)
            + (1.0 - alpha) * dense_scores.get(doc_id, 0.0),
        )
        for doc_id in doc_ids
    ]
    return sorted(fused, key=lambda item: (-item[1], item[0]))[:top_k]


def weighted_fusion(
    bm25_runs: Runs,
    dense_runs: Runs,
    normalization: str,
    alpha: float,
    top_k: int,
) -> Runs:
    query_ids = sorted(set(bm25_runs) | set(dense_runs))
    return {
        query_id: fuse_query(
            bm25_runs.get(query_id, []),
            dense_runs.get(query_id, []),
            normalization=normalization,
            alpha=alpha,
            top_k=top_k,
        )
        for query_id in query_ids
    }


def alpha_label(alpha: float) -> str:
    return f"{alpha:.1f}"


def method_name(normalization: str, alpha: float) -> str:
    return f"weighted_{normalization}_alpha_{alpha_label(alpha)}"


def write_results_tsv(path: str | Path, runs: Runs) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("query_id\tdoc_id\trank\tscore\n")
        for query_id, results in runs.items():
            for rank, (doc_id, score) in enumerate(results, start=1):
                handle.write(f"{query_id}\t{doc_id}\t{rank}\t{score:.8f}\n")


def metric_value(metrics: dict[str, float], prefix: str, metric: str) -> float:
    return metrics.get(f"{prefix} {metric}", 0.0)


def unprefixed_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {metric: metric_value(metrics, prefix, metric) for metric in METRIC_NAMES}


def metrics_for_method(runs: Runs, qrels, prefix: str) -> dict[str, float]:
    return unprefixed_metrics(compute_metrics(runs, qrels, metric_prefix=prefix), prefix)


def write_summary(path: str | Path, rows: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "normalization",
                "alpha",
                "MRR@10",
                "Recall@10",
                "NDCG@10",
                "Recall@100",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_ranked_table(rows: list[dict[str, object]]) -> None:
    ranked = sorted(rows, key=lambda row: float(row["MRR@10"]), reverse=True)
    print("Method | MRR@10 | Recall@10 | NDCG@10 | Recall@100")
    for row in ranked:
        print(
            f"{row['method']} | "
            f"{float(row['MRR@10']):.6f} | "
            f"{float(row['Recall@10']):.6f} | "
            f"{float(row['NDCG@10']):.6f} | "
            f"{float(row['Recall@100']):.6f}"
        )


def baseline_rows(
    bm25_runs: Runs,
    dense_runs: Runs,
    hybrid_rrf_path: str | Path,
    qrels,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, runs, prefix in (
        ("BM25", bm25_runs, "BM25"),
        ("Dense", dense_runs, "Dense"),
    ):
        metrics = metrics_for_method(runs, qrels, prefix)
        rows.append({"method": label, "normalization": "", "alpha": "", **metrics})

    hybrid_path = Path(hybrid_rrf_path)
    if hybrid_path.exists():
        metrics = metrics_for_method(read_results(hybrid_path), qrels, "Hybrid RRF")
        rows.append({"method": "Hybrid RRF", "normalization": "", "alpha": "", **metrics})

    return rows


def main() -> None:
    configure_stdout()
    args = parse_args()
    normalizations = parse_csv_list(args.normalizations)
    alphas = parse_alphas(args.alphas)

    unsupported = sorted(set(normalizations) - set(DEFAULT_NORMALIZATIONS))
    if unsupported:
        raise ValueError(f"Unsupported normalizations: {', '.join(unsupported)}")

    bm25_runs = read_results(args.bm25_results)
    dense_runs = read_results(args.dense_results)
    qrels = read_qrels(args.qrels)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    comparison_rows = baseline_rows(bm25_runs, dense_runs, args.hybrid_rrf_results, qrels)

    for normalization in normalizations:
        for alpha in alphas:
            method = method_name(normalization, alpha)
            runs = weighted_fusion(
                bm25_runs=bm25_runs,
                dense_runs=dense_runs,
                normalization=normalization,
                alpha=alpha,
                top_k=args.top_k,
            )
            result_path = output_dir / f"{method}.tsv"
            write_results_tsv(result_path, runs)

            metrics = metrics_for_method(runs, qrels, "Weighted")
            row: dict[str, object] = {
                "method": method,
                "normalization": normalization,
                "alpha": alpha_label(alpha),
                **metrics,
            }
            summary_rows.append(row)
            comparison_rows.append(row)

    write_summary(args.summary, summary_rows)
    print(f"Wrote weighted fusion summary: {args.summary}")
    print_ranked_table(comparison_rows)


if __name__ == "__main__":
    main()
