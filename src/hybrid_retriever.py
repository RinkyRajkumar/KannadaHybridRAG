"""Hybrid retrieval with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from .evaluate import compute_metrics, read_qrels, read_results
except ImportError:  # pragma: no cover - supports `python src/hybrid_retriever.py`
    from evaluate import compute_metrics, read_qrels, read_results

SearchResult = tuple[str, float]
Runs = dict[str, list[SearchResult]]


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse BM25 and dense runs with RRF.")
    parser.add_argument("--bm25-results", default="experiments/results/bm25_results.tsv")
    parser.add_argument("--dense-results", default="experiments/results/dense_results.tsv")
    parser.add_argument("--output", default="experiments/results/hybrid_rrf_results.tsv")
    parser.add_argument("--qrels", default="data/processed/qrels.tsv")
    parser.add_argument("--queries", default="data/processed/queries.jsonl")
    parser.add_argument("--corpus", default="data/processed/corpus.jsonl")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Print one query, top 10 from each method, and all metric summaries.",
    )
    parser.add_argument("--query-limit", type=int, default=100)
    parser.add_argument("--sample-query-id", default=None)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return rows
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def limit_runs(
    bm25_runs: Runs,
    dense_runs: Runs,
    qrels: dict[str, dict[str, float]],
    query_limit: int,
) -> tuple[Runs, Runs, dict[str, dict[str, float]]]:
    if query_limit <= 0:
        return bm25_runs, dense_runs, qrels

    if qrels:
        query_ids = [query_id for query_id in qrels if query_id in bm25_runs or query_id in dense_runs]
    else:
        query_ids = sorted(set(bm25_runs) | set(dense_runs))

    selected = set(query_ids[:query_limit])
    return (
        {query_id: results for query_id, results in bm25_runs.items() if query_id in selected},
        {query_id: results for query_id, results in dense_runs.items() if query_id in selected},
        {query_id: docs for query_id, docs in qrels.items() if query_id in selected},
    )


def reciprocal_rank_fusion(
    bm25_runs: Runs,
    dense_runs: Runs,
    rrf_k: int = 60,
    top_k: int = 100,
) -> Runs:
    fused_runs: Runs = {}
    query_ids = sorted(set(bm25_runs) | set(dense_runs))

    for query_id in query_ids:
        scores: dict[str, float] = {}
        for results in (bm25_runs.get(query_id, []), dense_runs.get(query_id, [])):
            for rank, (doc_id, _) in enumerate(results, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        fused_runs[query_id] = [(doc_id, score) for doc_id, score in ranked[:top_k]]

    return fused_runs


def write_results_tsv(path: str | Path, runs: Runs) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("query_id\tdoc_id\trank\tscore\n")
        for query_id, results in runs.items():
            for rank, (doc_id, score) in enumerate(results, start=1):
                handle.write(f"{query_id}\t{doc_id}\t{rank}\t{score:.8f}\n")


def choose_sample_query(
    queries: list[dict[str, Any]],
    qrels: dict[str, dict[str, float]],
    runs: Runs,
    sample_query_id: str | None,
) -> tuple[str | None, str]:
    if sample_query_id:
        text = next(
            (str(query.get("text", "")) for query in queries if str(query.get("query_id")) == sample_query_id),
            "",
        )
        return sample_query_id, text

    if qrels:
        for query_id in qrels:
            if query_id in runs:
                text = next(
                    (str(query.get("text", "")) for query in queries if str(query.get("query_id")) == query_id),
                    "",
                )
                return query_id, text

    if queries:
        query = queries[0]
        return str(query.get("query_id")), str(query.get("text", ""))

    if runs:
        query_id = next(iter(runs))
        return query_id, ""

    return None, ""


def print_top_results(
    title: str,
    query_id: str,
    runs: Runs,
    corpus_by_id: dict[str, dict[str, Any]],
) -> None:
    print(f"\nTop 10 {title} results")
    for rank, (doc_id, score) in enumerate(runs.get(query_id, [])[:10], start=1):
        text = str(corpus_by_id.get(doc_id, {}).get("text", ""))
        snippet = text[:180] + ("..." if len(text) > 180 else "")
        print(f"{rank}\t{doc_id}\t{score:.6f}\t{snippet}")


def table_value(metrics: dict[str, float], method: str, metric: str) -> float:
    return metrics.get(f"{method} {metric}", 0.0)


def print_comparison_table(
    bm25_metrics: dict[str, float],
    dense_metrics: dict[str, float],
    hybrid_metrics: dict[str, float],
) -> None:
    rows = [
        ("BM25", bm25_metrics, "BM25"),
        ("Dense", dense_metrics, "Dense"),
        ("Hybrid RRF", hybrid_metrics, "Hybrid RRF"),
    ]
    print("\nMethod | MRR@10 | Recall@10 | NDCG@10 | Recall@100")
    for label, metrics, prefix in rows:
        print(
            f"{label} | "
            f"{table_value(metrics, prefix, 'MRR@10'):.6f} | "
            f"{table_value(metrics, prefix, 'Recall@10'):.6f} | "
            f"{table_value(metrics, prefix, 'NDCG@10'):.6f} | "
            f"{table_value(metrics, prefix, 'Recall@100'):.6f}"
        )


def print_smoke_report(
    bm25_runs: Runs,
    dense_runs: Runs,
    hybrid_runs: Runs,
    qrels: dict[str, dict[str, float]],
    queries_path: str | Path,
    corpus_path: str | Path,
    sample_query_id: str | None,
) -> None:
    queries = read_jsonl(queries_path)
    corpus = read_jsonl(corpus_path)
    corpus_by_id = {str(row["doc_id"]): row for row in corpus}
    query_id, query_text = choose_sample_query(queries, qrels, hybrid_runs, sample_query_id)
    if query_id is None:
        print("No query available for smoke output.")
        return

    print("\nSample query")
    print(f"query_id: {query_id}")
    print(f"text: {query_text}")

    print_top_results("BM25", query_id, bm25_runs, corpus_by_id)
    print_top_results("dense", query_id, dense_runs, corpus_by_id)
    print_top_results("hybrid RRF", query_id, hybrid_runs, corpus_by_id)

    bm25_metrics = compute_metrics(bm25_runs, qrels, metric_prefix="BM25")
    dense_metrics = compute_metrics(dense_runs, qrels, metric_prefix="Dense")
    hybrid_metrics = compute_metrics(hybrid_runs, qrels, metric_prefix="Hybrid RRF")

    print("\nBM25 metrics")
    print(json.dumps(bm25_metrics, indent=2, sort_keys=True))
    print("\nDense metrics")
    print(json.dumps(dense_metrics, indent=2, sort_keys=True))
    print("\nHybrid RRF metrics")
    print(json.dumps(hybrid_metrics, indent=2, sort_keys=True))
    print_comparison_table(bm25_metrics, dense_metrics, hybrid_metrics)


def main() -> None:
    configure_stdout()
    args = parse_args()

    bm25_runs = read_results(args.bm25_results)
    dense_runs = read_results(args.dense_results)
    qrels = read_qrels(args.qrels)

    if args.smoke_test:
        bm25_runs, dense_runs, qrels = limit_runs(
            bm25_runs,
            dense_runs,
            qrels,
            query_limit=args.query_limit,
        )

    hybrid_runs = reciprocal_rank_fusion(
        bm25_runs=bm25_runs,
        dense_runs=dense_runs,
        rrf_k=args.rrf_k,
        top_k=args.top_k,
    )
    write_results_tsv(args.output, hybrid_runs)

    print(
        json.dumps(
            {
                "bm25_queries": len(bm25_runs),
                "dense_queries": len(dense_runs),
                "hybrid_queries": len(hybrid_runs),
                "rrf_k": args.rrf_k,
                "top_k": args.top_k,
                "output": args.output,
            },
            indent=2,
        )
    )

    if args.smoke_test:
        print_smoke_report(
            bm25_runs=bm25_runs,
            dense_runs=dense_runs,
            hybrid_runs=hybrid_runs,
            qrels=qrels,
            queries_path=args.queries,
            corpus_path=args.corpus,
            sample_query_id=args.sample_query_id,
        )


if __name__ == "__main__":
    main()
