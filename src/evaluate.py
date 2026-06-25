"""Evaluate saved retrieval results against qrels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SearchResult = tuple[str, float]
Runs = dict[str, list[SearchResult]]
Qrels = dict[str, dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval results.")
    parser.add_argument("--qrels", default="data/processed/qrels.tsv")
    parser.add_argument("--results", default="experiments/results/bm25_results.tsv")
    parser.add_argument("--output", default="experiments/results/bm25_metrics.json")
    parser.add_argument(
        "--metric-prefix",
        default="BM25",
        help="Metric label prefix, e.g. BM25 or Dense.",
    )
    return parser.parse_args()


def read_qrels(path: str | Path) -> Qrels:
    qrels: Qrels = {}
    with Path(path).open("r", encoding="utf-8") as handle:
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
            qrels.setdefault(query_id, {})[doc_id] = float(relevance_text)
    return qrels


def read_results(path: str | Path) -> Runs:
    runs: dict[str, list[tuple[int, str, float]]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if line_number == 1 and parts[:4] == ["query_id", "doc_id", "rank", "score"]:
                continue
            if len(parts) >= 6 and parts[1].upper() == "Q0":
                query_id, doc_id, rank_text, score_text = parts[0], parts[2], parts[3], parts[4]
            elif len(parts) >= 4:
                query_id, doc_id, rank_text, score_text = parts[0], parts[1], parts[2], parts[3]
            else:
                raise ValueError(f"Malformed results line {line_number}: {line}")
            runs.setdefault(query_id, []).append((int(rank_text), doc_id, float(score_text)))

    return {
        query_id: [(doc_id, score) for _, doc_id, score in sorted(rows, key=lambda item: item[0])]
        for query_id, rows in runs.items()
    }


def mrr_at_k(results: list[SearchResult], relevant_docs: dict[str, float], k: int) -> float:
    for rank, (doc_id, _) in enumerate(results[:k], start=1):
        if relevant_docs.get(doc_id, 0.0) > 0.0:
            return 1.0 / rank
    return 0.0


def recall_at_k(results: list[SearchResult], relevant_docs: dict[str, float], k: int) -> float:
    positives = {doc_id for doc_id, relevance in relevant_docs.items() if relevance > 0.0}
    if not positives:
        return 0.0
    retrieved = {doc_id for doc_id, _ in results[:k]}
    return len(positives & retrieved) / len(positives)


def ndcg_at_k(results: list[SearchResult], relevant_docs: dict[str, float], k: int) -> float:
    dcg = 0.0
    for rank, (doc_id, _) in enumerate(results[:k], start=1):
        relevance = relevant_docs.get(doc_id, 0.0)
        if relevance > 0.0:
            dcg += (2.0**relevance - 1.0) / math.log2(rank + 1.0)

    ideal_relevances = sorted(
        (relevance for relevance in relevant_docs.values() if relevance > 0.0),
        reverse=True,
    )[:k]
    idcg = sum(
        (2.0**relevance - 1.0) / math.log2(rank + 1.0)
        for rank, relevance in enumerate(ideal_relevances, start=1)
    )
    return dcg / idcg if idcg > 0.0 else 0.0


def metric_name(metric_prefix: str, name: str) -> str:
    prefix = metric_prefix.strip()
    return f"{prefix} {name}" if prefix else name


def compute_metrics(runs: Runs, qrels: Qrels, metric_prefix: str = "BM25") -> dict[str, float]:
    query_ids = sorted(qrels)
    if not query_ids:
        return {
            metric_name(metric_prefix, "MRR@10"): 0.0,
            metric_name(metric_prefix, "Recall@10"): 0.0,
            metric_name(metric_prefix, "NDCG@10"): 0.0,
            metric_name(metric_prefix, "Recall@100"): 0.0,
        }

    return {
        metric_name(metric_prefix, "MRR@10"): average(
            mrr_at_k(runs.get(query_id, []), qrels[query_id], 10) for query_id in query_ids
        ),
        metric_name(metric_prefix, "Recall@10"): average(
            recall_at_k(runs.get(query_id, []), qrels[query_id], 10) for query_id in query_ids
        ),
        metric_name(metric_prefix, "NDCG@10"): average(
            ndcg_at_k(runs.get(query_id, []), qrels[query_id], 10) for query_id in query_ids
        ),
        metric_name(metric_prefix, "Recall@100"): average(
            recall_at_k(runs.get(query_id, []), qrels[query_id], 100) for query_id in query_ids
        ),
    }


def average(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def main() -> None:
    args = parse_args()
    qrels = read_qrels(args.qrels)
    runs = read_results(args.results)
    metrics = compute_metrics(runs, qrels, metric_prefix=args.metric_prefix)

    payload = {
        "qrels": args.qrels,
        "results": args.results,
        "num_qrels_queries": len(qrels),
        "num_result_queries": len(runs),
        "metric_prefix": args.metric_prefix,
        "metrics": metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
