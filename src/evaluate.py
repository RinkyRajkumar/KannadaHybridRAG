"""Run retrieval and compute MRR@10, Recall@10, and NDCG@10."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from .bm25_retriever import BM25Retriever
    from .dense_retriever import DEFAULT_MODEL, DenseRetriever
    from .hybrid_retriever import HybridRetriever
except ImportError:  # pragma: no cover - supports `python src/evaluate.py`
    from bm25_retriever import BM25Retriever
    from dense_retriever import DEFAULT_MODEL, DenseRetriever
    from hybrid_retriever import HybridRetriever

SearchResult = tuple[str, float]
Qrels = dict[str, dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval over processed JSONL/TSV files.")
    parser.add_argument("--corpus", default="data/processed/corpus.jsonl")
    parser.add_argument("--queries", default="data/processed/queries.jsonl")
    parser.add_argument("--qrels", default="data/processed/qrels.tsv")
    parser.add_argument("--retriever", choices=("bm25", "dense", "hybrid"), default="bm25")
    parser.add_argument("--top-k", type=int, default=10, help="Evaluation cutoff.")
    parser.add_argument("--candidate-k", type=int, default=100, help="Hybrid candidate pool size.")
    parser.add_argument("--output", default="experiments/results/metrics.json")
    parser.add_argument("--run-output", default=None, help="Optional JSONL file for ranked results.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="SentenceTransformers model.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="Optional dense model device, e.g. cpu or cuda.")
    parser.add_argument("--fusion", choices=("rrf", "linear"), default="rrf")
    parser.add_argument("--sparse-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
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


def read_qrels(path: str | Path) -> Qrels:
    qrels: Qrels = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if line_idx == 0 and parts[:3] == ["query_id", "doc_id", "relevance"]:
                continue
            if len(parts) < 3:
                raise ValueError(f"Malformed qrels line {line_idx + 1}: {line}")
            query_id, doc_id, relevance_text = parts[:3]
            relevance = float(relevance_text)
            qrels.setdefault(query_id, {})[doc_id] = relevance
    return qrels


def build_retriever(args: argparse.Namespace, corpus: list[dict[str, Any]]):
    if args.retriever == "bm25":
        return BM25Retriever(corpus)

    if args.retriever == "dense":
        return DenseRetriever(
            corpus,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
        )

    sparse = BM25Retriever(corpus)
    dense = DenseRetriever(
        corpus,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
    )
    return HybridRetriever(
        sparse,
        dense,
        fusion=args.fusion,
        sparse_weight=args.sparse_weight,
        dense_weight=args.dense_weight,
        rrf_k=args.rrf_k,
        candidate_k=args.candidate_k,
    )


def run_retrieval(retriever, queries: list[dict[str, Any]], top_k: int) -> dict[str, list[SearchResult]]:
    runs: dict[str, list[SearchResult]] = {}
    for query in queries:
        query_id = str(query["query_id"])
        runs[query_id] = retriever.search(query.get("text", ""), top_k=top_k)
    return runs


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


def compute_metrics(runs: dict[str, list[SearchResult]], qrels: Qrels, k: int) -> dict[str, float]:
    query_ids = sorted(qrels)
    if not query_ids:
        return {f"MRR@{k}": 0.0, f"Recall@{k}": 0.0, f"NDCG@{k}": 0.0}

    mrr = [mrr_at_k(runs.get(query_id, []), qrels[query_id], k) for query_id in query_ids]
    recall = [recall_at_k(runs.get(query_id, []), qrels[query_id], k) for query_id in query_ids]
    ndcg = [ndcg_at_k(runs.get(query_id, []), qrels[query_id], k) for query_id in query_ids]

    return {
        f"MRR@{k}": sum(mrr) / len(query_ids),
        f"Recall@{k}": sum(recall) / len(query_ids),
        f"NDCG@{k}": sum(ndcg) / len(query_ids),
    }


def write_run(path: str | Path, runs: dict[str, list[SearchResult]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for query_id, results in runs.items():
            for rank, (doc_id, score) in enumerate(results, start=1):
                handle.write(
                    json.dumps(
                        {
                            "query_id": query_id,
                            "doc_id": doc_id,
                            "rank": rank,
                            "score": score,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )


def main() -> None:
    args = parse_args()
    corpus = read_jsonl(args.corpus)
    queries = read_jsonl(args.queries)
    qrels = read_qrels(args.qrels)

    retriever = build_retriever(args, corpus)
    runs = run_retrieval(retriever, queries, top_k=args.top_k)
    metrics = compute_metrics(runs, qrels, k=args.top_k)

    payload = {
        "retriever": args.retriever,
        "top_k": args.top_k,
        "num_documents": len(corpus),
        "num_queries": len(queries),
        "num_qrels_queries": len(qrels),
        "metrics": metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.run_output:
        write_run(args.run_output, runs)

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
