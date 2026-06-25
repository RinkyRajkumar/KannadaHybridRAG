"""Pure-Python BM25 retriever and BM25 run writer."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from heapq import nlargest
from pathlib import Path
from typing import Any

try:
    from .preprocess_kn import tokenize
except ImportError:  # pragma: no cover - supports `python src/bm25_retriever.py`
    from preprocess_kn import tokenize

Tokenizer = Callable[[object], list[str]]
SearchResult = tuple[str, float]
Runs = dict[str, list[SearchResult]]


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


class BM25Retriever:
    """BM25Okapi-style retriever over in-memory corpus rows."""

    def __init__(
        self,
        corpus: list[dict[str, Any]],
        tokenizer: Tokenizer = tokenize,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.doc_ids = [str(row["doc_id"]) for row in corpus]
        self.doc_lengths: list[int] = []
        self.avg_doc_length = 0.0
        self.idf: dict[str, float] = {}
        self.inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._build()

    def _build(self) -> None:
        document_frequencies: Counter[str] = Counter()
        term_frequencies: list[Counter[str]] = []

        for row in self.corpus:
            tokens = self.tokenizer(row.get("text", ""))
            counts = Counter(tokens)
            term_frequencies.append(counts)
            self.doc_lengths.append(sum(counts.values()))
            document_frequencies.update(counts.keys())

        doc_count = len(self.corpus)
        total_length = sum(self.doc_lengths)
        self.avg_doc_length = total_length / doc_count if doc_count else 0.0

        for term, df in document_frequencies.items():
            self.idf[term] = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))

        for doc_idx, counts in enumerate(term_frequencies):
            for term, frequency in counts.items():
                self.inverted_index[term].append((doc_idx, frequency))

    def search(self, query: object, top_k: int = 10) -> list[SearchResult]:
        if not self.corpus:
            return []

        scores = [0.0] * len(self.corpus)
        for term in set(self.tokenizer(query)):
            postings = self.inverted_index.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for doc_idx, frequency in postings:
                doc_len = self.doc_lengths[doc_idx] or 1
                norm = 1.0 - self.b + self.b * (doc_len / (self.avg_doc_length or 1.0))
                denominator = frequency + self.k1 * norm
                scores[doc_idx] += idf * (frequency * (self.k1 + 1.0)) / denominator

        ranked = nlargest(top_k, enumerate(scores), key=lambda item: item[1])
        return [(self.doc_ids[idx], float(score)) for idx, score in ranked if score > 0.0]

    def batch_search(self, queries: list[dict[str, Any]], top_k: int = 10) -> Runs:
        return {
            str(query["query_id"]): self.search(query.get("text", ""), top_k=top_k)
            for query in queries
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BM25 over processed Kannada retrieval files.")
    parser.add_argument("--corpus", default="data/processed/corpus.jsonl")
    parser.add_argument("--queries", default="data/processed/queries.jsonl")
    parser.add_argument("--qrels", default="data/processed/qrels.tsv")
    parser.add_argument("--output", default="experiments/results/bm25_results.tsv")
    parser.add_argument("--top-k", type=int, default=100, help="Number of documents per query.")
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Limit to 100 queries plus a small document subset and print sample output/metrics.",
    )
    parser.add_argument("--query-limit", type=int, default=100)
    parser.add_argument("--doc-limit", type=int, default=1000)
    parser.add_argument("--sample-query-id", default=None)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_qrels(path: str | Path) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = {}
    qrels_path = Path(path)
    if not qrels_path.exists():
        return qrels

    with qrels_path.open("r", encoding="utf-8") as handle:
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


def select_smoke_subset(
    corpus: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    qrels: dict[str, dict[str, float]],
    query_limit: int,
    doc_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if qrels:
        selected_queries = [
            query for query in queries if str(query["query_id"]) in qrels
        ][:query_limit]
    else:
        selected_queries = queries[:query_limit]

    selected_query_ids = {str(query["query_id"]) for query in selected_queries}
    relevant_doc_ids = {
        doc_id
        for query_id in selected_query_ids
        for doc_id, relevance in qrels.get(query_id, {}).items()
        if relevance > 0.0
    }

    selected_docs: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()

    for row in corpus:
        doc_id = str(row["doc_id"])
        if doc_id in relevant_doc_ids and doc_id not in seen_doc_ids:
            selected_docs.append(row)
            seen_doc_ids.add(doc_id)

    for row in corpus:
        if len(selected_docs) >= max(doc_limit, len(relevant_doc_ids)):
            break
        doc_id = str(row["doc_id"])
        if doc_id not in seen_doc_ids:
            selected_docs.append(row)
            seen_doc_ids.add(doc_id)

    return selected_docs, selected_queries


def write_results_tsv(path: str | Path, runs: Runs) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("query_id\tdoc_id\trank\tscore\n")
        for query_id, results in runs.items():
            for rank, (doc_id, score) in enumerate(results, start=1):
                handle.write(f"{query_id}\t{doc_id}\t{rank}\t{score:.8f}\n")


def print_sample_results(
    queries: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    runs: Runs,
    sample_query_id: str | None,
) -> None:
    if not queries:
        print("No queries available for sample output.")
        return

    query = next(
        (row for row in queries if sample_query_id and str(row["query_id"]) == sample_query_id),
        queries[0],
    )
    query_id = str(query["query_id"])
    corpus_by_id = {str(row["doc_id"]): row for row in corpus}

    print("\nSample query")
    print(f"query_id: {query_id}")
    print(f"text: {query.get('text', '')}")
    print("\nTop 10 BM25 documents")
    for rank, (doc_id, score) in enumerate(runs.get(query_id, [])[:10], start=1):
        text = str(corpus_by_id.get(doc_id, {}).get("text", ""))
        snippet = text[:180] + ("..." if len(text) > 180 else "")
        print(f"{rank}\t{doc_id}\t{score:.4f}\t{snippet}")


def main() -> None:
    configure_stdout()
    args = parse_args()
    corpus = read_jsonl(args.corpus)
    queries = read_jsonl(args.queries)
    qrels = read_qrels(args.qrels)

    if args.smoke_test:
        corpus, queries = select_smoke_subset(
            corpus,
            queries,
            qrels,
            query_limit=args.query_limit,
            doc_limit=args.doc_limit,
        )

    retriever = BM25Retriever(corpus, k1=args.k1, b=args.b)
    runs = retriever.batch_search(queries, top_k=args.top_k)
    write_results_tsv(args.output, runs)

    print(
        json.dumps(
            {
                "documents": len(corpus),
                "queries": len(queries),
                "top_k": args.top_k,
                "output": args.output,
            },
            indent=2,
        )
    )

    if args.smoke_test:
        print_sample_results(queries, corpus, runs, args.sample_query_id)
        if qrels:
            try:
                from .evaluate import compute_metrics
            except ImportError:  # pragma: no cover - supports script execution
                from evaluate import compute_metrics

            query_ids = {str(query["query_id"]) for query in queries}
            smoke_qrels = {query_id: qrels[query_id] for query_id in query_ids if query_id in qrels}
            metrics = compute_metrics(runs, smoke_qrels)
            print("\nSmoke metrics")
            print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
