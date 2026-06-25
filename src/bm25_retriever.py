"""Pure-Python BM25 retriever."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable
from heapq import nlargest
from typing import Any

try:
    from .preprocess_kn import tokenize
except ImportError:  # pragma: no cover - supports `python src/bm25_retriever.py`
    from preprocess_kn import tokenize

Tokenizer = Callable[[object], list[str]]
SearchResult = tuple[str, float]


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

    def batch_search(self, queries: list[dict[str, Any]], top_k: int = 10) -> dict[str, list[SearchResult]]:
        return {
            str(query["query_id"]): self.search(query.get("text", ""), top_k=top_k)
            for query in queries
        }

