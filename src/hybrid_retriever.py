"""Hybrid retrieval by fusing BM25 and dense rankings."""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

SearchResult = tuple[str, float]


class Retriever(Protocol):
    def search(self, query: object, top_k: int = 10) -> list[SearchResult]:
        ...


class HybridRetriever:
    """Combine sparse and dense retrievers with RRF or min-max linear fusion."""

    def __init__(
        self,
        sparse_retriever: Retriever,
        dense_retriever: Retriever,
        fusion: str = "rrf",
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
        rrf_k: int = 60,
        candidate_k: int = 100,
    ) -> None:
        if fusion not in {"rrf", "linear"}:
            raise ValueError("fusion must be one of: rrf, linear")
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.fusion = fusion
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def search(self, query: object, top_k: int = 10) -> list[SearchResult]:
        candidate_k = max(top_k, self.candidate_k)
        sparse_results = self.sparse_retriever.search(query, top_k=candidate_k)
        dense_results = self.dense_retriever.search(query, top_k=candidate_k)

        if self.fusion == "linear":
            scores = self._linear_scores(sparse_results, dense_results)
        else:
            scores = self._rrf_scores(sparse_results, dense_results)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [(doc_id, float(score)) for doc_id, score in ranked[:top_k]]

    def _rrf_scores(
        self, sparse_results: list[SearchResult], dense_results: list[SearchResult]
    ) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        for weight, results in (
            (self.sparse_weight, sparse_results),
            (self.dense_weight, dense_results),
        ):
            for rank, (doc_id, _) in enumerate(results, start=1):
                scores[doc_id] += weight / (self.rrf_k + rank)
        return dict(scores)

    def _linear_scores(
        self, sparse_results: list[SearchResult], dense_results: list[SearchResult]
    ) -> dict[str, float]:
        sparse_scores = minmax_scores(sparse_results)
        dense_scores = minmax_scores(dense_results)
        doc_ids = set(sparse_scores) | set(dense_scores)
        return {
            doc_id: self.sparse_weight * sparse_scores.get(doc_id, 0.0)
            + self.dense_weight * dense_scores.get(doc_id, 0.0)
            for doc_id in doc_ids
        }


def minmax_scores(results: list[SearchResult]) -> dict[str, float]:
    if not results:
        return {}
    values = [score for _, score in results]
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return {doc_id: 1.0 for doc_id, _ in results}
    return {doc_id: (score - min_score) / (max_score - min_score) for doc_id, score in results}

