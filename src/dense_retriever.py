"""Dense vector retriever using SentenceTransformers.

This module is optional at runtime. BM25 works without dense dependencies, but
installing `sentence-transformers` enables dense and hybrid experiments.
"""

from __future__ import annotations

from typing import Any

try:
    from .preprocess_kn import normalize_text
except ImportError:  # pragma: no cover - supports `python src/dense_retriever.py`
    from preprocess_kn import normalize_text

SearchResult = tuple[str, float]
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class DenseRetriever:
    """In-memory dense retriever with cosine similarity over normalized vectors."""

    def __init__(
        self,
        corpus: list[dict[str, Any]],
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        device: str | None = None,
        show_progress: bool = True,
    ) -> None:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise ImportError(
                "Dense retrieval requires sentence-transformers and numpy. "
                "Install them with: pip install -r requirements.txt"
            ) from exc

        self.np = np
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.doc_ids = [str(row["doc_id"]) for row in corpus]
        texts = [normalize_text(row.get("text", "")) for row in corpus]
        self.embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )

    def search(self, query: object, top_k: int = 10) -> list[SearchResult]:
        if not self.doc_ids:
            return []

        query_embedding = self.model.encode(
            [normalize_text(query)],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = self.embeddings @ query_embedding
        top_k = min(top_k, len(self.doc_ids))
        top_indices = self.np.argsort(-scores)[:top_k]
        return [(self.doc_ids[idx], float(scores[idx])) for idx in top_indices]

    def batch_search(self, queries: list[dict[str, Any]], top_k: int = 10) -> dict[str, list[SearchResult]]:
        return {
            str(query["query_id"]): self.search(query.get("text", ""), top_k=top_k)
            for query in queries
        }

