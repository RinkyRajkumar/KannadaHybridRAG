"""Dense vector retrieval with multilingual E5 embeddings and FAISS."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from .preprocess_kn import normalize_text
except ImportError:  # pragma: no cover - supports `python src/dense_retriever.py`
    from preprocess_kn import normalize_text

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
SearchResult = tuple[str, float]
Runs = dict[str, list[SearchResult]]


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense E5 + FAISS retrieval.")
    parser.add_argument("--corpus", default="data/processed/corpus.jsonl")
    parser.add_argument("--queries", default="data/processed/queries.jsonl")
    parser.add_argument("--qrels", default="data/processed/qrels.tsv")
    parser.add_argument("--output", default="experiments/results/dense_results.tsv")
    parser.add_argument("--cache-dir", default="data/processed/dense_cache")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=100, help="Number of documents per query.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu", help="SentenceTransformers device, e.g. cpu or cuda.")
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached document embeddings and rebuild them.",
    )
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


def e5_passage(text: object) -> str:
    return f"passage: {normalize_text(text)}"


def e5_query(text: object) -> str:
    return f"query: {normalize_text(text)}"


def safe_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")


def corpus_fingerprint(corpus: list[dict[str, Any]], model_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    for row in corpus:
        digest.update(str(row["doc_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(normalize_text(row.get("text", "")).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def cache_paths(cache_dir: str | Path, model_name: str, fingerprint: str) -> dict[str, Path]:
    root = Path(cache_dir) / safe_model_name(model_name)
    return {
        "root": root,
        "embeddings": root / f"{fingerprint}.embeddings.npy",
        "doc_ids": root / f"{fingerprint}.doc_ids.json",
        "index": root / f"{fingerprint}.faiss",
        "meta": root / f"{fingerprint}.meta.json",
    }


def load_model(model_name: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "Dense retrieval requires sentence-transformers, torch, faiss-cpu, numpy, and tqdm. "
            "Install them with: pip install -r requirements.txt"
        ) from exc
    return SentenceTransformer(model_name, device=device)


def encode_texts(
    model,
    texts: list[str],
    batch_size: int,
    show_progress: bool,
):
    import numpy as np

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return np.ascontiguousarray(embeddings.astype("float32"))


def load_or_create_document_embeddings(
    corpus: list[dict[str, Any]],
    model,
    model_name: str,
    cache_dir: str | Path,
    batch_size: int,
    force_recompute: bool,
):
    import faiss
    import numpy as np

    fingerprint = corpus_fingerprint(corpus, model_name)
    paths = cache_paths(cache_dir, model_name, fingerprint)
    paths["root"].mkdir(parents=True, exist_ok=True)

    doc_ids = [str(row["doc_id"]) for row in corpus]
    cache_ready = (
        paths["embeddings"].exists()
        and paths["doc_ids"].exists()
        and paths["index"].exists()
        and not force_recompute
    )

    if cache_ready:
        cached_doc_ids = json.loads(paths["doc_ids"].read_text(encoding="utf-8"))
        if cached_doc_ids == doc_ids:
            embeddings = np.load(paths["embeddings"])
            index = faiss.read_index(str(paths["index"]))
            return embeddings, index, doc_ids, True

    passage_texts = [e5_passage(row.get("text", "")) for row in corpus]
    embeddings = encode_texts(
        model,
        passage_texts,
        batch_size=batch_size,
        show_progress=True,
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    np.save(paths["embeddings"], embeddings)
    faiss.write_index(index, str(paths["index"]))
    paths["doc_ids"].write_text(json.dumps(doc_ids, indent=2) + "\n", encoding="utf-8")
    paths["meta"].write_text(
        json.dumps(
            {
                "model_name": model_name,
                "fingerprint": fingerprint,
                "documents": len(doc_ids),
                "embedding_dim": int(embeddings.shape[1]),
                "e5_passage_prefix": "passage:",
                "normalized_embeddings": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return embeddings, index, doc_ids, False


def dense_search(
    model,
    index,
    doc_ids: list[str],
    queries: list[dict[str, Any]],
    top_k: int,
    batch_size: int,
) -> Runs:
    query_texts = [e5_query(query.get("text", "")) for query in queries]
    query_embeddings = encode_texts(
        model,
        query_texts,
        batch_size=batch_size,
        show_progress=True,
    )
    search_k = min(top_k, len(doc_ids))
    scores, indices = index.search(query_embeddings, search_k)

    runs: Runs = {}
    for query, query_scores, query_indices in zip(queries, scores, indices):
        query_id = str(query["query_id"])
        results: list[SearchResult] = []
        for doc_idx, score in zip(query_indices, query_scores):
            if doc_idx < 0:
                continue
            results.append((doc_ids[int(doc_idx)], float(score)))
        runs[query_id] = results
    return runs


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
    print("\nTop 10 dense documents")
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

    model = load_model(args.model_name, args.device)
    _, index, doc_ids, cache_hit = load_or_create_document_embeddings(
        corpus=corpus,
        model=model,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        force_recompute=args.force_recompute,
    )
    runs = dense_search(
        model=model,
        index=index,
        doc_ids=doc_ids,
        queries=queries,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )
    write_results_tsv(args.output, runs)

    print(
        json.dumps(
            {
                "documents": len(corpus),
                "queries": len(queries),
                "top_k": args.top_k,
                "model_name": args.model_name,
                "cache_hit": cache_hit,
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
            metrics = compute_metrics(runs, smoke_qrels, metric_prefix="Dense")
            print("\nSmoke metrics")
            print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
