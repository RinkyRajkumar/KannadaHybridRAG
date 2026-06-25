# Kannada Hybrid Retrieval

Retrieval evaluation pipeline for Kannada using BM25, dense vector retrieval,
and hybrid retrieval. This project focuses only on retrieval and evaluation; it
does not implement LLM answer generation.

## Project Structure

```text
data/raw/
data/processed/
src/load_data.py
src/preprocess_kn.py
src/bm25_retriever.py
src/dense_retriever.py
src/hybrid_retriever.py
src/evaluate.py
experiments/results/
README.md
```

## Dataset

The default loader uses the Hugging Face dataset
[`ai4bharat/IndicMSMARCO`](https://huggingface.co/datasets/ai4bharat/IndicMSMARCO)
with config `kn` and split `train`. The Hugging Face Dataset Viewer currently
lists Kannada (`kn`) as an available config for this dataset.

The converter writes:

- `data/processed/corpus.jsonl`
- `data/processed/queries.jsonl`
- `data/processed/qrels.tsv`

## Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For BM25-only experiments, `sentence-transformers` is not used at runtime, but
it is included in `requirements.txt` so dense and hybrid commands work after a
single install.

## Prepare Data

Load and convert the Kannada split:

```powershell
python -m src.load_data --dataset ai4bharat/IndicMSMARCO --config kn --split train --output-dir data/processed
```

Run a smaller smoke conversion:

```powershell
python -m src.load_data --dataset ai4bharat/IndicMSMARCO --config kn --split train --limit 1000 --output-dir data/processed
```

If another INDIC-MARCO style dataset is needed, override the dataset/config:

```powershell
python -m src.load_data --dataset saifulhaq9/indicmarco --config kn --split train --output-dir data/processed
```

## Evaluate BM25

```powershell
python -m src.evaluate `
  --retriever bm25 `
  --corpus data/processed/corpus.jsonl `
  --queries data/processed/queries.jsonl `
  --qrels data/processed/qrels.tsv `
  --top-k 10 `
  --output experiments/results/bm25_metrics.json `
  --run-output experiments/results/bm25_run.jsonl
```

## Evaluate Dense Retrieval

```powershell
python -m src.evaluate `
  --retriever dense `
  --model-name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 `
  --corpus data/processed/corpus.jsonl `
  --queries data/processed/queries.jsonl `
  --qrels data/processed/qrels.tsv `
  --top-k 10 `
  --output experiments/results/dense_metrics.json `
  --run-output experiments/results/dense_run.jsonl
```

## Evaluate Hybrid Retrieval

Hybrid retrieval defaults to reciprocal rank fusion (`rrf`) over BM25 and dense
rankings.

```powershell
python -m src.evaluate `
  --retriever hybrid `
  --fusion rrf `
  --candidate-k 100 `
  --sparse-weight 1.0 `
  --dense-weight 1.0 `
  --model-name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 `
  --corpus data/processed/corpus.jsonl `
  --queries data/processed/queries.jsonl `
  --qrels data/processed/qrels.tsv `
  --top-k 10 `
  --output experiments/results/hybrid_metrics.json `
  --run-output experiments/results/hybrid_run.jsonl
```

For score-based fusion instead of rank fusion:

```powershell
python -m src.evaluate --retriever hybrid --fusion linear --sparse-weight 0.5 --dense-weight 0.5
```

## Metrics

`src/evaluate.py` reports:

- `MRR@10`
- `Recall@10`
- `NDCG@10`

The output JSON also records the retriever type, number of documents, number of
queries, and number of qrels queries.

## Notes

- Kannada text normalization is intentionally light: Unicode NFC normalization
  and whitespace cleanup only.
- BM25 is implemented in pure Python in `src/bm25_retriever.py`.
- Dense retrieval is isolated in `src/dense_retriever.py` so models, batching,
  caching, or ANN indexes can be added later without changing metric code.
- The hybrid retriever supports both reciprocal rank fusion and min-max linear
  fusion.

