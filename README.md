# KannadaHybridRAG

Kannada retrieval baselines for BM25, dense vector retrieval, and Hybrid RRF
evaluation. This project currently focuses only on dataset loading, light
Kannada preprocessing, retrieval, and retrieval metrics. RAG answer generation
is intentionally left for later.

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
src/compare_results.py
src/weighted_fusion.py
src/analyze_hybrid_failures.py
src/create_romanized_queries.py
src/run_query_variant_experiment.py
src/analyze_script_gap.py
experiments/results/
README.md
```

The processed files are:

- `data/processed/corpus.jsonl`
- `data/processed/queries.jsonl`
- `data/processed/qrels.tsv`

BM25 writes:

- `experiments/results/bm25_results.tsv`
- `experiments/results/bm25_metrics.json`

Dense retrieval writes:

- `experiments/results/dense_results.tsv`
- `experiments/results/dense_metrics.json`

Hybrid RRF writes:

- `experiments/results/hybrid_rrf_results.tsv`
- `experiments/results/hybrid_rrf_metrics.json`

Weighted fusion writes:

- `experiments/results/weighted_fusion_summary.csv`
- `experiments/results/weighted_minmax_alpha_0.1.tsv`
- `experiments/results/weighted_zscore_alpha_0.1.tsv`
- `experiments/results/weighted_rank_alpha_0.1.tsv`

The Romanized Kannada query experiment writes:

- `data/processed/queries_native_100.jsonl`
- `data/processed/queries_romanized_100.jsonl`
- `data/processed/queries_mixed_100.jsonl`
- `data/processed/qrels_100.tsv`
- `data/processed/romanized_query_mapping.csv`
- `experiments/results/script_variant_summary.csv`

## Install Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

BM25 itself is pure Python. Dense retrieval uses `sentence-transformers`, `torch`,
`numpy`, `faiss-cpu`, and `tqdm`.

## Prepare Data From Hugging Face

The default Hugging Face source is `ai4bharat/IndicMSMARCO`, config `kn`, split
`train`.

```powershell
python -m src.load_data `
  --source auto `
  --dataset ai4bharat/IndicMSMARCO `
  --config kn `
  --split train `
  --output-dir data/processed
```

For a smaller conversion:

```powershell
python -m src.load_data `
  --source auto `
  --dataset ai4bharat/IndicMSMARCO `
  --config kn `
  --split train `
  --limit 1000 `
  --output-dir data/processed
```

## Prepare Data From Local Raw Files

If the Hugging Face dataset is unavailable, place files in `data/raw/` and run:

```powershell
python -m src.load_data --source raw --raw-dir data/raw --output-dir data/processed
```

Supported local options:

- Prebuilt format: `data/raw/corpus.jsonl`, `data/raw/queries.jsonl`, and `data/raw/qrels.tsv`
- Paired format: one or more `.jsonl`, `.json`, `.tsv`, or `.csv` files with fields such as `query_id`, `query`, `passage_id`, `passage`, and `relevance`

Processed `corpus.jsonl` rows use:

```json
{"doc_id": "doc1", "text": "document text", "metadata": {}}
```

Processed `queries.jsonl` rows use:

```json
{"query_id": "q1", "text": "query text", "metadata": {}}
```

Processed `qrels.tsv` uses:

```text
query_id	doc_id	relevance
q1	doc1	1
```

## Run BM25 Retrieval

Use `--top-k 100` so evaluation can compute `Recall@100`.

```powershell
python -m src.bm25_retriever `
  --corpus data/processed/corpus.jsonl `
  --queries data/processed/queries.jsonl `
  --qrels data/processed/qrels.tsv `
  --top-k 100 `
  --output experiments/results/bm25_results.tsv
```

The result TSV format is:

```text
query_id	doc_id	rank	score
```

## Evaluate BM25

```powershell
python -m src.evaluate `
  --qrels data/processed/qrels.tsv `
  --results experiments/results/bm25_results.tsv `
  --output experiments/results/bm25_metrics.json `
  --metric-prefix BM25
```

Metrics:

- `BM25 MRR@10`
- `BM25 Recall@10`
- `BM25 NDCG@10`
- `BM25 Recall@100`

## BM25 Smoke Test

After preparing data, run:

```powershell
python -m src.bm25_retriever `
  --corpus data/processed/corpus.jsonl `
  --queries data/processed/queries.jsonl `
  --qrels data/processed/qrels.tsv `
  --smoke-test `
  --query-limit 100 `
  --doc-limit 1000 `
  --top-k 100 `
  --output experiments/results/bm25_results.tsv
```

Smoke mode:

- keeps the first 100 qrel-covered queries
- keeps all relevant documents for those queries plus a limited document subset
- prints the top 10 documents for one sample query
- prints the BM25 evaluation metrics

You can then run the evaluator separately against the same result file:

```powershell
python -m src.evaluate `
  --qrels data/processed/qrels.tsv `
  --results experiments/results/bm25_results.tsv `
  --output experiments/results/bm25_metrics.json `
  --metric-prefix BM25
```

## Run Dense Retrieval

Dense retrieval uses `intfloat/multilingual-e5-small` by default and follows the
E5 input format:

- documents: `passage: <document text>`
- queries: `query: <query text>`

Document embeddings and the FAISS index are cached under
`data/processed/dense_cache/` using a corpus fingerprint, so unchanged corpora do
not need to be re-embedded on every run.

```powershell
python -m src.dense_retriever `
  --corpus data/processed/corpus.jsonl `
  --queries data/processed/queries.jsonl `
  --qrels data/processed/qrels.tsv `
  --model-name intfloat/multilingual-e5-small `
  --device cpu `
  --batch-size 32 `
  --top-k 100 `
  --output experiments/results/dense_results.tsv `
  --cache-dir data/processed/dense_cache
```

The dense result TSV format is compatible with `src.evaluate`:

```text
query_id	doc_id	rank	score
```

## Evaluate Dense Retrieval

```powershell
python -m src.evaluate `
  --qrels data/processed/qrels.tsv `
  --results experiments/results/dense_results.tsv `
  --output experiments/results/dense_metrics.json `
  --metric-prefix Dense
```

Dense metrics:

- `Dense MRR@10`
- `Dense Recall@10`
- `Dense NDCG@10`
- `Dense Recall@100`

## Run Hybrid RRF

Hybrid retrieval uses Reciprocal Rank Fusion over the saved BM25 and dense runs:

```text
rrf_score(doc) = 1 / (60 + bm25_rank(doc)) + 1 / (60 + dense_rank(doc))
```

Run fusion:

```powershell
python -m src.hybrid_retriever `
  --bm25-results experiments/results/bm25_results.tsv `
  --dense-results experiments/results/dense_results.tsv `
  --rrf-k 60 `
  --top-k 100 `
  --output experiments/results/hybrid_rrf_results.tsv
```

Evaluate Hybrid RRF:

```powershell
python -m src.evaluate `
  --qrels data/processed/qrels.tsv `
  --results experiments/results/hybrid_rrf_results.tsv `
  --output experiments/results/hybrid_rrf_metrics.json `
  --metric-prefix "Hybrid RRF"
```

Hybrid metrics:

- `Hybrid RRF MRR@10`
- `Hybrid RRF Recall@10`
- `Hybrid RRF NDCG@10`
- `Hybrid RRF Recall@100`

## Compare All Methods

After BM25, dense, and Hybrid RRF result files exist, print a comparison table:

```powershell
python -m src.compare_results `
  --qrels data/processed/qrels.tsv `
  --bm25-results experiments/results/bm25_results.tsv `
  --dense-results experiments/results/dense_results.tsv `
  --hybrid-results experiments/results/hybrid_rrf_results.tsv
```

The terminal output is:

```text
Method | MRR@10 | Recall@10 | NDCG@10 | Recall@100
BM25 | ...
Dense | ...
Hybrid RRF | ...
```

## Weighted Fusion Sweep

RRF did not outperform dense retrieval on native Kannada queries in the current
run. Weighted fusion is tested to determine whether a smaller BM25 contribution
improves ranking performance.

Weighted fusion combines per-query normalized BM25 and dense scores:

```text
hybrid_score = alpha * normalized_bm25_score + (1 - alpha) * normalized_dense_score
```

Run all normalization methods and alpha values:

```powershell
python -m src.weighted_fusion `
  --bm25-results experiments/results/bm25_results.tsv `
  --dense-results experiments/results/dense_results.tsv `
  --hybrid-rrf-results experiments/results/hybrid_rrf_results.tsv `
  --qrels data/processed/qrels.tsv `
  --output-dir experiments/results `
  --summary experiments/results/weighted_fusion_summary.csv `
  --top-k 100
```

This tests:

- normalizations: `minmax`, `zscore`, `rank`
- alpha values: `0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0`

The command writes all weighted result TSV files, evaluates each variant, saves
`experiments/results/weighted_fusion_summary.csv`, and prints a ranked comparison
table sorted by `MRR@10` including BM25, Dense, Hybrid RRF, and every weighted
fusion variant.

## Hybrid Diagnostics

After running weighted fusion, identify where dense, Hybrid RRF, and BM25 differ:

```powershell
python -m src.analyze_hybrid_failures `
  --qrels data/processed/qrels.tsv `
  --queries data/processed/queries.jsonl `
  --bm25-results experiments/results/bm25_results.tsv `
  --dense-results experiments/results/dense_results.tsv `
  --hybrid-results experiments/results/hybrid_rrf_results.tsv `
  --weighted-dir experiments/results `
  --output-dir experiments/results
```

Diagnostic outputs:

- `experiments/results/dense_beats_hybrid_queries.csv`
- `experiments/results/hybrid_beats_dense_queries.csv`
- `experiments/results/bm25_helped_queries.csv`
- `experiments/results/bm25_hurt_queries.csv`

## Romanized Kannada Query Experiment

This experiment tests whether retrieval quality drops when Kannada users type
queries in Roman script instead of native Kannada script. The goal is to measure
the script gap and check whether dense or hybrid retrieval improves robustness.

Create the 100-query native, Romanized, and mixed-script subset:

```powershell
python -m src.create_romanized_queries `
  --mode auto `
  --limit 100 `
  --queries data/processed/queries.jsonl `
  --qrels data/processed/qrels.tsv `
  --output-dir data/processed `
  --mapping data/processed/romanized_query_mapping.csv `
  --overwrite-mapping
```

Automatic mode accepts a transliteration package only when it returns clean
ASCII output; otherwise it uses the built-in deterministic Kannada-to-Latin
draft transliteration. For research reporting, review the editable mapping CSV:

```powershell
notepad data\processed\romanized_query_mapping.csv
```

If you want a fully manual workflow, create blank editable columns first:

```powershell
python -m src.create_romanized_queries `
  --mode manual `
  --limit 100 `
  --queries data/processed/queries.jsonl `
  --qrels data/processed/qrels.tsv `
  --output-dir data/processed `
  --mapping data/processed/romanized_query_mapping.csv `
  --overwrite-mapping
```

After editing `romanized_query` and `mixed_query`, regenerate the JSONL files
from the CSV without overwriting your edits:

```powershell
python -m src.create_romanized_queries `
  --mode manual `
  --limit 100 `
  --queries data/processed/queries.jsonl `
  --qrels data/processed/qrels.tsv `
  --output-dir data/processed `
  --mapping data/processed/romanized_query_mapping.csv
```

Run BM25, dense retrieval, Hybrid RRF, and the best weighted fusion method for
all three query variants:

```powershell
python -m src.run_query_variant_experiment `
  --corpus data/processed/corpus.jsonl `
  --qrels data/processed/qrels_100.tsv `
  --weighted-summary experiments/results/weighted_fusion_summary.csv `
  --output-dir experiments/results `
  --summary experiments/results/script_variant_summary.csv `
  --model-name intfloat/multilingual-e5-small `
  --device cpu `
  --batch-size 16 `
  --top-k 100
```

This command evaluates every result file against `data/processed/qrels_100.tsv`
and saves:

- `experiments/results/bm25_native_100.tsv`
- `experiments/results/bm25_romanized_100.tsv`
- `experiments/results/bm25_mixed_100.tsv`
- `experiments/results/dense_native_100.tsv`
- `experiments/results/dense_romanized_100.tsv`
- `experiments/results/dense_mixed_100.tsv`
- `experiments/results/hybrid_rrf_native_100.tsv`
- `experiments/results/hybrid_rrf_romanized_100.tsv`
- `experiments/results/hybrid_rrf_mixed_100.tsv`
- `experiments/results/weighted_best_native_100.tsv`
- `experiments/results/weighted_best_romanized_100.tsv`
- `experiments/results/weighted_best_mixed_100.tsv`
- `experiments/results/script_variant_summary.csv`

The summary table contains:

```text
query_variant,method,MRR@10,Recall@10,NDCG@10,Recall@100
native,BM25,...
native,Dense,...
native,Hybrid RRF,...
native,Best weighted fusion,...
romanized,BM25,...
romanized,Dense,...
romanized,Hybrid RRF,...
romanized,Best weighted fusion,...
mixed,BM25,...
mixed,Dense,...
mixed,Hybrid RRF,...
mixed,Best weighted fusion,...
```

Run script-gap diagnostics:

```powershell
python -m src.analyze_script_gap `
  --qrels data/processed/qrels_100.tsv `
  --native-queries data/processed/queries_native_100.jsonl `
  --romanized-queries data/processed/queries_romanized_100.jsonl `
  --mixed-queries data/processed/queries_mixed_100.jsonl `
  --results-dir experiments/results `
  --output-dir experiments/results
```

Diagnostic outputs:

- `experiments/results/native_to_romanized_drop.csv`
- `experiments/results/romanized_dense_beats_bm25.csv`
- `experiments/results/romanized_hybrid_helped.csv`
- `experiments/results/all_methods_failed_romanized.csv`

## Dense Smoke Test

After preparing data, run:

```powershell
python -m src.dense_retriever `
  --corpus data/processed/corpus.jsonl `
  --queries data/processed/queries.jsonl `
  --qrels data/processed/qrels.tsv `
  --smoke-test `
  --query-limit 100 `
  --doc-limit 1000 `
  --model-name intfloat/multilingual-e5-small `
  --device cpu `
  --batch-size 16 `
  --top-k 100 `
  --output experiments/results/dense_results.tsv `
  --cache-dir data/processed/dense_cache
```

Smoke mode:

- keeps the first 100 qrel-covered queries
- keeps all relevant documents for those queries plus a limited document subset
- prints one sample Kannada query
- prints the top 10 dense documents
- prints dense retrieval metrics

## Hybrid RRF Smoke Test

After BM25 and dense result files exist, run:

```powershell
python -m src.hybrid_retriever `
  --bm25-results experiments/results/bm25_results.tsv `
  --dense-results experiments/results/dense_results.tsv `
  --qrels data/processed/qrels.tsv `
  --queries data/processed/queries.jsonl `
  --corpus data/processed/corpus.jsonl `
  --smoke-test `
  --query-limit 100 `
  --rrf-k 60 `
  --top-k 100 `
  --output experiments/results/hybrid_rrf_results.tsv
```

Smoke mode:

- prints one sample Kannada query
- prints top 10 BM25 results
- prints top 10 dense results
- prints top 10 Hybrid RRF results
- prints BM25, dense, and Hybrid RRF metric summaries
- prints the final comparison table

## Preprocessing

`src/preprocess_kn.py` applies only:

- Unicode NFC normalization
- whitespace cleanup

Kannada script is kept unchanged. No stemming, stopword removal, transliteration,
or aggressive normalization is applied yet.
