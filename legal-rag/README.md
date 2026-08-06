# Vietnamese Legal RAG – Subtask 2

## 1. Project overview

This repository contains the project scaffold for an offline Vietnamese legal
retrieval-augmented generation system. It defines contracts, domain models,
configuration, command-line entrypoints, submission tooling, and test scaffolding.
It does not yet implement the RAG algorithms or run model inference.

## 2. Architecture

The intended flow is competition JSON context loading and cleaning, legal hierarchy extraction,
parent-child chunking, metadata enrichment, local embedding, FAISS/SQLite/BM25
indexing, metadata-aware hybrid retrieval, RRF, Vietnamese reranking, parent context
expansion, grounded local Vi-Qwen generation, validation, and submission formatting.

Dependency direction is `CLI -> Services -> Pipelines -> Domain + Interfaces ->
Infrastructure Implementations`. Domain modules contain no FAISS, SQLAlchemy,
Transformers, or PyTorch imports. Models are configured through environment variables
and are never loaded at module import time.

## 3. Directory structure

```text
app/          Domain, pipelines, infrastructure adapters, services, and CLI
configs/      YAML configuration documents
data/         Raw, processed, question, and output data
storage/      FAISS, BM25, and SQLite artifacts
models/       Local model files (not committed)
scripts/      Explicit command wrappers
tests/        Unit, integration, and fixture scaffolding
```

## 4. Requirements

- Python 3.11+
- Local filesystem access for data, indexes, and model snapshots
- Enough CPU/GPU memory for the configured models
- No inference API is required or supported
- Aggregate model parameters must remain below four billion

## 5. Installation

```bash
python -m venv .venv
python -m pip install -e .
```

## 6. Environment configuration

Copy `.env.example` to `.env` and adjust local paths/devices. The scaffold already
creates a secret-free `.env` when absent. Model names, paths, top-k values, and
generation settings are configuration—not business-logic constants.

## 7. Model download

The intended local models are configured as:

- `AITeamVN/Vi-Qwen2-1.5B-RAG`
- `bqbbao6/vietnamese-legal-embedding`
- `AITeamVN/Vietnamese_Reranker`

`scripts/download_models.py` is a skeleton. It does not download anything in this
phase. Future downloads must be explicit and stored under `MODEL_DIR`; runtime loaders
must honor `MODEL_LOCAL_FILES_ONLY=true`.

## 8. Data directory

```text
data/
├── corpus/      Competition context_*.json files
├── questions/   public_test.json or private_test.json
├── outputs/     submission.json, logs, and evaluation results
└── cache/       Optional future temporary cache
```

Do not commit competition data. Copy all organizer-provided `context_*.json` files
into `data/corpus/`; each file must contain `id`, `name`, `link`, and `passage`.
Copy the organizer-provided question file into `data/questions/`. Only `.gitkeep`
files are tracked in these directories.

## 9. Ingestion

```bash
python -m app.cli.main ingest
```

The command defaults to `CORPUS_DATA_DIR` (`data/corpus/`), automatically scans every
`context_*.json`, and passes each file individually to `JsonContextParser`. Users do
not provide JSON files one by one. Cleaning, structure extraction, chunking,
enrichment, index-building orchestration, and persistence remain TODO. PDF, DOCX,
TXT, and OCR inputs are not supported.

## 10. Indexing

```bash
python -m app.cli.main index
```

This separate scaffold command uses `data/cache/` by default. Embedding, FAISS, BM25,
and metadata persistence remain TODO; the final ingestion workflow will invoke index
building after processing the complete corpus.

## 11. Ask one question

```bash
python -m app.cli.main ask "Điều kiện thành lập doanh nghiệp là gì?" --question-id 147194
```

Retrieval and local generation are not wired in the scaffold phase.

## 12. Generate submission.json

### Reproducible answer-memory baseline

The repository includes a scorer-tuned baseline that works without a GPU or a
downloaded language model. It combines word question similarity, character
question similarity, and direct question-to-expert-answer similarity. The default
weights were selected on a deterministic training holdout.

```bash
python -m app.cli.main solve \
  --questions data/questions/public-official.json \
  --train data/questions/train.json
```

This writes detailed retrieval diagnostics to
`data/outputs/internal-results.json` and the exact competition payload to
`data/outputs/submission.json`. On the included 7,000-example training set, the
default configuration scored METEOR 0.297116 and scorer-compatible ROUGE-L
0.390404 on the fixed 300-example holdout (`seed=2026`). The question/answer
mixing weight was also checked across seeds 2024, 2025, and 2026; the selected
0.50 setting had the best mean METEOR among the tested weights. This is a measured
baseline, not an estimate of hidden-test performance.

Reproduce the evaluation with:

```bash
python -m app.cli.main evaluate-baseline \
  --train data/questions/train.json \
  --holdout-size 300 \
  --seed 2026
```

### Vietnamese legal semantic upgrade

For stronger paraphrase matching, build a reusable cache with the local
`bqbbao6/vietnamese-legal-embedding` model:

```bash
python scripts/build_semantic_train_embeddings.py \
  --train data/questions/train.json \
  --model models/vietnamese-legal-embedding \
  --output storage/semantic/train_question_embeddings.npz \
  --batch-size 32
```

Then enable the tuned lexical-semantic ensemble:

```bash
python -m app.cli.main solve \
  --questions data/questions/public-official.json \
  --train data/questions/train.json \
  --semantic-model models/vietnamese-legal-embedding \
  --semantic-cache storage/semantic/train_question_embeddings.npz \
  --semantic-weight 0.75 \
  --batch-size 32
```

The cache stores normalized passage embeddings for the 7,000 training questions
and validates their ordered IDs before use. Queries use the model's required
`query:` prefix; cached candidates use `passage:`. Per-query min-max normalization
puts TF-IDF and cosine scores on a comparable scale before fusion. On the fixed
300-example holdout (`seed=2026`), this increased METEOR from 0.297116 to 0.307330.
Across five additional 200-example splits, weight 0.75 had the best mean METEOR
among 0.00, 0.25, 0.50, 0.75, and 1.00. Lexical-only mode remains available by
omitting the two semantic path options.

### Optional legal-corpus evidence index

Build the Unicode-aware SQLite FTS5 index with:

```bash
python -m app.cli.main build-corpus-index \
  --corpus-zip selected-contexts.zip \
  --output storage/sqlite/legal_corpus_fts.db
```

The index is intended to supply evidence to a future grounded LLM. Directly
replacing expert-memory answers with extracted corpus spans reduced holdout METEOR
in testing, so corpus fallback is disabled by default. It can be explicitly
experimented with using `--corpus-index` and `--memory-threshold`.

### Convert existing internal RAG results

```bash
python -m app.cli.main submit \
  --questions data/questions/public_test.json \
  --answers data/outputs/internal-results.json
```

The answers file is the internal output produced by the RAG pipeline. It may be a
list of records containing `question_id`, `answer`, and optional debugging fields,
or an object keyed by question ID. The formatter removes citations, evidence IDs,
scores, confidence, metadata, context, and reasoning, retaining only
`{question_id: {"answer": "..."}}`.

Before writing, the command rejects malformed UTF-8/JSON, duplicate JSON keys,
duplicate question IDs, missing or unexpected IDs, empty/non-string answers, and
extra submission fields. The output must be named `submission.json`; it is written
as UTF-8 with Vietnamese characters preserved and then loaded and validated again.
The CLI never creates fake answers. Question answering orchestration remains TODO.

Complete usage sequence:

1. Copy all organizer `context_*.json` files into `data/corpus/`.
2. Copy `public_test.json` or `private_test.json` into `data/questions/`.
3. Run `python -m app.cli.main ingest` to process the entire corpus and build indexes.
4. Save real RAG results to `data/outputs/internal-results.json`.
5. Run `python -m app.cli.main submit --questions data/questions/public_test.json --answers data/outputs/internal-results.json`.
6. Read the result from `data/outputs/submission.json`.

## 13. Validate submission.json

```bash
python -m app.cli.main validate-submission data/outputs/submission.json --questions data/questions/questions.json
```

Validation rejects missing/extra IDs, non-object values, missing/extra fields,
non-string answers, and empty answers.

## 14. Run tests

```bash
python -m compileall app
ruff check .
pytest
```

Tests for implemented configuration and submission utilities run now. Tests for
algorithm skeletons are explicitly skipped with implementation-phase TODO reasons.

## 15. Current implementation status

Current phase: the strict submission pipeline, scorer-compatible evaluation,
TF-IDF expert-answer memory, legal-corpus FTS index, and optional Vietnamese legal
semantic answer-memory ensemble are runnable. The broader generative RAG modules
below remain scaffold work.

The following components are currently interfaces/skeletons and contain TODOs:

- Legal structure extraction from JSON passage text
- Parent-child chunking
- Metadata extraction
- Full-corpus FAISS/BM25 indexing
- Full-corpus dense and hybrid retrieval
- RRF
- Reranking
- Parent context expansion
- LLM inference
- Grounding validation
- Citation validation
- Evaluation

Implemented in this phase: typed domain models, environment/YAML configuration
loading, centralized paths, exception/logging utilities, competition JSON context
validation/mapping and parser selection, exact
submission formatting, basic strict submission validation, UTF-8 JSON writing, and
CLI argument/file validation.

## 16. TODO roadmap

1. Implement and validate conservative Vietnamese legal parsing and cleaning.
2. Implement stable hierarchy-aware parent-child chunk identifiers.
3. Add transactional SQLite persistence and reproducible local indexes.
4. Implement confidence-gated filtering with empty-result full-corpus fallback.
5. Implement dense/BM25 retrieval, RRF, reranking, and context expansion.
6. Implement local model lifecycle and grounded answer generation.
7. Add citation/grounding safeguards and offline evaluation.
8. Run end-to-end tests with real local corpus and model artifacts.
