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

```bash
python -m app.cli.main submit --questions data/questions/public_test.json
```

The formatter contract produces only `{question_id: {"answer": "..."}}`. The CLI
defaults to `data/outputs/submission.json`. It does not create fake answers; question
answering orchestration remains TODO.

Complete usage sequence:

1. Copy all organizer `context_*.json` files into `data/corpus/`.
2. Copy `public_test.json` or `private_test.json` into `data/questions/`.
3. Run `python -m app.cli.main ingest` to process the entire corpus and build indexes.
4. Run `python -m app.cli.main submit --questions data/questions/public_test.json`.
5. Read the result from `data/outputs/submission.json`.

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

Current phase:
Project scaffold only.

The following components are currently interfaces/skeletons and contain TODOs:

- Legal structure extraction from JSON passage text
- Parent-child chunking
- Metadata extraction
- Embedding
- FAISS indexing
- BM25 indexing
- Dense retrieval
- Hybrid retrieval
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
