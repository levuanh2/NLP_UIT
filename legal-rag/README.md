# Vietnamese Legal RAG – Subtask 2

## 1. Project overview

This repository contains an offline Vietnamese legal RAG system. Its private-test-safe
path cleans the organizer corpus, extracts legal hierarchy, creates stable parent-child
chunks, persists SQLite metadata, builds BM25 and normalized dense FAISS indexes,
performs metadata-aware hybrid retrieval with RRF and Vietnamese reranking, expands
parent evidence, generates grounded answers with local Vi-Qwen, validates provenance,
and writes the exact submission schema. No question or answer file is indexed.

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
tests/        Unit, integration, and fixture coverage
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

Copy `.env.example` to `.env` and adjust local paths/devices. The repository already
creates a secret-free `.env` when absent. Model names, paths, top-k values, and
generation settings are configuration—not business-logic constants.

## 7. Model download

The intended local models are configured as:

- `AITeamVN/Vi-Qwen2-1.5B-RAG`
- `bqbbao6/vietnamese-legal-embedding`
- `AITeamVN/Vietnamese_Reranker`

Download all three snapshots explicitly with `python scripts/download_models.py`, or
use `--only embedding`, `--only reranker`, or `--only llm`. Files are stored under
`MODEL_DIR`; runtime loaders honor `MODEL_LOCAL_FILES_ONLY=true`.

## 8. Data directory

```text
data/
├── corpus/      Competition context_*.json files
├── questions/   public_test.json or private_test.json
├── outputs/     submission.json, logs, and evaluation results
└── cache/       Deterministic parent/child JSONL cache and build manifest
```

Do not commit competition data. Copy all organizer-provided `context_*.json` files
into `data/corpus/`; each file must contain `id`, `name`, `link`, and `passage`.
Copy the organizer-provided question file into `data/questions/`. Only `.gitkeep`
files are tracked in these directories.

## 9. Ingestion

```bash
python -m app.cli.main ingest \
  --source data/corpus \
  --output storage/sqlite/legal.db
```

The command defaults to `CORPUS_DATA_DIR` (`data/corpus/`), scans every
`context_*.json` in deterministic order, normalizes Unicode/whitespace, preserves
legal article boundaries where available, windows oversized articles with overlap,
creates deterministic chunk IDs, and persists content plus document metadata and a
build manifest. It never reads `train.json`, public questions, or answer keys.

## 10. Indexing

```bash
python -m app.cli.main index \
  --cache data/cache \
  --embedding-model models/vietnamese-legal-embedding \
  --faiss-output storage/faiss/legal.index \
  --bm25-output storage/bm25/legal.db \
  --metadata-output storage/sqlite/legal.db \
  --rebuild
```

This embeds every cached child with the required `passage:` prefix, builds a cosine
FAISS index, a Unicode-aware BM25 index, and transactional parent/child metadata.
`--max-children` exists only for diagnostic smoke tests; omit it for competition use.
Dense inputs are capped at 128 model tokens for bounded 4 GB GPU memory; BM25 and
parent evidence retain the complete legal text.

## 11. Ask one question

Full hybrid RAG:

```bash
python -m app.cli.main ask-rag \
  "Điều kiện thành lập doanh nghiệp là gì?" \
  --question-id 147194
```

The command loads only local artifacts, runs dense/BM25 retrieval, RRF, reranking,
parent expansion, Vi-Qwen generation, and grounding validation.

Corpus-extractive diagnostic fallback:

```bash
python -m app.cli.main ask "Điều kiện thành lập doanh nghiệp là gì?" \
  --question-id 147194 \
  --corpus-index storage/sqlite/legal.db
```

The returned JSON includes a corpus-grounded extractive answer and internal evidence
diagnostics. The answer is retrieved only from chunks stored during ingestion.

## 12. Generate submission.json

### Full public/private RAG path

```bash
python -m app.cli.main solve-rag \
  --questions data/questions/private-official.json \
  --internal-output data/outputs/internal-results.json \
  --submission-output data/outputs/submission.json
```

The batch command checkpoints `internal-results.json` after every question and resumes
by default. Public and private files follow the identical path and are never used while
building chunks or indexes.

### Corpus-only public/private test path

Use the same command for public and private questions; only the question path changes:

```bash
python -m app.cli.main solve-corpus \
  --questions data/questions/public-official.json \
  --corpus-index storage/sqlite/legal.db \
  --semantic-model models/vietnamese-legal-embedding \
  --semantic-weight 0.75
```

This writes evidence-rich diagnostics to `data/outputs/internal-results.json` and
the scorer payload to `data/outputs/submission.json`. The latter contains exactly
`{question_id: {"answer": "..."}}`. Public questions are used only as queries and
for local evaluation. They are never indexed or used to choose an answer.
The optional semantic reranker embeds only the unseen query and retrieved corpus
chunks with the model's required `query:`/`passage:` prefixes; it never embeds or
retrieves from the training answer memory.

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

### ZIP alternative for the legal-corpus evidence index

Build the Unicode-aware SQLite FTS5 index with:

```bash
python -m app.cli.main build-corpus-index \
  --corpus-zip selected-contexts.zip \
  --output storage/sqlite/legal_corpus_fts.db
```

This produces the same persisted chunk schema directly from the organizer ZIP. Use
`solve-corpus` to query it without any dependency on training questions or answers.
The older `solve` command can also use it as an experimental fallback to answer
memory through `--corpus-index` and `--memory-threshold`.

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
The CLI never creates fake answers.

Complete usage sequence:

1. Copy all organizer `context_*.json` files into `data/corpus/`.
2. Copy `public_test.json` or `private_test.json` into `data/questions/`.
3. Run `python -m app.cli.main ingest` to process the entire corpus and build indexes.
4. Run `python -m app.cli.main index --rebuild` and wait for the full-corpus artifacts.
5. Run `python -m app.cli.main solve-rag --questions data/questions/private-official.json`.
6. Validate and read `data/outputs/submission.json`.

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

The suite covers ingestion hierarchy, stable chunk links, persistence orchestration,
metadata filtering, RRF, parent expansion, grounded generation, and submission output.

## 15. Current implementation status

Implemented: legal cleaning and hierarchy extraction; stable parent-child chunking;
JSONL cache; SQLite metadata; normalized local embedding; FAISS; BM25; explicit legal
reference filtering with empty-result fallback; dense/lexical retrieval; RRF; local
Vietnamese reranking; parent expansion; bounded prompts; local Vi-Qwen inference;
citation, grounding, and abstention validation; resumable batch generation; offline
evaluation; and strict UTF-8 submission formatting.

The TF-IDF/semantic answer-memory commands remain reproducible benchmarks only. They
are not the recommended private-test path because they retrieve training answers.

## 16. Operational notes

- Model and competition artifacts are ignored by Git.
- Full-corpus embedding time depends heavily on CUDA availability; CPU-only builds can
  take many hours. A diagnostic subset is not a valid competition index.
- The aggregate configured model size remains below four billion parameters.
