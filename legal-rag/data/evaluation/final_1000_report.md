# LegalQA - final 1000-question run

## A. Best configuration

- checkpoint: `/home/ezycloudx-admin/NLP_UIT/legal-rag/models/qlora-lr5e4/checkpoint-350`
- index: `/home/ezycloudx-admin/NLP_UIT/legal-rag/storage/index-staging` (enriched)
- reranker: Vietnamese cross-encoder scoring `embedding_text`
- RERANKER_TOP_K: 20
- LLM_BATCH_SIZE: 1
- CONTEXT_MAX_TOKENS: 10000
- selected on dev200 METEOR=0.488 ROUGE-L=0.3345

## B. dev200 candidates

| run | checkpoint | index | k | METEOR | ROUGE-L | failures | median words | runtime s | status |
|---|---|---|---|---|---|---|---|---|---|
| enriched-ckpt350-k20 | checkpoint-350 | enriched | 20 | 0.488 | 0.3345 | 0 | 457 | 3716 | VALID |
| enriched-ckpt125-k40 | checkpoint-125 | enriched | 40 | 0.4574 | 0.3132 | 0 | 441 | - | VALID |
| enriched-ckpt125-k20 | checkpoint-125 | enriched | 20 | 0.4541 | 0.3124 | 0 | 432 | - | VALID |
| baseline-ckpt125-k20 | checkpoint-125 | baseline | 20 | 0.4503 | 0.3081 | 0 | 435 | 3619 | VALID |
| baseline-ckpt125-k10 | checkpoint-125 | baseline | 10 | 0.4369 | 0.2992 | 0 | 440 | - | VALID |

### Contribution decomposition

- A. top-k (baseline k20 - baseline k10): +0.0134
- B. enrichment (enriched k20 - baseline k20): +0.0038
- C. checkpoint (ckpt350 k20 - ckpt125 k20, both enriched): +0.0339

Checkpoint-350 comes from the `qlora-lr5e4` run and checkpoint-125 from `qlora-answerer`, so C mixes learning rate, step count and training run. It is not a pure step ablation.

## C. Full 1000-question run

- generated: 1000 / 1000
- failed: 0
- abstentions: 0
- answer words: median 450.0, mean 532.2
- wall clock: 18073s (5.02 h)
- allocator OOM warnings (recovered): 0
0
- hard CUDA OOM losing a question: none

## D. Submission

- expected questions: 1000
- generated questions: 1000
- missing: 0
- extra: 0
- zip contents: ['submission.json']
- validation: PASS
- submission.json: `data/outputs/final-1000/submission.json`
- submission.zip: `data/outputs/final-1000/submission.zip`
