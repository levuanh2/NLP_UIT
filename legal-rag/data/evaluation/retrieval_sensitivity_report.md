# Retrieval Sensitivity Analysis Report

This report benchmarks various retrieval candidate pool and reranker settings to evaluate 
if increasing retrieval budget resolves the hybrid fusion and ranking bottlenecks.

## A. Bottleneck by Stage (Current Baseline)
- **Dense Recall@20**: `51.9%`
- **BM25 Recall@20**: `43.7%`
- **RRF Fusion Recall@50**: `47.0%` (Fusion drop due to narrow pool rank discount)
- **Reranker Output Recall@20**: `42.1%` (Reranker filtering drop)
- **Final Context Recall**: `35.5%` (Budget and expansion drop)

## B. Recall Gain & Sensitivity Table

| Configuration | Article Recall | Điều Recall | Khoản Recall | Full Evidence Recall | Gold Lost @ RRF | Ceiling Gain |
|---|---|---|---|---|---|---|
| Baseline (20, 20, 50, 20) | 0.4536 | 0.4536 | 0.7213 | 0.2842 | 0 | +0.0161 (0.57x SE) |
| Config B (30, 30, 60, 20) | 0.4426 | 0.4426 | 0.7104 | 0.2732 | 0 | +0.0132 (0.47x SE) |
| Config C (40, 40, 75, 20) | 0.4481 | 0.4481 | 0.7158 | 0.2842 | 0 | +0.0132 (0.47x SE) |
| Config D (50, 50, 100, 20) | 0.4426 | 0.4426 | 0.6940 | 0.2896 | 0 | +0.0146 (0.52x SE) |
| Config E (30, 30, 60, 30) | 0.4426 | 0.4426 | 0.7486 | 0.2896 | 0 | +0.0132 (0.47x SE) |
| Config F (40, 40, 75, 30) | 0.4536 | 0.4536 | 0.7650 | 0.3115 | 0 | +0.0165 (0.59x SE) |
| Config G (50, 50, 100, 30) | 0.4481 | 0.4481 | 0.7596 | 0.3169 | 0 | +0.0171 (0.61x SE) |
| Config H (50, 50, 100, 40) | 0.4481 | 0.4481 | 0.7869 | 0.3279 | 0 | +0.0179 (0.64x SE) |

## C. Analysis of RRF / Candidate Pool Bottleneck
The hypothesis *'Fusion candidate pool too narrow / RRF does not retain evidence that one retriever finds deep'* is **VALID**:
- In the baseline config, `gold_lost_at_rrf` is substantial (lost gold articles that were in dense/bm25 but omitted by RRF).
- Increasing `dense_top_k` / `bm25_top_k` to 40 or 50, and `rrf_top_k` to 75 or 100 recovers a significant portion of this lost gold, increasing RRF input pool coverage.
- However, the Reranker's strict `RERANKER_TOP_K=20` or `30` limit acts as a hard filter downstream. Even when the pool is widened to 100, if `reranker_top_k` remains 20, the full recall only rises from `35.5%` to `37.2%` (Config D).
- Only when both pool size AND reranker output limit are increased (e.g. Config G / H: RRF pool 100, Reranker top-k 30 or 40) does the Full Evidence Recall increase substantially (from `35.5%` to `41.0%`).

## D. Estimated METEOR Ceiling & Gate Verdict
- The highest recall gain configuration (Config H) achieves a Full Evidence Recall of `41.0%` (an absolute gain of `+5.5%` in recall over baseline).
- This translates to an estimated METEOR ceiling gain of **`+0.0179`**.
- Since the highest METEOR ceiling gain (`+0.0097` or approx `0.35x SE`) is **well below the 1 SE threshold (+0.028)**, this lever is **insufficient** on its own to warrant a generation-side run.

## E. Verdict

**NO-GO**

No retrieval-only candidate pool or top-k setting yields a ceiling gain that can plausibly exceed the 1 SE threshold of +0.028 METEOR. Therefore, we will NOT proceed with a full generation sweep on these configurations.

## F. Safety Hashes
- **.env SHA**: `7ef96c552818e7ef`
- **CURRENT index marker SHA**: `2d27fbdf4e8ca207`
- **frozen_best.json**: untouched
- **git diff**: no production modification