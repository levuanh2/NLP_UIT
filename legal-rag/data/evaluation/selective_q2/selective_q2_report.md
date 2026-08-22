# Selective Q2 Fallback Retrieval Report

## 1. Safety hashes
- .env SHA prefix: `7ef96c552818e7ef` ✓
- CURRENT: `v1` ✓
- frozen_best SHA prefix: `8ed73d34e075b725` ✓

## 2. Baseline configuration
- `dense_top_k`: 20
- `bm25_top_k`: 20
- `rrf_top_k`: 30
- `reranker_top_k`: 20
- `index`: `storage/index-staging/v1-enriched`

## 3. Fallback Policy Definitions
- **F1**: Fallback if no candidate matching the query analyzer's detected document/article is present in the top-20 retrieved candidates.
- **F2**: Fallback if top reranker score < 0.5.
- **F3**: Fallback if top reranker score < P25 (-0.5830).
- **F4**: Fallback if document name / article detected but no matching candidate OR top score < P10 (-2.3984).

## 4. Policy Performance Comparison

| Policy | Article Recall | Full Evidence Recall | Fallback Queries | Rescued | Regression | Net Gain | Fallback Precision |
|---|---|---|---|---|---|---|---|
| Baseline | 0.4430 | 0.3550 | 0 | 0 | 0 | 0 | — |
| Global Q2 | 0.2450 | 0.1750 | 200 | 12 | 35 | -23 | 0.060 |
| F1 | 0.2450 | 0.1700 | 11 | 0 | 0 | +0 | 0.000 |
| F2 | 0.2450 | 0.1700 | 62 | 1 | 1 | +0 | 0.016 |
| F3 | 0.2450 | 0.1700 | 43 | 1 | 1 | +0 | 0.023 |
| F4 | 0.2450 | 0.1700 | 28 | 0 | 0 | +0 | 0.000 |

## 5. Rescued and Regression Query Lists (for Best Policy: F1)
- **Rescued QIDs (0)**: None
- **Regression QIDs (0)**: None

## 6. Decision Gate
- **Best Policy**: `F1`
- **Gate Verdict**: `NO-GO`
- **Action**: STOP. No fallback policy achieved a positive net gain (rescues > regressions) or improved Article Recall compared to the baseline without causing net regression.