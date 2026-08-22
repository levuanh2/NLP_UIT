# V6 End-to-End Experiment Report

## 1. Configuration

| Parameter | Frozen v5 | V6 |
|---|---|---|
| dense_top_k | 20 | 50 |
| bm25_top_k | 20 | 50 |
| rrf_top_k | 30 | 100 |
| reranker_top_k | 20 | 40 |
| System Prompt | baseline | +internal evidence identification |
| Everything else | identical | identical |

## 2. Metrics

| Metric | Frozen v5 | V6 | Delta |
|---|---|---|---|
| **METEOR** | 0.4880 | 0.4799 | -0.0081 |
| **ROUGE-L** | 0.3345 | 0.3353 | +0.0008 |
| Median Words | 457 | 428 | -29 |
| Article Hit Rate | — | 0.5500 (110/200) | — |

## 3. Decision Gate

- Gate threshold: +0.028 METEOR
- Actual delta:   -0.0081
- **Verdict: `NO-GO`**

## 4. Runtime

- Wall time: 3907s (65.1min)
- Avg per query: 19.5s

## 5. Safety

- .env SHA prefix: `7ef96c552818e7ef` ✓
- CURRENT: `v1` ✓
- frozen_best SHA prefix: `8ed73d34e075b725` ✓
