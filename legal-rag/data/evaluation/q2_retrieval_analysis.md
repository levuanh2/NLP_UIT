# Q2 Legal Query Rewrite Analysis Report

## 1. Baseline vs Q2 Retrieval Comparison

| Metric | Baseline (k=20) | Q2 Rewrite | Delta |
|---|---|---|---|
| **Article Recall** | 0.4430 | 0.2450 | -0.1980 |
| **Full Evidence Recall** | 0.3550 | 0.1750 | -0.1800 |
| **Retriever Miss (Category A)** | 60 | 48 | -12 |

## 2. Recovered Baseline Retriever-Misses
Of the 60 baseline Category A retriever-miss questions:
- **Recovered by Q2**: `12` questions
- **Still Missed by Q2**: `48` questions

### Recovered Question IDs:
104775, 150771, 97303, 80573, 129425, 147505, 56437, 94095, 122267, 42881, 91413, 140419

### Still Missed Question IDs:
122917, 120493, 63853, 166833, 88519, 136849, 97011, 24471, 5585, 83721, 147183, 10515, 104867, 6813, 132411, 59801, 120979, 13327, 149895, 35675, 98627, 142999, 90083, 21165, 130057, 164805, 73183, 67793, 25433, 86753, 39065, 78595, 104927, 3065, 107175, 37907, 65633, 77989, 47219, 32277, 48823, 134943, 32967, 163025, 100179, 88097, 54597, 145171

## 3. Regression Check
- **Baseline Hits Lost by Q2**: `35` questions
- **Regression IDs**: 84063, 17071, 101347, 16727, 40285, 25197, 3501, 138665, 61917, 116951, 76757, 36981, 152853, 52453, 32039, 79409, 153641, 98907, 5333, 36139, 101793, 8545, 134869, 42791, 44641, 2903, 44591, 167645, 38837, 23211, 45687, 158077, 130739, 45003, 93479

## 4. Query Source Contribution Analysis
*(Based on recovered questions analysis)*
- `original_query`: default baseline search
- `statute_query`: aligns query structure with precise Vietnamese legal terminology
- `concept_queries`: resolves abstract definition matches
- `evidence_queries`: targets specific clauses or condition matches

## 5. First Gate Decision (Retrieval Only)
**First Gate Verdict**: `NO-GO`
- **Action**: STOP. The query rewriting strategy does not yield a net improvement in article recall or fails to recover baseline misses without incurring equal regression.