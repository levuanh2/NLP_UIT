# Mutually-Exclusive Causal Audit Report

## 1. Taxonomy Breakdown

| Category | Description | Count | Percentage |
|---|---|---|---|
| **A** | Retriever miss gold hoàn toàn | 60 | 30.0% |
| **B** | Gold sống qua retrieval nhưng mất ở Reranker | 5 | 2.5% |
| **C** | Gold sống qua reranker nhưng mất ở Context Builder | 5 | 2.5% |
| **D** | Gold đầy đủ trong context nhưng answer sai | 23 | 11.5% |
| **E** | Gold article có nhưng evidence không đủ khoản/điểm | 7 | 3.5% |
| **Other** | Không phân loại được hoặc không có gold parseable | 17 | 8.5% |
| **OK** | Trả lời đúng (class ok) | 83 | 41.5% |

## 2. Overlap and Causal Path Analysis
- **Retrieval Level Bottleneck (A + B)**: `A` (retriever miss) and `B` (reranker drop) represent the absolute trần giới hạn của retrieval. Since these documents are never retrieved or are filtered out before context building, they cannot be resolved by downstream prompt engineering or context expansion.
- **Context Level Bottleneck (C + E)**: `C` (context builder drop) and `E` (incomplete clauses/points) represent context construction issues. Here, the article is hit but the exact clauses are missing.
- **Generation Level Bottleneck (D)**: `D` is the pure generation-side bottleneck (wrong article selection, citation formatting error, or hallucination). This represents the ceiling of prompt-only optimizations.

## 3. Ceiling Analysis for Combined Interventions

| Category | Count | Sum Baseline METEOR | Oracle METEOR Gain | Ceiling METEOR |
|---|---|---|---|---|
| A | 60 | 24.9725 | +0.0493 | 0.5373 |
| B | 5 | 1.4844 | +0.0063 | 0.4944 |
| C | 5 | 1.4795 | +0.0064 | 0.4944 |
| D | 23 | 12.4170 | +0.0107 | 0.4987 |
| E | 7 | 2.8867 | +0.0059 | 0.4940 |
| Other | 17 | 8.6707 | +0.0101 | 0.4981 |

- **Combined Intervention 1 (Prompt only - Category D)**: Ceiling = `+0.0107` METEOR
- **Combined Intervention 2 (Context Expansion + Prompt - C + E + D)**: Ceiling = `+0.0230` METEOR
- **Combined Intervention 3 (Perfect Oracle - A + B + C + D + E)**: Ceiling = `+0.0786` METEOR

## 4. Verdict
**NO-GO**
Even combining all context builder and generation prompt interventions (Categories C + E + D) yields a theoretical ceiling below the 1 SE threshold of +0.028 METEOR. Therefore, we declare a final STOP.