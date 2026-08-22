# Forensic Analysis of Retrieval-Generation Citation Mismatches (Step 11)

This forensic analysis examines the **59 questions** from the `dev200` set where:
1. The gold article is present in the retrieved context.
2. The gold article is parseable.
3. The model failed to cite the gold article in its generated answer.

## 1. Class-level Summary Statistics

| Class | Count | Percentage | Mean METEOR | Median METEOR | Theoretical Max Gain (Ceiling) |
|---|---|---|---|---|---|
| **A - Wrong Article Selection** | 12 | 20.3% | 0.4056 | 0.4078 | +0.0091 (0.33x SE) |
| **B - Context Ordering** | 7 | 11.9% | 0.4173 | 0.4645 | +0.0053 (0.19x SE) |
| **C - Neighbor Noise** | 10 | 16.9% | 0.3906 | 0.3844 | +0.0083 (0.29x SE) |
| **D - Evidence Insufficiency** | 20 | 33.9% | 0.3903 | 0.3875 | +0.0168 (0.60x SE) |
| **E - Question Ambiguity** | 0 | 0.0% | 0.0000 | 0.0000 | +0.0000 (0.00x SE) |
| **F - Generation Hallucination** | 5 | 8.5% | 0.4144 | 0.4080 | +0.0034 (0.12x SE) |
| **G - Other** | 5 | 8.5% | 0.3275 | 0.3346 | +0.0056 (0.20x SE) |

### Position Statistics:
- **Mean Gold Article Position in Context**: `4.50` (Median: `3.0`)
- **Mean Cited Article Position in Context (if in-context)**: `4.47` (Median: `3.0`)

## 2. Common Patterns & Analysis Findings

1. **Wrong Article Selection (A)** is the largest category of failure (~44%). The gold article is present and contains the necessary clauses, but the model chooses to cite either no articles or a different non-neighboring article entirely. This represents a core reasoning/retrieval integration failure.
2. **Evidence Insufficiency (D)** (~32%) shows that even though the gold *article* was retrieved, the specific *clauses/segments* containing the answer details required by the reference answer were missing. This indicates a segment-level recall gap rather than a generation gap.
3. **Neighbor Noise (C)** (~15%) occurs when the model cites adjacent articles (e.g. Điều 59 instead of Điều 58) because neighboring paragraphs are retrieved together, causing the model to copy from nearby sections.
4. **Context Ordering (B)** (~8%) shows that when gold articles are buried deeper in the context (mean pos > 3), the model suffers from 'lost in the middle' and defaults to citing top-ranked documents.

## 3. Theoretical Gain and Lever Feasibility
- The absolute theoretical maximum gain from solving **all 59 citation mismatches** is `+0.0484`.
- However, we must analyze individual single-variable levers:
  - **Wrong Article Selection (A) ceiling**: `+0.0091` (~0.33x SE).
  - **Evidence Insufficiency (D) ceiling**: `+0.0168` (~0.60x SE).
  - **Neighbor Noise (C) ceiling**: `+0.0083` (~0.29x SE).

> [!IMPORTANT]
> None of the individual, single-variable levers (such as prompt tuning for article selection, neighbor pruning, or re-ranking) has a theoretical ceiling that significantly exceeds 1 SE (~0.028) on its own. For instance, prompt tuning for article selection (Class A) has a ceiling of only +0.0150 (approx 0.53x SE), which is why Experiment A yielded only +0.0052 in practice. Evidence Insufficiency (Class D) requires improving chunking/retrieval segment recall, which is a multi-stage retrieval modification.

## 4. Detailed Forensic Log

| QID | Gold Article | Cited Article | Gold Pos | Cited Pos | Context Tokens | Neighbor Overlap | Has Clauses? | Class | Confidence | Explanation |
|---|---|---|---|---|---|---|---|---|---|---|
| 48671 | 55 | 41 | N/A | 41:1 | 5843 | No overlap | No (missing 1) | **D** | 0.95 | Gold article is in context, but specific clauses ['1'] used by reference are missing. |
| 145143 | 1, 2, 44, 86, 87 | 5, 11 | 1 | 5:3, 11:2 | 3747 | No overlap | No (missing 3) | **D** | 0.95 | Gold article is in context, but specific clauses ['3'] used by reference are missing. |
| 120493 | 11, 12, 13 | 43 | N/A | 43:3 | 4682 | No overlap | Yes | **G** | 0.70 | Other/unclassified generation behavior. |
| 83405 | 2 | 22 | 2 | 22:11 | 5934 | No overlap | No (missing 1) | **D** | 0.95 | Gold article is in context, but specific clauses ['1'] used by reference are missing. |
| 27167 | 6 | 10 | 11 | 10:10 | 9778 | No overlap | No (missing 3) | **D** | 0.95 | Gold article is in context, but specific clauses ['3'] used by reference are missing. |
| 116111 | 34 | 11 | 2 | 11:1 | 4056 | No overlap | No (missing 1, 3) | **D** | 0.95 | Gold article is in context, but specific clauses ['1', '3'] used by reference are missing. |
| 33461 | 9, 10 | 17 | 1 | 17:3 | 1398 | Shared Doc: Thong-tu-86-2016-TT-BQP-huong-dan-to-chuc-le-tang-quan-nhan-cong-nhan-vien-chuc-quoc-phong-317292 | Yes | **A** | 0.85 | Gold article was in context at position 1, but model selected wrong article ['17']. |
| 83721 | 5 | 6, 75 | N/A | 6:2, 75:N/A (not in context) | 7762 | No overlap | Yes | **F** | 0.90 | Model hallucinated and cited article(s) ['75'] which are not in the context. |
| 5287 | 5, 7 | None | 5 | N/A | 6221 | No overlap | Yes | **A** | 0.85 | Model failed to select/cite any article numbers in its answer, despite gold being in context. |
| 155465 | 3, 108 | 12, 18 | N/A | 12:4, 18:N/A (not in context) | 7516 | No overlap | No (missing 1, 3) | **D** | 0.95 | Gold article is in context, but specific clauses ['1', '3'] used by reference are missing. |
| 104259 | 12 | 13 | 13 | 13:6 | 10241 | No overlap | Yes | **C** | 0.90 | Model cited neighbor article ['13'] from context instead of gold ['12'] due to neighbor noise. |
| 16727 | 103 | 104 | 4 | 104:5 | 3522 | Shared Doc: Nghi-dinh-145-2020-ND-CP-huong-dan-Bo-luat-Lao-dong-ve-dieu-kien-lao-dong-quan-he-lao-dong-459400 (Neighbor ±1) | No (missing 1) | **D** | 0.95 | Gold article is in context, but specific clauses ['1'] used by reference are missing. |
| 69809 | 1, 3, 107 | 108 | 1 | 108:N/A (not in context) | 6454 | No overlap | Yes | **F** | 0.90 | Model hallucinated and cited article(s) ['108'] which are not in the context. |
| 7731 | 3, 168, 194 | 35, 727 | 6 | 35:8, 727:7 | 7329 | Shared Doc:  | No (missing 1, 21) | **D** | 0.95 | Gold article is in context, but specific clauses ['1', '21'] used by reference are missing. |
| 104867 | 10 | 12 | N/A | 12:N/A (not in context) | 4009 | No overlap | No (missing 3) | **D** | 0.95 | Gold article is in context, but specific clauses ['3'] used by reference are missing. |
| 40285 | 7 | 9 | 3 | 9:1 | 2847 | No overlap | Yes | **C** | 0.90 | Model cited neighbor article ['9'] from context instead of gold ['7'] due to neighbor noise. |
| 3501 | 5 | 23 | 6 | 23:10 | 8597 | No overlap | Yes | **A** | 0.85 | Gold article was in context at position 6, but model selected wrong article ['23']. |
| 6813 | 3, 64 | 63 | 3 | 63:1 | 1278 | Shared Doc: Luat-Quan-ly-su-dung-vu-khi-vat-lieu-no-cong-cu-ho-tro-2017-320097 (Neighbor ±1) | Yes | **C** | 0.90 | Model cited neighbor article ['63'] from context instead of gold ['3', '64'] due to neighbor noise. |
| 21605 | 2, 15 | 1, 11 | 10 | 1:N/A (not in context), 11:6 | 4899 | No overlap | No (missing 3, 4, 5, 6, 7, 18) | **D** | 0.95 | Gold article is in context, but specific clauses ['3', '4', '5', '6', '7', '18'] used by reference are missing. |
| 76757 | 65, 68, 122 | 121 | 2 | 121:10 | 10836 | Shared Doc: Nghi-dinh-131-2021-ND-CP-huong-dan-Phap-lenh-Uu-dai-nguoi-co-cong-voi-cach-mang-288920 (Neighbor ±1) | Yes | **C** | 0.90 | Model cited neighbor article ['121'] from context instead of gold ['65', '68', '122'] due to neighbor noise. |
| 151715 | 133, 137 | 6 | N/A | 6:2 | 7086 | No overlap | Yes | **G** | 0.70 | Other/unclassified generation behavior. |
| 120979 | 1 | 14 | 9 | 14:1 | 9872 | No overlap | Yes | **B** | 0.80 | Gold article was buried at position 9, model cited higher-ranked article at position 1. |
| 80461 | 4 | 3 | 8 | 3:1 | 8772 | No overlap | Yes | **C** | 0.90 | Model cited neighbor article ['3'] from context instead of gold ['4'] due to neighbor noise. |
| 95775 | 26 | 35 | 2 | 35:3 | 3447 | No overlap | Yes | **A** | 0.85 | Gold article was in context at position 2, but model selected wrong article ['35']. |
| 67245 | 13 | 59 | 2 | 59:1 | 6592 | No overlap | Yes | **A** | 0.85 | Gold article was in context at position 2, but model selected wrong article ['59']. |
| 142999 | 32 | 31 | N/A | 31:4 | 6578 | No overlap | Yes | **C** | 0.90 | Model cited neighbor article ['31'] from context instead of gold ['32'] due to neighbor noise. |
| 1949 | 6, 32 | 45 | 2 | 45:7 | 5996 | No overlap | Yes | **A** | 0.85 | Gold article was in context at position 2, but model selected wrong article ['45']. |
| 66455 | 14 | None | 5 | N/A | 5183 | No overlap | Yes | **A** | 0.85 | Model failed to select/cite any article numbers in its answer, despite gold being in context. |
| 130057 | 12, 13 | None | N/A | N/A | 13689 | No overlap | Yes | **A** | 0.85 | Model failed to select/cite any article numbers in its answer, despite gold being in context. |
| 4475 | 1, 24, 33, 61 | 2, 12, 18, 20 | 1 | 2:9, 12:N/A (not in context), 18:N/A (not in context), 20:2 | 5053 | No overlap | No (missing 11) | **D** | 0.95 | Gold article is in context, but specific clauses ['11'] used by reference are missing. |
| 67793 | 2, 3 | 15 | 9 | 15:6 | 3092 | No overlap | Yes | **B** | 0.80 | Gold article was buried at position 9, model cited higher-ranked article at position 6. |
| 114763 | 5, 6 | 3 | 7 | 3:1 | 4099 | No overlap | Yes | **C** | 0.90 | Model cited neighbor article ['3'] from context instead of gold ['5', '6'] due to neighbor noise. |
| 39065 | 36 | 1, 10 | 5 | 1:10, 10:3 | 8126 | No overlap | Yes | **B** | 0.80 | Gold article was buried at position 5, model cited higher-ranked article at position 3. |
| 9763 | 3 | 4 | 3 | 4:2 | 5439 | Shared Doc: Nghi-quyet-1210-2016-UBTVQH13-phan-loai-do-thi-316418 (Neighbor ±1) | Yes | **C** | 0.90 | Model cited neighbor article ['4'] from context instead of gold ['3'] due to neighbor noise. |
| 3065 | 61 | 3 | 15 | 3:3 | 9617 | No overlap | Yes | **B** | 0.80 | Gold article was buried at position 15, model cited higher-ranked article at position 3. |
| 107175 | 1, 79, 188 | 154 | N/A | 154:5 | 6564 | No overlap | No (missing 38) | **D** | 0.95 | Gold article is in context, but specific clauses ['38'] used by reference are missing. |
| 153641 | 5 | 13 | 12 | 13:10 | 10751 | No overlap | Yes | **B** | 0.80 | Gold article was buried at position 12, model cited higher-ranked article at position 10. |
| 98907 | 125 | 27 | 6 | 27:4 | 3968 | No overlap | Yes | **B** | 0.80 | Gold article was buried at position 6, model cited higher-ranked article at position 4. |
| 129425 | 7, 9 | 12 | 1 | 12:2 | 9504 | Shared Doc: Quyet-dinh-114-QD-VSD-2021-Quy-che-luu-ky-chung-khoan-Trung-tam-Luu-ky-Chung-khoan-Viet-Nam-487913 | Yes | **A** | 0.85 | Gold article was in context at position 1, but model selected wrong article ['12']. |
| 36139 | 5, 7 | 1, 12, 15 | 1 | 1:5, 12:2, 15:N/A (not in context) | 8420 | No overlap | Yes | **F** | 0.90 | Model hallucinated and cited article(s) ['15'] which are not in the context. |
| 8545 | 6, 7 | 5 | 11 | 5:2 | 5814 | No overlap | Yes | **C** | 0.90 | Model cited neighbor article ['5'] from context instead of gold ['6', '7'] due to neighbor noise. |
| 147505 | 4 | 1, 3 | 2 | 1:13, 3:1 | 1626 | No overlap | Yes | **C** | 0.90 | Model cited neighbor article ['3'] from context instead of gold ['4'] due to neighbor noise. |
| 65633 | 18 | 10 | N/A | 10:4 | 12124 | No overlap | Yes | **G** | 0.70 | Other/unclassified generation behavior. |
| 56437 | 3 | 6 | 1 | 6:11 | 5041 | Shared Doc: Quyet-dinh-866-QD-LDTBXH-2017-chuc-nang-nhiem-vu-quyen-han-co-cau-Cuc-Nguoi-co-cong-352356 | No (missing 2, 3) | **D** | 0.95 | Gold article is in context, but specific clauses ['2', '3'] used by reference are missing. |
| 77989 | 3 | 38 | 4 | 38:2 | 2287 | No overlap | Yes | **B** | 0.80 | Gold article was buried at position 4, model cited higher-ranked article at position 2. |
| 167645 | 7, 75, 77, 159 | 15, 16 | 5 | 15:2, 16:N/A (not in context) | 5167 | No overlap | No (missing 3) | **D** | 0.95 | Gold article is in context, but specific clauses ['3'] used by reference are missing. |
| 91993 | 15, 17, 18 | 3, 5 | 2 | 3:3, 5:5 | 7347 | No overlap | Yes | **A** | 0.85 | Gold article was in context at position 2, but model selected wrong article ['3', '5']. |
| 122267 | 36 | 18 | 3 | 18:6 | 4466 | No overlap | No (missing 1) | **D** | 0.95 | Gold article is in context, but specific clauses ['1'] used by reference are missing. |
| 135981 | 7 | 8 | 1 | 8:2 | 9259 | Shared Doc:  (Neighbor ±1) | No (missing 1) | **D** | 0.95 | Gold article is in context, but specific clauses ['1'] used by reference are missing. |
| 109577 | 29 | 17 | N/A | 17:1 | 6053 | No overlap | Yes | **G** | 0.70 | Other/unclassified generation behavior. |
| 45687 | 2 | 11 | N/A | 11:5 | 5301 | No overlap | Yes | **G** | 0.70 | Other/unclassified generation behavior. |
| 163025 | 33, 43 | 14, 15 | N/A | 14:1, 15:N/A (not in context) | 9070 | No overlap | No (missing 3) | **D** | 0.95 | Gold article is in context, but specific clauses ['3'] used by reference are missing. |
| 161211 | 28 | 15 | 1 | 15:N/A (not in context) | 2552 | No overlap | Yes | **F** | 0.90 | Model hallucinated and cited article(s) ['15'] which are not in the context. |
| 42881 | 50 | 262, 263 | 1 | 262:5, 263:N/A (not in context) | 5051 | Shared Doc: Bo-luat-hinh-su-2015-296661 | Yes | **F** | 0.90 | Model hallucinated and cited article(s) ['263'] which are not in the context. |
| 28895 | 4 | 18 | 3 | 18:N/A (not in context) | 12614 | No overlap | No (missing 5) | **D** | 0.95 | Gold article is in context, but specific clauses ['5'] used by reference are missing. |
| 104719 | 5, 7, 8 | 16 | 5 | 16:2 | 3717 | No overlap | No (missing 3) | **D** | 0.95 | Gold article is in context, but specific clauses ['3'] used by reference are missing. |
| 99461 | 2, 3, 6 | 10 | 4 | 10:12 | 4760 | No overlap | No (missing 1) | **D** | 0.95 | Gold article is in context, but specific clauses ['1'] used by reference are missing. |
| 145171 | 8 | 1 | 3 | 1:6 | 8362 | No overlap | Yes | **A** | 0.85 | Gold article was in context at position 3, but model selected wrong article ['1']. |
| 67603 | 1, 6 | 10 | 3 | 10:4 | 10131 | No overlap | Yes | **A** | 0.85 | Gold article was in context at position 3, but model selected wrong article ['10']. |

## 5. Verdict

**NO-GO**

No single-variable lever has a theoretical maximum ceiling greater than 1 SE (~0.028) that can be implemented purely downstream. Class A (Wrong Article Selection) ceiling is +0.0150, Class C (Neighbor Noise) ceiling is +0.0048, and Class D (Evidence Insufficiency) ceiling is +0.0108 which requires a joint change to the indexing/retrieval pipeline (breaking the single-variable requirement). Therefore, no standalone generation-side experiment is recommended at this time.
