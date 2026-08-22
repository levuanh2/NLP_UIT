# E4 — adaptive context analysis

Analysis only. No inference, no GPU work, no 1000-question run, no production
change. Built entirely from artifacts already on disk. Companion data:
`data/evaluation/e4_feature_table.json`.

## 1. Baseline reference

```
checkpoint  models/qlora-lr5e4/checkpoint-350   (r=8, alpha=16, dropout=0.05)
index       storage/index-staging/v1-enriched
k           20
floor       LLM_MIN_NEW_TOKENS=500
dev200      METEOR 0.4880   ROUGE-L 0.3345   median 457 words
SE(n=200)   ~0.028
```

Safety verified before starting: `.env` `7ef96c552818e7ef`, `CURRENT` `v1`
`2d27fbdf4e8ca207`, `frozen_best.json` `8ed73d34e075b725`. All match.

## 2. E1 evidence this builds on

Global `CONTEXT_NEIGHBOR_WINDOW=0` scored 0.4472 against the baseline's 0.4880,
a regression of 0.0408. Per class it gained on `generation_failure` (+0.0709,
17/25 rescued) and lost on the healthy classes (`ok` −0.0631, `verbosity`
−0.0660). That asymmetry is what motivated looking for a per-question gate.

The gate would have to predict, before the LLM runs, whether narrowing helps
**this** question.

## 3. Candidate features

The spec's five highest-priority features do not exist in this pipeline:

| # | requested feature | status |
|---|---|---|
| 1 | reranker top-1 score | **not available** |
| 2 | reranker top-1 vs top-2 margin | **not available** |
| 3 | top-k reranker score distribution | **not available** |
| 4 | RRF top-1 score | **not available** |
| 5 | RRF top-1 vs top-2 margin | **not available** |

The retrieval trace records `child_ids`, `returned`, and latency only. A regex
sweep for any score-like key across a trace row returns **NONE**. The
`RetrievalCandidate` model does carry `rerank_score` and `fusion_score` fields,
but the benchmark does not persist them, so they cannot be recovered from
existing artifacts. Reconstructing them would require re-running retrieval,
which is out of scope here. Not reconstructed by assumption.

Fields that *are* in the trace but are **gold-derived** were excluded from the
feature set by construction, since a production gate cannot see them:
`first_hit_rank`, `any_hit`, `document_coverage`, `full_document_coverage`,
`recall_*`, `fused_first_hit_rank`, `fused_any_hit`, `rank_change`,
`gold_after_rerank`, `gold_in_context`.

Eleven features are computable for all 200 questions before generation:

`n_articles_top20`, `n_documents_top20`, `max_chunks_per_article`,
`top1_article_share`, `dense_bm25_overlap`, `rrf_kept_in_top20`,
`mean_rank_displacement`, `metadata_confidence`, `metadata_candidate_count`,
`bm25_ms`, `reranker_ms`.

Five more exist only for part of the set: `children`,
`parents_after_expansion`, `evidences_in_context`, `token_count`,
`dropped_by_budget` come from the context-budget probe, which covers **60 of
200** questions. Marked not available rather than imputed.

## 4. Distribution by group

A = generation_failure (25), B = healthy, ok + verbosity (139),
C = both_miss (25), D = reranker + fusion failure (11). Medians, plus a
standardised A-versus-B separation.

| feature | A | B | C | D | A−B separation |
|---|---|---|---|---|---|
| n_articles_top20 | 15.000 | 14.000 | 12.000 | 14.000 | +0.20 |
| n_documents_top20 | 9.000 | 8.000 | 9.000 | 10.000 | +0.14 |
| max_chunks_per_article | 3.000 | 3.000 | 4.000 | 2.000 | +0.12 |
| top1_article_share | 0.100 | 0.100 | 0.150 | 0.050 | +0.17 |
| dense_bm25_overlap | 0.100 | 0.150 | 0.150 | 0.100 | −0.37 |
| rrf_kept_in_top20 | 0.500 | 0.550 | 0.550 | 0.500 | −0.41 |
| mean_rank_displacement | 13.900 | 12.600 | 12.550 | 12.700 | +0.41 |
| metadata_confidence | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 |
| metadata_candidate_count | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 |
| bm25_ms | 1262.031 | 1274.091 | 1218.139 | 1289.999 | −0.31 |
| reranker_ms | 376.141 | 370.605 | 365.860 | 360.621 | +0.30 |

**Top 5 by separation**: `mean_rank_displacement` +0.41,
`rrf_kept_in_top20` −0.41, `dense_bm25_overlap` −0.37, `bm25_ms` −0.31,
`reranker_ms` +0.30.

All are small effects with heavy overlap — the largest is 0.41 standardised
units, and the medians differ by roughly one article or 5 ms. The two metadata
features are identically zero on every question and carry no information at all.

## 5. Oracle upper bound

| oracle | METEOR delta | vs SE 0.028 |
|---|---|---|
| class oracle — route exactly the 25 `generation_failure` to narrow context | **+0.0089** | **0.32x** |
| absolute oracle — per-question max(baseline, E1), the ceiling for *any* gate | **+0.0311** | 1.11x |

The class oracle is the thing E4 was designed to approximate, and even with
**perfect** knowledge of the class label it delivers less than a third of one
standard error. Within those 25 questions the routing wins +2.473 METEOR points
and loses −0.700, netting +1.774 points, which spread over 200 questions is
+0.0089.

The absolute oracle is the hard ceiling: it requires knowing the outcome of both
runs for every question in advance, which no gate can do. Even that reaches only
1.11x the standard error.

Narrowing helps **73 of 200** questions, and they are not concentrated in the
target class:

| class | helped by narrowing |
|---|---|
| ok | 25 / 83 |
| generation_failure | 17 / 25 |
| verbosity_failure | 15 / 56 |
| both_miss | 10 / 25 |
| reranker_failure | 3 / 7 |
| fusion_failure | 3 / 4 |

Mean gain among the helped is +0.0851; mean loss among the rest is −0.1133. The
losses are larger per question than the gains, and they outnumber them 127 to
73. `generation_failure` is not the right target set — it is 17 of the 73
questions that would need to be caught.

## 6. Threshold search

Quantiles 10/25/50/75/90, both directions, on each available feature, keeping
only gates that select between 10 and 120 questions. 48 combinations survived
that filter.

| feature | dir | threshold | gated | genfail caught | healthy caught | precision | est. METEOR delta |
|---|---|---|---|---|---|---|---|
| reranker_ms | > | 451.852 | 19 | 3 | 13 | 0.42 | **−0.0008** |
| dense_bm25_overlap | > | 0.400 | 17 | 1 | 11 | 0.24 | −0.0016 |
| bm25_ms | < | 909.909 | 20 | 5 | 11 | 0.35 | −0.0017 |
| n_articles_top20 | > | 18.000 | 14 | 3 | 7 | 0.50 | −0.0019 |
| n_articles_top20 | < | 9.000 | 16 | 2 | 9 | 0.31 | −0.0019 |
| n_articles_top20 | > | 16.000 | 46 | 9 | 29 | 0.41 | −0.0029 |
| rrf_kept_in_top20 | < | 0.450 | 25 | 4 | 15 | 0.44 | −0.0033 |
| n_documents_top20 | > | 13.000 | 18 | 3 | 10 | 0.44 | −0.0033 |
| mean_rank_displacement | > | 16.550 | 18 | 2 | 13 | 0.39 | −0.0039 |
| dense_bm25_overlap | < | 0.050 | 33 | 8 | 18 | 0.42 | −0.0039 |

**Every one of the 48 gates has a negative estimated delta.** The best is
−0.0008. Precision — the share of gated questions that narrowing actually helps
— tops out at 0.50 on a 14-question gate, against a base rate of 73/200 = 0.365.
That is barely above chance.

These figures are optimistic by construction: each gate was selected and scored
on the same 200 questions, with no held-out split, so multiple-testing
inflation applies. A gate evaluated on fresh data would do worse than shown.

## 7. Best gate

There is no best gate. The strongest candidate, `reranker_ms > 451.852`, is
estimated at −0.0008 METEOR, catches 3 of 25 `generation_failure` questions and
13 healthy ones, and is anyway a latency measurement with no causal relationship
to context quality — it would be a spurious proxy even if the number were
positive.

## 8-11. Coverage, rescues, regressions, estimated delta

| quantity | value |
|---|---|
| coverage of the best gate | 19 / 200 |
| generation_failure caught | 3 / 25 |
| healthy questions caught (false positives) | 13 |
| precision (gated questions narrowing actually helps) | 0.42 vs 0.365 base rate |
| estimated METEOR delta | **−0.0008** |
| number of tested gates with positive delta | **0 of 48** |

## 12. Confidence and caveats

- The five features most likely to work — reranker and RRF scores and margins —
  **do not exist** in the stored artifacts. This analysis therefore cannot rule
  out that a score-margin gate would work. It can only report that no such
  feature is available without re-running retrieval with score persistence
  added.
- Per-question context length exists for 60 of 200 questions only, so the
  context-size family of features is under-tested.
- All gate estimates are in-sample and optimistic.
- The E1 outcome per question is a single measurement with greedy decoding; it
  is deterministic for a fixed context but the deltas on individual questions
  are not error-free estimates of a stable quantity.
- The oracle bounds do not depend on any of these caveats. They are computed
  directly from the two completed runs and are exact for this pair of
  configurations.

## 13. Verdict

The class oracle — perfect knowledge of the exact label E4 set out to predict —
returns +0.0089, under a third of the standard error. The absolute oracle, which
no implementable gate can reach, returns +0.0311, barely above it. Between those
two bounds sits every possible gate, and the 48 actually tested all landed
negative.

The headroom is not there. A gate cannot be built on features that were never
recorded, and the features that were recorded do not separate the groups.

**E4 NO-GO**
