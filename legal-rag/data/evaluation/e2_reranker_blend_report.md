# E2 — RRF / reranker rank blend, retrieval only

Run 2026-08-21 16:14–16:44 on the GPU box. No generation, no LLM inference, no
1000-question run. Artifacts: `data/evaluation/e2-off/`, `data/evaluation/e2-on/`,
`data/evaluation/e2_reranker_blend_per_question.jsonl`.

## What changed

`VietnameseReranker.rerank()` sorted purely by cross-encoder logit and discarded
the fusion ordering it had been handed — which is exactly where the 7
`reranker_failure` questions lose their gold. E2 fuses the two *rankings*
instead of the two *scores*, because a cross-encoder logit and an RRF score
share no scale:

```python
blend_k = int(os.environ.get("RERANKER_BLEND_RRF", "0") or 0)
if blend_k > 0:
    # 1/(k + rank_rerank) + 1/(k + rank_fusion), k = 60 to match rrf_k
```

Unset or `0` leaves the reranker authoritative. Dense retrieval, BM25, RRF
candidate generation and the reranker model are all untouched; `RERANKER_TOP_K`
stays 20 and the index stays `v1-enriched`.

## Control

The patched code with the flag unset was re-run over all 200 questions and
compared against the pre-existing baseline artifact
(`retrieval_enrichment_ab/B-enriched-index-enriched-reranker-k20`):

| stage | rows differing |
|---|---|
| dense | 0 / 200 |
| bm25 | 0 / 200 |
| rrf | 0 / 200 |
| rrf_reranker | 0 / 200 |

The patch is inert when the flag is unset, and retrieval is bit-for-bit
reproducible across runs. In the preflight the `rrf` stage was also identical
between flag-off and flag-on, confirming the blend acts only after fusion and
does not perturb candidate generation.

## Result

| measure | baseline (blend off) | E2 (blend k=60) | delta |
|---|---|---|---|
| recall_article@20 after rerank | **0.7652** | 0.7577 | **−0.0075** |
| gold lost after RRF | 12 | 12 | 0 |
| gold lost after rerank / blend | 19 | **21** | +2 |
| median gold rank after RRF | 2 | 2 | 0 |
| median gold rank after rerank | 2 | **1** | −1 |

Scored over the 183 questions that carry a parseable gold citation.

Per question: **improved 7, regressed 11, unchanged 165.**

## The 7 known reranker_failure questions

| qid | gold rank in BM25 | gold rank after RRF | rank after baseline rerank | rank after E2 blend | baseline recall@20 | E2 recall@20 | rescued |
|---|---|---|---|---|---|---|---|
| 130739 | 27 | 14 | lost | **17** | 0.00 | **1.00** | yes |
| 5333 | 40 | 10 | lost | **11** | 0.00 | **1.00** | yes |
| 24471 | 9 | 20 | lost | lost | 0.00 | 0.00 | no |
| 136849 | 20 | 40 | lost | lost | 0.00 | 0.00 | no |
| 25197 | 33 | 23 | lost | lost | 0.00 | 0.00 | no |
| 5585 | 33 | 38 | lost | lost | 0.00 | 0.00 | no |
| 59801 | 15 | 28 | lost | lost | 0.00 | 0.00 | no |

**Rescued 2 / 7.**

The pattern explains the ceiling. The blend can only recover a gold that is
already inside the fusion top-20, because that is the window the reranker
selects from. Only 130739 (RRF rank 14) and 5333 (RRF rank 10) sit inside it,
and exactly those two were rescued. The other five sit at RRF rank 20, 23, 28,
38 and 40 — at or past the cut — so no reweighting of the two rankings could
reach them. E2's ceiling on this group was 2/7 before the run started.

## Verdict: REJECT

Recall did not increase; it fell by 0.0075, and the count of golds lost after
the ranking stage rose from 19 to 21. Under the stated rule — recall does not
increase, or regresses materially, so E2 is dropped — the answer is REJECT.

The mechanism worked exactly as designed on the two questions it could reach,
and the median gold rank among surviving golds improved from 2 to 1. But
reinstating the fusion signal also promotes candidates the reranker had
correctly demoted, and that pushes gold out of the top 20 on more questions than
it saves. This is the same shape as E1: a targeted repair that costs more
globally than it recovers.

Nothing here argues the reranker is well behaved on those 7 — BM25 finds their
gold 7/7 and RRF still holds it 7/7 before the reranker drops all 7. It argues
that a symmetric rank blend is the wrong instrument, and that 5 of the 7 are out
of reach of any post-fusion reweighting at k=20.

## Status of the code change

`app/retrieval/reranking/vietnamese_reranker.py` carries the blend block,
uncommitted, gated off by default. The control above is the evidence that it
changes nothing unless `RERANKER_BLEND_RRF` is set. Not committed, not pushed.

## Production safety

```
.env sha256   7ef96c552818e7ef   unchanged
CURRENT       v1, 2d27fbdf4e8ca207   unchanged
frozen_best   8ed73d34e075b725   not overwritten
```

No symlink touched, no index rebuilt, no training, no generation, no submission
built, nothing uploaded to any external service.

## Note on a defect in the first analysis pass

The first run of the analysis populated its child-text cache from the `rrf` and
`rrf_reranker` stages only, so BM25 candidates ranked deeper than the fusion
window resolved to nothing and three of the seven questions showed a blank BM25
column — contradicting the earlier finding that BM25 retrieves gold on 7/7. The
cache now covers all four stages and the column reads 27, 20, 9, 33, 40, 33, 15,
which agrees with the earlier measurement. The headline numbers were never
affected: they derive from the `rrf` and `rrf_reranker` stages, both of which
were fully cached in the first pass.
