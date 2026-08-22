# E5 — retrieval score diagnostics

Run 2026-08-21 17:44–18:01 on the GPU box, retrieval only. No generation, no
LLM, no training, no submission, nothing uploaded. Artifacts:
`data/evaluation/e5-full/per_question.jsonl`,
`data/evaluation/e5_retrieval_score_diagnostics.jsonl`.

E4 stopped because the five features most likely to gate on — reranker and RRF
top-1 scores and margins — were not recorded anywhere. E5 records them. The
question is no longer "would a score gate work"; it is "what do the scores say".

## 1. What was instrumented

Nothing was estimated. The scores already existed in the pipeline and were
being discarded at write time:

| score | produced by | previously |
|---|---|---|
| `dense_score`, `bm25_score`, `fusion_score` | `app/retrieval/fusion/rrf.py` | set on the candidate, dropped by the benchmark |
| `rerank_score` | `VietnameseReranker.rerank()` | same |

One real gap needed closing. The reranker scores **all 50** fused candidates and
then returns only `RERANKER_TOP_K=20`, so the score of a gold chunk that got
dropped was not in the return value at all. `rerank()` now publishes the full
table on the instance (`last_scores`, `last_input_order`) after sorting and
before truncation; the benchmark reads it. Confirmed on every question:
`rerank_scored_count = 50`, no nulls in any diagnostic field across 200 rows.

**Control (15 questions, flag-off):** `dense`, `bm25`, `rrf` and `rrf_reranker`
child_id lists identical 15/15 to the `e2-off` reference. Instrumentation is
observational; ranking is unchanged. No ABORT condition triggered.

Diff against the pre-patch backups: `vietnamese_reranker.py` **+13/−0**,
`retrieval_benchmark.py` **+64/−1** (the deleted line is the `build_row` call,
extended with the new argument).

## 2. Where gold is actually lost

183 of 200 questions carry a parseable gold citation; the other 17 have none, so
recall is undefined for them and they are excluded from every rate below.

| | count |
|---|---|
| gold kept in the returned top-20 | **164** |
| gold never reached the 50-candidate pool | **12** |
| gold in the pool, dropped by the reranker | **7** |

The 7 are exactly the questions the census labelled `reranker_failure`, and the
12 upstream losses are the 4 `fusion_failure` plus 8 of the `both_miss`. The
labels hold up against the scores.

So the reranker's own contribution to lost gold is **7 of 183 questions, 3.8%**.
Nearly twice as much gold (12) never survives fusion to reach it.

## 3. The 7 reranker_failure questions

Pool rank is position among the 50 scored candidates; the cut is 20.

| qid | BM25 | RRF | pool rank | gold score | top-1 | top-2 | margin | cut@20 | gap to top-1 |
|---|---|---|---|---|---|---|---|---|---|
| 130739 | 27 | 14 | 21 | −7.32 | 1.82 | 0.09 | 1.72 | −7.15 | 9.13 |
| 5333 | 40 | 10 | 22 | −6.89 | 0.19 | −0.77 | 0.96 | −6.75 | 7.07 |
| 5585 | 33 | 38 | 30 | −5.29 | 1.04 | 0.43 | 0.61 | −3.38 | 6.33 |
| 24471 | 9 | 20 | 35 | −11.48 | −2.58 | −2.83 | 0.25 | −6.10 | 8.90 |
| 25197 | 33 | 23 | 38 | −10.09 | −0.99 | −2.70 | 1.71 | −8.66 | 9.10 |
| 136849 | 20 | 40 | 44 | −11.48 | −0.90 | −1.57 | 0.67 | −5.16 | 10.58 |
| 59801 | 15 | 28 | 47 | −11.48 | 1.44 | 1.13 | 0.31 | −6.37 | 12.93 |

Classified: **B near-miss at the cut** (rank 21–25) 130739, 5333 — 2 questions;
**C mid-pool demotion** (26–40) 5585, 24471, 25197 — 3; **D hard rejection**
(41–50) 136849, 59801 — 2. No question falls in class A or E here.

**Why gold dropped, read off the data:** the cross-encoder does not narrowly
miss these. It scores them near the bottom of its own range. The gap to top-1 is
**median 9.10** for dropped gold against **0.128** for kept gold — roughly
seventy times larger. The two distributions barely touch: kept gold runs
−10.01 … +11.23 with median +0.881, dropped gold runs −11.48 … −5.29 with
median −10.09.

The nearest blockers show the same thing. For 130739 the three candidates above
gold score −6.77, −6.85, −7.15 against gold's −7.32 — all from
`doc:230594`, articles 82, 86 and 83, siblings of the gold article 87. For
136849 and 24471 gold sits at −11.48 among blockers at −11.44 to −11.46:
differences of 0.01–0.04 in a compressed low tail where the ordering carries no
real signal. But that is cosmetic — those golds are 24 and 15 places below the
cut, so tie-breaking is not what lost them.

## 4. The 4 fusion_failure and 25 both_miss

All 4 `fusion_failure` are class **A**: gold never entered the pool, so there is
no reranker score for it — `not available`, not reconstructed. All four share
one mechanism, and it is a genuine fusion loss rather than a recall loss:

| qid | dense rank | BM25 rank | RRF rank |
|---|---|---|---|
| 98627 | — | **29** | lost |
| 120493 | **32** | — | lost |
| 65633 | **34** | — | lost |
| 83721 | **35** | — | lost |

In every case **exactly one retriever finds the gold, deep in its list, and the
other misses it entirely**. A candidate seen by one list at rank ~30 scores
1/(60+30) ≈ 0.011 under RRF, while anything appearing in both lists near the top
approaches 2/61 ≈ 0.033. Fifty candidates clear that bar, so the gold never
reaches the pool. The retrieval system as a whole *did* find these documents —
fusion is where they were discarded.

Of the 25 `both_miss`: **17 are class E**, no parseable gold in the expert
answer at all — recall is 0 by definition and no retrieval change can affect
them. The remaining **8 are class A**, gold missing from the pool. **Zero** are
reranker-caused. This confirms the earlier reading that true retrieval failure
is 19/200, not 18%.

## 5. The six questions

**Q1 — Is the reranker really the bottleneck?** No. It loses gold on 7 of 183
questions; fusion and recall lose 12 before it ever runs. It is the smaller of
the two retrieval leaks.

**Q2 — Is there a clear threshold or margin pattern?** For *gold score*, yes and
it is stark — dropped gold sits ~9 points below top-1, kept gold ~0.1. For the
*gateable* features, no. The top1–top2 margin, the single feature E4 most wanted,
does **not** separate: median 1.038 when gold is kept versus 0.674 when it is
dropped, with fully overlapping ranges (0.00–12.77 vs 0.25–1.72). A gate cannot
use the gold score, because knowing which candidate is gold is the entire
problem. Margin is available at inference time and carries no signal.

**Q3 — What rank/score range loses gold?** Pool ranks 21, 22, 30, 35, 38, 44, 47
— spread across the whole tail, not bunched at the cut. Gold scores −11.48 to
−5.29, entirely below the median rank-20 cut score of −5.32.

**Q4 — Could a threshold or blend rescue them?** **No, and this is provable
rather than empirical.** Dropped gold scoring at or above its own question's
rank-20 cut score: **0 of 7**. Every dropped gold scores below 20 other
candidates on the very score any threshold would read. A monotone rule on that
score cannot admit gold without admitting all 20 above it. This is the same wall
E2 hit from the other side, and it explains why E2 rescued only 2/7.

Widening the cut does work mechanically — `RERANKER_TOP_K` 25 recovers 2/7, 30
recovers 3/7, 40 recovers 5/7, 50 recovers 7/7 — but 50 *is* the pool, so
"recover all 7" is identical to not reranking at all, and each extra chunk is
paid for out of the context budget that E1 showed is load-bearing.

**Q5 — Upper bound on dev200.** Assumption stated openly: a question whose gold
is retrieved scores the healthy `ok`-class mean of **0.5506**. That is generous
— these are the hardest questions in the set — so each figure is a ceiling, not
a forecast.

| repair | n | METEOR | vs SE 0.028 |
|---|---|---|---|
| reranker-dropped gold only | 7 | **+0.0114** | 0.41× |
| reranker + fusion failures | 11 | +0.0168 | 0.60× |
| every lost gold, perfect retrieval | 19 | **+0.0260** | 0.93× |

**Q6 — Is it above SE?** No. Even perfect retrieval on every one of the 19
questions with recoverable gold lands at 0.93× the standard error — undetectable
on dev200. The reranker-specific slice is 0.41×.

## 6. Note on the benchmark's own recommendation

The run printed `RECOMMENDATION: REMOVE RERANKER` (Recall@5 −0.0123, gold
dropped on 7, improved 59 / worsened 40). That is a retrieval-metric verdict
computed at k=5, and it is worth recording, but it does not escape the bound
above: removing the reranker is itself a retrieval change and so is capped by
the same +0.0260 ceiling, while also discarding the 59 questions it promotes and
raising MRR from 0.5024 to 0.5095. Not acted on here — E5 is observational.

## 7. Verdict

The instrumentation did its job: the scores E4 needed now exist, and they answer
the question rather than leaving it open. The answer is that the reranker is not
where this competition is lost. It drops gold on 7 of 183 questions, it does so
by margins of nine score points rather than rounding errors, no gate on any
inference-time feature separates those 7 from the healthy 164, and repairing
**all** retrieval failures — the reranker's 7 plus the 12 that never reach it —
is worth at most 0.93 of one standard error.

Retrieval is closed as a lever at dev200 resolution. The remaining gap is
content accuracy in generation, consistent with the census finding that
`article_hit` runs 66–69% in scoring classes against 29% in
`generation_failure`.

**E5 NO-GO** — no follow-up retrieval experiment proposed, because the estimated
gain does not clear SE.

## 8. Production safety

```
.env sha256   7ef96c552818e7ef   unchanged
CURRENT       v1, 2d27fbdf4e8ca207   unchanged
frozen_best   8ed73d34e075b725   not overwritten
```

No symlink touched, no index rebuilt or deleted, no `.env` write, no training,
no generation, no 1000-question run, no submission built, nothing committed,
pushed, or uploaded to any external service. The E2 `RERANKER_BLEND_RRF` block
remains in the working tree, uncommitted and default-off, as instructed — not
removed blind.
