# E1 — CONTEXT_NEIGHBOR_WINDOW=0

Run 2026-08-21 14:55–15:58 on the GPU box, tmux `e1`, script `~/e1.sh`
(sha `4f7e51538efb7897`). Artifacts: `data/outputs/dev200-e1-nbwindow0/`,
`data/evaluation/e1_per_question.jsonl`.

## Hypothesis under test

`CONTEXT_NEIGHBOR_WINDOW=1` injects the neighbouring chunk on each side of every
retrieved child, so adjacent articles sit next to the gold article in the
context. The 25 `generation_failure` questions have near-perfect retrieval
(gold at median rank 5, 25/25) yet cite the wrong article, often an adjacent one
(103 → 104, 10 → 11, 122 → 121). Setting the window to 0 should remove the
distractor and raise the article-hit rate.

## Setup

Only `CONTEXT_NEIGHBOR_WINDOW` changed, 1 → 0. Everything else was pinned
explicitly as process-level environment, including `LLM_MIN_NEW_TOKENS=500` so
no value could leak in from the preceding floor sweep:

```
INDEX_ROOT_DIR=.../storage/index-staging   RERANKER_TOP_K=20
LLM_ADAPTER_PATH=.../qlora-lr5e4/checkpoint-350   LLM_BATCH_SIZE=1
LLM_MIN_NEW_TOKENS=500   CONTEXT_NEIGHBOR_WINDOW=0
```

Before running, the env binding itself was verified: `Settings()` reports
`context_neighbor_window = 1` by default and `0` with the variable set. Without
that check a non-binding variable would have produced a baseline rerun that
looked like a valid experiment.

## Validity

| field | value |
|---|---|
| preflight | 5 answered, 0 failed |
| answered | 200/200 |
| failed | 0 |
| abstain (empty or under 20 words) | 0 |
| allocator warnings | 0 |
| runtime | 3637 s |

VALID.

## Result

| run | METEOR | ROUGE-L | median words |
|---|---|---|---|
| baseline (window=1) | **0.4880** | **0.3345** | 457 |
| E1 (window=0) | 0.4472 | 0.3039 | 451 |
| delta | **−0.0408** | −0.0306 | −6 |

Rescued (delta > +0.02): 62. Regressed (delta < −0.02): 109. Unchanged: 29.

The METEOR loss of 0.0408 is larger than the n=200 standard error of 0.028, so
this is a real regression, not noise.

## Per class

| class | n | METEOR base → E1 | delta | rescued | regressed | words base → E1 | ref words |
|---|---|---|---|---|---|---|---|
| generation_failure | 25 | 0.2419 → 0.3128 | **+0.0709** | 17 | 8 | 690 → 624 | 298 |
| fusion_failure | 4 | 0.2786 → 0.3009 | +0.0223 | 3 | 1 | 643 → 740 | 438 |
| reranker_failure | 7 | 0.2262 → 0.2077 | −0.0185 | 3 | 2 | 840 → 495 | 545 |
| both_miss | 25 | 0.4495 → 0.4108 | −0.0387 | 10 | 13 | 625 → 499 | 280 |
| ok | 83 | 0.5506 → 0.4874 | −0.0631 | 18 | 50 | 409 → 433 | 334 |
| verbosity_failure | 56 | 0.5702 → 0.5041 | −0.0660 | 11 | 35 | 558 → 432 | 188 |

Net METEOR points: `generation_failure` **+1.774**, everything else **−9.947**,
for a total of −8.17 over 200 questions.

## The hypothesised mechanism is not what happened

| measure | baseline | E1 |
|---|---|---|
| article hit, all 200 | 55% (96/175) | 57% (99/175) |
| article hit, generation_failure | 29% (7/24) | 29% (7/24) |
| right document / wrong article, all | 22 | 21 |
| right document / wrong article, generation_failure | 5 | 3 |

Citation selection barely moved, and did not move at all on the target group.
Removing the neighbours did **not** make the model pick the right article. The
hypothesis as stated is **not supported**.

What changed instead is answer length. Dropping the neighbouring chunks removes
supporting text, answers get shorter, and recall-weighted METEOR follows:
`reranker_failure` 840 → 495 words, `both_miss` 625 → 499, `verbosity_failure`
558 → 432. The `generation_failure` group gained because its baseline answers
were bloated at 690 words against 298-word references and largely wrong — there,
trimming helped. The healthy classes were *using* that neighbour text, so losing
it cost them.

Even the rescue is partial: the 17 rescued questions move from a mean METEOR of
0.2279 to 0.3734, still well below the 0.5585 mean of the scoring classes.

## Conclusion

The neighbour window is load-bearing for the 139 healthy questions and harmful
for the 25 broken ones. A global flip is the wrong instrument for a localised
problem: it trades 1.77 points of gain for 9.95 points of loss.

**DECISION: KEEP BASELINE.** `frozen_best.json` is unchanged
(sha `8ed73d34e075b725`). No 1000-question run. Proceed to E2/E3.

The result is still informative rather than wasted: it rules out the
adjacent-article-distractor explanation, and it shows the `generation_failure`
group responds strongly to context composition — which makes a *conditional*
narrowing (applied only when the reranker is confident, rather than globally) a
better-motivated follow-up than it was before this run.

## Production safety

```
.env sha256    7ef96c552818e7ef  before  ->  7ef96c552818e7ef  after
CURRENT        v1, 2d27fbdf4e8ca207      ->  v1, 2d27fbdf4e8ca207
frozen_best    8ed73d34e075b725, not overwritten
```

The script asserted both shas before starting and would have aborted on a
mismatch. Nothing committed, nothing pushed, no submission built, nothing
uploaded to any external service.
