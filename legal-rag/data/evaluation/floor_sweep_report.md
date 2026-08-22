# Floor sweep — dev200, frozen v5 config

Run 2026-08-21 11:55–14:34 on the GPU box, tmux `floorsweep`, script
`~/floor_ckpt350.sh` (sha `0b6e84a3e02744ea`, verified to contain zero `sed -i`
and to touch `.env` and `CURRENT` only through `sha256sum` / `cat`).

Everything except `LLM_MIN_NEW_TOKENS` was held fixed:

```
checkpoint  models/qlora-lr5e4/checkpoint-350   (r=8, alpha=16, dropout=0.05)
base        AITeamVN/Vi-Qwen2-3B-RAG
index       storage/index-staging/v1-enriched
k           RERANKER_TOP_K=20
batch       LLM_BATCH_SIZE=1
decode      temperature 0.0, top_p 1.0, do_sample false,
            repetition_penalty 1.1, LLM_MAX_NEW_TOKENS 1100,
            CONTEXT_MAX_TOKENS 10000
```

All overrides were process-level environment prefixes. `.env` was not written.

## Results

| floor | METEOR | ROUGE-L | median words | answered | failed | abstain | allocator warnings | runtime |
|---|---|---|---|---|---|---|---|---|
| 200 | 0.4816 | **0.4010** | 316 | 200/200 | 0 | 0 | 0 | 2869 s |
| 300 | 0.4895 | 0.3873 | 335 | 200/200 | 0 | 0 | 0 | 3030 s |
| **400** | **0.4928** | 0.3623 | 376 | 200/200 | 0 | 0 | 0 | 3329 s |
| 500 (baseline) | 0.4880 | 0.3345 | 457 | 200/200 | 0 | — | — | 3716 s |

Reference median is 281 words. Abstention was checked directly: zero empty
answers and zero answers under 20 words in every run. All three new runs are
VALID.

## Reading

**METEOR traces an inverted U with a peak near 376 words.** 0.4816 → 0.4895 →
0.4928 → 0.4880 across 316 → 335 → 376 → 457 words. The best value, floor 400,
beats the baseline by **+0.0048**.

**ROUGE-L is strictly monotone decreasing in length** across all four points:
0.4010 → 0.3873 → 0.3623 → 0.3345. Floor 200 is +0.0665 over the baseline. A
consistent direction across four independent measurements is stronger evidence
than any single pairwise delta.

**Runtime is monotone in length too** — floor 400 finishes 387 s (10.4%) faster
than floor 500 for the same 200 questions.

## Verdict: the frozen winner does not change

The dev200 standard error at n=200 is about 0.028. Floor 400's METEOR advantage
of +0.0048 is roughly one sixth of that, so by the stated rule it is **not** a
winner and `frozen_best.json` is left untouched at floor 500.

Applying the measured dev200 → leaderboard discount of 0.60 (derived from the
v4 → v5 move) to +0.0048 predicts about +0.0029 on the official set, below the
n=1000 standard error of 0.006. Even if the dev200 difference is real, it is not
expected to be measurable on the leaderboard. That is the reason not to spend a
five-hour 1000-question run on it.

One thing worth recording anyway: **floor 400 is not worse than floor 500 on
anything measured** — higher on both metrics and 10% cheaper to generate. If a
future experiment needs a faster dev200 loop, floor 400 is a free swap. This is
an efficiency observation, not a metric claim.

## Correction to an earlier prediction

Before this sweep ran I predicted that lowering the floor would *reduce* METEOR,
reasoning from the failure census: the `verbosity_failure` class (answers more
than twice the reference length) had the highest mean METEOR of any class,
0.5702 against 0.5506 for `ok`. That prediction was wrong — floors 300 and 400
both scored above floor 500.

The reasoning error was treating a within-run correlation as a causal effect.
Questions whose references are long naturally draw long answers and high recall,
which is what made the verbose class look strong. Forcing extra tokens onto
questions whose references are short is a different intervention, and past about
380 words it costs METEOR rather than buying it. Correlation across questions
did not license a claim about the effect of the knob.

## Production safety

Checked before the sweep, printed by the script at both ends, and re-checked
after:

```
.env sha256    7ef96c552818e7ef  (before)  ->  7ef96c552818e7ef  (after)
CURRENT        v1, sha 2d27fbdf4e8ca207    ->  v1, sha 2d27fbdf4e8ca207
frozen_best    sha 8ed73d34e075b725, not overwritten
git            cd8a79e, nothing committed, nothing pushed
```

No index was rebuilt or published, no checkpoint was modified, no submission was
built, and nothing was uploaded to any external service.
