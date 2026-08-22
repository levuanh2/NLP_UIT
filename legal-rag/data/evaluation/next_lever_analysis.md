# Next lever analysis — dev200, frozen v5 config

Generated 2026-08-21. Sources, all pre-existing, nothing re-run:

- `data/evaluation/retrieval_enrichment_ab/B-enriched-index-enriched-reranker-k20/per_question.jsonl`
- `data/evaluation/step8_retrieval_failure_analysis.json`, `step8_per_question.jsonl`
- `data/evaluation/step9_failure_breakdown.jsonl` (candidate ids resolved against
  `storage/indexes/v1/metadata/legal.sqlite` opened `mode=ro`)
- `data/outputs/dev200-enriched-k20-ckpt350/submission.json`

Config under analysis: checkpoint-350 (r=8), `v1-enriched`, k=20, batch 1,
floor 500. dev200 METEOR 0.4880 / ROUGE-L 0.3345.

Rank recovery was validated before use: recomputing recall@20 from the resolved
child ids reproduced the recall already stored in the trace on **200/200
questions, 0 mismatches**.

## 1. Failure census

| class | n | % | mean METEOR | mean ROUGE-L | article cited correctly |
|---|---|---|---|---|---|
| ok | 83 | 41.5 | 0.5506 | 0.4401 | 66% |
| verbosity_failure | 56 | 28.0 | 0.5702 | 0.3131 | 69% |
| generation_failure | 25 | 12.5 | 0.2419 | 0.1626 | 29% |
| both_miss | 25 | 12.5 | 0.4495 | 0.2855 | 0% |
| reranker_failure | 7 | 3.5 | 0.2262 | 0.1263 | 0% |
| fusion_failure | 4 | 2.0 | 0.2786 | 0.1868 | 0% |
| context_failure | 0 | 0 | — | — | — |
| metadata_entity_failure | 0 | 0 | — | — | — |

### both_miss is mostly not a retrieval failure

17 of the 25 carry **no parseable legal citation in the expert answer**, so gold
is the empty set and recall is 0 by definition rather than by failure. Their
mean METEOR is 0.4495, close to the run average — consistent with retrieval
having worked. True retrieval failures are 8 + 7 + 4 = **19/200 (9.5%)**, not
the 18% a naive read of the census gives.

### Length is not the discriminator

| group | median pred/ref word ratio | mean ROUGE-L |
|---|---|---|
| generation_failure | 1.65 | 0.1626 |
| ok + verbosity_failure | 1.72 | 0.3889 |

The failing group is *slightly shorter* relative to its references than the
scoring group. Separately, `verbosity_failure` (predicted more than 2x reference
length) has the **highest** mean METEOR of any class, 0.5702 against 0.5506 for
`ok`. METEOR here is alpha=0.9 recall-weighted, so padding is close to free. Any
hypothesis of the form "shorter answers score better" is contradicted by this
data.

## 2. The dominant failure mode: right law, wrong article

| class | right document, wrong article |
|---|---|
| generation_failure | 20% (5/25) |
| reranker_failure | 29% (2/7) |
| ok | 13% (11/83) |
| verbosity_failure | 5% (3/56) |

`generation_failure` has near-perfect retrieval — the reranker put gold at
median rank 5, on all 25/25 questions, and `context_failure = 0` confirms
nothing was lost to the token budget. The evidence is in the context and the
model still quotes the wrong article, frequently an adjacent one:

| qid | gold article | cited article | gold rank in context |
|---|---|---|---|
| 16727 | 103 | 104 | 6 |
| 134869 | 10 | 11 | 7 |
| 76757 | 122, 65, 68 | 121 | 2 |
| 40285 | 7 | 9 | 4 |
| 116951 | 11 | 6 | 8 |

`CONTEXT_NEIGHBOR_WINDOW=1` injects the neighbouring chunk on each side of every
retrieved child, so adjacent articles are present in the context by
construction. That is a mechanism consistent with the plus-or-minus-one
confusions, and it is testable by changing one variable.

## 3. Lever ranking

Ceilings assume the class is repaired **perfectly** to the mean of the scoring
classes (0.5585), which no real change will achieve. They are upper bounds for
prioritisation, not forecasts. The dev200 standard error is 0.028.

| # | lever | class | n | ceiling on dev200 METEOR | detectable at n=200 | cause confidence | cost | regression risk |
|---|---|---|---|---|---|---|---|---|
| 1 | context neighbour window | generation_failure | 25 | **+0.0396** | yes | medium | 1 h | medium |
| 2 | reranker demotes fusion's gold | reranker_failure | 7 | +0.0116 | **no** | high | 15 min retrieval-only | low |
| 3 | candidate depth before RRF | fusion_failure | 4 | +0.0056 | **no** | high | 15 min retrieval-only | low |
| — | metadata filter | — | 0 | 0 | — | — | — | closed: 0 firings, 0 false positives |
| — | context budget | — | 0 | 0 | — | — | — | closed: 0 truncations, 0 gold lost at k=20 |
| — | index enrichment | — | — | +0.0038 measured | no | high | — | closed: inside SE |

The important consequence: **levers 2 and 3 cannot be validated on dev200 by
METEOR.** Repairing all 11 questions perfectly moves the mean by 0.0172, well
inside one standard error. They must be measured as retrieval recall, where the
signal is per-question and exact, and only promoted to a generation run if the
recall moves.

## 4. Proposed experiments (not yet run)

### E1 — Does the neighbour window cause the adjacent-article confusion?

- **Hypothesis**: `CONTEXT_NEIGHBOR_WINDOW=1` places adjacent articles beside the
  gold article in the context, and the model quotes the wrong one. Setting it to
  0 removes the distractor.
- **Single variable**: `CONTEXT_NEIGHBOR_WINDOW=0`. Checkpoint, index, k, batch,
  floor and prompt all unchanged.
- **Command shape**: process-level env only —
  `INDEX_ROOT_DIR=.../index-staging RERANKER_TOP_K=20 LLM_BATCH_SIZE=1
  LLM_ADAPTER_PATH=.../checkpoint-350 CONTEXT_NEIGHBOR_WINDOW=0
  python scripts/run_ingestion.py submit --questions data/questions/dev200.json`
- **Ceiling**: +0.0396. **Realistic**: unknown; the same window may be what lets
  other questions find their evidence, so this can regress.
- **Runtime**: about 1 h at 200 questions, batch 1, plus a 5-question preflight.
- **Metric**: METEOR primary; also the `article_hit` rate on the 25-question
  `generation_failure` set, which is the mechanism-level signal and far less
  noisy than the mean.
- **Rollback**: none needed — process-level env only, nothing written outside
  `data/outputs/`.
- **Risk**: medium. A narrower context could cost recall on questions that
  currently rely on the neighbour.

### E2 — Is the reranker demoting gold that fusion already found?

- **Hypothesis**: on all 7 `reranker_failure` questions BM25 returns gold (7/7,
  median rank 27) and RRF keeps it (7/7, median rank 23), then the reranker drops
  it out of the top 20 entirely (0/7). A score blend that keeps the fusion
  signal, or reserving slots for the top RRF candidates, would retain it.
- **Single variable**: reranker score combination. No model change.
- **Metric**: `recall_article@20` after reranking on dev200, measured with
  `scripts/retrieval_benchmark.py`. **Not METEOR** — see section 3.
- **Runtime**: about 15 min, retrieval only, no generation.
- **Rollback**: code change behind an env flag; unset to restore.
- **Risk**: low to measure, medium to adopt — the reranker is worth +0.0134 on
  the top-k axis and must not be degraded for the other 193 questions.

### E3 — Are the fusion candidates deep enough?

- **Hypothesis**: on all 4 `fusion_failure` questions gold sits at rank 32-35 in
  dense and 29 in BM25, deeper than the pool RRF fuses, so it is cut before
  ranking. Raising `DENSE_TOP_K` / `BM25_TOP_K` / `RRF_TOP_K` from 40/40/50
  recovers it.
- **Single variable**: one pool size at a time, process-level env.
- **Metric**: `recall_article@20` on dev200, retrieval only. Also record BM25
  latency — it is already the dominant cost at 1323 ms and deeper pools raise it.
- **Runtime**: about 15 min per setting.
- **Rollback**: process-level env only.
- **Risk**: low correctness risk, real latency risk.

## 5. Explicitly not worth trying yet

- **More training.** checkpoint-400 (r=8) scored 0.4747 against checkpoint-350's
  0.4880, and r16-350 scored 0.4540. Training further along the same recipe has
  measured negative return.
- **Higher LoRA rank.** r16-400 reached 0.4851, inside one SE of the winner, at
  double the adapter size. No measured gain.
- **Lower generation floor**, unless the sweep in flight says otherwise: the
  class analysis shows padding is not penalised by this metric.
- **Metadata / entity retrieval.** 0 firings and 0 false positives on dev200,
  and only 6/1000 test questions carry a structured identifier.
- **Context budget work.** 0 truncations and 0 gold lost at k=20.
- **Index enrichment beyond the current staging copy.** Measured at +0.0038,
  inside SE, with a BM25 latency cost of 1169 to 1323 ms.
