# QueryLens — Evaluation Results (model_v2_span, span-optimized)

Model: `model_v2_span/model-best`. Gold: `gold_v2.json` (40) + `gold_v2_extra.json` (33)
= 73 hand-labeled held-out queries; **71 evaluated** (2 auto-excluded as present in
training — no leakage). Metric: per-label set-match P/R/F; macro over the 6 priority
span labels. Full per-query predictions in `EVAL_RESULTS_v2.json`.

## Per-label span scores
| label | P | R | F1 | # gold |
|---|---|---|---|---|
| CONCEPT | 0.83 | 0.83 | 0.83 | 63 |
| GIVEN_NAME | 0.92 | 1.00 | 0.96 | 11 |
| STYLE | 0.92 | 0.79 | 0.85 | 14 |
| REQUIRE_TOKEN | 0.93 | 0.72 | 0.81 | 18 |
| QUALIFIER | 0.89 | 0.71 | 0.79 | 24 |
| EXCLUDE_TOKEN | 0.91 | 0.42 | 0.57 | 24 |
| **MACRO span F1** | | | **0.80** | |
| intent (secondary) | | | 0.92 acc | 71 |

## Read
- **Precision is high on every label (0.83–0.93)** — when the model emits a span it is
  almost always correct (few false positives).
- **Recall** is the varying axis: CONCEPT/GIVEN_NAME excellent; STYLE/REQUIRE/QUALIFIER
  good (0.72–0.79); **EXCLUDE_TOKEN is the weak spot (0.42)** — negation is the hardest
  label (see `TRAINING_DATA_BACKLOG.md` R5). Precision there is still 0.91.
- intent shown for completeness only — it is the accepted-secondary axis (span quality
  is the objective); on the intent-only neutral hand-gold it is ~0.68.

## Caveat on gold size
The plan targeted ~150 held-out queries; current gold is 73 (71 unseen). The reliable
labels (CONCEPT n=63, QUALIFIER/EXCLUDE n=24) are adequately sampled; the smaller ones
(GIVEN_NAME n=11, STYLE n=14) are noisier. Expanding the gold to ~150 (more GIVEN_NAME,
STYLE, and real EXCLUDE cases) would tighten those estimates — recommended before the
next model-selection decision.
