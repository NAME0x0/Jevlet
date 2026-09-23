# Research decisions

- Reconstruct public System-One ideas; do not claim or attempt a proprietary code clone.
- Start from random initialization and use programmatically verifiable local data.
- Prefer dynamic text-option scoring over a fixed class vocabulary.
- Keep the vault outside the autoresearch code path.
- Preserve the Pareto frontier instead of reporting one opaque aggregate score.

## 2026-09-23: what is public about Jev, and what follows from it

TypeSafe announced Jev on 2026-09-15. Public material (the TypeSafe launch post, SDK guides,
and third-party write-ups) states:

| Public claim | Consequence for Jevlet |
|---|---|
| Unstructured state in, typed decisions out; Noul, Choice, Score | `system_one.SystemOne.evaluate(state, {id: question})` |
| Questions are evaluated independently, in parallel, against one state | Packed branches with sibling isolation; tests prove packed == separate |
| Choice criteria are runtime `name -> description` pairs, up to 255 | `option_text(name, description)`; hard 255 cap |
| Beyond 255 options: score independently, then choose | Two-stage API; `option_isolated` scales options without growing positions |
| Score has 2-10 ordered levels; result is the probability-weighted mean | `ScoreQuestion` enforces 2-10 levels; expectation over 1..n |
| ~64k tokens shared, state + longest question within ~32k | Branch position ids restart after the state |
| Trained with RLCD: rewards calibrated probabilities, not rater approval | Proper scoring rules (log, Brier, CE+Brier), soft targets |
| Workflow evals compare against averaged frontier-model probabilities | Teacher soft labels are the intended next data source |
| Guidance: act at high confidence, confirm at medium, escalate at low; 0.9 for destructive ops | Gate is `verify` until user-validated; risky tasks never auto-execute |
| Poor at arithmetic, counting, date/measure comparison, generation | `rules` family is expected to stay weak; not a daily-driver target |

Not public: parameter count, backbone, context mechanism, the RLCD algorithm, training data.
Everything below is therefore a reconstruction to test, not a claim about Jev's internals.

## 2026-09-23: Jevlet-P replaces the scratch model for the daily driver

A random-init byte transformer cannot learn language semantics in one laptop night (46%
dev after the smoke run). A pretrained encoder under the identical packed topology reached
83.1% in 8 minutes on the RTX A2000. The scratch track remains for topology research where
pretrained representations would confound the result.

Encoder, not decoder: the topology needs bidirectional attention inside a branch so options
can compare each other before `[DECIDE]`. Custom 4D masks pass straight through transformers
5.x, so any BERT-family backbone works without modifying model code.
