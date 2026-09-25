# Lab notebook

Machine-generated experiment records are written to `research/experiment_log*.jsonl` and
`results/runs/`. This file holds concise human interpretation after completed cycles.

## 2026-09-23 — Jevlet-P first results (RTX A2000 4 GB)

**Jevlet-P vs zero-shot router, mixture v1 dev (5,655 questions).** MiniLM-L6 backbone,
2,500 steps, 8 min training, 618 MB peak VRAM.

| | Accuracy | ECE | Latency |
|---|---|---|---|
| Jevlet-P | 83.1% | 0.018 | 2.4 ms/question (GPU) |
| Zero-shot MiniLM router | 52.2% | 0.094 | 12.9 ms/question (CPU) |

Jevlet-P wins every family; weakest are BoolQ 65.6% (near majority class) and MNLI 60.5%.
CPU latency for one state with three questions in one pass: 41.5 ms median.

**Daily-driver benchmark (78 hand-written laptop tasks) exposes the real gap.** Route accuracy
30.8% (Jevlet-P) vs 44.9% (zero-shot); risk detection near chance for both. Cause: no route or
risk decisions in mixture v1. Response: `daily_synthetic.py` plus MASSIVE/CLINC intents in
mixture v2 (75.5k rows). The v2 model has **not been trained yet**.

**Ablation search `research-p1` (successive halving, 9 hypotheses).**

| Stage (steps) | Candidate | Accuracy | ECE | Order agreement | Questions/s |
|---|---|---|---|---|---|
| 1 (400) | baseline | 73.2% | 0.036 | 0.875 | 368 |
| 1 | option_isolated | 71.8% | 0.037 | **1.000** | 392 |
| 1 | full (leaky) | 73.9% | 0.059 | 0.844 | 348 |
| 1 | END_OPTION pooling | 68.4% | 0.026 | 0.750 | 396 |
| 1 | bilinear head | 72.2% | 0.077 | 1.000 | 381 |
| 1 | MLP head | 72.3% | 0.043 | 0.969 | 389 |
| 1 | CE+Brier | 74.0% | 0.031 | 0.938 | 393 |
| 1 | label smoothing | 73.2% | 0.050 | 0.875 | 398 |
| 1 | bge-small backbone | 74.5% | 0.043 | 0.938 | 220 |
| 3 (2500) | bge-small backbone | **84.4%** | **0.024** | 0.961 | 193 |
| 3 | CE+Brier | 83.6% | 0.036 | 0.922 | 326 |

Readings (exploratory, single seed):
- Exact option-order invariance (`option_isolated`) costs ~1.4 points at 400 steps; it lost the
  in-layer accuracy tie-break for stage 2 despite sitting on the Pareto front.
- Sibling leakage (`full`) buys ~0.7 points of accuracy but worsens calibration; isolation,
  which Jev's independent-question design implies, is the better default.
- CE+Brier improves accuracy and calibration together at 400 steps, consistent with training
  on proper scoring rules as an RLCD proxy. The effect shrinks by stage 3.
- bge-small (12 layers) is the most accurate but halves throughput.

**Interrupted.** The 2-seed final verification of bge-small was stopped at ~step 2,200 of
4,000 (seed 1337) for battery. The run's deadline was wall-clock (`started_at + 3 h`), so
resuming `results/runs/research-p1` now would skip the finals; rerun the finals in a new
output directory or on Colab. The daily v2 training was stopped before its first checkpoint.

## 2026-09-24 — daily data, teacher labels, grounding, per-kind calibration

**Route/risk data transfers.** v2 (mixture v2: + compositional daily data, MASSIVE, CLINC;
5,000 steps, 16 min) takes the held-out 78-task benchmark from 30.8% to 91.0% route accuracy
and escalates all vague requests. v3 adds the ~220-task teacher corpus (soft route and risk
probabilities) and 15k screen-grounding rows. The teacher corpus was written before any v2
benchmark errors were inspected, and no data was added in response to them.

| Held-out benchmark | Route acc. | Route ECE | Risky tasks passing the P(safe) ≥ 0.9 gate | Safe tasks needing confirmation |
|---|---|---|---|---|
| zero-shot router | 44.9% | 0.081 | — | — |
| v2, global temperature | 91.0% | 0.071 | — | — |
| v3, global temperature (T=3.05) | 92.3% | 0.157 (underconfident) | — | — |
| v2, per-kind temperatures | 91.0% | 0.061 | 1 / 11 | 0% |
| **v3, per-kind temperatures** | **92.3%** | **0.062** | **0 / 11** | 3% |

**A single temperature is the wrong calibration unit.** Fitted on the mixed v3 dev set, one
temperature (3.05) made the daily driver underconfident. Temperatures per question kind,
fitted on deployment-like data (daily generator dev + teacher dev, never the benchmark),
restore ECE to 0.06. Promoted: v3 per-kind calibrated → `data/daily/current.pt` (v2 kept as
`previous.pt`).

**Remaining benchmark misses are lexical traps and unseen risk types**: "image resize"
(code) → Gemini, "take a screenshot" → Gemini, force-push and drive formatting not flagged
risky at 0.5 (both still blocked by the 0.9 safe gate). n = 78 and 11 risky cases: treat as
direction, not precision.

**Screen grounding on unseen apps** (Teams, Spotify; never in training): the correct control
is picked 45–56% of the time and "none of these" is right 81–89% of the time (zero-shot
router 36% overall). It is overconfident there (ECE 0.19), so grounding stays suggest-and-confirm. Live check on
the real Windows Terminal: 3/3 correct (Close Tab, New Tab, Minimize); UIA read ~170 ms cold,
decision ~250 ms.

## 2026-09-24 (afternoon) — command -> action assistant, v4

v4 = bge-small (33M) on mixture v4 (v3 sources + 40k generated assistant commands over 20
skills), 8,000 steps, 71 min on the A2000 (sharing the GPU part of the time), per-kind
temperatures fitted on command/daily/teacher dev sets.

| Benchmark (held out) | v3 | v4 |
|---|---|---|
| Daily route accuracy (78 tasks) | 92.3% | **97.4%** |
| Daily risk recall (11 risky) | 73% | **91%** |
| Risky tasks passing the P(safe) ≥ 0.9 gate | 0/11 | 0/11 |
| Unseen-app grounding (Teams, Spotify) | 77.6% | **82.4%** |
| Assistant commands, end to end (80, natural phrasing) | — | 73.8% |
| Assistant: wrong plan that would run on one Enter | — | 3/80 |
| Assistant: destructive request runnable | — | 0/7 |

The mixture dev set reaches 93.2%, but held-out natural phrasing reaches 73.8%: the gap is
coverage of how people actually talk, not training length. A step-5,500 preview scored the same
on commands. Misses cluster in three groups:

1. colloquial verbs the generator never uses ("get X going", "tuck this away", "kill the sound");
2. symptom-to-fix requests ("my headphones won't connect" -> Bluetooth settings, "the screen
   is too dim" -> display, which v4 turns into volume down and would run);
3. reaching an open window through what it holds ("back to the thesis", "where's my terminal").

These misses have now been seen, so benchmark v1 can no longer measure fixes for them. The next data
round gets a fresh held-out benchmark written before the data, and v1 is reported as
"inspected".

## 2026-09-25 — v6 data: 10x rows, real human commands, span recall

v5 (44 skills) was never trained: its 44 "name: description" options needed ~730 positions, past
BERT's 512, and the collator would have raised on every command row. v6 grows the catalogue to 50
skills (weather, directions, list/cancel reminders, timer and alarm control), offers skill names
only (312 positions; names were reworded to carry the description), and lets the collator trim the
state only as far as the longest branch needs. Benchmark v2 gained 14 cases for the new skills,
written before their templates (98 cases).

**Data (mixture v6, 1,350,835 train rows, 9x v5).** 600k composed commands (grammar-built fillers
instead of 6-13-item lists; surface variation on 99.9% of rows: typos, text-speak, contractions,
casing, courtesy and filler words), 183k real human commands (TOPv2, MASSIVE, CLINC150 intents
mapped to skills, two variants each), 240k daily decisions, 150k grounding, 96k public NLU
(MNLI 10x), 60k System-One synthetic, 20k intents, 2k teacher. Rows within 0.8 token Jaccard
(digits collapsed) of a benchmark-v2 case are dropped: 123 of 600k composed rows, mostly
"what's N percent of M".

**Finding: span extraction was the hidden real-world bottleneck.** On generated commands the
span rules offered the gold argument 99.9% of the time; on TOPv2's human commands only ~70%
(directions 62%, reminders 78%, alarm names 36%). Human phrasing puts the argument mid-sentence
("what time should I leave for Boston if I want to arrive by 5"). Rules tuned on TOPv2 *train*
only (preposition phrases, capitalized runs, weekday and clause cuts, colon/dash labels, a
time-stripped second pass) raise recall@10 to 85.3% (directions 84%, weather 89%, reminders 87%,
alarm names 75%); TOPv2 *test* shows the same 86-96% per skill, so the rules did not overfit.
The rules now live in `shared/text_rules.json` for the C# port.

Training moves to Colab (`notebooks/jevlet_colab.ipynb`): at ~381 tokens per row the laptop would
need ~25 h for 1.6M examples. The Colab recipe keeps the example budget at effective batch 64
(25k steps, learning rates x sqrt(4)) and keeps snapshots at 160k/400k/800k examples for a scaling
curve on the held-out human commands.

## Next

1. Train v6 on Colab; report the held-out human-command results, the scaling curve, and
   benchmark v2 on the laptop.
2. Natural-phrasing data (colloquial verbs, symptom -> setting, window-by-content) plus a new
   held-out command benchmark written first; then v5 (done as v6 above).
2. Grounding needs more app inventories, ideally recorded from live UIA trees of the apps
   actually used, plus demonstrations (the user performs the task; the clicked control is the
   label).
2. Resume the interrupted bge-small finals: `python -m scripts.run_overnight --config
   configs/overnight_pretrained.json --output results/runs/research-p1 --extend-hours 2`.
3. Colab: full-scale v3 recipe and the 4 h ablation search via `notebooks/jevlet_colab.ipynb`.

## Earlier next steps (2026-09-23, done)

1. Train daily v2: `python -m scripts.train --config configs/pretrained_daily.json --run-dir results/runs/jevlet-p-daily-v2`
   (~12 min on the A2000, on mains power), then `python -m scripts.eval_daily --checkpoint results/runs/jevlet-p-daily-v2/best.pt`.
2. Export and use it: `python -m scripts.export_checkpoint <best.pt> data/daily/current.pt --fp16`,
   then `python -m scripts.desktop --checkpoint data/daily/current.pt`; rate suggestions and run
   `python -m scripts.personalize` once enough feedback accumulates.
3. On Colab: run `notebooks/jevlet_colab.ipynb` for the full-scale daily model and the 4 h search.
