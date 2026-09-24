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

## Next

1. Train daily v2: `python -m scripts.train --config configs/pretrained_daily.json --run-dir results/runs/jevlet-p-daily-v2`
   (~12 min on the A2000, on mains power), then `python -m scripts.eval_daily --checkpoint results/runs/jevlet-p-daily-v2/best.pt`.
2. Export and use it: `python -m scripts.export_checkpoint <best.pt> data/daily/current.pt --fp16`,
   then `python -m scripts.desktop --checkpoint data/daily/current.pt`; rate suggestions and run
   `python -m scripts.personalize` once enough feedback accumulates.
3. On Colab: run `notebooks/jevlet_colab.ipynb` for the full-scale daily model and the 4 h search.
