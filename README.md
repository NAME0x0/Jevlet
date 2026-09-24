# Jevlet

Jevlet is a from-scratch research reconstruction of a Jev-like **System-One decision model**
plus a laptop daily driver built on it. It is not TypeSafe's implementation and makes no claim
about Jev's unpublished internals; `research/decisions.md` lists which public facts each design
choice rests on.

A System-One model takes a state and several typed questions (`Noul`, `Choice`, `Score`) and
returns calibrated probabilities over runtime-defined options in one parallel pass, with no
text generation. It cannot answer outside the options it was given.

## Architecture

One packed sequence holds the shared state and every question branch:

```text
[CLS] [STATE] state tokens
   ├─ [QUESTION] q1 … [OPTION] a [END_OPTION] [OPTION] b [END_OPTION] … [DECIDE]
   └─ [QUESTION] q2 … [OPTION] … [DECIDE]
```

- State tokens see only the state. A branch sees the state and itself, never a sibling, so
  one packed pass gives exactly the answers of separate calls (tested) while encoding the state
  once.
- Branch position ids restart after the state, so only `state + longest question` must fit
  the backbone's positions; more questions cost compute, not position range.
- `[DECIDE]` is projected to a query and each option to a key; their dot products are the
  listwise logits (bilinear, linear, and MLP heads are ablations).
- `option_isolated` topology gives options identical positions and hides them from each
  other, so only `[DECIDE]` compares them: probabilities are exactly order-invariant.

Two backbones share this topology, training loop, metrics, and API:

| Family | Backbone | Use |
|---|---|---|
| `pretrained` (Jevlet-P) | any BERT-family encoder, default MiniLM-L6 (22.9M) | daily driver |
| `scratch` | byte-level transformer, random init | topology research without pretrained confounds |

## Results so far (RTX A2000 4 GB)

| Model | Benchmark route | Route ECE | Risky tasks passing gate | Unseen-app grounding |
|---|---|---|---|---|
| Zero-shot MiniLM router | 44.9% | 0.081 | — | 36% |
| Jevlet-P v1 (generic data) | 30.8% | 0.102 | — | — |
| Jevlet-P v2 (+ daily route/risk data) | 91.0% | 0.061 | 1 / 11 | — |
| **Jevlet-P v3 (+ teacher, grounding; per-kind calibration)** | **92.3%** | **0.062** | **0 / 11** | 45–56% control, 81–89% "none" |

The benchmark is 78 hand-written laptop tasks kept out of all training data
(`jevlet/benchmarks.py`); the gate runs a task unattended only if P(safe) ≥ 0.9. CPU latency
is ~40 ms for three questions in one pass. Details and caveats: `research/lab-notebook.md`.

## Quick start

Python 3.11+ and PyTorch 2.3+. From `D:\Jevlet`:

```powershell
python -m pip install -e ".[dev,semantic,desktop,colab]"
```

Use the daily-driver checkpoint (`data/daily/current.pt`, a 44 MB fp16 export):

```powershell
python -m scripts.router_cli --checkpoint data/daily/current.pt route --question "Fix the failing tests in auth.py"
python -m scripts.desktop --checkpoint data/daily/current.pt
python -m scripts.ground_live --checkpoint data/daily/current.pt --task "close this tab"
```

Without `--checkpoint`, the router and panel fall back to the zero-shot MiniLM router.

## System-One API

```python
from jevlet.system_one import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOne

engine = SystemOne("data/daily/current.pt")
answers = engine.evaluate(
    "Task: the invoice total is wrong and the customer is angry\nActive window: Outlook - Inbox",
    {
        "route": ChoiceQuestion("Who should handle this task?", {
            "Claude": "Draft and explain written material.",
            "Human": "Ask a person for anything sensitive or irreversible.",
        }),
        "risky": NoulQuestion("Could this move money, delete data, or share private data?"),
        "urgency": ScoreQuestion("How urgent is this?", ["Low", "Medium", "High"]),
    },
)
```

All questions run in one packed forward pass. Choices accept `name -> description` criteria
and up to 255 options (score independently and shortlist beyond that). Put anything every
question needs into the state: branches cannot see each other's question text.

## Desktop harness

The panel (`scripts.desktop`) runs in your signed-in Windows session. It reads the foreground
window through Windows UI Automation using one cached `FindAllBuildCache` call (~80 ms, versus ~630 ms
through pywinauto), with no Playwright, Selenium, or headless browser. It can launch
allowlisted apps, switch windows, focus a control, or type text, and every action is previewed
and explicitly confirmed. Route and risk are asked in one pass; a task with P(risky) ≥ 0.5 can never reach
`execute`, and nothing executes unattended until the model has been validated on 20 of your own
held-out ratings. Screens and typed text are never saved.

Screen grounding asks which visible control fits a task: the options are the window's
actionable UI Automation controls, the Jev-style runtime criteria for a desktop.

## Learning from use

Thumbs up records the suggestion as correct; thumbs down with a correction records the
correction; a bare thumbs down is never a training label. Then:

```powershell
python -m scripts.personalize --checkpoint data/daily/current.pt
```

Feedback splits 60/20/20 by decision id into fine-tune, temperature, and gate sets. The new
model replaces `current.pt` only if held-out feedback accuracy holds, NLL improves, and general
accuracy on replayed data drops by at most one point; the old model is kept as `previous.pt`.

## Data

| Source | Module | Notes |
|---|---|---|
| Synthetic System-One families | `synthetic.py` | rules, missing evidence, contradictions, known probabilities |
| Public sets | `public_data.py` | BANKING77, BoolQ, MNLI, MASSIVE, CLINC150 (out-of-scope → abstain); pinned revisions |
| Daily route/risk decisions | `daily_synthetic.py` | paraphrased criteria, option subsets, soft targets on real overlaps |
| Teacher corpus | `corpus/`, `teacher.py` | ~220 hand-labeled tasks with soft route and risk probabilities |
| Screen grounding | `grounding_synthetic.py` | 12 app inventories; Teams and Spotify held out |

`scripts.build_mixture` combines sources into one hash-pinned train/dev/vault mixture; vault
rows are never read by training or research.

## Training

Laptop (on mains power):

```powershell
python -m scripts.train --config configs/pretrained_daily_v3.json --run-dir results/runs/jevlet-p-daily-v3
python -m scripts.calibrate results/runs/jevlet-p-daily-v3/best.pt results/runs/jevlet-p-daily-v3/calibrated.pt --data data/daily/dev.jsonl --data data/teacher/dev.jsonl --fp16
python -m scripts.eval_daily --checkpoint results/runs/jevlet-p-daily-v3/calibrated.pt
```

Runs save a resumable `last.pt` (model, optimizer, scaler, RNG, sampler) and resume with
`--resume-from`. Colab: `notebooks/jevlet_colab.ipynb` pulls every dataset, trains in BF16,
evaluates, exports, and runs the ablation search. It syncs to Drive or a private Hugging Face
repo and survives VM loss by resuming. It never installs CUDA drivers.

## Research

`python -m scripts.run_overnight --config configs/overnight_pretrained.json` runs successive
halving over one-axis hypotheses (`research/search_space.py`), promotes by Pareto layers over
quality, calibration, robustness, throughput, latency, and VRAM, and never opens the vault.
Interrupted runs resume in the same output directory; `--extend-hours N` gives them a new
deadline. Results stay exploratory until finalists repeat across seeds.

## Laptop safety

Training uses FP16, a GPU memory cap, and saves progress periodically. Jevlet does not change
power limits. Train on mains power with the laptop on a hard, ventilated surface.
