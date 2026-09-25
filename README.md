# Jevlet

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NAME0x0/Jevlet/blob/main/notebooks/jevlet_colab.ipynb)
[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-NAME0x0%2FJevlet-yellow)](https://huggingface.co/NAME0x0/Jevlet)

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
| `pretrained` (Jevlet-P) | any BERT-family encoder; bge-small-en-v1.5 (33M) since v4 | daily driver |
| `scratch` | byte-level transformer, random init | topology research without pretrained confounds |

## Results so far (RTX A2000 4 GB)

| Model | Benchmark route | Route ECE | Risky tasks passing gate | Unseen-app grounding |
|---|---|---|---|---|
| Zero-shot MiniLM router | 44.9% | 0.081 | — | 36% |
| Jevlet-P v1 (generic data) | 30.8% | 0.102 | — | — |
| Jevlet-P v2 (+ daily route/risk data) | 91.0% | 0.061 | 1 / 11 | — |
| Jevlet-P v3 (+ teacher, grounding; per-kind calibration) | 92.3% | 0.062 | 0 / 11 | 77.6% |
| **Jevlet-P v4 (bge-small, + 40k assistant commands)** | **97.4%** | **0.056** | **0 / 11** | **82.4%** |

v4 also plans assistant commands: 73.8% end-to-end on 80 held-out natural-language commands,
0 of 7 destructive requests runnable, ~35 ms median latency on the GPU.

The benchmark is 78 hand-written laptop tasks kept out of all training data
(`jevlet/benchmarks.py`); the gate runs a task unattended only if P(safe) ≥ 0.9. CPU latency
is ~40 ms for three questions in one pass. Details and caveats: `research/lab-notebook.md`.

**v6 (training on Colab):** 50 skills, 1.35M training rows (9x v5): 183k real human commands from
TOPv2, MASSIVE, and CLINC150 mapped onto the skills, 600k composed commands with typing noise, and the
earlier sources scaled up. It is evaluated on the held-out test splits of those human corpora and on
`jevlet/assistant/benchmark_v2.py` (98 commands). Results go to the
[model card](https://huggingface.co/NAME0x0/Jevlet) when the run finishes.

## Quick start

Python 3.11+ and PyTorch 2.3+. From `D:\Jevlet`:

```powershell
python -m pip install -e ".[dev,semantic,desktop,colab]"
```

Start the assistant (tray icon, runs in the background):

```powershell
pythonw -m jevlet.app                   # or: python -m scripts.desktop
python -m scripts.startup --install     # start it at sign-in; --remove to undo
```

Press **Alt+Space** (falls back to Alt+Shift+Space or Ctrl+Alt+J if taken) and type what you want.

## The assistant

| Key | Does |
|---|---|
| type | live interpretation: action, target, confidence, risk, latency |
| Enter | run it; anything risky or uncertain needs Enter twice |
| ↓ / ↑ | alternatives the model also considered |
| Tab | **show me**: click the right control yourself; Jevlet records it as training data |
| Esc | close; Enter on the empty bar undoes the last action |

Skills (50, defined once in `shared/skills.json` for the Python planner and the app): apps and
windows; playback, music, volume, brightness, Wi-Fi and Bluetooth; Settings pages and dark/light
mode; web search, websites, folders, and files; typing, shortcuts, and clicking visible controls;
timers, stopwatch, alarms (set, list, cancel, snooze or stop), reminders, calendar events (add,
show, move, cancel), to-dos, and notes; calculator and unit conversion, world clock, weather,
directions, date and battery; screenshots, lock, and power; email drafts; handing a request to
Claude/ChatGPT/Gemini; and asking for clarification. "Open Spotify then play the next song" runs
step by step, re-planning each step against the live screen.

Every decision is a Jev-style choice over options found on your machine at that moment: installed
apps, open windows, visible UI Automation controls, or spans of your own words. Jevlet never
generates text and never automates a browser. Destructive requests (deleting, paying, sending on
your behalf) are not skills, so they can only produce "I need more detail".

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

## Screen reading

Windows UI Automation reads the foreground window in one cached `FindAllBuildCache` call
(~80 ms, versus ~630 ms through pywinauto). Controls are activated through UIA patterns (Invoke,
Select, Toggle, Expand) before falling back to a real click. Windows are focused without
synthesized keys, because an Alt tap opens Office ribbon key tips. Screens are never saved.

## Learning from use

Jevlet learns from what you do, not from forms:

- Running the top suggestion records it as correct; choosing an alternative with ↓ records a
  correction.
- **Tab → click** records a demonstration: the command, the window's controls, and the control you
  actually clicked (low-level mouse hook + UI Automation element-at-point, stored locally).
- `python -m scripts.capture_inventory` records the controls of your open apps (read-only), so
  grounding training covers the apps you really use.

```powershell
python -m scripts.personalize --checkpoint data/daily/current.pt
```

Ratings and demonstrations split 60/20/20 by stable id into fine-tune, temperature, and gate sets.
A new model replaces `current.pt` only if held-out accuracy holds, NLL improves, and accuracy on
replayed general data drops by at most one point. The old model is kept as `previous.pt`.

## Data

| Source | Module | Notes |
|---|---|---|
| Synthetic System-One families | `synthetic.py` | rules, missing evidence, contradictions, known probabilities |
| Public sets | `public_data.py` | BANKING77, BoolQ, MNLI, MASSIVE, CLINC150 (out-of-scope → abstain); pinned revisions |
| Daily route/risk decisions | `daily_synthetic.py` | paraphrased criteria, option subsets, soft targets on real overlaps |
| Teacher corpus | `corpus/`, `teacher.py` | ~220 hand-labeled tasks with soft route and risk probabilities |
| Screen grounding | `grounding_synthetic.py` | 22 app inventories; Teams and Spotify held out |
| Real human commands | `assistant/real_commands.py` | TOPv2, MASSIVE, CLINC150 intents mapped to skills; TOPv2 parses give argument spans |
| Composed commands | `assistant/commands_synthetic.py`, `lexicon.py`, `augment.py` | grammar-built fillers; typos, text-speak, casing, courtesy words |

Every generated or imported command is dropped if it comes within 0.8 token Jaccard (digits
collapsed) of a `benchmark_v2` case (`assistant/decontam.py`). Free-text arguments are spans of
the command chosen by the model from candidates built by `shared/text_rules.json`
(`python -m scripts.audit_spans` reports their recall). `python -m scripts.build_v6_data` builds
every source and the hash-pinned train/dev/vault mixture; vault rows are never read by training
or research. Personal inputs (your Start-menu apps and UI captures) are used only on Windows
with `--personal`; Colab builds never contain them.

## Training

**Colab (recommended for v6).** Open `notebooks/jevlet_colab.ipynb` with the badge above, pick an
L4 or A100 runtime, add `HF_TOKEN` to Colab's Secrets, and run all cells. The notebook builds the
data (or restores it from a hash-verified Drive cache), trains, calibrates, evaluates on held-out
human commands, and publishes the weights with a full model card to Hugging Face. Everything is
driven by `jevlet/colab.py`, which is tested locally:

- the trainer runs detached from the kernel; closing the tab or restarting the kernel does not stop
  it, and rerunning the cell re-attaches;
- `last.pt` (with `last.prev.pt` one save back) and step snapshots mirror to Drive atomically; a
  reset runtime resumes from the newest readable checkpoint, at the same row even on a different GPU;
- out-of-memory halves the micro-batch and keeps the effective batch by gradient accumulation;
  other crashes resume from the last save, up to five times;
- a resumed run checks out the git commit it started with.

Laptop (on mains power):

```powershell
python -m scripts.build_v6_data
python -m scripts.train --config configs/pretrained_daily_v6.json --run-dir results/runs/jevlet-p-daily-v6
python -m scripts.calibrate results/runs/jevlet-p-daily-v6/best.pt data/models/jevlet-v6.pt --fp16 --data data/real_commands/dev.jsonl --data data/commands_v6/dev.jsonl
python -m scripts.eval_assistant data/models/jevlet-v6.pt --benchmark v2
```

Runs save a resumable `last.pt` (model, optimizer, scaler, RNG, position in the epoch) and
resume with `--resume-from`.

## Research

`python -m scripts.run_overnight --config configs/overnight_pretrained.json` runs successive
halving over one-axis hypotheses (`research/search_space.py`), promotes by Pareto layers over
quality, calibration, robustness, throughput, latency, and VRAM, and never opens the vault.
Interrupted runs resume in the same output directory; `--extend-hours N` gives them a new
deadline. Results stay exploratory until finalists repeat across seeds.

## Laptop safety

Training uses FP16, a GPU memory cap, and saves progress periodically. Jevlet does not change
power limits. Train on mains power with the laptop on a hard, ventilated surface.

## License

Code: [MIT](LICENSE). Datasets keep their own licenses (listed in the model card); TOPv2 and BoolQ are share-alike.
