# Jevlet

Jevlet has two deliberately separate tracks. The day-one router uses a small pretrained text
encoder to choose among described, dynamic options on the laptop CPU. The research track is a
from-scratch reconstruction of a Jev-like **System-One decision model**. It is not TypeSafe's
proprietary implementation and makes no claim to reproduce unpublished details. The research
model consumes a state plus typed questions and dynamic options, then returns probabilities
without autoregressive text generation.

The research defaults target an RTX A2000 4 GB laptop. Colab can train a larger scratch model and
download public datasets; only validated improvements should be promoted to the daily driver.

## Architecture

One packed transformer sequence represents a state and all of its question branches:

```text
<STATE> shared state
    ├─ <QUESTION> ... <OPTION> ... <END_OPTION> ... <DECIDE>
    └─ <QUESTION> ... <OPTION> ... <END_OPTION> ... <DECIDE>
```

State tokens use causal attention. A question branch can attend to the whole state and to earlier
tokens in its own branch, including earlier options, but never to sibling questions. `<DECIDE>` is
projected into a query and each option boundary into a key; their scaled dot products form the
listwise logits. Bilinear, linear-compatibility, and MLP heads are available as ablations.

`Noul`, `Choice`, and ordered `Score` are exposed in `jevlet.api`. A callback-based two-stage API can
cheaply shortlist high-cardinality candidates before the listwise final decision. The byte tokenizer
has a fixed vocabulary and accepts unseen option strings, preventing fixed-class shortcuts.

## Quick start

Python 3.11+ and PyTorch 2.3+ are required. From `D:\Jevlet` on Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,semantic,desktop,colab]"
```

The pretrained `all-MiniLM-L6-v2` weights download on first use to `data/model_cache` (ignored by
Git). Subsequent launches reuse the cache. No Hugging Face token is required for public weights.
The locally downloaded cache can be copied with the repository to avoid another first-use download.

Run a read-only routing smoke set and then ask for a suggestion:

```powershell
python -m scripts.router_benchmark
python -m scripts.router_cli route --question "Fix failing Python tests"
python -m scripts.desktop
```

The desktop panel must run in your signed-in Windows session. It can inspect accessible controls
(with `pywinauto`), preview the screen locally, and launch an allowlisted app, switch windows,
focus a control, or type text. Every action is previewed and explicitly confirmed. It does not
use Playwright, Selenium, a headless browser, or automatic clicks. Screen images and typed text
are not saved; route requests, window title, and feedback are stored locally in SQLite. Use
`python -m scripts.desktop --manual` if you want the panel without the model. Do not install it as
an unattended Windows service: services cannot interact with the logged-in desktop. After a live
interactive smoke test, install the opt-in per-user sign-in launcher:

```powershell
python -m scripts.startup --install
python -m scripts.startup --remove
```

The first command creates only `Jevlet Desktop.cmd` in your user Startup folder; the second
removes only that Jevlet-marked file. The panel runs in the logged-in desktop session and stays
available until you close it. Startup errors are logged under `data/desktop_startup_error.log`.

Feedback for a CLI suggestion uses its returned decision ID:

```powershell
python -m scripts.router_cli feedback 1 --up
python -m scripts.router_cli feedback 2 --down --correct Codex
python -m scripts.router_cli adapt
```

In the panel, use thumbs up, or thumbs down with an optional correct-route selection. A bare
downvote is not treated as a positive training label. Adaptation splits explicit labels into
train/tune/validation partitions and promotes an adapter only when held-out accuracy does not
decrease and NLL improves. Until at least 20 held-out labels support calibration, suggestions
remain confirmation-gated. This is a conservative personalized memory/temperature update, not
unbounded recursive self-training or proof of autonomous desktop competence.

## Colab notebook and public datasets

Open `notebooks/jevlet_colab.ipynb` in Colab after pushing the repository or placing the source
folder in Drive. Choose a GPU runtime; L4 is preferred if offered, T4 works. The notebook checks
the actual Torch/CUDA runtime and selects BF16 only where supported (otherwise FP16); it never
installs a CUDA driver. Set `REPO_URL` or `DRIVE_SOURCE_FOLDER`, and choose Drive or a **private**
Hugging Face model repository for checkpoint persistence. Training runs on `/content` and syncs
atomic `last.pt` snapshots off the VM; restarting with the same dataset revisions resumes model,
optimizer, scaler, RNG, and sampler position. Colab availability and runtime length are not
guaranteed, even with a paid plan.

The default public-data pull uses BANKING77, BoolQ, and MNLI with pinned source revisions,
deduplication, split hashes, and a separate vault. AG News, SST-2, and Yelp are opt-in due to
unclear or special license terms. The notebook runs a short public-data scratch baseline and a
fixed-budget pointer-vs-bilinear ablation. Public-data adapters can also be run locally:

```powershell
python -m scripts.download_public_data --output data/public
```

The first public-data pull downloads full source datasets to the Hugging Face cache before taking
bounded train/dev/vault slices; allow disk space and network time. Do not redistribute datasets
without reviewing their source licenses. The notebook is a reproducible training path, not the
always-on desktop host.

Generate the fixed 80k benchmark:

```powershell
python -m scripts.generate_data --count 80000 --seed 1337
```

Run one proxy baseline and evaluate it:

```powershell
python -m scripts.train --config configs/proxy.json --run-dir results/runs/baseline
python -m scripts.evaluate results/runs/baseline/best.pt
```

For an interrupted run, increase the target step count and resume from `last.pt`:

```powershell
python -m scripts.train --config configs/proxy.json --run-dir results/runs/baseline --resume-from results/runs/baseline/last.pt --max-steps 1000
```

Run the bounded overnight search:

```powershell
python -m scripts.run_overnight --config configs/overnight.json --hours 8
```

The overnight runner prints each hypothesis and mechanism before training, checks train/dev hashes,
never opens the hidden vault, records every failure or result, and writes a Pareto frontier after
each stage. Promoted stages continue their earlier checkpoint; interrupted runs resume within the
same output directory and time budget. Early stages use the same fixed small dev sample for every
candidate, expanding evaluation for finalists. Vault evaluation is an explicit, separate action:

```powershell
python -m scripts.evaluate results/runs/baseline/best.pt --split vault --unlock-vault
```

For a very fast end-to-end validation:

```powershell
python -m scripts.generate_data --count 600 --seed 1337
python -m scripts.train --config configs/smoke.json --run-dir results/runs/smoke
pytest
ruff check .
```

## Scratch-model API (research only)

```python
from jevlet.api import JevletRouter, confidence_gate

router = JevletRouter("results/runs/baseline/best.pt")
decision = router.choice(
    "Rename Python files and update all imports.",
    "Which worker should handle this task?",
    ["local worker", "Codex", "retrieval", "human"],
)
action = confidence_gate(decision)
```

This API exercises the experimental topology. An eight-step or one-night scratch checkpoint is
**not** a reliable daily-driver. The pretrained `SemanticRouter` above is the default for actual
routing, and its suggestions still require user verification.

## Data and metrics

The local generator mixes semantic routing, compositional rules, explicit missing evidence,
contradictions, option permutations, and known categorical probabilities. It creates train, dev,
and a separately stored hidden-vault split plus content hashes. Development/vault examples include
unseen domains and option classes.

Every run records accuracy, OOD and unknown accuracy, Brier score, ECE, NLL, option-order
robustness, state-sharing speedup, incremental question cost, latency, throughput, parameter count,
and peak VRAM. Post-hoc temperature scaling is fit only on dev predictions.

## Research discipline

`research/program.md` defines the immutable/mutable boundary. Night one varies one conceptual axis
at a time: state pooling, option pooling, decision head, loss, or topology. Results remain
exploratory until finalists are repeated across seeds and evaluated once on the vault.

The next scaling check is **Jevlet-P**: replace the byte embedding/transformer with a pretrained
approximately 0.5–0.6B causal backbone while preserving the packed branch mask, option-boundary
representations, decision heads, loss functions, splits, and evaluator. The optional `pretrained`
dependency group reserves the Transformers dependency for that work; the scratch baseline has no
download requirement. Keeping the decision topology and evaluation unchanged makes it possible to
test which discoveries survive pretrained representations.

## Laptop safety

The proxy defaults to FP16, batch size 2, accumulation 16, activation checkpointing, and an 80%
per-process CUDA memory fraction. Jevlet does not silently change the GPU's power limit. Set your
preferred NVIDIA/Windows power and thermal policy before an unattended run, keep the laptop on a
hard ventilated surface, and shorten `--hours` if cooling is marginal.
