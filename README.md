# Jevlet

Jevlet is a from-scratch research reconstruction of a Jev-like **System-One decision model**. It is
not TypeSafe's proprietary implementation and makes no claim to reproduce unpublished details.
The model consumes a state plus typed questions and dynamic text options, then returns probability
distributions without autoregressive text generation.

The repository is designed for an RTX A2000 4 GB laptop: a small proxy searches architectural
choices, successive halving promotes only promising candidates, and all GPU-memory-heavy settings
are conservative and configurable.

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

Python 3.11+ and PyTorch 2.3+ are required. The current environment can be used directly, or:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Generate the fixed 80k benchmark:

```powershell
python -m scripts.generate_data --count 80000 --seed 1337
```

Run one proxy baseline and evaluate it:

```powershell
python -m scripts.train --config configs/proxy.json --run-dir results/runs/baseline
python -m scripts.evaluate results/runs/baseline/best.pt
```

Run the bounded overnight search:

```powershell
python -m scripts.run_overnight --config configs/overnight.json --hours 8
```

The overnight runner prints each hypothesis and mechanism before training, checks train/dev hashes,
never opens the hidden vault, records every failure or result, and writes a Pareto frontier after
each stage. Vault evaluation is an explicit, separate action:

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

## Daily-driver API

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

This scratch model is intended for routing, tool selection, notification classification, and
confidence-gated execution—not general knowledge generation.

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

