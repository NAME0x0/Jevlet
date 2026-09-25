"""Hugging Face model card for a trained Jevlet release, rendered from its results.json."""

from __future__ import annotations

from typing import Any

DATASETS = (
    (
        "WillHeld/top_v2",
        "TOPv2 (Facebook): human assistant commands, mapped to skills",
        "CC BY-SA 4.0",
    ),
    (
        "mteb/amazon_massive_intent",
        "MASSIVE (Amazon), English: intents and mapped commands",
        "CC BY 4.0",
    ),
    ("clinc/clinc_oos", "CLINC150: intents and mapped commands", "CC BY 3.0"),
    ("mteb/banking77", "BANKING77: fine-grained intent choice", "CC BY 4.0"),
    ("google/boolq", "BoolQ: yes/no reading questions (Noul)", "CC BY-SA 3.0"),
    ("nyu-mll/multi_nli", "MultiNLI: entailment as typed questions", "mixed; see the dataset card"),
)
TAGS = (
    "jevlet", "decision-model", "calibration", "system-one", "intent-classification",
    "slot-filling", "desktop-assistant", "windows", "bge-small", "typed-questions",
)  # fmt: skip


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _row(name: str, block: dict[str, Any] | None) -> str:
    if not block:
        return f"| {name} | n/a | n/a | n/a | n/a |"
    return (
        f"| {name} | {block['questions']:,} | {_pct(block['accuracy'])} | {_num(block['ece'])} | "
        f"{_num(block.get('calibrated_ece'))} |"
    )


def render_model_card(
    results: dict[str, Any], *, repo_id: str, version: str, github: str, license: str = "other"
) -> str:
    final = results.get("final", {})
    vault = final.get("real_vault", {})
    dev = final.get("mixture_dev", {})
    training = results.get("training", {})
    metrics = training.get("metrics", {})
    config = training.get("config", {})
    hyper = config.get("training", {})
    sessions = config.get("sessions", [])
    gpus = sorted({session["gpu"]["name"] for session in sessions if session.get("gpu")})
    manifest = training.get("data_manifest", {})
    sources = {
        name: info.get("splits", {}).get("train", {}).get("rows")
        for name, info in manifest.get("sources", {}).items()
    }
    effective = int(hyper.get("effective_batch", hyper.get("batch_size", 1)))
    scaling_rows = []
    for name, report in sorted(
        results.get("scaling", {}).items(), key=lambda kv: int(kv[0].split("-")[1])
    ):
        step = int(name.split("-")[1])
        skill = report.get("assistant/skill", {})
        scaling_rows.append(
            f"| {step:,} | {step * effective:,} | {_pct(skill.get('accuracy'))} | {_num(skill.get('ece'))} |"
        )
    final_skill = vault.get("assistant/skill", {})
    scaling_rows.append(
        f"| {metrics.get('steps', 0):,} (final) | {metrics.get('steps', 0) * effective:,} | "
        f"{_pct(final_skill.get('accuracy'))} | {_num(final_skill.get('ece'))} |"
    )
    temperatures = results.get("temperatures") or {}
    commit = config.get("provenance", {}).get("git_commit", "unknown")
    yaml_datasets = "\n".join(f"  - {repo}" for repo, _, _ in DATASETS)
    yaml_tags = "\n".join(f"  - {tag}" for tag in TAGS)
    dataset_rows = "\n".join(
        f"| [{repo}](https://huggingface.co/datasets/{repo}) | {use} | {lic} |"
        for repo, use, lic in DATASETS
    )
    source_rows = "\n".join(
        f"| {name} | {rows:,} |" for name, rows in sorted(sources.items()) if rows
    )
    total_rows = sum(rows for rows in sources.values() if rows)
    return f"""---
language: en
license: {license}
library_name: pytorch
base_model: BAAI/bge-small-en-v1.5
pipeline_tag: text-classification
datasets:
{yaml_datasets}
tags:
{yaml_tags}
---

# Jevlet {version}

Jevlet is a small **System-One decision model**: it reads a state (a typed command plus what is on
screen) and a set of typed questions, and returns **calibrated probabilities over the options it
is given**. It never generates text. It is a research replication of the idea behind TypeSafe's Jev,
built to drive an always-on Windows command palette where every action is chosen, not written.

- **Code, data pipeline, and training notebook:** [{github}]({github}) (commit `{commit[:12]}`)
- **Base encoder:** [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), {metrics.get("parameter_count", 0):,} parameters in total
- **Skills:** 50 (`shared/skills.json`), each with typed argument questions answered from live options

## How it decides

One packed forward pass answers several questions about the same state at once:

| Question | Kind | Options come from |
|---|---|---|
| Which action does the command ask for? | Choice | the 50 skill names |
| Is it risky to run without asking? | Noul (yes/no) | fixed |
| Which app / window / file / reminder …? | Choice | the live environment (installed apps, open windows, stores) |
| Which part of the command is the text to use? | Choice | spans copied verbatim from the command (`shared/text_rules.json`) |

Branches restart their positions after the shared state (`block_bidir` topology), so questions do not
see each other. A pointer head scores each option at its boundary token. Per-kind temperatures
(`temperatures` in the checkpoint) calibrate choices and yes/no answers separately. The app runs an
action on one keypress only when P(safe) ≥ 0.9 and every answer's confidence ≥ 0.75; otherwise it asks.

## Results

### Held-out human commands (never used for training or for tuning the span rules)

Test splits of TOPv2, MASSIVE, and CLINC150 mapped to Jevlet's skills
({vault.get("examples", 0):,} commands).

| Question type | Questions | Accuracy | ECE | ECE (calibrated) |
|---|---|---|---|---|
{_row("Skill", vault.get("assistant/skill"))}
{_row("Risk (Noul)", vault.get("assistant/risk"))}
{_row("Arguments", vault.get("assistant/slot"))}

### Mixture dev split (all sources)

| Question type | Questions | Accuracy | ECE | ECE (calibrated) |
|---|---|---|---|---|
{_row("All questions", dev.get("all"))}
{_row("Commands: skill", dev.get("assistant/skill"))}
{_row("Commands: arguments", dev.get("assistant/slot"))}
{_row("Grounding (which control)", dev.get("grounding"))}

### Scaling with training examples (skill accuracy on the held-out human commands)

| Step | Examples seen | Skill accuracy | ECE |
|---|---|---|---|
{chr(10).join(scaling_rows)}

Calibration temperatures: `{temperatures or "in the checkpoint"}`.
The hand-written laptop benchmark (`jevlet/assistant/benchmark_v2.py`, 98 commands, 6 dangerous)
runs against the live Windows environment and is reported in the GitHub repository.

## Training data

{total_rows:,} training rows. Nothing personal: the Colab build excludes the author's installed-app
list and UI captures. Every row is dropped if it comes within 0.8 token Jaccard (digits collapsed) of a
held-out benchmark command.

| Source in the mixture | Train rows |
|---|---|
{source_rows}

Public data used:

| Dataset | Use | License (as reported by the source) |
|---|---|---|
{dataset_rows}

Synthetic sources are generated by the repository: composed commands with surface variation (typos,
text-speak, casing, courtesy words), grounding (which on-screen control), daily decisions, and
System-One families (contradiction, missing information, calibration, rules).

## Training procedure

| Setting | Value |
|---|---|
| Steps | {int(metrics.get("steps", 0)):,} at effective batch {effective} |
| Learning rate | head {hyper.get("learning_rate")}, encoder {hyper.get("backbone_learning_rate")}, cosine, warmup {hyper.get("warmup_steps")} |
| Loss | {hyper.get("loss")} (cross-entropy plus Brier, a proper scoring rule) |
| Precision | {hyper.get("amp_dtype")} mixed precision |
| Hardware | {", ".join(gpus) or "n/a"} (Google Colab) |
| Training time | {_num(metrics.get("train_seconds", 0) / 3600, 2)} h, {int(metrics.get("train_tokens_per_second", 0)):,} tokens/s |
| Peak VRAM | {_num(metrics.get("peak_vram_mb", 0) / 1024, 1)} GB |

`progress.jsonl` holds the loss curve; `run_config.json` the exact configuration and every session's
GPU; `results.json` every number on this page.

## Files

| File | What |
|---|---|
| `jevlet-{version}.pt` | fp16 checkpoint with per-kind temperatures (loaded by `jevlet.system_one.SystemOne`) |
| `model.safetensors`, `config.json` | the same weights without pickle, plus model config and temperatures |
| `shared/skills.json`, `shared/text_rules.json` | the skill catalogue and span rules the model was trained against |
| `results.json`, `progress.jsonl`, `run_config.json` | evaluation, training curve, configuration |

## Use

```python
from huggingface_hub import hf_hub_download
from jevlet.benchmarks import RISK_QUESTION, daily_state
from jevlet.system_one import ChoiceQuestion, NoulQuestion, SystemOne

engine = SystemOne(hf_hub_download("{repo_id}", "jevlet-{version}.pt"))
answers = engine.evaluate(
    daily_state("remind me to call the bank at 5", "OUTLOOK: Inbox - Outlook"),
    {{
        "skill": ChoiceQuestion("Which action does the command ask for?", ["Set a reminder", "Set an alarm clock", "Search the web"]),
        "risk": NoulQuestion(RISK_QUESTION),
    }},
)
print(answers["skill"].selected, answers["skill"].confidence, answers["risk"].probability_true)
```

The planner in `jevlet/assistant/planner.py` builds the full question set from the live environment.

## Limitations

- English only; tuned for a Windows laptop's apps, windows, and settings pages.
- It chooses among the options it is given. If the right app, file, or span is not offered it cannot
  answer correctly, and should pick "Not applicable" or ask.
- Free-text arguments are spans of the command. On held-out human commands, span extraction offers the
  right span for about 85% of reminders, directions, music, and weather requests.
- Arithmetic, dates, and times are parsed by code, not predicted.
- The risk question is a safety net, not a guarantee: destructive actions also have fixed risk floors.
"""
