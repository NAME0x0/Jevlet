"""Model-card figures from a released run's own files (no values are estimated or filled in).

    python -m scripts.plot_training data/models/v6 figures/

Reads ``progress.jsonl``, ``results.json`` and ``run_config.json`` from the release folder and
writes two PNGs plus ``training_figures.json`` with every plotted number:

- ``training_loss_after_resume.png``: logged training loss at the optimizer steps actually
  recorded. Steps the log does not contain are left empty and marked, never interpolated.
- ``training_progression.png``: held-out human-command accuracy at the kept checkpoints of the
  same run. One cosine-scheduled run confounds examples seen with learning-rate annealing, so this
  is a training progression, not a scaling curve; the learning-rate factor at each checkpoint is
  printed next to it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jevlet.training import learning_rate_factor  # noqa: E402

QUESTIONS = (("assistant/skill", "Skill"), ("assistant/slot", "Arguments (slots)"))


def _progression(results: dict, training: dict, effective: int) -> list[dict]:
    rows = []
    for name, report in sorted(results.get("scaling", {}).items(), key=lambda kv: int(kv[0][5:])):
        rows.append((int(name[5:]), report))
    rows.append((int(results["training"]["metrics"]["steps"]), results["final"]["real_vault"]))
    return [
        {
            "step": step,
            "examples_seen": step * effective,
            # The factor used for the optimizer step just before this checkpoint.
            "lr_factor": learning_rate_factor(step - 1, training),
            **{key: report[key]["accuracy"] for key, _ in QUESTIONS},
            "assistant/risk": report["assistant/risk"]["accuracy"],
        }
        for step, report in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", help="folder with progress.jsonl, results.json, run_config.json")
    parser.add_argument("output")
    parser.add_argument("--version", default="v6")
    args = parser.parse_args()
    release, output = Path(args.release), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = json.loads((release / "results.json").read_text(encoding="utf-8"))
    config = json.loads((release / "run_config.json").read_text(encoding="utf-8"))
    training = config["training"]
    effective = int(training.get("effective_batch", training.get("batch_size", 1)))
    max_steps = int(training["max_steps"])
    progress = [
        json.loads(line)
        for line in (release / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    steps = [row["step"] for row in progress]
    losses = [row["loss"] for row in progress]
    first = min(steps)

    # Figure 1: loss at the recorded steps only.
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    if first > 0:
        ax.axvspan(0, first, facecolor="0.94", hatch="//", edgecolor="0.78", linewidth=0)
        ax.text(first / 2, max(losses), "not recorded\n(first session's log\nnot preserved)",
                ha="center", va="top", fontsize=8, color="0.35")  # fmt: skip
    ax.plot(steps, losses, ".", markersize=3, color="#3b6ea5", label="logged loss (one batch)")
    lr = ax.twinx()
    lr.plot(steps, [row["lr_factor"] for row in progress], "--", color="0.55", linewidth=1,
            label="learning-rate factor (logged)")  # fmt: skip
    lr.set_ylim(0, 1.05)
    lr.set_ylabel("learning-rate factor")
    ax.set_xlim(0, max_steps)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("training loss (cross-entropy + 0.25 Brier)")
    ax.set_title(f"{args.version} training loss after resume")
    handles = ax.get_legend_handles_labels()[0] + lr.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + lr.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="upper right", fontsize=8)
    fig.text(0.01, 0.01, f"Logged steps {first:,}–{max(steps):,}, every "
             f"{steps[1] - steps[0]} steps. Missing values are not reconstructed.",
             fontsize=7, color="0.35")  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output / "training_loss_after_resume.png")
    plt.close(fig)

    # Figure 2: held-out human-command accuracy at the kept checkpoints of the same run.
    rows = _progression(results, training, effective)
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4), dpi=150, width_ratios=(2, 1))
    x = [row["examples_seen"] for row in rows]
    for (key, label), marker in zip(QUESTIONS, ("o", "s"), strict=True):
        left.plot(x, [100 * row[key] for row in rows], marker=marker, label=label)
    for row in rows:
        point = (row["examples_seen"], 100 * row["assistant/skill"])
        left.annotate(f"LR×{row['lr_factor']:.2f}", point, textcoords="offset points",
                      xytext=(0, 8), ha="center", fontsize=7, color="0.4")  # fmt: skip
    right.plot(x, [100 * row["assistant/risk"] for row in rows], marker="^", color="#8c564b")
    for axis, title in ((left, "Skill and arguments"), (right, "Risk question")):
        axis.set_xscale("log")
        axis.set_xticks(
            x, [f"{value / 1e6:.2g}M" if value >= 1e6 else f"{value // 1000}k" for value in x]
        )
        axis.minorticks_off()
        axis.set_xlabel("training examples seen")
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.3)
    left.margins(y=0.12)  # room for the learning-rate labels above the points
    left.set_ylabel("accuracy on held-out human commands (%)")
    left.legend(fontsize=8)
    fig.suptitle(f"{args.version} training progression (one run; not a scaling curve)")
    note = (
        "Checkpoints of one cosine-scheduled run: examples seen and learning-rate annealing are "
        "confounded. LR×: learning-rate factor at the checkpoint."
    )
    fig.text(0.01, 0.01, note, fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output / "training_progression.png")
    plt.close(fig)

    record = {
        "loss": {"first_logged_step": first, "last_logged_step": max(steps), "rows": len(steps)},
        "progression": rows,
    }
    (output / "training_figures.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
