"""Build the v6 training sources (10x v5) and the v6 mixture.

    python -m scripts.build_v6_data --steps real commands grounding daily public mixture

Sources: real human commands (TOPv2/MASSIVE/CLINC mapped to skills), composed synthetic
commands with surface variation, grounding, daily decisions, and public NLU data. Each step
writes its own directory and manifest; the mixture step reads them all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STEPS = ("real", "commands", "grounding", "daily", "public", "mixture")
MIXTURE = {
    "commands": ("data/commands_v6", None),
    "real": ("data/real_commands", None),
    "grounding": ("data/grounding_v6", {"train": 150_000, "dev": 3_000}),
    "daily": ("data/daily_v6", None),
    "public": ("data/public_v6", None),
    "intents": ("data/public_intents", None),
    "synthetic": ("data", {"train": 60_000, "dev": 800, "vault": 400}),
    "teacher": ("data/teacher", None),
}


def _summary(manifest: dict) -> str:
    splits = manifest.get("splits", manifest)
    rows = {k: v.get("rows", v.get("count")) for k, v in splits.items() if isinstance(v, dict)}
    return json.dumps(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if "real" in args.steps:
        from jevlet.assistant.real_commands import generate_real_dataset

        print("real", _summary(generate_real_dataset("data/real_commands")), flush=True)
    if "commands" in args.steps:
        from jevlet.assistant.commands_synthetic import generate_command_dataset
        from jevlet.assistant.environment import installed_apps

        apps = tuple(app.name for app in installed_apps())
        manifest = generate_command_dataset(
            "data/commands_v6", (600_000, 20_000), seed=606, local_apps=apps, workers=args.workers
        )
        print("commands", _summary(manifest), flush=True)
    if "grounding" in args.steps:
        from jevlet.grounding_live import inventory_apps
        from jevlet.grounding_synthetic import generate_grounding_dataset

        manifest = generate_grounding_dataset(
            "data/grounding_v6", counts=(300_000, 30_000), extra_apps=inventory_apps()
        )
        print("grounding", _summary(manifest), flush=True)
    if "daily" in args.steps:
        from jevlet.daily_synthetic import generate_daily_dataset

        # Same seed and dev/vault counts as v5: dev and vault rows are unchanged.
        manifest = generate_daily_dataset("data/daily_v6", counts=(240_000, 3_000, 1_500))
        print("daily", _summary(manifest), flush=True)
    if "public" in args.steps:
        from jevlet.public_data import download_public_data

        # BANKING77 and BoolQ are already used almost whole; MNLI grows 10x. Same seed and
        # dev/vault limits as v5, so dev and vault rows are unchanged.
        manifest = download_public_data(
            "data/public_v6",
            ("banking77", "boolq", "mnli"),
            train_limit=80_000,
            dev_limit=1_000,
            vault_limit=1_000,
        )
        print("public", json.dumps(manifest.get("sources", {}), default=str)[:400], flush=True)
    if "mixture" in args.steps:
        from jevlet.mixture import build_mixture

        sources = {name: path for name, (path, _) in MIXTURE.items()}
        limits = {name: limit for name, (_, limit) in MIXTURE.items() if limit}
        manifest = build_mixture(sources, "data/mixture_v6", limits=limits)
        rows = {split: info["rows"] for split, info in manifest["splits"].items()}
        per_source = {
            name: info["splits"].get("train", {}).get("rows")
            for name, info in manifest["sources"].items()
        }
        print("mixture", json.dumps(rows), json.dumps(per_source), flush=True)
        Path("data/mixture_v6/sources.json").write_text(json.dumps(per_source, indent=2) + "\n")


if __name__ == "__main__":
    main()
