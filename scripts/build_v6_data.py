"""Build the v6 training sources (10x v5) and the v6 mixture, on the laptop or on Colab.

    python -m scripts.build_v6_data                      # every step, skipping finished ones
    python -m scripts.build_v6_data --root /content/data --steps mixture check

Each step builds into ``<name>.partial`` and is renamed into place only when it succeeds, so a
crash mid-step never leaves a half-written source; rerunning skips completed steps. Personal
inputs (this PC's Start-menu apps and captured UI inventories) are used only with
``--personal``, which defaults to on for Windows and off elsewhere: Colab data never contains
them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

STEPS = (
    "topv2", "real", "commands", "grounding", "daily", "public", "intents", "synthetic", "teacher",
    "mixture", "check",
)  # fmt: skip
TOPV2_REPO = "WillHeld/top_v2"
TOPV2_REVISION = "refs/convert/parquet"
# (directory, per-split row limits) relative to the data root.
MIXTURE = {
    "commands": ("commands_v6", None),
    "real": ("real_commands", None),
    "grounding": ("grounding_v6", {"train": 150_000, "dev": 3_000}),
    "daily": ("daily_v6", None),
    "public": ("public_v6", None),
    "intents": ("public_intents", None),
    "synthetic": ("synthetic", {"train": 60_000, "dev": 800, "vault": 400}),
    "teacher": ("teacher", None),
}


def _summary(manifest: dict) -> str:
    splits = manifest.get("splits", manifest)
    rows = {k: v.get("rows", v.get("count")) for k, v in splits.items() if isinstance(v, dict)}
    return json.dumps(rows)


def _step(root: Path, name: str, build) -> None:
    """Run ``build(partial_dir)`` unless ``root/name`` is complete; publish it atomically."""
    final = root / name
    if (final / ".complete").exists():
        print(f"{name}: done already", flush=True)
        return
    partial = root / f"{name}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    result = build(partial)
    (partial / ".complete").write_text(json.dumps(result or {}, default=str)[:2000] + "\n")
    shutil.rmtree(final, ignore_errors=True)
    partial.rename(final)
    print(f"{name}: {_summary(result) if isinstance(result, dict) else 'ok'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data")
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--personal", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    personal = sys.platform == "win32" if args.personal is None else args.personal

    if "topv2" in args.steps:

        def topv2(out: Path) -> dict:
            from huggingface_hub import hf_hub_download

            from jevlet.training import file_sha256

            files = {}
            for split in ("train", "eval", "test"):
                path = hf_hub_download(
                    TOPV2_REPO, f"default/{split}/0000.parquet", repo_type="dataset",
                    revision=TOPV2_REVISION, local_dir=out,
                )  # fmt: skip
                files[split] = file_sha256(path)
            return {"repo": TOPV2_REPO, "revision": TOPV2_REVISION, "sha256": files}

        _step(root, "raw_top_v2", topv2)
    if "real" in args.steps:
        from jevlet.assistant.real_commands import generate_real_dataset

        _step(root, "real_commands", lambda out: generate_real_dataset(out, root / "raw_top_v2"))
    if "commands" in args.steps:
        from jevlet.assistant.commands_synthetic import generate_command_dataset

        apps: tuple[str, ...] = ()
        if personal:
            from jevlet.assistant.environment import installed_apps

            apps = tuple(app.name for app in installed_apps())
        _step(
            root,
            "commands_v6",
            lambda out: generate_command_dataset(
                out, (600_000, 20_000), seed=606, local_apps=apps, workers=args.workers
            ),
        )
    if "grounding" in args.steps:
        from jevlet.grounding_synthetic import generate_grounding_dataset

        extra = ()
        if personal:
            from jevlet.grounding_live import inventory_apps

            extra = inventory_apps()
        _step(
            root,
            "grounding_v6",
            lambda out: generate_grounding_dataset(out, counts=(300_000, 30_000), extra_apps=extra),
        )
    if "daily" in args.steps:
        from jevlet.daily_synthetic import generate_daily_dataset

        # Same seed and dev/vault counts as v5: dev and vault rows are unchanged.
        _step(root, "daily_v6", lambda out: generate_daily_dataset(out, (240_000, 3_000, 1_500)))
    if "public" in args.steps:
        from jevlet.public_data import download_public_data

        # BANKING77 and BoolQ are already used almost whole; MNLI grows 10x.
        _step(
            root,
            "public_v6",
            lambda out: download_public_data(
                out, ("banking77", "boolq", "mnli"), train_limit=80_000, dev_limit=1_000,
                vault_limit=1_000,
            ),
        )  # fmt: skip
    if "intents" in args.steps:
        from jevlet.public_data import download_public_data

        _step(
            root,
            "public_intents",
            lambda out: download_public_data(
                out, ("massive", "clinc"), train_limit=10_000, dev_limit=1_500, vault_limit=1_500
            ),
        )
    if "synthetic" in args.steps:
        from jevlet.synthetic import generate_dataset

        _step(root, "synthetic", lambda out: generate_dataset(out, count=80_000, seed=1337))
    if "teacher" in args.steps:
        from jevlet.teacher import generate_teacher_dataset

        _step(root, "teacher", lambda out: generate_teacher_dataset(out))
    if "mixture" in args.steps:
        from jevlet.mixture import build_mixture

        def mixture(out: Path) -> dict:
            sources = {name: root / path for name, (path, _) in MIXTURE.items()}
            limits = {name: limit for name, (_, limit) in MIXTURE.items() if limit}
            manifest = build_mixture(sources, out, limits=limits)
            manifest["personal_inputs"] = personal
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
            return manifest

        _step(root, "mixture_v6", mixture)
    if "check" in args.steps:
        check_collation(root / "mixture_v6")


def check_collation(mixture: Path, config_path: str = "configs/pretrained_daily_v6.json") -> None:
    """Collate every 25th train row and every dev row with the real collator: a position or
    packing overflow must fail here, not hours into training."""
    from jevlet.data import JsonlDecisionDataset
    from jevlet.training import build_collator, build_model

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model = build_model(config["model"], load_weights=False)
    collator = build_collator(model, config["data"])
    longest, checked = 0, 0
    for name, stride in (("train.jsonl", 25), ("dev.jsonl", 1)):
        dataset = JsonlDecisionDataset(mixture / name, lazy=True)
        for index in range(0, len(dataset), stride):
            longest = max(longest, collator([dataset[index]])["input_ids"].shape[1])
            checked += 1
    print(f"check: {checked} rows collate, longest {longest} tokens", flush=True)


if __name__ == "__main__":
    main()
