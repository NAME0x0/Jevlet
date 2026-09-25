"""Write notebooks/jevlet_colab.ipynb (the notebook is generated so it stays reviewable here).

python -m scripts.make_colab_notebook
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = "NAME0x0/Jevlet"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# Train Jevlet on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/@REPO@/blob/main/notebooks/jevlet_colab.ipynb)

Builds the v6 data, trains Jevlet-P, calibrates it, evaluates it on held-out human commands, and
publishes the weights with a full model card to Hugging Face. Checkpoints go to Google Drive.

**Runtime:** *Runtime → Change runtime type →* **L4** or **A100** (bf16). T4 works at a smaller
micro-batch. Colab Pro's background execution lets training continue with the tab closed.

**If anything goes wrong** (disconnect, crash, runtime reset, out of memory): reconnect and
**Runtime → Run all**. Nothing is lost that was saved:

| What | Where | Survives a runtime reset |
|---|---|---|
| Built training data (hash-verified cache) | `Drive/Jevlet/data_cache/` | yes |
| `last.pt` every 500 steps (+ `last.prev.pt`), step snapshots, progress | `Drive/Jevlet/runs/<version>/` | yes |
| Calibrated export, `results.json` | `Drive/Jevlet/exports/<version>/` | yes |
| Working copies, the running trainer | `/content` | no (resumed from Drive) |

A resumed run uses the git commit it started with, so later pushes to the repository never change a
run half way. Out-of-memory halves the micro-batch and keeps the effective batch through gradient
accumulation; other crashes resume from the last save, up to five times.""",
    ),
    (
        "code",
        """# Settings. Change RUN_VERSION to start a separate run instead of resuming this one.
RUN_VERSION = "v6"
REPO_URL = "https://github.com/@REPO@.git"
BRANCH = "main"
DRIVE_DIR = "/content/drive/MyDrive/Jevlet"
BASE_CONFIG = "configs/pretrained_daily_v6_colab.json"
HF_REPO_ID = ""  # empty: <your Hugging Face username>/Jevlet
HF_PRIVATE = False
MODEL_LICENSE = "mit"  # license field of the model card (the repository is MIT too)
PUBLISH = True
ALLOW_CODE_UPDATE = False  # True: a resumed run picks up newer code from the branch""",
    ),
    (
        "code",
        """# Drive and GPU.
import subprocess

from google.colab import drive

drive.mount("/content/drive")
gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True, text=True,
)
if gpu.returncode != 0:
    raise SystemExit("No GPU: Runtime -> Change runtime type -> L4 or A100, then Run all.")
print("GPU:", gpu.stdout.strip())
if "T4" in gpu.stdout:
    print("T4: works, but an L4 or A100 is several times faster.")""",
    ),
    (
        "code",
        """# Code: clone once; a resumed run returns to the commit it started with.
import json
import os
from pathlib import Path

REPO_DIR = Path("/content/Jevlet")
run_file = Path(DRIVE_DIR) / "runs" / RUN_VERSION / "run_config.json"
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, str(REPO_DIR)], check=True)
subprocess.run(["git", "fetch", "origin", BRANCH], cwd=REPO_DIR, check=True)
pinned = None
if run_file.exists() and not ALLOW_CODE_UPDATE:
    pinned = json.loads(run_file.read_text()).get("provenance", {}).get("git_commit")
target = pinned or f"origin/{BRANCH}"
subprocess.run(["git", "checkout", "--quiet", "--detach", target], cwd=REPO_DIR, check=True)
commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, capture_output=True, text=True).stdout.strip()
print("code at", commit, "(pinned by the run)" if pinned else "(latest)")
os.chdir(REPO_DIR)""",
    ),
    (
        "code",
        """# Dependencies, pinned to the versions the laptop trains and tests with.
subprocess.run(
    [
        "pip", "install", "--quiet", "transformers==5.5.0", "tokenizers==0.22.2", "datasets==4.3.0",
        "huggingface_hub>=1.0", "safetensors", "pandas", "pyarrow",
    ],
    check=True,
)
import sys

sys.path.insert(0, str(REPO_DIR))
import torch
import transformers

print("torch", torch.__version__, "| transformers", transformers.__version__)""",
    ),
    (
        "code",
        """# Hugging Face token: add HF_TOKEN in Colab's Secrets panel (key icon), or paste it when asked.
from getpass import getpass

from huggingface_hub import HfApi

try:
    from google.colab import userdata

    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = None
if not HF_TOKEN:
    HF_TOKEN = getpass("Hugging Face token (write access): ")
os.environ["HF_TOKEN"] = HF_TOKEN  # downloads use it too; never printed
HF_USER = HfApi(token=HF_TOKEN).whoami()["name"]
HF_REPO_ID = HF_REPO_ID or f"{HF_USER}/Jevlet"
print("Hugging Face user:", HF_USER, "| model repo:", HF_REPO_ID)""",
    ),
    (
        "code",
        """# Data: verified local copy, else the Drive cache, else a full build (then cached).
from jevlet.colab import Paths, finish, gpu_profile, prepare_data, publish, stop, train

paths = Paths(drive=Path(DRIVE_DIR), work=Path("/content"), repo=REPO_DIR, version=RUN_VERSION)
profile = gpu_profile()
print("hardware:", profile)
manifest = prepare_data(paths, profile["workers"])
rows = {name: info["splits"].get("train", {}).get("rows") for name, info in manifest["sources"].items()}
print(f"mixture: {manifest['splits']['train']['rows']:,} train rows", rows)""",
    ),
    (
        "markdown",
        """## Train
Runs as a background process with progress every 30 s. Stopping this cell (■) only stops watching:
training continues, and running the cell again re-attaches. After a runtime reset it resumes from the
newest readable checkpoint on Drive.""",
    ),
    (
        "code",
        """status = train(paths, profile, BASE_CONFIG)
print("training:", status)""",
    ),
    (
        "code",
        """# Calibrate per question kind, evaluate the final model and every snapshot on held-out human
# commands, and write results.json next to the export on Drive. Safe to rerun.
results = finish(paths)
vault = results["final"]["real_vault"]
for name in ("assistant/skill", "assistant/risk", "assistant/slot"):
    block = vault.get(name, {})
    print(f"{name:18} accuracy {block.get('accuracy', 0):.3f}  ECE {block.get('ece', 0):.3f}")
for step, report in results.get("scaling", {}).items():
    print(step, "skill accuracy", round(report.get("assistant/skill", {}).get("accuracy", 0), 4))""",
    ),
    (
        "code",
        """# Publish weights, catalogues, results, and the model card; tag the version.
if PUBLISH:
    url = publish(
        paths, HF_REPO_ID, HF_TOKEN, HF_PRIVATE, "https://github.com/@REPO@", license=MODEL_LICENSE
    )
    print(url)""",
    ),
    (
        "markdown",
        """## On the laptop
The laptop benchmark needs the live Windows environment:

```powershell
cd D:\\Jevlet
python -c "from huggingface_hub import hf_hub_download as d; print(d('<user>/Jevlet', 'jevlet-v6.pt', local_dir='data/models'))"
python -m scripts.eval_assistant data\\models\\jevlet-v6.pt --benchmark v2
```

## Maintenance""",
    ),
    (
        "code",
        """# Optional: show the log tail, or stop training (the last save stays on Drive).
print(Path(paths.run / "train.log").read_text()[-3000:] if (paths.run / "train.log").exists() else "no log yet")
# stop(paths)""",
    ),
]


def main() -> None:
    cells = []
    for kind, source in CELLS:
        source = source.replace("@REPO@", REPO)
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "L4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = Path("notebooks/jevlet_colab.ipynb")
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
