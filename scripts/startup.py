"""Install or remove Jevlet's per-user Windows sign-in launcher."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MARKER = ":: Jevlet user-session startup shortcut"
FILENAME = "Jevlet Desktop.cmd"


def launcher_text(project: Path, pythonw: Path) -> str:
    for path in (project, pythonw):
        if '"' in str(path) or "\n" in str(path):
            raise ValueError("unsafe character in launcher path")
    return f'{MARKER}\n@echo off\ncd /d "{project}"\nstart "" "{pythonw}" -m jevlet.app\n'


def install(startup_dir: Path, project: Path, pythonw: Path) -> Path:
    if not pythonw.is_file():
        raise FileNotFoundError(f"pythonw.exe not found: {pythonw}")
    startup_dir.mkdir(parents=True, exist_ok=True)
    target = startup_dir / FILENAME
    if target.exists() and not target.read_text(encoding="utf-8").startswith(MARKER):
        raise FileExistsError("refusing to overwrite an unrelated Startup launcher")
    target.write_text(launcher_text(project, pythonw), encoding="utf-8")
    return target


def remove(startup_dir: Path) -> bool:
    target = startup_dir / FILENAME
    if not target.exists():
        return False
    if not target.read_text(encoding="utf-8").startswith(MARKER):
        raise ValueError("refusing to remove an unrelated Startup launcher")
    target.unlink()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--install", action="store_true")
    choice.add_argument("--remove", action="store_true")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("Startup launcher requires Windows")
    startup_dir = (
        Path(os.environ["APPDATA"])
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Startup"
    )
    if args.remove:
        print("Removed Jevlet startup launcher" if remove(startup_dir) else "No launcher installed")
        return
    project = Path(__file__).resolve().parents[1]
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    print(f"Installed {install(startup_dir, project, pythonw)}")


if __name__ == "__main__":
    main()
