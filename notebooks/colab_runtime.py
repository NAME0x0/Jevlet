"""Resumable Colab runtime helpers; no CUDA driver installation or hardcoded tokens."""

from __future__ import annotations

import hashlib
import shutil
import threading
from pathlib import Path
from typing import Any


def amp_dtype(torch_module: Any) -> str:
    if not torch_module.cuda.is_available():
        return "none"
    return "bf16" if torch_module.cuda.is_bf16_supported() else "fp16"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointSync:
    """Periodically save checkpoint files to Drive or a private HF model repo.

    The local training directory stays under /content for speed. Only metadata and
    checkpoints are synced; public dataset contents are not uploaded by default.
    """

    def __init__(
        self,
        local_root: str | Path,
        *,
        drive_root: str | Path | None = None,
        hf_repo_id: str | None = None,
        interval_seconds: int = 180,
    ) -> None:
        if (drive_root is None) == (hf_repo_id is None):
            raise ValueError("select exactly one of drive_root or hf_repo_id")
        self.local_root = Path(local_root)
        self.drive_root = Path(drive_root) if drive_root else None
        self.hf_repo_id = hf_repo_id
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.errors: list[str] = []
        self.synced: dict[str, str] = {}
        self._api = None
        if hf_repo_id:
            from huggingface_hub import HfApi

            self._api = HfApi()
            info = self._api.repo_info(hf_repo_id, repo_type="model")
            if not info.private:
                raise ValueError("HF checkpoint repo must be private")

    def _files(self) -> list[Path]:
        names = {
            "last.pt",
            "best.pt",
            "metrics.json",
            "config.json",
            "summary.json",
            "dataset_manifest.json",
        }
        return sorted(
            path
            for path in self.local_root.rglob("*")
            if path.is_file() and (path.name in names or path.name.endswith(".jsonl"))
        )

    def sync_once(self) -> dict[str, str]:
        completed: dict[str, str] = {}
        for path in self._files():
            relative = path.relative_to(self.local_root)
            digest = _sha256(path)
            key = relative.as_posix()
            if self.synced.get(key) == digest:
                continue
            if self.drive_root is not None:
                destination = self.drive_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".uploading")
                shutil.copy2(path, temporary)
                if _sha256(temporary) != digest:
                    raise RuntimeError(f"checkpoint copy failed integrity check: {relative}")
                temporary.replace(destination)
            else:
                assert self._api is not None and self.hf_repo_id is not None
                self._api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=key,
                    repo_id=self.hf_repo_id,
                    repo_type="model",
                    commit_message=f"Sync Jevlet {key}",
                )
            completed[key] = digest
            self.synced[key] = digest
        return completed

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("sync already started")

        def worker() -> None:
            while not self._stop.wait(self.interval_seconds):
                try:
                    self.sync_once()
                except Exception as error:  # keep training alive; surface failure at stop
                    self.errors.append(repr(error))

        self._thread = threading.Thread(target=worker, name="jevlet-checkpoint-sync", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, str]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        final = self.sync_once()
        if self.errors:
            raise RuntimeError(f"intermediate checkpoint sync failed: {self.errors}")
        return final


def restore_checkpoint(
    relative: str,
    local_root: str | Path,
    *,
    drive_root: str | Path | None = None,
    hf_repo_id: str | None = None,
) -> Path | None:
    """Restore one last.pt from configured persistence, if present."""
    local = Path(local_root) / relative
    if drive_root is not None:
        remote = Path(drive_root) / relative
        if not remote.exists():
            return None
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote, local)
        if _sha256(local) != _sha256(remote):
            raise RuntimeError("restored checkpoint checksum mismatch")
        return local
    if hf_repo_id is not None:
        from huggingface_hub import hf_hub_download

        try:
            downloaded = hf_hub_download(hf_repo_id, filename=relative, repo_type="model")
        except Exception as error:
            # Missing checkpoint is normal on first run; auth/network errors are not.
            from huggingface_hub.errors import EntryNotFoundError

            if isinstance(error, EntryNotFoundError):
                return None
            raise
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, local)
        return local
    raise ValueError("select Drive or HF persistence")
