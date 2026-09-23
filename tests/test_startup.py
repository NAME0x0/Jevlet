from __future__ import annotations

import pytest

from scripts.startup import FILENAME, install, remove


def test_install_and_remove_only_owned_launcher(tmp_path) -> None:
    pythonw = tmp_path / "pythonw.exe"
    pythonw.write_bytes(b"placeholder")
    startup = tmp_path / "Startup"
    project = tmp_path / "Jevlet"
    target = install(startup, project, pythonw)
    assert target.name == FILENAME
    assert str(project) in target.read_text(encoding="utf-8")
    assert remove(startup)
    assert not target.exists()
    assert not remove(startup)


def test_unrelated_launcher_is_preserved(tmp_path) -> None:
    startup = tmp_path / "Startup"
    startup.mkdir()
    target = startup / FILENAME
    target.write_text("another program", encoding="utf-8")
    with pytest.raises(ValueError, match="unrelated"):
        remove(startup)
    assert target.read_text(encoding="utf-8") == "another program"
