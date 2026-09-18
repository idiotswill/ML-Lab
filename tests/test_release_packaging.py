from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from ml_lab.release.windows import create_windows_portable


def _build_fixture(root: Path) -> tuple[Path, Path]:
    dist = root / "main.dist"
    (dist / "ml_lab" / "ui" / "qml").mkdir(parents=True)
    (dist / "MLLab.exe").write_bytes(b"compiled-executable")
    (dist / "ml_lab" / "ui" / "qml" / "Main.qml").write_text(
        "import QtQuick\n",
        encoding="utf-8",
    )
    lock = root / "uv.lock"
    lock.write_text("version = 1\n", encoding="utf-8")
    return dist, lock


def test_windows_portable_is_deterministic_and_truthful(tmp_path: Path) -> None:
    dist, lock = _build_fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    archive_one, manifest_one = create_windows_portable(
        dist_dir=dist,
        output_dir=first,
        source_commit="abc123",
        lock_path=lock,
        version="0.1.0.dev0",
        release_label="v0.1.0.dev0",
        python_version="3.12.10",
        qt_version="6.11.2",
        nuitka_version="2.8.9",
    )
    archive_two, manifest_two = create_windows_portable(
        dist_dir=dist,
        output_dir=second,
        source_commit="abc123",
        lock_path=lock,
        version="0.1.0.dev0",
        release_label="v0.1.0.dev0",
        python_version="3.12.10",
        qt_version="6.11.2",
        nuitka_version="2.8.9",
    )

    assert archive_one.read_bytes() == archive_two.read_bytes()
    assert manifest_one.read_bytes() == manifest_two.read_bytes()

    payload = json.loads(manifest_one.read_text(encoding="utf-8"))
    assert payload["testing_ready"] is False
    assert payload["integration_gate"] == "NO_GO"
    assert payload["source"]["commit"] == "abc123"
    assert payload["inputs"]["installer_recipe"] is None
    assert payload["standalone_tree"]["file_count"] == 2
    assert payload["artifacts"]["portable_zip"]["filename"] == archive_one.name
    assert len(payload["artifacts"]["portable_zip"]["sha256"]) == 64
    assert len(payload["inputs"]["dependency_lock"]["sha256"]) == 64

    with zipfile.ZipFile(archive_one) as packaged:
        assert packaged.namelist() == [
            "MLLab/MLLab.exe",
            "MLLab/ml_lab/ui/qml/Main.qml",
        ]
        assert packaged.read("MLLab/MLLab.exe") == b"compiled-executable"


def test_windows_portable_rejects_unsafe_release_label(tmp_path: Path) -> None:
    dist, lock = _build_fixture(tmp_path)
    with pytest.raises(ValueError, match="release_label"):
        create_windows_portable(
            dist_dir=dist,
            output_dir=tmp_path / "out",
            source_commit="abc123",
            lock_path=lock,
            version="0.1.0.dev0",
            release_label="../testing-ready",
            python_version="3.12.10",
            qt_version="6.11.2",
            nuitka_version="2.8.9",
        )
