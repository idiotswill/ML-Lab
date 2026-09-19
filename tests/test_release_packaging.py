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
    assert payload["build_kind"] == "development_portable"
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



def test_windows_package_records_installer_and_recipe_hashes(tmp_path: Path) -> None:
    dist, lock = _build_fixture(tmp_path)
    recipe = tmp_path / "MLLab.iss"
    recipe.write_text("PrivilegesRequired=lowest\n", encoding="utf-8")
    installer = tmp_path / "MLLab-Setup-0.1.0.dev0-x64.exe"
    installer.write_bytes(b"installer-binary")

    _, manifest = create_windows_portable(
        dist_dir=dist,
        output_dir=tmp_path / "out",
        source_commit="deadbeef",
        lock_path=lock,
        version="0.1.0.dev0",
        release_label="v0.1.0.dev0",
        python_version="3.12.10",
        qt_version="6.11.2",
        nuitka_version="2.8.9",
        installer_recipe=recipe,
        installer_path=installer,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["build_kind"] == "development_windows_package"
    assert payload["inputs"]["installer_recipe"]["path"] == recipe.as_posix()
    assert len(payload["inputs"]["installer_recipe"]["sha256"]) == 64
    assert payload["artifacts"]["installer_exe"]["filename"] == installer.name
    assert payload["artifacts"]["installer_exe"]["size"] == len(b"installer-binary")
    assert len(payload["artifacts"]["installer_exe"]["sha256"]) == 64


def test_windows_package_requires_installer_and_recipe_together(tmp_path: Path) -> None:
    dist, lock = _build_fixture(tmp_path)
    installer = tmp_path / "MLLab-Setup.exe"
    installer.write_bytes(b"installer")

    with pytest.raises(ValueError, match="supplied together"):
        create_windows_portable(
            dist_dir=dist,
            output_dir=tmp_path / "out",
            source_commit="abc123",
            lock_path=lock,
            version="0.1.0.dev0",
            release_label="v0.1.0.dev0",
            python_version="3.12.10",
            qt_version="6.11.2",
            nuitka_version="2.8.9",
            installer_path=installer,
        )


def test_installer_recipe_stays_per_user_and_x64_targeted() -> None:
    recipe = (
        Path(__file__).resolve().parents[1] / "packaging" / "windows" / "MLLab.iss"
    ).read_text(encoding="utf-8")

    assert "PrivilegesRequired=lowest" in recipe
    assert "DefaultDirName={localappdata}" in recipe
    assert "ArchitecturesAllowed=x64compatible" in recipe
    assert "ArchitecturesInstallIn64BitMode=x64compatible" in recipe


def test_windows_package_can_be_marked_testing_candidate_without_testing_ready(
    tmp_path: Path,
) -> None:
    dist, lock = _build_fixture(tmp_path)
    recipe = tmp_path / "MLLab.iss"
    recipe.write_text("PrivilegesRequired=lowest\n", encoding="utf-8")
    installer = tmp_path / "MLLab-Setup-v0.1.0-testing.1-x64.exe"
    installer.write_bytes(b"candidate-installer")

    _, manifest = create_windows_portable(
        dist_dir=dist,
        output_dir=tmp_path / "out",
        source_commit="candidate123",
        lock_path=lock,
        version="0.1.0-testing.1",
        release_label="v0.1.0-testing.1",
        python_version="3.12.10",
        qt_version="6.11.2",
        nuitka_version="2.8.9",
        installer_recipe=recipe,
        installer_path=installer,
        testing_candidate=True,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["build_kind"] == "testing_candidate_windows_package"
    assert payload["application"]["version"] == "0.1.0-testing.1"
    assert payload["application"]["release_label"] == "v0.1.0-testing.1"
    assert payload["testing_ready"] is False
    assert payload["integration_gate"] == "NO_GO"
