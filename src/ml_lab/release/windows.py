from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from importlib import metadata
from pathlib import Path
from typing import Sequence

from ml_lab import __version__

_MANIFEST_SCHEMA_VERSION = 1
_FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_RELEASE_LABEL = re.compile(r"^[A-Za-z0-9._-]+$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _standalone_files(dist_dir: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for path in sorted(dist_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Portable package refuses symbolic link: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(dist_dir).as_posix()
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not files:
        raise ValueError(f"Standalone directory is empty: {dist_dir}")
    return files


def _tree_digest(files: Sequence[dict[str, object]]) -> str:
    encoded = json.dumps(
        list(files),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_deterministic_zip(
    dist_dir: Path,
    archive_path: Path,
    files: Sequence[dict[str, object]],
) -> None:
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for item in files:
            relative = str(item["path"])
            source = dist_dir / Path(relative)
            info = zipfile.ZipInfo(f"MLLab/{relative}", date_time=_FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with source.open("rb") as source_handle, archive.open(info, "w") as target:
                shutil.copyfileobj(source_handle, target, length=1024 * 1024)


def create_windows_portable(
    *,
    dist_dir: Path,
    output_dir: Path,
    source_commit: str,
    lock_path: Path,
    version: str,
    release_label: str,
    python_version: str,
    qt_version: str,
    nuitka_version: str,
    installer_recipe: Path | None = None,
) -> tuple[Path, Path]:
    if not source_commit.strip():
        raise ValueError("source_commit is required")
    if not _RELEASE_LABEL.fullmatch(release_label):
        raise ValueError("release_label may contain only letters, digits, '.', '_' and '-'")
    if not dist_dir.is_dir():
        raise ValueError(f"Standalone directory does not exist: {dist_dir}")
    if not lock_path.is_file():
        raise ValueError(f"Dependency lock does not exist: {lock_path}")
    if installer_recipe is not None and not installer_recipe.is_file():
        raise ValueError(f"Installer recipe does not exist: {installer_recipe}")

    output_dir.mkdir(parents=True, exist_ok=True)
    files = _standalone_files(dist_dir)
    tree_sha256 = _tree_digest(files)

    archive_name = f"MLLab-{release_label}-windows-x64-portable.zip"
    manifest_name = f"MLLab-{release_label}-build-manifest.json"
    archive_path = output_dir / archive_name
    manifest_path = output_dir / manifest_name

    _write_deterministic_zip(dist_dir, archive_path, files)
    archive_sha256 = _sha256_file(archive_path)

    installer_recipe_value: dict[str, object] | None = None
    if installer_recipe is not None:
        installer_recipe_value = {
            "path": installer_recipe.as_posix(),
            "sha256": _sha256_file(installer_recipe),
        }

    manifest: dict[str, object] = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "build_kind": "development_portable",
        "testing_ready": False,
        "integration_gate": "NO_GO",
        "application": {
            "name": "Frankenhomie ML Lab",
            "version": version,
            "release_label": release_label,
            "platform": "windows-x64",
        },
        "source": {
            "commit": source_commit,
        },
        "toolchain": {
            "python": python_version,
            "qt_pyside6": qt_version,
            "nuitka": nuitka_version,
        },
        "inputs": {
            "dependency_lock": {
                "path": lock_path.as_posix(),
                "sha256": _sha256_file(lock_path),
            },
            "installer_recipe": installer_recipe_value,
        },
        "standalone_tree": {
            "file_count": len(files),
            "sha256": tree_sha256,
            "files": files,
        },
        "artifacts": {
            "portable_zip": {
                "filename": archive_name,
                "size": archive_path.stat().st_size,
                "sha256": archive_sha256,
            }
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return archive_path, manifest_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the Windows standalone build.")
    parser.add_argument("--dist", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--release-label", default=f"v{__version__}")
    parser.add_argument("--installer-recipe", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    archive_path, manifest_path = create_windows_portable(
        dist_dir=args.dist,
        output_dir=args.output,
        source_commit=args.source_commit,
        lock_path=args.lock,
        version=__version__,
        release_label=args.release_label,
        python_version=sys.version.split()[0],
        qt_version=metadata.version("PySide6"),
        nuitka_version=metadata.version("Nuitka"),
        installer_recipe=args.installer_recipe,
    )
    print(archive_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
