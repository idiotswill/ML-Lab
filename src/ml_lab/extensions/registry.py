from __future__ import annotations

import json
from pathlib import Path

from ml_lab.extensions.manifests import ExtensionManifest


class ManifestRegistry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[tuple[str, str], ExtensionManifest] = {}

    def register(self, manifest: ExtensionManifest) -> None:
        if manifest.kind != self.kind:
            raise ValueError(f"Expected {self.kind} manifest, got {manifest.kind}.")
        key = (manifest.extension_id, manifest.version)
        if key in self._items:
            raise ValueError(f"Duplicate {self.kind} manifest {key[0]} {key[1]}.")
        self._items[key] = manifest

    def load_file(self, path: Path) -> ExtensionManifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = ExtensionManifest.from_dict(payload, expected_kind=self.kind)
        self.register(manifest)
        return manifest

    def list(self) -> list[ExtensionManifest]:
        return sorted(self._items.values(), key=lambda item: (item.display_name.casefold(), item.version))
