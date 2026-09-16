from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml_lab.extensions.manifests import ExtensionManifest, ManifestError
from ml_lab.extensions.registry import ManifestRegistry

BUILTIN_ADAPTER = {
    "id": "generic",
    "version": "1.0.0",
    "protocol_version": 1,
    "display_name": "Generic ML Project",
    "capabilities": ["project-host"],
    "config_schema": {"type": "object", "additionalProperties": False},
}

BUILTIN_RUNTIME = {
    "id": "builtin-core",
    "version": "1.0.0",
    "protocol_version": 1,
    "display_name": "ML Lab Core Runtime",
    "capabilities": ["core.self_test", "core.hash_file"],
    "config_schema": {"type": "object", "additionalProperties": False},
}


class ExtensionCatalog:
    """Manifest-only extension discovery; no arbitrary plugin code is imported."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.adapters = ManifestRegistry("adapter")
        self.runtimes = ManifestRegistry("runtime")
        self.errors: list[str] = []

    def load(self) -> None:
        self.adapters = ManifestRegistry("adapter")
        self.runtimes = ManifestRegistry("runtime")
        self.errors.clear()
        self.adapters.register(ExtensionManifest.from_dict(BUILTIN_ADAPTER, expected_kind="adapter"))
        self.runtimes.register(ExtensionManifest.from_dict(BUILTIN_RUNTIME, expected_kind="runtime"))
        self._scan(self.workspace_root / "extensions" / "adapters", self.adapters)
        self._scan(self.workspace_root / "extensions" / "runtimes", self.runtimes)

    def adapter_options(self) -> list[dict[str, str]]:
        return [
            {"id": item.extension_id, "name": item.display_name, "version": item.version}
            for item in self.adapters.list()
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "adapters": len(self.adapters.list()),
            "runtimes": len(self.runtimes.list()),
            "errors": list(self.errors),
        }

    def _scan(self, directory: Path, registry: ManifestRegistry) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for path in sorted(directory.glob("*.json")):
            try:
                registry.load_file(path)
            except (OSError, json.JSONDecodeError, ManifestError, ValueError) as exc:
                self.errors.append(f"{path.name}: {exc}")
