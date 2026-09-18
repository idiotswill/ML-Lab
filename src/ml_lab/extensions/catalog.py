from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml_lab.adapters.phase_a import PHASE_A_ADAPTER_ID, PHASE_A_ADAPTER_VERSION
from ml_lab.extensions.manifests import ExtensionManifest, ManifestError
from ml_lab.extensions.registry import ManifestRegistry
from ml_lab.trainers.service import SPARSE_RUNTIME_PACK_ID, SPARSE_TRAINER_ID

BUILTIN_ADAPTERS = (
    {
        "id": "generic",
        "version": "1.0.0",
        "protocol_version": 1,
        "display_name": "Generic ML Project",
        "capabilities": ["project-host"],
        "config_schema": {"type": "object", "additionalProperties": False},
    },
    {
        "id": PHASE_A_ADAPTER_ID,
        "version": PHASE_A_ADAPTER_VERSION,
        "protocol_version": 1,
        "display_name": "Frankenhomie Phase A — Residual Semantics",
        "capabilities": [
            "contract-snapshot",
            "dataset-validation",
            "bounded-semantic-evaluation",
            "reference-corpus",
        ],
        "config_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "frankenhomie_repository": {"type": "string"},
                "frankenhomie_ref": {"type": "string", "default": "master"},
            },
        },
    },
)

BUILTIN_RUNTIMES = (
    {
        "id": "builtin-core",
        "version": "1.0.0",
        "protocol_version": 1,
        "display_name": "ML Lab Core Runtime",
        "capabilities": ["core.self_test", "core.hash_file"],
        "config_schema": {"type": "object", "additionalProperties": False},
    },
    {
        "id": SPARSE_RUNTIME_PACK_ID,
        "version": "1.0.0",
        "protocol_version": 1,
        "display_name": "Built-in Sparse CPU Runtime",
        "capabilities": [SPARSE_TRAINER_ID],
        "config_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "feature_dim": {"type": "integer", "minimum": 256},
                "alpha": {"type": "number", "exclusiveMinimum": 0},
                "text_key": {"type": "string"},
                "label_key": {"type": "string"},
            },
        },
    },
)


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
        for payload in BUILTIN_ADAPTERS:
            self.adapters.register(
                ExtensionManifest.from_dict(payload, expected_kind="adapter")
            )
        for payload in BUILTIN_RUNTIMES:
            self.runtimes.register(
                ExtensionManifest.from_dict(payload, expected_kind="runtime")
            )
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
