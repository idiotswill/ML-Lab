from __future__ import annotations

import hashlib
from pathlib import Path

from ml_lab.adapters.phase_a_expanded_freeze import freeze_expanded_phase_a
from ml_lab.adapters.phase_a_protected_evidence import build_phase_a_protected_evidence
from ml_lab.datasets.leakage import canonical_json
from ml_lab.storage.workspace import Workspace


def complete_expanded_phase_a(
    *,
    expanded_root: Path,
    frankenhomie_repository: Path,
) -> dict[str, object]:
    root = expanded_root.expanduser().resolve()
    working_workspace_path = root / "working-workspace"
    authored_root = root / "authored"
    protected_root = root / "protected"
    frozen_workspace = root / "frozen-workspace"
    cache_root = root / "case-cache"

    workspace = Workspace.open(working_workspace_path)
    protected = build_phase_a_protected_evidence(
        workspace=workspace,
        frankenhomie_repository=frankenhomie_repository,
        authored_root=authored_root,
        output_dir=protected_root,
        case_cache_dir=cache_root,
    )

    frozen: dict[str, object] | None = None
    if protected.get("ok") is True:
        frozen = freeze_expanded_phase_a(
            frankenhomie_repository=frankenhomie_repository,
            expanded_root=root,
            protected_root=protected_root,
            workspace_path=frozen_workspace,
        )

    base: dict[str, object] = {
        "schema": "ml-lab-phase-a-expanded-completion/1",
        "ok": protected.get("ok") is True and frozen is not None and frozen.get("ok") is True,
        "expanded_root": str(root),
        "protected_payload_sha256": protected.get("payload_sha256"),
        "protected_residual_case_count": protected.get("residual_case_count"),
        "protected_residual_emitted_count": protected.get("residual_emitted_count"),
        "zero_model_case_count": protected.get("zero_model_case_count"),
        "zero_model_passed_count": protected.get("zero_model_passed_count"),
        "protected_errors": protected.get("errors"),
        "frozen_payload_sha256": frozen.get("payload_sha256") if frozen else None,
        "frozen_dataset_example_count": (
            frozen.get("dataset", {}).get("example_count")
            if isinstance(frozen, dict) and isinstance(frozen.get("dataset"), dict)
            else None
        ),
        "frozen": frozen is not None and frozen.get("frozen") is True,
        "model_training_started": False,
        "integration_gate": "NO_GO",
    }
    payload_sha = hashlib.sha256(canonical_json(base).encode()).hexdigest()
    receipt = {**base, "payload_sha256": payload_sha}
    (root / "expanded-completion-receipt.json").write_bytes(
        (canonical_json(receipt) + "\n").encode("utf-8")
    )
    return receipt
