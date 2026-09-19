from pathlib import Path

from ml_lab.release.smoke import (
    prepare_clean_machine_smoke,
    reopen_clean_machine_smoke,
)


def test_clean_machine_smoke_persists_and_fresh_verifies(tmp_path: Path) -> None:
    workspace = tmp_path / "ML Lab Ünicode Workspace"
    prepared = prepare_clean_machine_smoke(workspace)
    assert prepared["ok"] is True
    assert prepared["stage"] == "prepare"

    reopened = reopen_clean_machine_smoke(workspace)
    assert reopened["ok"] is True
    assert reopened["stage"] == "reopen"
    assert reopened["fresh_verification"] == "PASS"
    assert reopened["integration_gate"] == "NO_GO"
    assert reopened["bundle_sha256"] == prepared["bundle_sha256"]
