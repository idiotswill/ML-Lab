from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import ml_lab.adapters.phase_a_freeze as freeze_module
from ml_lab.adapters.phase_a_freeze import freeze_phase_a_train_dev
from ml_lab.contracts.snapshot import ContractFileSpec
from ml_lab.core.models import ContractSnapshot, DatasetSplit, utc_now_iso


def _request(declaration: str) -> dict[str, object]:
    return {
        "version": "semantic-residual-v2",
        "request_id": "semantic:test",
        "assessment": {
            "version": "input-assessment-v1",
            "declaration": declaration,
            "route": "SEMANTIC_REQUIRED",
            "model_required_because": "COMMITMENT:UNRESOLVED_DECLARATION",
            "mechanics_authorized": False,
            "unsupported_family": None,
            "candidates": [
                {
                    "entity_key": "object:hatch",
                    "name": "Brass Hatch",
                    "aliases": [],
                    "source_fact_keys": ["campaign.entity.object.hatch"],
                    "focus_rank": None,
                }
            ],
            "clauses": [
                {
                    "index": 0,
                    "raw": declaration,
                    "commitment": "UNRESOLVED_DECLARATION",
                    "family": "UNKNOWN",
                    "mention": None,
                    "entity_key": None,
                    "evidence": [],
                }
            ],
            "trace": [],
        },
        "context": {
            "version": "semantic-context-v1",
            "actor_id": "tester",
            "audience": "TABLE",
            "snapshot_revision": "test",
            "scene_facts": [],
            "discourse_facts": [],
            "campaign_facts": [],
            "combat_facts": [],
            "inventory_facts": [],
            "rules_facts": [],
            "candidates": [
                {
                    "entity_key": "object:hatch",
                    "name": "Brass Hatch",
                    "aliases": [],
                    "source_fact_keys": ["campaign.entity.object.hatch"],
                    "focus_rank": None,
                }
            ],
            "context": None,
        },
        "failed_deterministic_stage": "COMMITMENT:UNRESOLVED_DECLARATION",
        "allowed_action_families": ["SEARCH_INSPECT"],
        "family_slots": [
            {
                "family": "SEARCH_INSPECT",
                "slots": [
                    {
                        "name": "SUBJECT",
                        "required": True,
                        "allowed_values": ["object:hatch"],
                        "prebound": None,
                    }
                ],
            }
        ],
        "clarification_answer": None,
        "previous_question": None,
        "previous_reason_code": None,
        "unresolved_slots": [],
        "question_history": [],
        "selected_primary_action": None,
        "permitted_decisions": ["RESOLVE", "ASK_PLAYER"],
        "instruction": "INTERPRET_ENGLISH_ONLY_WITHIN_ENVELOPE",
    }


def _row() -> dict[str, object]:
    return {
        "example_id": "train-one",
        "split": "TRAIN",
        "source_id": "synthetic:test",
        "lineage_group": "train/one",
        "request": _request("I appraise Brass Hatch."),
        "expected": {
            "version": "semantic-residual-v2",
            "decision": "RESOLVE",
            "action_family": "SEARCH_INSPECT",
            "entity_keys": [],
            "slots": [{"name": "SUBJECT", "value": "object:hatch"}],
            "reason_code": "SYNTHETIC_AUTHOR_LABEL",
            "facts_used": [],
            "question": None,
            "missing_slots": [],
            "candidate_keys": [],
        },
        "tags": ["synthetic"],
    }


def _write_candidate(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    dataset_path = root / "phase-a-train-dev-v1.jsonl"
    dataset_bytes = (json.dumps(_row(), sort_keys=True, separators=(",", ":")) + "\n").encode()
    dataset_path.write_bytes(dataset_bytes)
    base = {
        "schema": "ml-lab-phase-a-dataset-factory-receipt/1",
        "ok": True,
        "candidate_training_data": True,
        "case_count": 1,
        "emitted_count": 1,
        "split_counts": {"TRAIN": 1, "DEV": 0},
        "dataset_file": dataset_path.name,
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "errors": [],
        "frozen": False,
        "full_payload_leakage": {"blocking_count": 0},
        "protected_language_leakage": {"blocking_count": 0},
        "production_data": False,
        "transcript_derived": False,
        "protected_splits_in_output": False,
        "trainer_visible_splits": ["TRAIN", "DEV"],
        "target_contract": "semantic-residual-v2",
        "target_frankenhomie_commit": "1" * 40,
        "integration_gate": "NO_GO",
    }
    payload_sha = hashlib.sha256(
        freeze_module.canonical_json(base).encode("utf-8")
    ).hexdigest()
    receipt = {**base, "payload_sha256": payload_sha}
    receipt_path = root / "factory-receipt.json"
    receipt_path.write_bytes(
        (freeze_module.canonical_json(receipt) + "\n").encode("utf-8")
    )
    return receipt_path, dataset_path


class _FakeContractSnapshotService:
    def __init__(self, workspace):
        self.workspace = workspace
        self._manifest: dict[str, object] | None = None

    def capture(
        self,
        *,
        project_id: str,
        adapter_id: str,
        adapter_version: str,
        repository: Path,
        ref: str,
        contract_version: str,
        files: tuple[ContractFileSpec, ...],
    ) -> ContractSnapshot:
        del repository, files
        created_at = utc_now_iso()
        manifest = {
            "working_tree_dirty_at_capture": True,
            "commit_sha": ref,
            "contract_version": contract_version,
        }
        artifact = self.workspace.artifacts.commit_bytes(
            (json.dumps(manifest, sort_keys=True) + "\n").encode(),
            media_type="application/json",
            metadata={"kind": "test-contract"},
        )
        snapshot = ContractSnapshot(
            id="snapshot-1",
            project_id=project_id,
            adapter_id=adapter_id,
            adapter_version=adapter_version,
            repo_path="fake",
            repo_identity="fake",
            commit_sha=ref,
            contract_version=contract_version,
            compatibility_signature="c" * 64,
            manifest_artifact_digest=artifact.digest,
            created_at=created_at,
        )
        with self.workspace.database.transaction() as conn:
            conn.execute(
                "INSERT INTO contract_snapshots"
                "(id,project_id,adapter_id,adapter_version,repo_path,repo_identity,"
                "commit_sha,contract_version,compatibility_signature,"
                "manifest_artifact_digest,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    snapshot.id,
                    snapshot.project_id,
                    snapshot.adapter_id,
                    snapshot.adapter_version,
                    snapshot.repo_path,
                    snapshot.repo_identity,
                    snapshot.commit_sha,
                    snapshot.contract_version,
                    snapshot.compatibility_signature,
                    snapshot.manifest_artifact_digest,
                    snapshot.created_at,
                ),
            )
        self._manifest = manifest
        return snapshot

    def manifest(self, _snapshot_id: str) -> dict[str, object]:
        assert self._manifest is not None
        return self._manifest


def test_freeze_uses_ordinary_dataset_service_and_hides_protected_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        freeze_module,
        "ContractSnapshotService",
        _FakeContractSnapshotService,
    )
    receipt_path, dataset_path = _write_candidate(tmp_path / "candidate")
    workspace = tmp_path / "freeze-workspace"

    result = freeze_phase_a_train_dev(
        frankenhomie_repository=tmp_path / "frankenhomie",
        factory_receipt_path=receipt_path,
        dataset_path=dataset_path,
        workspace_path=workspace,
    )

    assert result["ok"] is True
    assert result["frozen"] is True
    assert result["ordinary_dataset_service"] is True
    assert result["ordinary_contract_snapshot_service"] is True
    assert result["model_training_started"] is False
    assert result["integration_gate"] == "NO_GO"
    assert result["trainer_visible_splits"] == ["DEV", "TRAIN"]
    assert result["protected_splits_in_trainer_handles"] is False
    assert set(result["trainer_handles"]) == {"TRAIN", "DEV"}
    assert set(result["evaluation_handles"]) == {
        split.value for split in DatasetSplit
    }
    partitions = result["dataset"]["partitions"]
    assert partitions["TRAIN"]["example_count"] == 1
    assert partitions["DEV"]["example_count"] == 0
    assert partitions["TEST"]["example_count"] == 0
    assert partitions["REDTEAM"]["example_count"] == 0
    assert (workspace / "phase-a-freeze-receipt.json").is_file()


def test_freeze_rejects_physical_dataset_hash_mismatch(
    tmp_path: Path,
) -> None:
    receipt_path, dataset_path = _write_candidate(tmp_path / "candidate")
    dataset_path.write_bytes(dataset_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="SHA-256"):
        freeze_phase_a_train_dev(
            frankenhomie_repository=tmp_path / "frankenhomie",
            factory_receipt_path=receipt_path,
            dataset_path=dataset_path,
            workspace_path=tmp_path / "freeze-workspace",
        )
    assert not (tmp_path / "freeze-workspace").exists()


def test_freeze_rejects_factory_with_blocking_leakage(
    tmp_path: Path,
) -> None:
    receipt_path, dataset_path = _write_candidate(tmp_path / "candidate")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["full_payload_leakage"]["blocking_count"] = 1
    base = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    receipt["payload_sha256"] = hashlib.sha256(
        freeze_module.canonical_json(base).encode("utf-8")
    ).hexdigest()
    receipt_path.write_bytes(
        (freeze_module.canonical_json(receipt) + "\n").encode()
    )

    with pytest.raises(ValueError, match="blocking leakage"):
        freeze_phase_a_train_dev(
            frankenhomie_repository=tmp_path / "frankenhomie",
            factory_receipt_path=receipt_path,
            dataset_path=dataset_path,
            workspace_path=tmp_path / "freeze-workspace",
        )
