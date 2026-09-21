from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import ml_lab.adapters.phase_a_first_experiment as runner
from ml_lab.core.models import (
    DatasetSplit,
    JobRecord,
    JobStatus,
    MetricDirection,
    utc_now_iso,
)
from ml_lab.datasets.leakage import canonical_json
from ml_lab.datasets.service import DatasetService, ValidatedExampleInput
from ml_lab.evaluation.phase_a_metrics import PhaseAMetricValue
from ml_lab.evaluation.service import EvaluationSummary
from ml_lab.experiments.service import ExperimentService
from ml_lab.storage.workspace import Workspace
from ml_lab.trainers.service import TrainingState

TARGET_COMMIT = "1ff3e2155a3c2d2b316023e9933edf3a6d53697f"


def _setup_frozen_workspace(tmp_path: Path) -> tuple[Workspace, dict[str, object]]:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project(
        "Phase A Semantic ML",
        "frankenhomie.phase-a-residual",
    )
    manifest = workspace.artifacts.commit_bytes(
        b'{"format_version":1}\n',
        media_type="application/vnd.ml-lab.contract-manifest+json",
        metadata={"kind": "test-contract"},
    )
    snapshot_id = "snapshot-1"
    with workspace.database.transaction() as conn:
        conn.execute(
            "INSERT INTO contract_snapshots"
            "(id,project_id,adapter_id,adapter_version,repo_path,repo_identity,commit_sha,"
            "contract_version,compatibility_signature,manifest_artifact_digest,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                snapshot_id,
                project.id,
                "frankenhomie.phase-a-residual",
                "1.0.0",
                str(tmp_path / "frankenhomie"),
                "test-repo",
                TARGET_COMMIT,
                "semantic-residual-v2",
                "a" * 64,
                manifest.digest,
                utc_now_iso(),
            ),
        )

    datasets = DatasetService(workspace)
    dataset = datasets.create(
        project.id,
        "Phase A frozen",
        contract_snapshot_id=snapshot_id,
    )
    for split, example_id, text in (
        (DatasetSplit.TRAIN, "train-1", "synthetic train sentence"),
        (DatasetSplit.DEV, "dev-1", "synthetic dev sentence"),
    ):
        datasets.add_example(
            dataset.id,
            ValidatedExampleInput(
                example_id=example_id,
                split=split,
                source_id=f"synthetic:{example_id}",
                lineage_group=f"lineage:{example_id}",
                payload={"request": {"text": text}},
                label={"decision": "ASK_PLAYER"},
                tags=("synthetic",),
            ),
        )
    frozen = datasets.freeze(dataset.id)
    handles = datasets.trainer_partition_handles(dataset.id)
    partitions = {
        item.split.value: {
            "artifact_digest": item.artifact_digest,
            "sha256": item.partition_sha256,
            "example_count": item.example_count,
        }
        for item in datasets.partitions(dataset.id)
    }

    base: dict[str, object] = {
        "schema": "ml-lab-phase-a-freeze-receipt/1",
        "ok": True,
        "frozen": True,
        "project_id": project.id,
        "dataset": {
            "id": dataset.id,
            "state": frozen.state.value,
            "example_count": frozen.example_count,
            "partitions": partitions,
        },
        "contract_snapshot": {"id": snapshot_id},
        "target_frankenhomie_commit": TARGET_COMMIT,
        "target_contract": "semantic-residual-v2",
        "trainer_handles": handles,
        "trainer_visible_splits": ["DEV", "TRAIN"],
        "protected_splits_in_trainer_handles": False,
        "ordinary_dataset_service": True,
        "ordinary_contract_snapshot_service": True,
        "model_training_started": False,
        "production_data": False,
        "transcript_derived": False,
        "integration_gate": "NO_GO",
    }
    payload_sha = hashlib.sha256(canonical_json(base).encode("utf-8")).hexdigest()
    receipt = {**base, "payload_sha256": payload_sha}
    (workspace.root / "phase-a-freeze-receipt.json").write_bytes(
        (canonical_json(receipt) + "\n").encode("utf-8")
    )
    return workspace, receipt


class _FakeTrainingService:
    job: JobRecord | None = None

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def launch(self, experiment_id: str) -> TrainingState:
        experiments = ExperimentService(self.workspace)
        experiments.start(experiment_id)
        model = self.workspace.artifacts.commit_bytes(
            b'{"kind":"phase-a-bounded-sparse-v1"}\n',
            media_type="application/vnd.ml-lab.phase-a-bounded-sparse+json",
            metadata={"kind": "test-model"},
        )
        completed = experiments.complete(
            experiment_id,
            metrics=[],
            model_artifact_digest=model.digest,
        )
        result = self.workspace.artifacts.commit_bytes(
            b'{"ok":true}\n',
            media_type="application/json",
            metadata={"kind": "test-training-result"},
        )
        now = utc_now_iso()
        job = JobRecord(
            id="job-1",
            task_type="trainer.phase_a_sparse.v1",
            status=JobStatus.COMPLETED,
            created_at=now,
            updated_at=now,
            progress=1.0,
            message="Completed",
            staging_dir=self.workspace.root / "jobs" / "job-1",
            result_artifact_digest=result.digest,
        )
        type(self).job = job
        return TrainingState(completed, job)

    def refresh_job(self, experiment_id: str, job_id: str) -> TrainingState:
        assert job_id == "job-1"
        assert self.job is not None
        return TrainingState(
            ExperimentService(self.workspace).get(experiment_id),
            self.job,
        )

    def cancel_job(self, experiment_id: str, job_id: str) -> TrainingState:
        raise AssertionError((experiment_id, job_id))

    def job_for_experiment(self, experiment_id: str) -> JobRecord:
        assert ExperimentService(self.workspace).get(experiment_id)
        assert self.job is not None
        return self.job

    def shutdown(self) -> None:
        pass


def _metric(metric_id: str, value: float | None, veto: bool = False) -> PhaseAMetricValue:
    return PhaseAMetricValue(
        metric_id=metric_id,
        label=metric_id,
        value=value,
        direction=MetricDirection.ZERO if veto else MetricDirection.HIGHER,
        veto=veto,
    )


def test_first_sparse_experiment_is_train_only_and_emits_dev_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, _receipt = _setup_frozen_workspace(tmp_path)
    monkeypatch.setattr(runner, "TrainingService", _FakeTrainingService)

    def fake_eval(
        _workspace: Workspace,
        experiment_id: str,
        *,
        splits,
    ):
        assert tuple(splits) == (DatasetSplit.DEV,)
        return (
            EvaluationSummary(
                experiment_id=experiment_id,
                split=DatasetSplit.DEV,
                total=1,
                correct=0,
                failures=1,
                veto_failures=0,
                mean_latency_ms=1.5,
                p95_latency_ms=1.5,
            ),
        )

    monkeypatch.setattr(runner, "evaluate_phase_a_sparse_experiment", fake_eval)
    monkeypatch.setattr(
        runner,
        "summarize_phase_a_experiment",
        lambda *_args, **_kwargs: (
            _metric("useful_resolution_coverage", 0.0),
            _metric("false_commitments", 0.0, veto=True),
            _metric("contract_failures", 0.0, veto=True),
        ),
    )

    result = runner.run_phase_a_first_sparse_experiment(
        workspace_path=workspace.root,
    )

    assert result["ok"] is True
    assert result["stage"] == "EXPERIMENT"
    assert result["promotion_allowed"] is False
    assert result["integration_gate"] == "NO_GO"
    assert result["dev_was_used_for_training"] is False
    assert result["test_was_used_for_training"] is False
    assert result["redteam_was_used_for_training"] is False
    assert set(result["training_partition_handles"]) == {"TRAIN"}
    assert result["experiment"]["status"] == "COMPLETED"
    assert result["dev_evaluation"]["total"] == 1
    assert result["veto_failure_count"] == 0
    assert (workspace.root / runner.EXPERIMENT_RECEIPT_NAME).is_file()


def test_first_sparse_experiment_rejects_tampered_freeze_receipt(
    tmp_path: Path,
) -> None:
    workspace, receipt = _setup_frozen_workspace(tmp_path)
    receipt["trainer_visible_splits"] = ["TRAIN"]
    (workspace.root / "phase-a-freeze-receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        runner.run_phase_a_first_sparse_experiment(
            workspace_path=workspace.root,
        )
