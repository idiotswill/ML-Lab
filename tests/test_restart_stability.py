from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from ml_lab.core.models import TERMINAL_JOB_STATUSES, DatasetSplit, JobStatus, utc_now_iso
from ml_lab.datasets.service import DatasetService, ValidatedExampleInput
from ml_lab.evaluation.service import CaseOutcome, EvaluationService
from ml_lab.experiments.service import ExperimentService
from ml_lab.jobs.manager import JobManager
from ml_lab.jobs.protocol import JobSpec
from ml_lab.storage.workspace import Workspace

pytestmark = pytest.mark.scale


def _wait_terminal(manager: JobManager, job_id: str, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = manager.get(job_id)
        if record.status in TERMINAL_JOB_STATUSES:
            return record
        time.sleep(0.03)
    raise AssertionError("job did not finish")


def test_hash_restart_marks_interrupted_and_clean_retry_completes(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    source = tmp_path / "hash-source.bin"
    source.write_bytes(b"restart-safe-hash\n" * 131_072)
    expected = hashlib.sha256(source.read_bytes()).hexdigest()

    interrupted_id = "interrupted-hash"
    interrupted_staging = workspace.root / "jobs" / interrupted_id
    interrupted_staging.mkdir(parents=True)
    JobSpec(
        job_id=interrupted_id,
        task_type="core.hash_file",
        staging_dir=str(interrupted_staging),
        payload={"path": str(source)},
    ).write(interrupted_staging / "job_spec.json")
    (interrupted_staging / "events.jsonl").write_text(
        json.dumps(
            {
                "kind": "progress",
                "progress": 0.5,
                "message": "Hashing interrupted before application restart",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    now = utc_now_iso()
    with workspace.database.transaction() as conn:
        conn.execute(
            "INSERT INTO jobs("
            "id,task_type,status,progress,message,staging_dir,correlation_id,"
            "created_at,updated_at"
            ") VALUES(?,?,?,?,?,?,?,?,?)",
            (
                interrupted_id,
                "core.hash_file",
                JobStatus.RUNNING.value,
                0.5,
                "Running",
                str(interrupted_staging),
                "restart-hash",
                now,
                now,
            ),
        )

    reopened = Workspace.open(workspace.root)
    manager = JobManager(reopened.root, reopened.database, reopened.artifacts)
    try:
        assert manager.reconcile_startup() == 1
        interrupted = manager.get(interrupted_id)
        assert interrupted.status is JobStatus.INTERRUPTED
        assert interrupted.result_artifact_digest is None
        assert not (interrupted.staging_dir / "result_manifest.json").exists()

        retry = manager.start("core.hash_file", {"path": str(source)})
        finished = _wait_terminal(manager, retry.id)
        assert finished.status is JobStatus.COMPLETED
        assert finished.result_artifact_digest is not None
        result = json.loads(
            reopened.artifacts.resolve(finished.result_artifact_digest).read_text(
                encoding="utf-8"
            )
        )
        assert result["sha256"] == expected
        assert result["size_bytes"] == source.stat().st_size
        assert retry.id != interrupted_id
        assert manager.get(interrupted_id).status is JobStatus.INTERRUPTED
    finally:
        manager.shutdown()


def test_import_interruption_rolls_back_then_restart_retry_is_atomic(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Import restart")
    service = DatasetService(workspace)
    dataset = service.create(project.id, "draft")
    source = tmp_path / "dataset.jsonl"
    rows = [
        {
            "example_id": f"case-{index}",
            "split": split,
            "source_id": f"synthetic:{index}",
            "lineage_group": f"lineage:{index}",
            "payload": {"text": text},
            "label": {"class": label},
        }
        for index, (split, text, label) in enumerate(
            (
                ("TRAIN", "amber lighthouse cedar orbit", "A"),
                ("DEV", "violet canyon marble signal", "B"),
                ("TEST", "silver orchard comet harbor", "C"),
            ),
            start=1,
        )
    ]
    source.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    def interrupting_validator(
        row: dict[str, object],
        line_number: int,
        _source_name: str,
    ) -> ValidatedExampleInput:
        if line_number == 2:
            raise InterruptedError("simulated import interruption")
        return ValidatedExampleInput(
            example_id=str(row["example_id"]),
            split=DatasetSplit(str(row["split"])),
            source_id=str(row["source_id"]),
            lineage_group=str(row["lineage_group"]),
            payload=row["payload"],
            label=row["label"],
            tags=(),
        )

    with pytest.raises(InterruptedError, match="simulated import interruption"):
        service.import_jsonl(
            dataset.id,
            source,
            validator=interrupting_validator,
        )

    with workspace.database.connection() as conn:
        example_count = conn.execute(
            "SELECT COUNT(*) FROM dataset_examples WHERE dataset_id=?",
            (dataset.id,),
        ).fetchone()[0]
        import_count = conn.execute(
            "SELECT COUNT(*) FROM dataset_imports WHERE dataset_id=?",
            (dataset.id,),
        ).fetchone()[0]
    assert example_count == 0
    assert import_count == 0
    assert service.get(dataset.id).example_count == 0

    reopened = Workspace.open(workspace.root)
    restarted = DatasetService(reopened)
    summary = restarted.import_jsonl(dataset.id, source)
    assert summary.imported == 3
    assert summary.rejected == 0
    assert restarted.get(dataset.id).example_count == 3
    with reopened.database.connection() as conn:
        import_count = conn.execute(
            "SELECT COUNT(*) FROM dataset_imports WHERE dataset_id=?",
            (dataset.id,),
        ).fetchone()[0]
    assert import_count == 1


def test_evaluation_restart_resumes_only_missing_protected_cases(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Evaluation restart")
    datasets = DatasetService(workspace)
    dataset = datasets.create(project.id, "frozen")
    rows = (
        (DatasetSplit.TRAIN, "train-a", "training-only phrase", "TRAIN"),
        (DatasetSplit.TEST, "test-a", "first protected phrase", "A"),
        (DatasetSplit.TEST, "test-b", "second protected phrase", "B"),
    )
    for split, example_id, text, label in rows:
        datasets.add_example(
            dataset.id,
            ValidatedExampleInput(
                example_id=example_id,
                split=split,
                source_id=f"synthetic:{example_id}",
                lineage_group=f"lineage:{example_id}",
                payload={"text": text},
                label={"class": label},
                tags=(),
            ),
        )
    datasets.freeze(dataset.id)
    experiment = ExperimentService(workspace).create(
        project_id=project.id,
        dataset_id=dataset.id,
        trainer_id="test",
        runtime_pack_id="test",
    )
    first_service = EvaluationService(workspace)

    def interrupted(row: dict[str, object]) -> CaseOutcome:
        if row["example_id"] == "test-b":
            raise InterruptedError("simulated evaluation interruption")
        return CaseOutcome(observed=row["label"], correct=True, latency_ms=0.1)

    with pytest.raises(InterruptedError, match="simulated evaluation interruption"):
        first_service.evaluate_partition(
            experiment.id,
            DatasetSplit.TEST,
            interrupted,
        )

    before_restart = first_service.page_cases(experiment.id, DatasetSplit.TEST)
    assert [case.example_id for case in before_restart] == ["test-a"]
    first_case_id = before_restart[0].id

    reopened = Workspace.open(workspace.root)
    restarted = EvaluationService(reopened)
    progress = restarted.progress(experiment.id, DatasetSplit.TEST)
    assert progress.expected == 2
    assert progress.evaluated == 1
    assert not progress.complete

    resumed_calls: list[str] = []

    def resumed(row: dict[str, object]) -> CaseOutcome:
        resumed_calls.append(str(row["example_id"]))
        return CaseOutcome(observed=row["label"], correct=True, latency_ms=0.2)

    summary = restarted.evaluate_partition(
        experiment.id,
        DatasetSplit.TEST,
        resumed,
    )
    assert resumed_calls == ["test-b"]
    assert summary.total == 2
    assert summary.correct == 2
    after_restart = restarted.page_cases(experiment.id, DatasetSplit.TEST)
    assert [case.example_id for case in after_restart] == ["test-a", "test-b"]
    assert after_restart[0].id == first_case_id
    assert restarted.progress(experiment.id, DatasetSplit.TEST).complete
