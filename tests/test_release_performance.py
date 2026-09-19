from __future__ import annotations

from pathlib import Path

import pytest

from ml_lab.jobs.protocol import JobSpec
from ml_lab.jobs.worker import task_performance_load
from ml_lab.release.performance import _split_for, _write_dataset_source


def test_performance_dataset_source_is_bounded_and_split(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_dataset_source(source, 20, prefix="test")
    lines = source.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20
    assert [_split_for(index) for index in range(10)] == [
        "TRAIN",
        "TRAIN",
        "TRAIN",
        "TRAIN",
        "TRAIN",
        "TRAIN",
        "TRAIN",
        "DEV",
        "TEST",
        "REDTEAM",
    ]


def test_performance_load_honors_preexisting_cancellation(tmp_path: Path) -> None:
    staging = tmp_path / "job"
    staging.mkdir()
    (staging / "cancel.request").write_text("cancel", encoding="utf-8")
    spec = JobSpec(
        job_id="performance-test",
        task_type="core.performance_load",
        staging_dir=str(staging),
        payload={"duration_seconds": 1.0},
    )
    with pytest.raises(InterruptedError, match="Cancellation requested"):
        task_performance_load(spec, staging)
