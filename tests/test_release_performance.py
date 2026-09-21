from __future__ import annotations

from pathlib import Path

import pytest

from ml_lab.jobs.protocol import JobSpec
from ml_lab.jobs.worker import task_performance_load
from ml_lab.release.performance import (
    _split_for,
    _summarize_navigation_rounds,
    _write_dataset_source,
)


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



def _navigation_round(*, navigation_stalls: int, scheduler_stalls: int = 0) -> dict[str, object]:
    return {
        "completed": True,
        "samples": 20,
        "max_stall_ms": 130.0 if navigation_stalls or scheduler_stalls else 40.0,
        "stalls_over_100ms": navigation_stalls + scheduler_stalls,
        "max_navigation_process_ms": 116.0 if navigation_stalls else 35.0,
        "navigation_stalls_over_100ms": navigation_stalls,
        "max_scheduler_delay_ms": 125.0 if scheduler_stalls else 5.0,
        "scheduler_delays_over_100ms": scheduler_stalls,
    }


def test_ordinary_navigation_allows_one_isolated_over_100ms_gui_sample() -> None:
    summary = _summarize_navigation_rounds(
        [
            _navigation_round(navigation_stalls=1),
            _navigation_round(navigation_stalls=0),
            _navigation_round(navigation_stalls=0),
        ]
    )
    assert summary["rounds_completed"] == 3
    assert summary["rounds_with_navigation_over_100ms"] == 1
    assert summary["total_navigation_stalls_over_100ms"] == 1
    assert summary["repeatable_navigation_over_100ms"] is False


def test_ordinary_navigation_rejects_repeated_over_100ms_gui_samples() -> None:
    summary = _summarize_navigation_rounds(
        [
            _navigation_round(navigation_stalls=1),
            _navigation_round(navigation_stalls=0),
            _navigation_round(navigation_stalls=1),
        ]
    )
    assert summary["rounds_with_navigation_over_100ms"] == 2
    assert summary["repeatable_navigation_over_100ms"] is True


def test_scheduler_delay_is_recorded_but_not_mislabeled_as_gui_stall() -> None:
    summary = _summarize_navigation_rounds(
        [
            _navigation_round(navigation_stalls=0, scheduler_stalls=1),
            _navigation_round(navigation_stalls=0),
            _navigation_round(navigation_stalls=0),
        ]
    )
    assert summary["total_scheduler_delays_over_100ms"] == 1
    assert summary["total_navigation_stalls_over_100ms"] == 0
    assert summary["repeatable_navigation_over_100ms"] is False
