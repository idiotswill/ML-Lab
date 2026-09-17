from __future__ import annotations

from pathlib import Path

import pytest

from ml_lab.core.models import FailureSeverity, FailureStatus
from ml_lab.failures.service import FailureService
from ml_lab.storage.workspace import Workspace
from ml_lab.ui.redteam_failures import RedTeamFailuresController

_FAILURE_COUNT = 10_000
_REGRESSION_COUNT = 5_000
_VETO_COUNT = 2_500
_BATCH_SIZE = 1_000


def _seed_failures(workspace: Workspace, project_id: str) -> None:
    failure_sql = (
        "INSERT INTO failures("
        "id,project_id,experiment_id,dataset_id,redteam_run_id,example_id,split,kind,"
        "severity,status,expected_json,observed_json,evidence_artifact_digest,created_at"
        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    )
    regression_sql = (
        "INSERT INTO regression_cases(failure_id,suite_name,promoted_at) VALUES(?,?,?)"
    )
    with workspace.database.transaction() as conn:
        for start in range(0, _FAILURE_COUNT, _BATCH_SIZE):
            failures = []
            regressions = []
            for index in range(start, min(start + _BATCH_SIZE, _FAILURE_COUNT)):
                failure_id = f"failure-{index:05d}"
                severity = (
                    FailureSeverity.VETO
                    if index % 4 == 0
                    else FailureSeverity.NON_VETO
                )
                split = "TEST" if index % 2 == 0 else "REDTEAM"
                created_at = f"2026-01-01T00:00:00.{index:06d}+00:00"
                failures.append(
                    (
                        failure_id,
                        project_id,
                        None,
                        None,
                        None,
                        f"example-{index:05d}",
                        split,
                        f"SCALE_KIND_{index % 7}",
                        severity.value,
                        FailureStatus.OPEN.value,
                        f'{{"expected":{index}}}',
                        f'{{"observed":{index + 1}}}',
                        None,
                        created_at,
                    )
                )
                if index % 2 == 0:
                    regressions.append(
                        (
                            failure_id,
                            "default",
                            f"2026-01-02T00:00:00.{index:06d}+00:00",
                        )
                    )
            conn.executemany(failure_sql, failures)
            conn.executemany(regression_sql, regressions)


@pytest.mark.scale
def test_failure_library_pages_ten_thousand_with_regression_membership(
    tmp_path: Path,
) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Failure scale", "generic")
    _seed_failures(workspace, project.id)

    service = FailureService(workspace)
    assert service.count_for_project(project.id) == _FAILURE_COUNT
    assert service.count_for_project(
        project.id,
        severity=FailureSeverity.VETO,
    ) == _VETO_COUNT
    assert service.count_for_project(
        project.id,
        regression_only=True,
    ) == _REGRESSION_COUNT
    assert service.count_for_project(
        project.id,
        severity=FailureSeverity.VETO,
        regression_only=True,
    ) == _VETO_COUNT
    assert service.regression_count(project.id) == _REGRESSION_COUNT

    first_entries = service.page_entries_for_project(project.id, limit=100)
    assert len(first_entries) == 100
    assert first_entries[0].failure.id == "failure-09999"
    assert first_entries[-1].failure.id == "failure-09900"
    assert sum(entry.is_regression for entry in first_entries) == 50

    regression_entries = service.page_entries_for_project(
        project.id,
        regression_only=True,
        limit=100,
    )
    assert len(regression_entries) == 100
    assert all(entry.is_regression for entry in regression_entries)
    assert regression_entries[0].failure.id == "failure-09998"

    controller = RedTeamFailuresController()
    controller.bind_project(workspace, project.id, "generic")
    try:
        assert controller.failureTotal == _FAILURE_COUNT
        assert controller.vetoFailureTotal == _VETO_COUNT
        assert controller.regressionTotal == _REGRESSION_COUNT
        assert controller.failurePageNumber == 1
        assert controller.canPreviousFailurePage is False
        assert controller.canNextFailurePage is True

        first_page = controller.failureRows
        assert len(first_page) == 100
        assert first_page[0]["id"] == "failure-09999"
        assert first_page[-1]["id"] == "failure-09900"

        controller.nextFailurePage()
        second_page = controller.failureRows
        assert controller.failurePageNumber == 2
        assert len(second_page) == 100
        assert second_page[0]["id"] == "failure-09899"
        assert second_page[-1]["id"] == "failure-09800"

        controller.setFailureSeverity("VETO")
        assert controller.failureTotal == _VETO_COUNT
        veto_page = controller.failureRows
        assert len(veto_page) == 100
        assert all(row["severity"] == FailureSeverity.VETO.value for row in veto_page)
        assert veto_page[0]["id"] == "failure-09996"

        controller.selectFailure("failure-09996")
        selected = controller.selectedFailure
        assert selected["id"] == "failure-09996"
        assert selected["isRegression"] is True
        assert '"expected": 9996' in selected["expected"]
        assert '"observed": 9997' in selected["observed"]

        controller.setFailureSeverity("NON_VETO")
        controller.selectFailure("failure-09999")
        before = service.immutable_payload("failure-09999")
        assert controller.selectedFailure["isRegression"] is False
        controller.promoteSelectedFailure("default")
        assert controller.regressionTotal == _REGRESSION_COUNT + 1
        assert controller.selectedFailure["isRegression"] is True
        assert service.immutable_payload("failure-09999") == before
    finally:
        controller.shutdown()
