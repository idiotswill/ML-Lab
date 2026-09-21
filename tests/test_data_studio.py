from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_lab.adapters.builtin import dataset_validator_for
from ml_lab.core.models import DatasetSplit, DatasetState
from ml_lab.datasets.import_stage import prepare_import_stage
from ml_lab.datasets.service import DatasetService
from ml_lab.storage.workspace import Workspace
from ml_lab.ui.data_studio import (
    DataStudioController,
    _perform_dataset_import_via_child,
    _perform_dataset_operation,
)


def test_data_studio_import_and_freeze_use_real_dataset_contract(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Data Studio", "generic")
    dataset = DatasetService(workspace).create(project.id, "v1")
    source = tmp_path / "dataset.jsonl"
    rows = [
        {
            "example_id": "train-1",
            "split": "TRAIN",
            "source_id": "synthetic:train-1",
            "lineage_group": "train-1",
            "payload": {"text": "amber lighthouse cedar orbit"},
            "label": {"class": "A"},
        },
        {
            "example_id": "dev-1",
            "split": "DEV",
            "source_id": "synthetic:dev-1",
            "lineage_group": "dev-1",
            "payload": {"text": "violet canyon marble signal"},
            "label": {"class": "B"},
        },
        {
            "example_id": "test-1",
            "split": "TEST",
            "source_id": "synthetic:test-1",
            "lineage_group": "test-1",
            "payload": {"text": "silver orchard comet harbor"},
            "label": {"class": "C"},
        },
        {
            "example_id": "red-1",
            "split": "REDTEAM",
            "source_id": "synthetic:red-1",
            "lineage_group": "red-1",
            "payload": {"text": "obsidian glacier kettle phoenix"},
            "label": {"class": "ABSTAIN"},
        },
    ]
    source.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    imported = _perform_dataset_import_via_child(
        workspace_root=workspace.root,
        dataset_id=dataset.id,
        adapter_id="generic",
        source=source,
    )
    assert imported["imported"] == 4
    assert imported["rejected"] == 0
    import_view = imported["view_snapshot"]
    assert isinstance(import_view, dict)
    assert import_view["split_counts"]["ALL"] == 4
    assert len(import_view["examples"]) == 4
    assert import_view["selected_dataset"]["example_count"] == 4

    frozen = _perform_dataset_operation(
        workspace_root=workspace.root,
        operation="freeze",
        dataset_id=dataset.id,
        adapter_id="generic",
        source=None,
    )
    assert frozen["state"] == DatasetState.FROZEN.value
    assert frozen["manifest_sha256"]
    assert frozen["leakage_report_sha256"]
    freeze_view = frozen["view_snapshot"]
    assert isinstance(freeze_view, dict)
    assert freeze_view["selected_dataset"]["state"] == DatasetState.FROZEN.value

    service = DatasetService(workspace)
    assert {item.split for item in service.partitions(dataset.id)} == set(DatasetSplit)
    assert set(service.trainer_partition_handles(dataset.id)) == {"TRAIN", "DEV"}
    assert set(service.evaluation_partition_handles(dataset.id)) == {
        "TRAIN",
        "DEV",
        "TEST",
        "REDTEAM",
    }


def test_builtin_dataset_validator_bridge_is_explicit() -> None:
    assert dataset_validator_for("generic") is None
    assert dataset_validator_for("frankenhomie.phase-a-residual") is not None



def test_import_completion_uses_worker_prepared_view_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.create(tmp_path / "workspace-cache")
    project = workspace.create_project("Data Studio cache", "generic")
    dataset = DatasetService(workspace).create(project.id, "cached")
    source = tmp_path / "cached.jsonl"
    source.write_text(
        json.dumps(
            {
                "example_id": "cached-1",
                "split": "TRAIN",
                "source_id": "synthetic:cached-1",
                "lineage_group": "cached-1",
                "payload": {"text": "prepared off gui thread"},
                "label": {"class": "A"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = _perform_dataset_operation(
        workspace_root=workspace.root,
        operation="import",
        dataset_id=dataset.id,
        adapter_id="generic",
        source=source,
    )

    controller = DataStudioController()
    controller._context_token = 9
    controller._selected_dataset_id = dataset.id
    controller._busy = True

    def forbidden_refresh() -> None:
        raise AssertionError("GUI completion path must not re-query the dataset database")

    monkeypatch.setattr(controller, "_refresh_view_cache", forbidden_refresh)
    try:
        controller._operation_completed(9, "import", dataset.id, result)
        assert controller.busy is False
        assert controller.splitCounts["ALL"] == 1
        assert len(controller.examples) == 1
        assert controller.examples[0]["exampleId"] == "cached-1"
        assert controller.selectedDataset["example_count"] == 1
    finally:
        controller.shutdown()



def test_import_stage_never_mutates_workspace_before_parent_commit(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "workspace-stage")
    project = workspace.create_project("Import stage boundary", "generic")
    service = DatasetService(workspace)
    dataset = service.create(project.id, "v1")

    seed = tmp_path / "seed.jsonl"
    seed.write_text(
        json.dumps(
            {
                "example_id": "existing",
                "split": "TRAIN",
                "source_id": "synthetic:existing",
                "lineage_group": "existing",
                "payload": {"text": "already authoritative"},
                "label": {"class": "A"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    seeded = service.import_jsonl(dataset.id, seed)
    assert seeded.imported == 1

    source = tmp_path / "stage.jsonl"
    rows = [
        {
            "example_id": "existing",
            "split": "TRAIN",
            "source_id": "synthetic:existing-again",
            "lineage_group": "existing-again",
            "payload": {"text": "duplicate against authoritative dataset"},
            "label": {"class": "A"},
        },
        {
            "example_id": "fresh",
            "split": "DEV",
            "source_id": "synthetic:fresh",
            "lineage_group": "fresh",
            "payload": {"text": "fresh staged row"},
            "label": {"class": "B"},
        },
        {
            "example_id": "fresh",
            "split": "TEST",
            "source_id": "synthetic:fresh-duplicate",
            "lineage_group": "fresh-duplicate",
            "payload": {"text": "duplicate inside staged source"},
            "label": {"class": "C"},
        },
    ]
    source.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    staged = tmp_path / "prepared.sqlite"
    summary = prepare_import_stage(
        source=source,
        staged_database=staged,
        adapter_id="generic",
    )

    assert service.get(dataset.id).example_count == 1
    assert len(service.page_examples(dataset.id, limit=100)) == 1
    assert summary["accepted"] == 2
    assert summary["rejected"] == 1

    committed = service.commit_prepared_import(dataset.id, staged, summary)
    assert committed.imported == 1
    assert committed.rejected == 2
    assert {error.code for error in committed.errors} == {"DUPLICATE_OR_CONSTRAINT"}
    assert service.get(dataset.id).example_count == 2
    assert {item.example_id for item in service.page_examples(dataset.id)} == {
        "existing",
        "fresh",
    }
