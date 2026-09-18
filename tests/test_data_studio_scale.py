from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_lab.core.models import DatasetSplit, DatasetState
from ml_lab.datasets.service import DatasetService
from ml_lab.storage.workspace import Workspace
from ml_lab.ui.data_studio import DataStudioController

_ROW_COUNT = 100_000
_EXPECTED_COUNTS = {
    DatasetSplit.TRAIN: 70_000,
    DatasetSplit.DEV: 10_000,
    DatasetSplit.TEST: 10_000,
    DatasetSplit.REDTEAM: 10_000,
}


def _split_for(index: int) -> DatasetSplit:
    if index < 70_000:
        return DatasetSplit.TRAIN
    if index < 80_000:
        return DatasetSplit.DEV
    if index < 90_000:
        return DatasetSplit.TEST
    return DatasetSplit.REDTEAM


def _write_jsonl_source(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(_ROW_COUNT):
            example_id = f"scale-{index:06d}"
            row = {
                "example_id": example_id,
                "split": _split_for(index).value,
                "source_id": f"synthetic:{example_id}",
                "lineage_group": f"lineage:{example_id}",
                "payload": index,
                "label": {"class": "A"},
                "tags": ["scale"],
            }
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


@pytest.mark.scale
def test_data_studio_imports_freezes_and_pages_one_hundred_thousand_examples(
    tmp_path: Path,
) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Data scale", "generic")
    datasets = DatasetService(workspace)
    dataset = datasets.create(project.id, "scale-100k")
    source = tmp_path / "scale-100k.jsonl"
    _write_jsonl_source(source)

    imported = datasets.import_jsonl(dataset.id, source)
    assert imported.imported == _ROW_COUNT
    assert imported.rejected == 0
    assert imported.errors == ()
    assert datasets.get(dataset.id).example_count == _ROW_COUNT

    controller = DataStudioController()
    controller.bind_project(workspace, project.id, "generic")
    controller.selectDataset(dataset.id)
    try:
        assert controller.splitCounts["ALL"] == _ROW_COUNT
        assert controller.splitCounts["TRAIN"] == 70_000
        assert controller.splitCounts["DEV"] == 10_000
        assert controller.splitCounts["TEST"] == 10_000
        assert controller.splitCounts["REDTEAM"] == 10_000

        first_page = controller.examples
        assert len(first_page) == 100
        assert first_page[0]["exampleId"] == "scale-000000"
        assert first_page[-1]["exampleId"] == "scale-000099"
        assert controller.pageNumber == 1
        assert controller.canNextPage is True

        controller.nextPage()
        second_page = controller.examples
        assert len(second_page) == 100
        assert second_page[0]["exampleId"] == "scale-000100"
        assert second_page[-1]["exampleId"] == "scale-000199"
        assert controller.pageNumber == 2

        controller.setSplitFilter("REDTEAM")
        redteam_page = controller.examples
        assert len(redteam_page) == 100
        assert redteam_page[0]["exampleId"] == "scale-090000"
        assert redteam_page[-1]["exampleId"] == "scale-090099"
        assert controller.canNextPage is True

        frozen = datasets.freeze(dataset.id)
        assert frozen.state is DatasetState.FROZEN
        assert frozen.example_count == _ROW_COUNT

        partitions = {item.split: item for item in datasets.partitions(dataset.id)}
        assert set(partitions) == set(DatasetSplit)
        assert {
            split: partition.example_count for split, partition in partitions.items()
        } == _EXPECTED_COUNTS
        for partition in partitions.values():
            path = workspace.artifacts.resolve(partition.artifact_digest)
            assert path.is_file()
            assert path.stat().st_size > 0

        assert set(datasets.trainer_partition_handles(dataset.id)) == {"TRAIN", "DEV"}
        assert set(datasets.evaluation_partition_handles(dataset.id)) == {
            "TRAIN",
            "DEV",
            "TEST",
            "REDTEAM",
        }
    finally:
        controller.shutdown()
