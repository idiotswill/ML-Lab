from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml_lab.core.models import DatasetSplit, DatasetState
from ml_lab.datasets.service import DatasetService
from ml_lab.storage.workspace import Workspace
from ml_lab.ui.data_studio import DataStudioController

_ROW_COUNT = 100_000
_BATCH_SIZE = 5_000
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


def _seed_rows(workspace: Workspace, dataset_id: str) -> None:
    created_at = "2026-01-01T00:00:00+00:00"
    insert_sql = (
        "INSERT INTO dataset_examples("
        "dataset_id,example_id,split,source_id,lineage_group,fingerprint,"
        "normalized_fingerprint,near_signature,payload_json,label_json,tags_json,created_at"
        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?)"
    )
    with workspace.database.transaction() as conn:
        for start in range(0, _ROW_COUNT, _BATCH_SIZE):
            batch = []
            for index in range(start, min(start + _BATCH_SIZE, _ROW_COUNT)):
                example_id = f"scale-{index:06d}"
                split = _split_for(index)
                payload_json = json.dumps(
                    {"text": f"scale payload {index:06d}"},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                fingerprint = hashlib.sha256(example_id.encode("utf-8")).hexdigest()
                near = hashlib.blake2b(
                    f"near:{index}".encode("utf-8"),
                    digest_size=8,
                ).hexdigest()
                batch.append(
                    (
                        dataset_id,
                        example_id,
                        split.value,
                        f"synthetic:{example_id}",
                        f"lineage:{example_id}",
                        fingerprint,
                        fingerprint,
                        near,
                        payload_json,
                        '{"class":"A"}',
                        "[]",
                        created_at,
                    )
                )
            conn.executemany(insert_sql, batch)
        conn.execute(
            "UPDATE dataset_versions SET example_count=? WHERE id=?",
            (_ROW_COUNT, dataset_id),
        )


@pytest.mark.scale
def test_data_studio_freezes_and_pages_one_hundred_thousand_examples(
    tmp_path: Path,
) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Data scale", "generic")
    datasets = DatasetService(workspace)
    dataset = datasets.create(project.id, "scale-100k")
    _seed_rows(workspace, dataset.id)

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
