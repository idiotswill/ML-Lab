import json
from pathlib import Path

import pytest

from ml_lab.core.models import DatasetSplit, DatasetState
from ml_lab.datasets.leakage import LeakageExample, scan_leakage
from ml_lab.datasets.service import DatasetLeakageError, DatasetService, ValidatedExampleInput
from ml_lab.storage.workspace import Workspace


def _workspace(tmp_path: Path) -> tuple[Workspace, str, DatasetService]:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Dataset tests")
    return workspace, project.id, DatasetService(workspace)


def test_streaming_jsonl_import_reports_row_errors(tmp_path: Path) -> None:
    workspace, project_id, datasets = _workspace(tmp_path)
    dataset = datasets.create(project_id, "import")
    source = tmp_path / "examples.jsonl"
    source.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "example_id": "ok-1",
                        "split": "train",
                        "payload": {"text": "open the north door"},
                        "label": {"decision": "resolve"},
                    }
                ),
                "{broken json",
                json.dumps({"example_id": "missing-label", "split": "DEV", "payload": {}}),
                json.dumps(
                    {
                        "example_id": "ok-2",
                        "split": "REDTEAM",
                        "payload": {"text": "do not hit it"},
                        "label": {"decision": "no_action"},
                        "lineage_group": "negation-1",
                        "tags": ["negation"],
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    summary = datasets.import_jsonl(dataset.id, source)
    assert summary.imported == 2
    assert summary.rejected == 2
    assert [error.code for error in summary.errors] == ["INVALID_JSON", "INVALID_EXAMPLE"]
    assert datasets.get(dataset.id).example_count == 2
    page = datasets.page_examples(dataset.id, limit=1)
    assert len(page) == 1
    assert page[0].example_id == "ok-1"
    assert workspace.database.schema_version() >= 2


def test_cross_split_duplicate_blocks_freeze(tmp_path: Path) -> None:
    workspace, project_id, datasets = _workspace(tmp_path)
    dataset = datasets.create(project_id, "leaky")
    shared = {"text": "strike the goblin with the sword"}
    datasets.add_example(
        dataset.id,
        ValidatedExampleInput(
            example_id="train-1",
            split=DatasetSplit.TRAIN,
            source_id="synthetic:1",
            lineage_group="attack-a",
            payload=shared,
            label={"decision": "resolve"},
            tags=(),
        ),
    )
    datasets.add_example(
        dataset.id,
        ValidatedExampleInput(
            example_id="test-1",
            split=DatasetSplit.TEST,
            source_id="synthetic:2",
            lineage_group="attack-b",
            payload=shared,
            label={"decision": "resolve"},
            tags=(),
        ),
    )

    with pytest.raises(DatasetLeakageError) as captured:
        datasets.freeze(dataset.id)
    assert captured.value.blocking_count >= 1
    after = datasets.get(dataset.id)
    assert after.state is DatasetState.DRAFT
    assert after.leakage_report_artifact_digest is not None
    assert datasets.partitions(dataset.id) == []
    assert workspace.artifacts.resolve(captured.value.report_digest).exists()


def test_freeze_creates_four_partitions_and_hides_protected_handles(tmp_path: Path) -> None:
    workspace, project_id, datasets = _workspace(tmp_path)
    dataset = datasets.create(project_id, "clean")
    payloads = {
        DatasetSplit.TRAIN: "carry crimson lantern toward distant harbor",
        DatasetSplit.DEV: "inspect silver astrolabe beneath ruined tower",
        DatasetSplit.TEST: "whisper amber password beside frozen waterfall",
        DatasetSplit.REDTEAM: "cancel violet bargain before midnight bell",
    }
    for index, (split, text) in enumerate(payloads.items(), start=1):
        datasets.add_example(
            dataset.id,
            ValidatedExampleInput(
                example_id=f"case-{index}",
                split=split,
                source_id=f"synthetic:{index}",
                lineage_group=f"lineage-{index}",
                payload={"text": text},
                label={"expected": split.value.lower()},
                tags=(),
            ),
        )

    frozen = datasets.freeze(dataset.id)
    assert frozen.state is DatasetState.FROZEN
    assert frozen.manifest_artifact_digest is not None
    partitions = datasets.partitions(dataset.id)
    assert {partition.split for partition in partitions} == set(DatasetSplit)
    assert all(partition.example_count == 1 for partition in partitions)
    for partition in partitions:
        assert workspace.artifacts.resolve(partition.artifact_digest).exists()

    trainer = datasets.trainer_partition_handles(dataset.id)
    assert set(trainer) == {"TRAIN", "DEV"}
    assert "TEST" not in trainer
    assert "REDTEAM" not in trainer

    evaluation = datasets.evaluation_partition_handles(dataset.id)
    assert set(evaluation) == {split.value for split in DatasetSplit}


def test_lineage_group_crossing_partitions_is_blocking(tmp_path: Path) -> None:
    _, project_id, datasets = _workspace(tmp_path)
    dataset = datasets.create(project_id, "lineage")
    for example_id, split, text in (
        ("a", DatasetSplit.TRAIN, "open an iron gate"),
        ("b", DatasetSplit.DEV, "close a paper window"),
    ):
        datasets.add_example(
            dataset.id,
            ValidatedExampleInput(
                example_id=example_id,
                split=split,
                source_id=example_id,
                lineage_group="same-template",
                payload={"text": text},
                label={"value": example_id},
                tags=(),
            ),
        )
    report = datasets.scan_leakage(dataset.id)
    assert any(issue.kind == "LINEAGE_LEAKAGE" and issue.blocking for issue in report.issues)


def test_streaming_leakage_scan_preserves_normalized_and_near_duplicate_checks() -> None:
    def examples():
        yield LeakageExample(
            example_id="train-normalized",
            split=DatasetSplit.TRAIN,
            lineage_group="lineage-a",
            fingerprint="exact-a",
            normalized_fingerprint="normalized-shared",
            near_signature="0000000000000000",
        )
        yield LeakageExample(
            example_id="test-normalized",
            split=DatasetSplit.TEST,
            lineage_group="lineage-b",
            fingerprint="exact-b",
            normalized_fingerprint="normalized-shared",
            near_signature="ffffffffffffffff",
        )
        yield LeakageExample(
            example_id="train-near",
            split=DatasetSplit.TRAIN,
            lineage_group="lineage-c",
            fingerprint="exact-c",
            normalized_fingerprint="normalized-c",
            near_signature="1234567890abcdef",
        )
        yield LeakageExample(
            example_id="test-near",
            split=DatasetSplit.TEST,
            lineage_group="lineage-d",
            fingerprint="exact-d",
            normalized_fingerprint="normalized-d",
            near_signature="1234567890abcdee",
        )

    report = scan_leakage(examples())
    blocking = {
        (issue.kind, issue.left_id, issue.right_id)
        for issue in report.issues
        if issue.blocking
    }
    assert (
        "NORMALIZED_DUPLICATE",
        "train-normalized",
        "test-normalized",
    ) in blocking
    assert ("NEAR_DUPLICATE", "train-near", "test-near") in blocking
