import json
import time
from pathlib import Path

import pytest

from ml_lab.core.models import TERMINAL_JOB_STATUSES, JobStatus
from ml_lab.jobs.manager import JobManager
from ml_lab.storage.workspace import Workspace
from ml_lab.trainers.sparse_nb import SparseNBModel, train_sparse_nb


def test_sparse_nb_learns_and_round_trips_deterministically(tmp_path: Path) -> None:
    records = [
        {"payload": {"text": "open the oak door"}, "label": {"class": "OPEN"}},
        {"payload": {"text": "please open that door"}, "label": {"class": "OPEN"}},
        {"payload": {"text": "shut the iron gate"}, "label": {"class": "CLOSE"}},
        {"payload": {"text": "close that gate now"}, "label": {"class": "CLOSE"}},
    ]
    model, count = train_sparse_nb(records, feature_dim=2048)
    assert count == 4
    assert model.predict("open the door")[0] == "OPEN"
    assert model.predict("close the gate")[0] == "CLOSE"

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    model.save(first)
    SparseNBModel.load(first).save(second)
    assert first.read_bytes() == second.read_bytes()


def test_sparse_training_worker_commits_named_model_artifact(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    train.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "payload": {"text": "red apple"},
                        "label": {"class": "FRUIT"},
                    }
                ),
                json.dumps(
                    {
                        "payload": {"text": "green apple"},
                        "label": {"class": "FRUIT"},
                    }
                ),
                json.dumps(
                    {
                        "payload": {"text": "blue hammer"},
                        "label": {"class": "TOOL"},
                    }
                ),
                json.dumps(
                    {
                        "payload": {"text": "steel hammer"},
                        "label": {"class": "TOOL"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dev.write_text(
        json.dumps(
            {"payload": {"text": "apple"}, "label": {"class": "FRUIT"}}
        )
        + "\n",
        encoding="utf-8",
    )
    manager = JobManager(workspace.root, workspace.database, workspace.artifacts)
    record = manager.start(
        "trainer.sparse_nb.v1",
        {
            "train_paths": [str(train)],
            "dev_paths": [str(dev)],
            "feature_dim": 2048,
        },
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        record = manager.get(record.id)
        if record.status in TERMINAL_JOB_STATUSES:
            break
        time.sleep(0.03)
    manager.shutdown()

    assert record.status is JobStatus.COMPLETED
    assert record.result_artifact_digest is not None
    result = json.loads(
        workspace.artifacts.resolve(record.result_artifact_digest).read_text(
            encoding="utf-8"
        )
    )
    model_output = result["output_artifacts"]["model"]
    model_digest = model_output["sha256"]
    model_path = workspace.artifacts.resolve(model_digest)
    loaded = SparseNBModel.load(model_path)
    assert loaded.predict("red apple")[0] == "FRUIT"
    assert result["dev_accuracy"] == 1.0


def test_worker_output_cannot_escape_staging(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    manager = JobManager(workspace.root, workspace.database, workspace.artifacts)
    staging = workspace.root / "jobs" / "malicious"
    staging.mkdir(parents=True)
    outside = workspace.root / "outside.bin"
    outside.write_bytes(b"outside")
    manifest = staging / "result_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "outputs": [
                    {
                        "name": "escape",
                        "path": "../outside.bin",
                        "media_type": "application/octet-stream",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsafe worker output path"):
        manager._finalize_declared_outputs("malicious", staging, manifest)
    assert outside.read_bytes() == b"outside"
