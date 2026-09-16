import hashlib
import json
import zipfile
from pathlib import Path

from ml_lab.bundles.service import BundleService
from ml_lab.bundles.verify import verify_bundle_file
from ml_lab.core.models import DatasetSplit, FailureSeverity, MetricDirection, MetricValue, ModelStage
from ml_lab.datasets.service import DatasetService, ValidatedExampleInput
from ml_lab.experiments.service import ExperimentService
from ml_lab.failures.service import FailureService
from ml_lab.models.registry import ModelRegistryService
from ml_lab.storage.workspace import Workspace


def _release_candidate(tmp_path: Path) -> tuple[Workspace, str]:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Bundle tests")
    datasets = DatasetService(workspace)
    dataset = datasets.create(project.id, "frozen")
    texts = {
        DatasetSplit.TRAIN: "carry bronze compass toward western ridge",
        DatasetSplit.DEV: "inspect blue lantern under stone arch",
        DatasetSplit.TEST: "whisper quiet phrase beside amber fountain",
        DatasetSplit.REDTEAM: "do not strike the silver statue",
    }
    for index, (split, text) in enumerate(texts.items()):
        datasets.add_example(
            dataset.id,
            ValidatedExampleInput(
                example_id=f"bundle-{index}",
                split=split,
                source_id=f"synthetic:{index}",
                lineage_group=f"bundle-lineage-{index}",
                payload={"text": text},
                label={"class": split.value},
                tags=(),
            ),
        )
    datasets.freeze(dataset.id)

    experiments = ExperimentService(workspace)
    experiment = experiments.create(
        project_id=project.id,
        dataset_id=dataset.id,
        trainer_id="sparse-v1",
        runtime_pack_id="builtin-sparse",
        config={"alpha": 0.2},
        seed=17,
        environment={"runtime": "test"},
    )
    experiments.start(experiment.id)
    model_artifact = workspace.artifacts.commit_bytes(
        b"deterministic-model-bytes",
        media_type="application/octet-stream",
    )
    experiments.complete(
        experiment.id,
        model_artifact_digest=model_artifact.digest,
        metrics=(
            MetricValue("precision", 0.99, MetricDirection.HIGHER, False),
            MetricValue("false_commitments", 0.0, MetricDirection.ZERO, True),
        ),
    )
    FailureService(workspace).record(
        project_id=project.id,
        experiment_id=experiment.id,
        dataset_id=dataset.id,
        kind="KNOWN_NON_VETO",
        severity=FailureSeverity.NON_VETO,
        expected={"decision": "ASK"},
        observed={"decision": "ASK", "wording": "verbose"},
        split=DatasetSplit.DEV,
    )

    registry = ModelRegistryService(workspace)
    model = registry.register_from_experiment(
        experiment.id,
        compatibility={"adapter": "test", "contract": "v1"},
    )
    model = registry.promote(model.id, ModelStage.SHADOW)
    model = registry.promote(model.id, ModelStage.ADVISORY)
    model = registry.promote(model.id, ModelStage.RELEASE_CANDIDATE)
    return workspace, model.id


def test_bundle_is_deterministic_split_safe_and_fresh_verifiable(tmp_path: Path) -> None:
    workspace, model_id = _release_candidate(tmp_path)
    service = BundleService(workspace)

    first = service.build_release_candidate(model_id)
    second = service.build_release_candidate(model_id)
    assert first.bundle_artifact_digest == second.bundle_artifact_digest

    bundle_path = workspace.artifacts.resolve(first.bundle_artifact_digest)
    direct = verify_bundle_file(bundle_path)
    assert direct["status"] == "PASS"
    assert direct["integration_gate"] == "NO_GO"

    with zipfile.ZipFile(bundle_path, "r") as archive:
        reproduction = json.loads(archive.read("reproduction.json"))
        assert set(reproduction["input_partitions"]) == {"TRAIN", "DEV"}
        known_failures = json.loads(archive.read("known_failures.json"))
        assert len(known_failures) == 1
        assert known_failures[0]["kind"] == "KNOWN_NON_VETO"

    receipt = service.verify_fresh(first.id)
    assert receipt.status == "PASS"
    receipt_payload = json.loads(
        workspace.artifacts.resolve(receipt.receipt_artifact_digest).read_text(
            encoding="utf-8"
        )
    )
    assert receipt_payload["fresh_process"] is True
    assert receipt_payload["status"] == "PASS"


def test_bundle_hash_tampering_is_rejected(tmp_path: Path) -> None:
    workspace, model_id = _release_candidate(tmp_path)
    bundle = BundleService(workspace).build_release_candidate(model_id)
    source = workspace.artifacts.resolve(bundle.bundle_artifact_digest)
    tampered = tmp_path / "tampered.zip"
    members = _read_members(source)
    members["metrics.json"] = b'{"tampered":true}\n'
    _write_members(tampered, members)

    result = verify_bundle_file(tampered)
    assert result["status"] == "FAIL"
    assert "SHA-256 mismatch" in str(result["error"])


def test_bundle_semantically_rejects_protected_trainer_partition(tmp_path: Path) -> None:
    workspace, model_id = _release_candidate(tmp_path)
    bundle = BundleService(workspace).build_release_candidate(model_id)
    source = workspace.artifacts.resolve(bundle.bundle_artifact_digest)
    malicious = tmp_path / "protected.zip"
    members = _read_members(source)
    reproduction = json.loads(members["reproduction.json"])
    reproduction["input_partitions"]["TEST"] = "0" * 64
    members["reproduction.json"] = (
        json.dumps(reproduction, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    members["hashes.sha256"] = _hash_manifest(members)
    _write_members(malicious, members)

    result = verify_bundle_file(malicious)
    assert result["status"] == "FAIL"
    assert "protected partition" in str(result["error"])


def _read_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path, "r") as archive:
        return {
            info.filename: archive.read(info.filename)
            for info in archive.infolist()
            if not info.is_dir()
        }


def _hash_manifest(members: dict[str, bytes]) -> bytes:
    lines = []
    for name in sorted(members):
        if name == "hashes.sha256":
            continue
        digest = hashlib.sha256(members[name]).hexdigest()
        lines.append(f"{digest}  {name}\n")
    return "".join(lines).encode("utf-8")


def _write_members(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(members):
            archive.writestr(name, members[name])
