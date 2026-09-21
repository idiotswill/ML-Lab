from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from ml_lab.adapters.phase_a import PhaseAResidualAdapter
from ml_lab.adapters.phase_a_dataset_factory import (
    _cached_export,
    _cached_reference_validation,
    _hidden_fixture_leaks,
    _proposal_from_author_label,
)
from ml_lab.adapters.phase_a_fixture_export import PhaseAFixtureExporter
from ml_lab.adapters.phase_a_reference import PhaseAReferenceValidator
from ml_lab.datasets.leakage import canonical_json
from ml_lab.storage.workspace import Workspace


def build_phase_a_protected_evidence(
    *,
    workspace: Workspace,
    frankenhomie_repository: Path,
    authored_root: Path,
    output_dir: Path,
    case_cache_dir: Path,
) -> dict[str, object]:
    residual_seed_path = authored_root / "phase-a-expanded-protected-residual-seed-v1.json"
    zero_seed_path = authored_root / "phase-a-expanded-zero-model-seed-v1.json"
    residual_seed = _read_object(residual_seed_path)
    zero_seed = _read_object(zero_seed_path)
    _validate_seed_pair(residual_seed, zero_seed)

    target_commit = str(residual_seed["target_frankenhomie_commit"])
    target_contract = str(residual_seed["target_contract"])
    actor_id = str(residual_seed["actor_id"])
    audience = str(residual_seed["audience"])
    scenes = _mapping(residual_seed, "scenes")
    zero_scenes = _mapping(zero_seed, "scenes")

    exporter = PhaseAFixtureExporter(workspace)
    reference = PhaseAReferenceValidator(workspace)
    adapter = PhaseAResidualAdapter()
    cache_root = case_cache_dir.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    residual_rows: list[dict[str, object]] = []
    zero_results: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    export_hits = export_misses = reference_hits = reference_misses = 0

    for case in _sequence_mapping(residual_seed, "cases"):
        case_id = _required_text(case, "case_id")
        scene_id = _required_text(case, "scene_id")
        scene = _mapping(scenes, scene_id)
        fixture = _fixture(
            fixture_id=f"protected:{case_id}",
            declaration=_required_text(case, "declaration"),
            actor_id=actor_id,
            audience=audience,
            scene=scene,
        )
        receipt, hit = _cached_export(
            exporter=exporter,
            repository=frankenhomie_repository,
            target_commit=target_commit,
            fixture=fixture,
            cache_root=cache_root,
        )
        export_hits += int(hit)
        export_misses += int(not hit)
        if receipt.status != "RESIDUAL_EXPORTED" or receipt.request is None:
            errors.append(
                {
                    "case_id": case_id,
                    "code": "NOT_RESIDUAL_EXPORTED",
                    "status": receipt.status,
                    "route": receipt.route,
                    "failed_deterministic_stage": receipt.failed_deterministic_stage,
                    "error_code": receipt.error_code,
                }
            )
            continue

        hidden = _hidden_fixture_leaks(
            facts=_sequence_mapping(scene, "facts"),
            request=receipt.request,
        )
        if hidden:
            errors.append(
                {
                    "case_id": case_id,
                    "code": "HIDDEN_FIXTURE_FACT_LEAK",
                    "detail": hidden,
                }
            )
            continue

        expected = _proposal_from_author_label(case)
        check = adapter.validate_proposal(receipt.request, expected)
        if not check.accepted:
            errors.append(
                {
                    "case_id": case_id,
                    "code": "AUTHOR_LABEL_REJECTED_BY_ADAPTER",
                    "detail": check.error_code,
                }
            )
            continue

        pinned, ref_hit = _cached_reference_validation(
            reference=reference,
            repository=frankenhomie_repository,
            target_commit=target_commit,
            request=receipt.request,
            proposal=expected,
            cache_root=cache_root,
        )
        reference_hits += int(ref_hit)
        reference_misses += int(not ref_hit)
        if pinned.status != "ACCEPTED":
            errors.append(
                {
                    "case_id": case_id,
                    "code": "AUTHOR_LABEL_REJECTED_BY_PINNED_REFERENCE",
                    "detail": pinned.error_code,
                }
            )
            continue

        residual_rows.append(
            {
                "example_id": case_id,
                "split": _required_text(case, "split"),
                "source_id": f"synthetic-corpus-protected-v1:{case_id}",
                "lineage_group": _required_text(case, "lineage_group"),
                "request": receipt.request,
                "expected": expected,
                "tags": [str(tag) for tag in _sequence(case, "tags")],
            }
        )

    for case in _sequence_mapping(zero_seed, "cases"):
        case_id = _required_text(case, "case_id")
        scene_id = _required_text(case, "scene_id")
        scene = _mapping(zero_scenes, scene_id)
        fixture = _fixture(
            fixture_id=f"zero:{case_id}",
            declaration=_required_text(case, "declaration"),
            actor_id=str(zero_seed["actor_id"]),
            audience=str(zero_seed["audience"]),
            scene=scene,
        )
        receipt, hit = _cached_export(
            exporter=exporter,
            repository=frankenhomie_repository,
            target_commit=target_commit,
            fixture=fixture,
            cache_root=cache_root,
        )
        export_hits += int(hit)
        export_misses += int(not hit)

        expected_status = _required_text(case, "expected_status")
        expected_route = _required_text(case, "expected_route")
        ok = (
            receipt.status == expected_status
            and receipt.route == expected_route
            and receipt.request is None
        )
        zero_results.append(
            {
                "case_id": case_id,
                "status": receipt.status,
                "route": receipt.route,
                "failed_deterministic_stage": receipt.failed_deterministic_stage,
                "passed": ok,
                "receipt_artifact_digest": receipt.receipt_artifact_digest,
            }
        )
        if not ok:
            errors.append(
                {
                    "case_id": case_id,
                    "code": "ZERO_MODEL_INVARIANT_FAILED",
                    "expected_status": expected_status,
                    "actual_status": receipt.status,
                    "expected_route": expected_route,
                    "actual_route": receipt.route,
                    "request_exported": receipt.request is not None,
                }
            )

    output_root = output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    residual_path = output_root / "phase-a-protected-residual-v1.jsonl"
    zero_path = output_root / "phase-a-zero-model-evidence-v1.json"
    residual_sha: str | None = None

    if not errors:
        residual_bytes = (
            "".join(canonical_json(row) + "\n" for row in residual_rows)
        ).encode("utf-8")
        residual_path.write_bytes(residual_bytes)
        residual_sha = hashlib.sha256(residual_bytes).hexdigest()

    zero_payload = {
        "schema": "ml-lab-phase-a-zero-model-evidence/1",
        "target_frankenhomie_commit": target_commit,
        "target_contract": target_contract,
        "case_count": len(zero_results),
        "passed_count": sum(int(row["passed"] is True) for row in zero_results),
        "results": zero_results,
        "integration_gate": "NO_GO",
    }
    zero_bytes = (canonical_json(zero_payload) + "\n").encode("utf-8")
    zero_path.write_bytes(zero_bytes)

    split_counts = {
        "TEST": sum(row["split"] == "TEST" for row in residual_rows),
        "REDTEAM": sum(row["split"] == "REDTEAM" for row in residual_rows),
    }
    base: dict[str, object] = {
        "schema": "ml-lab-phase-a-protected-authority-receipt/1",
        "ok": not errors,
        "target_frankenhomie_commit": target_commit,
        "target_contract": target_contract,
        "residual_case_count": len(_sequence_mapping(residual_seed, "cases")),
        "residual_emitted_count": len(residual_rows) if not errors else 0,
        "residual_split_counts": split_counts if not errors else {},
        "residual_dataset_sha256": residual_sha,
        "zero_model_case_count": len(zero_results),
        "zero_model_passed_count": sum(int(row["passed"] is True) for row in zero_results),
        "zero_model_evidence_sha256": hashlib.sha256(zero_bytes).hexdigest(),
        "errors": errors,
        "case_cache": {
            "export_hits": export_hits,
            "export_misses": export_misses,
            "reference_hits": reference_hits,
            "reference_misses": reference_misses,
        },
        "production_data": False,
        "transcript_derived": False,
        "training_allowed": False,
        "integration_gate": "NO_GO",
    }
    payload_sha = hashlib.sha256(canonical_json(base).encode()).hexdigest()
    final_receipt = {**base, "payload_sha256": payload_sha}
    (output_root / "protected-authority-receipt.json").write_bytes(
        (canonical_json(final_receipt) + "\n").encode("utf-8")
    )
    return final_receipt


def _fixture(
    *,
    fixture_id: str,
    declaration: str,
    actor_id: str,
    audience: str,
    scene: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": "ml-lab-phase-a-residual-fixture/1",
        "fixture_id": fixture_id,
        "fixture_kind": "NON_CANON_FIXTURE",
        "production_data": False,
        "declaration": declaration,
        "actor_id": actor_id,
        "audience": audience,
        "snapshot_revision": _required_text(scene, "snapshot_revision"),
        "facts": [dict(row) for row in _sequence_mapping(scene, "facts")],
    }


def _validate_seed_pair(
    residual: Mapping[str, object],
    zero: Mapping[str, object],
) -> None:
    if residual.get("schema") != "ml-lab-phase-a-protected-residual-seed/1":
        raise ValueError("Unsupported protected residual seed")
    if zero.get("schema") != "ml-lab-phase-a-zero-model-seed/1":
        raise ValueError("Unsupported zero-model seed")
    for value in (residual, zero):
        if value.get("status") != "PROTECTED_EVALUATION_ONLY":
            raise ValueError("Protected seed status changed")
        if value.get("production_data") is not False:
            raise ValueError("Protected evidence cannot use production data")
        if value.get("transcript_derived") is not False:
            raise ValueError("Protected evidence cannot be transcript-derived")
        if value.get("target_contract") != "semantic-residual-v2":
            raise ValueError("Protected evidence targets wrong contract")
    if residual["target_frankenhomie_commit"] != zero["target_frankenhomie_commit"]:
        raise ValueError("Protected seeds target different commits")
    for case in _sequence_mapping(residual, "cases"):
        if _required_text(case, "split") not in {"TEST", "REDTEAM"}:
            raise ValueError("Protected residual split must be TEST/REDTEAM")
        if case.get("training_allowed") is not False:
            raise ValueError("Protected residual case became training-visible")
    for case in _sequence_mapping(zero, "cases"):
        if _required_text(case, "split") != "REDTEAM":
            raise ValueError("Zero-model case must remain REDTEAM")
        if case.get("training_allowed") is not False:
            raise ValueError("Zero-model case became training-visible")


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    raw = value.get(key)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{key} must be an object")
    return raw


def _sequence(value: Mapping[str, object], key: str) -> Sequence[object]:
    raw = value.get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{key} must be a list")
    return raw


def _sequence_mapping(
    value: Mapping[str, object],
    key: str,
) -> list[Mapping[str, object]]:
    raw = _sequence(value, key)
    if any(not isinstance(item, Mapping) for item in raw):
        raise ValueError(f"{key} must contain objects")
    return [item for item in raw if isinstance(item, Mapping)]


def _required_text(value: Mapping[str, object], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return raw.strip()
