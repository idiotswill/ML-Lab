from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from ml_lab.adapters.phase_a import PhaseAResidualAdapter
from ml_lab.adapters.phase_a_fixture_export import (
    PhaseAFixtureExporter,
    PhaseAFixtureExportReceipt,
)
from ml_lab.adapters.phase_a_reference import PhaseAReferenceValidator
from ml_lab.core.models import DatasetSplit
from ml_lab.datasets.leakage import (
    LeakageExample,
    LeakageReport,
    canonical_json,
    content_fingerprint,
    near_signature,
    normalized_fingerprint,
    scan_leakage,
)
from ml_lab.storage.workspace import Workspace

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SYNTHETIC_SEED = (
    _REPO_ROOT / "benchmarks" / "phase_a" / "synthetic_train_dev_seed_v1.json"
)
DEFAULT_PROTECTED_SEED = (
    _REPO_ROOT / "benchmarks" / "phase_a" / "authoritative_fixture_seed_v1.json"
)


def run_phase_a_dataset_factory(
    *,
    workspace: Workspace,
    frankenhomie_repository: Path,
    output_dir: Path,
    seed_path: Path = DEFAULT_SYNTHETIC_SEED,
    protected_seed_path: Path = DEFAULT_PROTECTED_SEED,
    case_cache_dir: Path | None = None,
) -> dict[str, object]:
    seed_bytes = seed_path.read_bytes()
    protected_bytes = protected_seed_path.read_bytes()
    seed = json.loads(seed_bytes)
    protected = json.loads(protected_bytes)
    if not isinstance(seed, dict) or not isinstance(protected, dict):
        raise ValueError("Phase A dataset factory inputs must be JSON objects")
    _validate_seed(seed)
    _validate_protected_seed(protected)

    target_commit = _required_text(seed, "target_frankenhomie_commit")
    target_contract = _required_text(seed, "target_contract")
    if target_commit != protected.get("target_frankenhomie_commit"):
        raise ValueError("Synthetic and protected seeds target different Frankenhomie commits")
    if target_contract != protected.get("target_contract"):
        raise ValueError("Synthetic and protected seeds target different contracts")

    exporter = PhaseAFixtureExporter(workspace)
    reference = PhaseAReferenceValidator(workspace)
    adapter = PhaseAResidualAdapter()
    cache_root = (
        case_cache_dir.expanduser().resolve()
        if case_cache_dir is not None
        else None
    )
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
    cache_hits = 0
    cache_misses = 0
    source_id_prefix = str(
        seed.get("source_id_prefix") or "synthetic-train-dev-v1"
    )

    rows: list[dict[str, object]] = []
    case_results: list[dict[str, object]] = []
    split_counts: Counter[str] = Counter()
    errors: list[dict[str, object]] = []
    scenes = _mapping(seed, "scenes")

    for raw_case in _sequence_mapping(seed, "cases"):
        case_id = _required_text(raw_case, "case_id")
        split = DatasetSplit(_required_text(raw_case, "split"))
        scene_id = _required_text(raw_case, "scene_id")
        scene_raw = scenes.get(scene_id)
        if not isinstance(scene_raw, Mapping):
            raise ValueError(f"Unknown synthetic scene {scene_id!r}")
        fixture = {
            "fixture_id": f"dataset:{case_id}",
            "fixture_kind": "NON_CANON_FIXTURE",
            "production_data": False,
            "declaration": _required_text(raw_case, "declaration"),
            "actor_id": _required_text(seed, "actor_id"),
            "audience": _required_text(seed, "audience"),
            "snapshot_revision": _required_text(scene_raw, "snapshot_revision"),
            "facts": list(_sequence_mapping(scene_raw, "facts")),
        }
        receipt, cache_hit = _cached_export(
            exporter=exporter,
            repository=frankenhomie_repository,
            target_commit=target_commit,
            fixture=fixture,
            cache_root=cache_root,
        )
        if cache_hit:
            cache_hits += 1
        else:
            cache_misses += 1
        result: dict[str, object] = {
            "case_id": case_id,
            "split": split.value,
            "status": receipt.status,
            "route": receipt.route,
            "failed_deterministic_stage": receipt.failed_deterministic_stage,
            "request_sha256": receipt.request_sha256,
            "fixture_sha256": receipt.fixture_sha256,
            "error_code": receipt.error_code,
            "error_message": receipt.error_message,
        }
        case_results.append(result)

        if receipt.status != "RESIDUAL_EXPORTED" or receipt.request is None:
            errors.append(
                {
                    "case_id": case_id,
                    "code": "NOT_RESIDUAL_EXPORTED",
                    "detail": {
                        "status": receipt.status,
                        "route": receipt.route,
                        "error_code": receipt.error_code,
                    },
                }
            )
            continue

        hidden_leaks = _hidden_fixture_leaks(
            facts=_sequence_mapping(scene_raw, "facts"),
            request=receipt.request,
        )
        if hidden_leaks:
            errors.append(
                {
                    "case_id": case_id,
                    "code": "HIDDEN_FIXTURE_FACT_LEAK",
                    "detail": hidden_leaks,
                }
            )
            continue

        expected = _proposal_from_author_label(raw_case)
        validation = adapter.validate_proposal(receipt.request, expected)
        if not validation.accepted:
            errors.append(
                {
                    "case_id": case_id,
                    "code": "LABEL_OUT_OF_ENVELOPE",
                    "detail": validation.error_code,
                }
            )
            continue

        pinned = reference.validate(
            repository=frankenhomie_repository,
            ref=target_commit,
            request=receipt.request,
            proposal=expected,
        )
        result["reference_status"] = pinned.status
        result["reference_error_code"] = pinned.error_code
        if pinned.status != "ACCEPTED":
            errors.append(
                {
                    "case_id": case_id,
                    "code": "PINNED_LABEL_REJECTED",
                    "detail": pinned.error_code,
                }
            )
            continue

        row: dict[str, object] = {
            "example_id": case_id,
            "split": split.value,
            "source_id": f"{source_id_prefix}:{case_id}",
            "lineage_group": _required_text(raw_case, "lineage_group"),
            "request": receipt.request,
            "expected": expected,
            "tags": [str(value) for value in _sequence(raw_case, "tags")],
        }
        rows.append(row)
        split_counts[split.value] += 1

    full_report = _full_payload_leakage(rows)
    language_report = _language_leakage(rows, protected)
    if full_report.has_blockers:
        errors.append(
            {
                "case_id": None,
                "code": "FULL_PAYLOAD_LEAKAGE",
                "detail": full_report.to_dict(),
            }
        )
    if language_report.has_blockers:
        errors.append(
            {
                "case_id": None,
                "code": "PROTECTED_LANGUAGE_LEAKAGE",
                "detail": language_report.to_dict(),
            }
        )

    output_root = output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    full_report_path = output_root / "leakage-full-payload.json"
    language_report_path = output_root / "leakage-protected-language.json"
    full_report_path.write_text(
        canonical_json(full_report.to_dict()) + "\n",
        encoding="utf-8",
    )
    language_report_path.write_text(
        canonical_json(language_report.to_dict()) + "\n",
        encoding="utf-8",
    )

    dataset_path = output_root / "phase-a-train-dev-v1.jsonl"
    if errors:
        if dataset_path.exists():
            dataset_path.unlink()
        dataset_sha: str | None = None
    else:
        ordered_rows = sorted(
            rows,
            key=lambda item: str(item["example_id"]),
        )
        serialized = "".join(
            canonical_json(row) + "\n"
            for row in ordered_rows
        )
        dataset_bytes = serialized.encode("utf-8")
        dataset_path.write_bytes(dataset_bytes)
        dataset_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    base_receipt: dict[str, object] = {
        "schema": "ml-lab-phase-a-dataset-factory-receipt/1",
        "ok": not errors,
        "target_frankenhomie_commit": target_commit,
        "target_contract": target_contract,
        "seed_sha256": hashlib.sha256(seed_bytes).hexdigest(),
        "protected_seed_sha256": hashlib.sha256(protected_bytes).hexdigest(),
        "case_count": len(_sequence_mapping(seed, "cases")),
        "emitted_count": len(rows) if not errors else 0,
        "split_counts": dict(sorted(split_counts.items())) if not errors else {},
        "dataset_file": dataset_path.name if not errors else None,
        "dataset_sha256": dataset_sha,
        "full_payload_leakage": full_report.to_dict(),
        "protected_language_leakage": language_report.to_dict(),
        "case_results": case_results,
        "errors": errors,
        "case_cache": {
            "enabled": cache_root is not None,
            "hits": cache_hits,
            "misses": cache_misses,
        },
        "production_data": False,
        "transcript_derived": False,
        "trainer_visible_splits": ["TRAIN", "DEV"],
        "protected_splits_in_output": False,
        "candidate_training_data": not errors,
        "frozen": False,
        "integration_gate": "NO_GO",
    }
    payload_sha = hashlib.sha256(
        canonical_json(base_receipt).encode("utf-8")
    ).hexdigest()
    factory_receipt: dict[str, object] = {
        **base_receipt,
        "payload_sha256": payload_sha,
    }
    (output_root / "factory-receipt.json").write_text(
        canonical_json(factory_receipt) + "\n",
        encoding="utf-8",
    )
    return factory_receipt


def _cached_export(
    *,
    exporter: PhaseAFixtureExporter,
    repository: Path,
    target_commit: str,
    fixture: Mapping[str, object],
    cache_root: Path | None,
) -> tuple[PhaseAFixtureExportReceipt, bool]:
    input_sha = hashlib.sha256(
        canonical_json(fixture).encode("utf-8")
    ).hexdigest()
    cache_path = (
        cache_root / f"{target_commit[:12]}-{input_sha}.json"
        if cache_root is not None
        else None
    )
    if cache_path is not None and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(cached, dict):
            raise ValueError("Phase A case cache entry must be a JSON object")
        if cached.get("schema") != "ml-lab-phase-a-factory-case-cache/1":
            raise ValueError("Unsupported Phase A case cache schema")
        if cached.get("input_sha256") != input_sha:
            raise ValueError("Phase A case cache input hash mismatch")
        if cached.get("target_commit") != target_commit:
            raise ValueError("Phase A case cache target commit mismatch")
        raw_receipt = cached.get("receipt")
        if not isinstance(raw_receipt, dict):
            raise ValueError("Phase A case cache receipt must be an object")
        receipt = _receipt_from_cache(raw_receipt)
        _validate_cached_receipt(
            receipt,
            target_commit=target_commit,
            fixture_id=str(fixture["fixture_id"]),
        )
        return receipt, True

    receipt = exporter.export(
        repository=repository,
        ref=target_commit,
        fixture=fixture,
    )
    if cache_path is not None:
        payload = {
            "schema": "ml-lab-phase-a-factory-case-cache/1",
            "input_sha256": input_sha,
            "target_commit": target_commit,
            "fixture_id": fixture["fixture_id"],
            "receipt": _receipt_to_cache(receipt),
        }
        cache_path.write_bytes(
            (canonical_json(payload) + "\n").encode("utf-8")
        )
    return receipt, False


def _receipt_to_cache(
    receipt: PhaseAFixtureExportReceipt,
) -> dict[str, object]:
    return {
        "status": receipt.status,
        "commit_sha": receipt.commit_sha,
        "contract_version": receipt.contract_version,
        "fixture_id": receipt.fixture_id,
        "fixture_sha256": receipt.fixture_sha256,
        "route": receipt.route,
        "failed_deterministic_stage": receipt.failed_deterministic_stage,
        "request_sha256": receipt.request_sha256,
        "request": receipt.request,
        "fresh_process": receipt.fresh_process,
        "ephemeral_sqlite_only": receipt.ephemeral_sqlite_only,
        "network_access_allowed": receipt.network_access_allowed,
        "resolver_dispatch_available": receipt.resolver_dispatch_available,
        "authority_mutation_allowed": receipt.authority_mutation_allowed,
        "error_code": receipt.error_code,
        "error_message": receipt.error_message,
        "receipt_artifact_digest": receipt.receipt_artifact_digest,
    }


def _receipt_from_cache(
    value: Mapping[str, object],
) -> PhaseAFixtureExportReceipt:
    request_raw = value.get("request")
    if request_raw is not None and not isinstance(request_raw, dict):
        raise ValueError("Cached Phase A request must be an object")
    return PhaseAFixtureExportReceipt(
        status=str(value["status"]),
        commit_sha=str(value["commit_sha"]),
        contract_version=str(value["contract_version"]),
        fixture_id=str(value["fixture_id"]),
        fixture_sha256=str(value["fixture_sha256"]),
        route=(str(value["route"]) if value.get("route") is not None else None),
        failed_deterministic_stage=(
            str(value["failed_deterministic_stage"])
            if value.get("failed_deterministic_stage") is not None
            else None
        ),
        request_sha256=(
            str(value["request_sha256"])
            if value.get("request_sha256") is not None
            else None
        ),
        request=(
            {str(key): item for key, item in request_raw.items()}
            if isinstance(request_raw, dict)
            else None
        ),
        fresh_process=value.get("fresh_process") is True,
        ephemeral_sqlite_only=value.get("ephemeral_sqlite_only") is True,
        network_access_allowed=value.get("network_access_allowed") is True,
        resolver_dispatch_available=value.get("resolver_dispatch_available") is True,
        authority_mutation_allowed=value.get("authority_mutation_allowed") is True,
        error_code=(
            str(value["error_code"])
            if value.get("error_code") is not None
            else None
        ),
        error_message=(
            str(value["error_message"])
            if value.get("error_message") is not None
            else None
        ),
        receipt_artifact_digest=str(value["receipt_artifact_digest"]),
    )


def _validate_cached_receipt(
    receipt: PhaseAFixtureExportReceipt,
    *,
    target_commit: str,
    fixture_id: str,
) -> None:
    if receipt.commit_sha != target_commit:
        raise ValueError("Cached Phase A receipt commit mismatch")
    if receipt.contract_version != "semantic-residual-v2":
        raise ValueError("Cached Phase A receipt contract mismatch")
    if receipt.fixture_id != fixture_id:
        raise ValueError("Cached Phase A receipt fixture mismatch")
    if receipt.request is None:
        if receipt.request_sha256 is not None:
            raise ValueError("Cached missing request cannot have a request hash")
        return
    PhaseAResidualAdapter().validate_exported_request(receipt.request)
    request_sha = hashlib.sha256(
        canonical_json(receipt.request).encode("utf-8")
    ).hexdigest()
    if request_sha != receipt.request_sha256:
        raise ValueError("Cached Phase A request hash mismatch")


def _hidden_fixture_leaks(
    *,
    facts: Sequence[Mapping[str, object]],
    request: Mapping[str, object],
) -> list[dict[str, str]]:
    visible_strings: set[str] = set()
    context = request.get("context")
    if isinstance(context, Mapping):
        visible_strings.update(_all_strings(context))
    assessment = request.get("assessment")
    if isinstance(assessment, Mapping):
        candidates = assessment.get("candidates")
        if candidates is not None:
            visible_strings.update(_all_strings(candidates))
    family_slots = request.get("family_slots")
    if family_slots is not None:
        visible_strings.update(_all_strings(family_slots))
    leaks: list[dict[str, str]] = []
    for fact in facts:
        if fact.get("visibility") != "GM_ONLY":
            continue
        key = fact.get("key")
        if isinstance(key, str) and key in visible_strings:
            leaks.append({"kind": "KEY", "value": key})
        value = fact.get("value")
        if isinstance(value, str) and value.strip() and value in visible_strings:
            leaks.append({"kind": "VALUE", "value": value})
    return leaks


def _all_strings(value: object) -> set[str]:
    strings: set[str] = set()
    if isinstance(value, str):
        strings.add(value)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            strings.add(str(key))
            strings.update(_all_strings(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            strings.update(_all_strings(item))
    return strings


def _proposal_from_author_label(case: Mapping[str, object]) -> dict[str, object]:
    expected = _mapping(case, "expected")
    decision = _required_text(expected, "decision")
    family_raw = expected.get("action_family")
    family = str(family_raw) if family_raw is not None else None
    slots = [
        {"name": _required_text(slot, "name"), "value": _required_text(slot, "value")}
        for slot in _sequence_mapping(expected, "slots")
    ]
    if decision == "RESOLVE":
        return {
            "version": "semantic-residual-v2",
            "decision": "RESOLVE",
            "action_family": family,
            "entity_keys": [],
            "slots": slots,
            "reason_code": "SYNTHETIC_AUTHOR_LABEL",
            "facts_used": [],
            "question": None,
            "missing_slots": [],
            "candidate_keys": [],
        }
    if decision != "ASK_PLAYER":
        raise ValueError(f"Unsupported synthetic decision {decision!r}")
    return {
        "version": "semantic-residual-v2",
        "decision": "ASK_PLAYER",
        "action_family": family,
        "entity_keys": [],
        "slots": slots,
        "reason_code": "SYNTHETIC_AUTHOR_LABEL",
        "facts_used": [],
        "question": _required_text(expected, "question"),
        "missing_slots": [
            str(value) for value in _sequence(expected, "missing_slots")
        ],
        "candidate_keys": [
            str(value) for value in _sequence(expected, "candidate_keys")
        ],
    }


def _full_payload_leakage(
    rows: Sequence[Mapping[str, object]],
) -> LeakageReport:
    examples = []
    for row in rows:
        split = DatasetSplit(str(row["split"]))
        payload = {"request": row["request"]}
        examples.append(
            LeakageExample(
                example_id=str(row["example_id"]),
                split=split,
                lineage_group=str(row["lineage_group"]),
                fingerprint=content_fingerprint(payload),
                normalized_fingerprint=normalized_fingerprint(payload),
                near_signature=near_signature(payload),
            )
        )
    return scan_leakage(examples)


def _language_leakage(
    rows: Sequence[Mapping[str, object]],
    protected: Mapping[str, object],
) -> LeakageReport:
    examples: list[LeakageExample] = []
    for row in rows:
        request = _mapping(row, "request")
        assessment = _mapping(request, "assessment")
        declaration = _required_text(assessment, "declaration")
        examples.append(
            _text_leakage_example(
                example_id=str(row["example_id"]),
                split=DatasetSplit(str(row["split"])),
                lineage_group=str(row["lineage_group"]),
                declaration=declaration,
            )
        )
    for raw in _sequence_mapping(protected, "cases"):
        split = DatasetSplit(_required_text(raw, "split"))
        examples.append(
            _text_leakage_example(
                example_id=f"protected:{_required_text(raw, 'case_id')}",
                split=split,
                lineage_group=f"protected:{_required_text(raw, 'lineage_group')}",
                declaration=_required_text(raw, "declaration"),
            )
        )
    return scan_leakage(examples)


def _text_leakage_example(
    *,
    example_id: str,
    split: DatasetSplit,
    lineage_group: str,
    declaration: str,
) -> LeakageExample:
    payload = {"declaration": declaration}
    return LeakageExample(
        example_id=example_id,
        split=split,
        lineage_group=lineage_group,
        fingerprint=content_fingerprint(payload),
        normalized_fingerprint=normalized_fingerprint(payload),
        near_signature=near_signature(payload),
    )


def _validate_seed(seed: Mapping[str, object]) -> None:
    if seed.get("schema") != "ml-lab-phase-a-synthetic-seed/1":
        raise ValueError("Unsupported Phase A synthetic seed schema")
    if seed.get("status") != "SYNTHETIC_NON_CANON":
        raise ValueError("Phase A training seed must be synthetic non-canon")
    if seed.get("production_data") is not False:
        raise ValueError("Phase A training seed cannot use Production data")
    if seed.get("transcript_derived") is not False:
        raise ValueError("Phase A training seed cannot be transcript-derived")
    if seed.get("target_contract") != "semantic-residual-v2":
        raise ValueError("Phase A training seed must target semantic-residual-v2")
    scenes = _mapping(seed, "scenes")
    if not scenes:
        raise ValueError("Phase A training seed requires scenes")
    for scene_id, raw_scene in scenes.items():
        if not isinstance(raw_scene, Mapping):
            raise ValueError(f"Scene {scene_id!r} must be an object")
        facts = _sequence_mapping(raw_scene, "facts")
        if not facts:
            raise ValueError(f"Scene {scene_id!r} requires fixture facts")
        for fact in facts:
            source = fact.get("source")
            if not isinstance(source, str) or not source.startswith("fixture:"):
                raise ValueError("Synthetic seed facts must use fixture sources")
    cases = _sequence_mapping(seed, "cases")
    if not cases:
        raise ValueError("Phase A training seed requires cases")
    ids: set[str] = set()
    for case in cases:
        case_id = _required_text(case, "case_id")
        if case_id in ids:
            raise ValueError("Synthetic seed case IDs must be unique")
        ids.add(case_id)
        split = DatasetSplit(_required_text(case, "split"))
        if split not in {DatasetSplit.TRAIN, DatasetSplit.DEV}:
            raise ValueError("Synthetic factory accepts TRAIN and DEV only")
        if _required_text(case, "scene_id") not in scenes:
            raise ValueError(f"Synthetic case {case_id} references an unknown scene")
        _mapping(case, "expected")


def _validate_protected_seed(protected: Mapping[str, object]) -> None:
    if protected.get("schema") != "ml-lab-phase-a-authoritative-fixture-seed/1":
        raise ValueError("Unsupported protected Phase A seed schema")
    if protected.get("status") != "PROTECTED_EVALUATION_ONLY":
        raise ValueError("Protected Phase A seed status changed")
    for case in _sequence_mapping(protected, "cases"):
        split = DatasetSplit(_required_text(case, "split"))
        if split not in {DatasetSplit.TEST, DatasetSplit.REDTEAM}:
            raise ValueError("Protected Phase A seed must remain TEST/REDTEAM only")
        if case.get("training_allowed") is not False:
            raise ValueError("Protected Phase A cases cannot become training data")


def _required_text(value: Mapping[str, object], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return raw.strip()


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
