from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from ml_lab.adapters.phase_a_fixture_export import PhaseAFixtureExporter
from ml_lab.datasets.leakage import canonical_json
from ml_lab.storage.workspace import Workspace

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PHASE_A_FIXTURE_SEED = (
    _REPO_ROOT / "benchmarks" / "phase_a" / "authoritative_fixture_seed_v1.json"
)


def run_phase_a_fixture_seed_evidence(
    *,
    frankenhomie_repository: Path,
    output_path: Path,
    seed_path: Path = DEFAULT_PHASE_A_FIXTURE_SEED,
) -> dict[str, object]:
    """Run the protected seed against one exact Frankenhomie commit and freeze evidence."""
    seed_bytes = seed_path.read_bytes()
    seed = json.loads(seed_bytes)
    if not isinstance(seed, dict):
        raise ValueError("Phase A fixture seed must be a JSON object")
    _validate_seed(seed)

    target_commit = _required_text(seed, "target_frankenhomie_commit")
    target_contract = _required_text(seed, "target_contract")
    cases = seed["cases"]
    if not isinstance(cases, list):
        raise ValueError("Phase A fixture seed cases must be a list")

    evidence_rows: list[dict[str, object]] = []
    mismatches: list[dict[str, object]] = []
    status_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    request_count = 0

    with tempfile.TemporaryDirectory(prefix="ml-lab-phase-a-fixture-evidence-") as temp:
        workspace = Workspace.create(Path(temp) / "workspace")
        exporter = PhaseAFixtureExporter(workspace)
        for raw_case in cases:
            if not isinstance(raw_case, dict):
                raise ValueError("Phase A fixture case must be an object")
            fixture = _case_fixture(seed, raw_case)
            receipt = exporter.export(
                repository=frankenhomie_repository,
                ref=target_commit,
                fixture=fixture,
            )
            actual = {
                "status": receipt.status,
                "route": receipt.route,
                "failed_deterministic_stage": receipt.failed_deterministic_stage,
                "request_sha256": receipt.request_sha256,
                "request": receipt.request,
                "fixture_sha256": receipt.fixture_sha256,
                "commit_sha": receipt.commit_sha,
                "contract_version": receipt.contract_version,
                "fresh_process": receipt.fresh_process,
                "ephemeral_sqlite_only": receipt.ephemeral_sqlite_only,
                "network_access_allowed": receipt.network_access_allowed,
                "resolver_dispatch_available": receipt.resolver_dispatch_available,
                "authority_mutation_allowed": receipt.authority_mutation_allowed,
                "error_code": receipt.error_code,
                "error_message": receipt.error_message,
            }
            expected = {
                "status": raw_case["expected_status"],
                "route": raw_case["expected_route"],
                "failed_deterministic_stage": raw_case[
                    "expected_failed_deterministic_stage"
                ],
                "permitted_decisions": raw_case["expected_permitted_decisions"],
            }
            mismatch = _mismatch(expected, actual)
            status_counts[receipt.status] += 1
            split_counts[str(raw_case["split"])] += 1
            if receipt.request is not None:
                request_count += 1
            row = {
                "case_id": raw_case["case_id"],
                "split": raw_case["split"],
                "lineage_group": raw_case["lineage_group"],
                "declaration": raw_case["declaration"],
                "training_allowed": raw_case["training_allowed"],
                "tags": raw_case["tags"],
                "expected": expected,
                "actual": actual,
                "matches_expectation": mismatch is None,
            }
            evidence_rows.append(row)
            if mismatch is not None:
                mismatches.append(
                    {
                        "case_id": raw_case["case_id"],
                        "mismatch": mismatch,
                    }
                )

    base_payload: dict[str, object] = {
        "schema": "ml-lab-phase-a-fixture-evidence/1",
        "ok": not mismatches,
        "target_frankenhomie_commit": target_commit,
        "target_contract": target_contract,
        "seed_path": seed_path.name,
        "seed_sha256": hashlib.sha256(seed_bytes).hexdigest(),
        "case_count": len(evidence_rows),
        "request_count": request_count,
        "split_counts": dict(sorted(split_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "cases": evidence_rows,
        "training_allowed": False,
        "production_data": False,
        "integration_gate": "NO_GO",
    }
    payload_sha256 = hashlib.sha256(
        canonical_json(base_payload).encode("utf-8")
    ).hexdigest()
    payload = {**base_payload, "payload_sha256": payload_sha256}
    _write_atomic(output_path, canonical_json(payload) + "\n")
    return payload


def _case_fixture(
    seed: dict[str, object],
    case: dict[str, object],
) -> dict[str, object]:
    return {
        "fixture_id": case["case_id"],
        "fixture_kind": seed["fixture_kind"],
        "production_data": seed["production_data"],
        "declaration": case["declaration"],
        "actor_id": seed["actor_id"],
        "audience": seed["audience"],
        "snapshot_revision": seed["snapshot_revision"],
        "facts": seed["facts"],
    }


def _mismatch(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
) -> dict[str, object] | None:
    fields = ("status", "route", "failed_deterministic_stage")
    differences: dict[str, object] = {}
    for field in fields:
        if expected[field] != actual[field]:
            differences[field] = {
                "expected": expected[field],
                "actual": actual[field],
            }

    permitted = expected["permitted_decisions"]
    request = actual["request"]
    actual_permitted: object = None
    if isinstance(request, dict):
        actual_permitted = request.get("permitted_decisions")
    if permitted != actual_permitted:
        differences["permitted_decisions"] = {
            "expected": permitted,
            "actual": actual_permitted,
        }
    return differences or None


def _validate_seed(seed: dict[str, object]) -> None:
    if seed.get("schema") != "ml-lab-phase-a-authoritative-fixture-seed/1":
        raise ValueError("Unsupported Phase A fixture seed schema")
    if seed.get("status") != "PROTECTED_EVALUATION_ONLY":
        raise ValueError("Phase A fixture seed must remain protected evaluation only")
    if seed.get("target_contract") != "semantic-residual-v2":
        raise ValueError("Phase A fixture seed contract changed")
    if seed.get("fixture_kind") != "NON_CANON_FIXTURE":
        raise ValueError("Phase A fixture seed must be non-canon")
    if seed.get("production_data") is not False:
        raise ValueError("Phase A fixture seed cannot use Production data")
    cases = seed.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Phase A fixture seed requires cases")
    if any(
        not isinstance(case, dict)
        or case.get("split") not in {"TEST", "REDTEAM"}
        or case.get("training_allowed") is not False
        for case in cases
    ):
        raise ValueError("Phase A fixture seed cases must remain protected TEST/REDTEAM")


def _required_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Phase A fixture seed is missing {key}")
    return value


def _write_atomic(path: Path, text: str) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)
