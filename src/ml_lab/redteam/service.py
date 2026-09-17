from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ml_lab.core.models import ExperimentStatus, utc_now_iso
from ml_lab.datasets.leakage import canonical_json
from ml_lab.storage.workspace import Workspace


@dataclass(frozen=True, slots=True)
class RedTeamCase:
    case_id: str
    payload: object
    expected: object
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RedTeamMutator:
    mutator_id: str
    mutate: Callable[[object, random.Random], object]


@dataclass(frozen=True, slots=True)
class RedTeamRunRecord:
    id: str
    project_id: str
    seed: int
    mutator_version: str
    status: ExperimentStatus
    created_at: str
    dataset_id: str | None = None
    experiment_id: str | None = None
    manifest_artifact_digest: str | None = None
    completed_at: str | None = None


class RedTeamService:
    """Deterministic adapter-supplied mutation suites.

    Mutators own project semantics. The host only guarantees deterministic seed derivation,
    provenance, immutable generated-case artifacts, and truthful run state.
    """

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.database = workspace.database
        self.artifacts = workspace.artifacts

    def start_run(
        self,
        *,
        project_id: str,
        seed: int,
        mutator_version: str,
        dataset_id: str | None = None,
        experiment_id: str | None = None,
    ) -> RedTeamRunRecord:
        clean_version = mutator_version.strip()
        if not clean_version:
            raise ValueError("mutator_version is required")
        run_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        with self.database.transaction() as conn:
            conn.execute(
                "INSERT INTO redteam_runs"
                "(id,project_id,dataset_id,experiment_id,seed,mutator_version,status,created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    project_id,
                    dataset_id,
                    experiment_id,
                    seed,
                    clean_version,
                    ExperimentStatus.RUNNING.value,
                    created_at,
                ),
            )
        return self.get(run_id)

    def generate_cases(
        self,
        *,
        seed: int,
        base_cases: Sequence[RedTeamCase],
        mutators: Sequence[RedTeamMutator],
    ) -> list[dict[str, object]]:
        _validated_mutator_ids(mutators)
        generated: list[dict[str, object]] = []
        ordered_cases = sorted(base_cases, key=lambda case: case.case_id)
        ordered_mutators = sorted(mutators, key=lambda mutator: mutator.mutator_id)
        for case in ordered_cases:
            for mutator in ordered_mutators:
                derived_seed = _derived_seed(seed, case.case_id, mutator.mutator_id)
                rng = random.Random(derived_seed)
                payload = mutator.mutate(case.payload, rng)
                case_hash = hashlib.sha256(
                    canonical_json(
                        {
                            "base_case_id": case.case_id,
                            "mutator_id": mutator.mutator_id,
                            "derived_seed": derived_seed,
                            "payload": payload,
                            "expected": case.expected,
                        }
                    ).encode("utf-8")
                ).hexdigest()
                generated.append(
                    {
                        "case_id": f"rt-{case_hash[:24]}",
                        "base_case_id": case.case_id,
                        "mutator_id": mutator.mutator_id,
                        "derived_seed": derived_seed,
                        "payload": payload,
                        "expected": case.expected,
                        "tags": sorted(set(case.tags) | {f"mutator:{mutator.mutator_id}"}),
                    }
                )
        return generated

    def complete_run(
        self,
        run_id: str,
        *,
        generated_cases: Sequence[Mapping[str, object]],
        mutator_ids: Sequence[str],
        metadata: Mapping[str, object] | None = None,
    ) -> RedTeamRunRecord:
        run = self.get(run_id)
        if run.status is not ExperimentStatus.RUNNING:
            raise RuntimeError("Only RUNNING red-team runs can be completed.")
        clean_ids = _validated_mutator_id_values(mutator_ids)
        completed_at = utc_now_iso()
        manifest = {
            "format_version": 1,
            "run_id": run.id,
            "project_id": run.project_id,
            "dataset_id": run.dataset_id,
            "experiment_id": run.experiment_id,
            "seed": run.seed,
            "mutator_version": run.mutator_version,
            "mutators": sorted(clean_ids),
            "metadata": dict(metadata or {}),
            "generated_cases": [dict(case) for case in generated_cases],
            "created_at": run.created_at,
            "completed_at": completed_at,
        }
        ref = self.artifacts.commit_bytes(
            (canonical_json(manifest) + "\n").encode("utf-8"),
            media_type="application/vnd.ml-lab.redteam-manifest+json",
            metadata={"redteam_run_id": run.id, "project_id": run.project_id},
        )
        with self.database.transaction() as conn:
            cursor = conn.execute(
                "UPDATE redteam_runs SET status=?,manifest_artifact_digest=?,completed_at=? "
                "WHERE id=? AND status=?",
                (
                    ExperimentStatus.COMPLETED.value,
                    ref.digest,
                    completed_at,
                    run.id,
                    ExperimentStatus.RUNNING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Red-team run state changed concurrently.")
        return self.get(run.id)

    def interrupt_run(self, run_id: str) -> RedTeamRunRecord:
        return self._finish_without_manifest(run_id, ExperimentStatus.INTERRUPTED)

    def fail_run(self, run_id: str) -> RedTeamRunRecord:
        return self._finish_without_manifest(run_id, ExperimentStatus.FAILED)

    def run_suite(
        self,
        *,
        project_id: str,
        seed: int,
        mutator_version: str,
        base_cases: Sequence[RedTeamCase],
        mutators: Sequence[RedTeamMutator],
        dataset_id: str | None = None,
        experiment_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> RedTeamRunRecord:
        ids = _validated_mutator_ids(mutators)
        run = self.start_run(
            project_id=project_id,
            seed=seed,
            mutator_version=mutator_version,
            dataset_id=dataset_id,
            experiment_id=experiment_id,
        )
        try:
            generated = self.generate_cases(
                seed=seed,
                base_cases=base_cases,
                mutators=mutators,
            )
            return self.complete_run(
                run.id,
                generated_cases=generated,
                mutator_ids=ids,
                metadata=metadata,
            )
        except Exception:
            self.fail_run(run.id)
            raise

    def get(self, run_id: str) -> RedTeamRunRecord:
        with self.database.connection() as conn:
            row = conn.execute("SELECT * FROM redteam_runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown red-team run {run_id}")
        return _run_from_row(row)

    def page_for_project(
        self,
        project_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> list[RedTeamRunRecord]:
        if offset < 0:
            raise ValueError("offset must be >= 0")
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        with self.database.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM redteam_runs WHERE project_id=? "
                "ORDER BY created_at DESC,id LIMIT ? OFFSET ?",
                (project_id, limit, offset),
            ).fetchall()
        return [_run_from_row(row) for row in rows]

    def count_for_project(self, project_id: str) -> int:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM redteam_runs WHERE project_id=?",
                (project_id,),
            ).fetchone()
        return int(row[0]) if row else 0

    def generated_cases(self, run_id: str) -> list[dict[str, object]]:
        run = self.get(run_id)
        if run.manifest_artifact_digest is None:
            return []
        path = self.artifacts.resolve(run.manifest_artifact_digest)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        cases = manifest.get("generated_cases", [])
        if not isinstance(cases, list):
            raise ValueError("Red-team manifest generated_cases is not a list.")
        return [dict(case) for case in cases if isinstance(case, dict)]

    def _finish_without_manifest(
        self,
        run_id: str,
        status: ExperimentStatus,
    ) -> RedTeamRunRecord:
        if status not in {ExperimentStatus.INTERRUPTED, ExperimentStatus.FAILED}:
            raise ValueError("Unsupported terminal red-team status")
        self.get(run_id)
        with self.database.transaction() as conn:
            cursor = conn.execute(
                "UPDATE redteam_runs SET status=?,completed_at=? "
                "WHERE id=? AND status=?",
                (
                    status.value,
                    utc_now_iso(),
                    run_id,
                    ExperimentStatus.RUNNING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Red-team run state changed concurrently.")
        return self.get(run_id)


def _validated_mutator_ids(mutators: Sequence[RedTeamMutator]) -> list[str]:
    if not mutators:
        raise ValueError("At least one red-team mutator is required.")
    return _validated_mutator_id_values([mutator.mutator_id for mutator in mutators])


def _validated_mutator_id_values(mutator_ids: Sequence[str]) -> list[str]:
    ids = [mutator_id.strip() for mutator_id in mutator_ids]
    if not ids or any(not mutator_id for mutator_id in ids):
        raise ValueError("mutator_id cannot be empty")
    if len(ids) != len(set(ids)):
        raise ValueError("mutator_id values must be unique")
    return ids


def _derived_seed(seed: int, case_id: str, mutator_id: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{case_id}\0{mutator_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _run_from_row(row: sqlite3.Row) -> RedTeamRunRecord:
    return RedTeamRunRecord(
        id=row["id"],
        project_id=row["project_id"],
        dataset_id=row["dataset_id"],
        experiment_id=row["experiment_id"],
        seed=int(row["seed"]),
        mutator_version=row["mutator_version"],
        status=ExperimentStatus(row["status"]),
        manifest_artifact_digest=row["manifest_artifact_digest"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
    )
