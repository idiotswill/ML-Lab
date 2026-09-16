from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Mapping

from ml_lab.core.models import (
    ExperimentStatus,
    ModelStage,
    RegisteredModel,
    utc_now_iso,
)
from ml_lab.datasets.leakage import canonical_json
from ml_lab.experiments.service import ExperimentService
from ml_lab.failures.service import FailureService
from ml_lab.storage.workspace import Workspace

_ALLOWED_NEXT: dict[ModelStage, ModelStage] = {
    ModelStage.EXPERIMENT: ModelStage.SHADOW,
    ModelStage.SHADOW: ModelStage.ADVISORY,
    ModelStage.ADVISORY: ModelStage.RELEASE_CANDIDATE,
}


class ModelRegistryService:
    """Versioned model registry with explicit, non-skippable promotion stages.

    Ordinary Lab code can promote through RELEASE_CANDIDATE only. There is intentionally
    no normal transition to INTEGRATION_APPROVED.
    """

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.database = workspace.database
        self.artifacts = workspace.artifacts
        self.experiments = ExperimentService(workspace)
        self.failures = FailureService(workspace)

    def register_from_experiment(
        self,
        experiment_id: str,
        *,
        compatibility: Mapping[str, object] | None = None,
    ) -> RegisteredModel:
        experiment = self.experiments.get(experiment_id)
        if experiment.status is not ExperimentStatus.COMPLETED:
            raise RuntimeError("Only completed experiments can register models.")
        if experiment.model_artifact_digest is None:
            raise RuntimeError("Experiment has no model artifact to register.")
        self.artifacts.resolve(experiment.model_artifact_digest)
        now = utc_now_iso()
        compatibility_json = canonical_json(dict(compatibility or {}))
        manifest = {
            "format_version": 1,
            "project_id": experiment.project_id,
            "experiment_id": experiment.id,
            "experiment_manifest_sha256": experiment.manifest_artifact_digest,
            "model_artifact_sha256": experiment.model_artifact_digest,
            "initial_stage": ModelStage.EXPERIMENT.value,
            "compatibility": json.loads(compatibility_json),
            "created_at": now,
        }
        manifest_ref = self.artifacts.commit_bytes(
            (canonical_json(manifest) + "\n").encode("utf-8"),
            media_type="application/vnd.ml-lab.model-manifest+json",
            metadata={
                "experiment_id": experiment.id,
                "project_id": experiment.project_id,
            },
        )
        model = RegisteredModel(
            id=str(uuid.uuid4()),
            project_id=experiment.project_id,
            experiment_id=experiment.id,
            model_artifact_digest=experiment.model_artifact_digest,
            stage=ModelStage.EXPERIMENT,
            compatibility_json=compatibility_json,
            manifest_artifact_digest=manifest_ref.digest,
            created_at=now,
            updated_at=now,
        )
        with self.database.transaction() as conn:
            conn.execute(
                "INSERT INTO models"
                "(id,project_id,experiment_id,model_artifact_digest,stage,"
                "compatibility_json,manifest_artifact_digest,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    model.id,
                    model.project_id,
                    model.experiment_id,
                    model.model_artifact_digest,
                    model.stage.value,
                    model.compatibility_json,
                    model.manifest_artifact_digest,
                    model.created_at,
                    model.updated_at,
                ),
            )
            conn.execute(
                "INSERT INTO model_stage_history"
                "(model_id,from_stage,to_stage,evidence_json,created_at) "
                "VALUES(?,?,?,?,?)",
                (model.id, None, ModelStage.EXPERIMENT.value, "{}", now),
            )
        return model

    def get(self, model_id: str) -> RegisteredModel:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT * FROM models WHERE id=?", (model_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown model {model_id}")
        return _model_from_row(row)

    def list_for_project(
        self,
        project_id: str,
        *,
        limit: int = 200,
    ) -> list[RegisteredModel]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self.database.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM models "
                "WHERE project_id=? ORDER BY updated_at DESC,id LIMIT ?",
                (project_id, limit),
            ).fetchall()
        return [_model_from_row(row) for row in rows]

    def promote(
        self,
        model_id: str,
        to_stage: ModelStage,
        *,
        evidence: Mapping[str, object] | None = None,
    ) -> RegisteredModel:
        model = self.get(model_id)
        if to_stage is ModelStage.INTEGRATION_APPROVED:
            raise PermissionError(
                "INTEGRATION_APPROVED cannot be granted by the ordinary ML Lab "
                "promotion API."
            )
        expected = _ALLOWED_NEXT.get(model.stage)
        if expected is None or to_stage is not expected:
            raise ValueError(
                f"Model stage must advance one step at a time; {model.stage.value} -> "
                f"{to_stage.value} is not allowed."
            )
        if to_stage is ModelStage.RELEASE_CANDIDATE:
            self._assert_release_candidate_eligible(model)

        now = utc_now_iso()
        evidence_json = canonical_json(dict(evidence or {}))
        with self.database.transaction() as conn:
            cursor = conn.execute(
                "UPDATE models SET stage=?,updated_at=? WHERE id=? AND stage=?",
                (to_stage.value, now, model.id, model.stage.value),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Model stage changed concurrently.")
            conn.execute(
                "INSERT INTO model_stage_history"
                "(model_id,from_stage,to_stage,evidence_json,created_at) "
                "VALUES(?,?,?,?,?)",
                (model.id, model.stage.value, to_stage.value, evidence_json, now),
            )
        return self.get(model.id)

    def stage_history(self, model_id: str) -> list[dict[str, object]]:
        self.get(model_id)
        with self.database.connection() as conn:
            rows = conn.execute(
                "SELECT from_stage,to_stage,evidence_json,created_at "
                "FROM model_stage_history WHERE model_id=? ORDER BY id",
                (model_id,),
            ).fetchall()
        return [
            {
                "from_stage": row["from_stage"],
                "to_stage": row["to_stage"],
                "evidence": json.loads(row["evidence_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _assert_release_candidate_eligible(self, model: RegisteredModel) -> None:
        metrics = self.experiments.metrics(model.experiment_id)
        failing_vetoes = [
            metric for metric in metrics if metric.veto and metric.value != 0.0
        ]
        if failing_vetoes:
            names = ", ".join(metric.metric_id for metric in failing_vetoes)
            raise RuntimeError(
                f"Release candidate blocked by non-zero veto metric(s): {names}."
            )
        open_vetoes = self.failures.open_veto_count(
            experiment_id=model.experiment_id
        )
        if open_vetoes:
            raise RuntimeError(
                f"Release candidate blocked by {open_vetoes} unresolved veto "
                "failure record(s)."
            )


def _model_from_row(row: sqlite3.Row) -> RegisteredModel:
    return RegisteredModel(
        id=row["id"],
        project_id=row["project_id"],
        experiment_id=row["experiment_id"],
        model_artifact_digest=row["model_artifact_digest"],
        stage=ModelStage(row["stage"]),
        compatibility_json=row["compatibility_json"],
        manifest_artifact_digest=row["manifest_artifact_digest"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
