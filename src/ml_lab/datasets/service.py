from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from ml_lab.core.models import (
    TRAINER_VISIBLE_SPLITS,
    DatasetExample,
    DatasetPartition,
    DatasetSplit,
    DatasetState,
    DatasetVersion,
    utc_now_iso,
)
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


@dataclass(frozen=True, slots=True)
class ImportErrorRecord:
    line_number: int
    code: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "line_number": self.line_number,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ImportSummary:
    imported: int
    rejected: int
    errors: tuple[ImportErrorRecord, ...]
    import_id: str | None = None
    error_report_artifact_digest: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "imported": self.imported,
            "rejected": self.rejected,
            "errors": [error.to_dict() for error in self.errors],
            "import_id": self.import_id,
            "error_report_artifact_digest": self.error_report_artifact_digest,
        }


@dataclass(frozen=True, slots=True)
class ValidatedExampleInput:
    example_id: str
    split: DatasetSplit
    source_id: str
    lineage_group: str
    payload: object
    label: object
    tags: tuple[str, ...]


ExampleValidator = Callable[[dict[str, object], int, str], ValidatedExampleInput]


class DatasetLeakageError(RuntimeError):
    def __init__(self, report_digest: str, blocking_count: int):
        super().__init__(
            f"Dataset freeze blocked by {blocking_count} cross-split leakage issue(s). "
            f"Report artifact: {report_digest}"
        )
        self.report_digest = report_digest
        self.blocking_count = blocking_count


class DatasetService:
    """Own draft examples and immutable frozen partitions.

    Trainer-facing handles are intentionally separate from evaluation handles. A trainer
    can receive TRAIN and optional DEV only; TEST and REDTEAM are absent by construction.
    """

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.database = workspace.database
        self.artifacts = workspace.artifacts

    def create(
        self,
        project_id: str,
        name: str,
        *,
        contract_snapshot_id: str | None = None,
    ) -> DatasetVersion:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Dataset name is required.")
        dataset = DatasetVersion(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name=clean_name,
            state=DatasetState.DRAFT,
            contract_snapshot_id=contract_snapshot_id,
            created_at=utc_now_iso(),
        )
        with self.database.transaction() as conn:
            conn.execute(
                "INSERT INTO dataset_versions"
                "(id,project_id,name,state,contract_snapshot_id,example_count,created_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    dataset.id,
                    dataset.project_id,
                    dataset.name,
                    dataset.state.value,
                    dataset.contract_snapshot_id,
                    dataset.example_count,
                    dataset.created_at,
                ),
            )
        return dataset

    def get(self, dataset_id: str) -> DatasetVersion:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT * FROM dataset_versions WHERE id=?", (dataset_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown dataset {dataset_id}")
        return _dataset_from_row(row)

    def list_for_project(self, project_id: str) -> list[DatasetVersion]:
        with self.database.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM dataset_versions "
                "WHERE project_id=? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        return [_dataset_from_row(row) for row in rows]

    def import_jsonl(
        self,
        dataset_id: str,
        source: Path,
        *,
        validator: ExampleValidator | None = None,
        max_reported_errors: int = 1000,
    ) -> ImportSummary:
        dataset = self.get(dataset_id)
        if dataset.state is not DatasetState.DRAFT:
            raise RuntimeError("Only DRAFT datasets can accept imported examples.")
        errors: list[ImportErrorRecord] = []
        imported = 0
        rejected = 0
        source_name = source.name

        with (
            source.open("r", encoding="utf-8-sig") as handle,
            self.database.transaction() as conn,
        ):
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    decoded = json.loads(raw_line)
                    if not isinstance(decoded, dict):
                        raise ValueError("JSONL row must be an object.")
                    row = {str(key): value for key, value in decoded.items()}
                    item = validate_example_row(
                        row,
                        line_number,
                        source_name,
                        validator=validator,
                    )
                    _insert_example(conn, dataset_id, item)
                    imported += 1
                except (
                    json.JSONDecodeError,
                    ValueError,
                    TypeError,
                    sqlite3.IntegrityError,
                ) as exc:
                    rejected += 1
                    if len(errors) < max_reported_errors:
                        errors.append(
                            ImportErrorRecord(
                                line_number=line_number,
                                code=_import_error_code(exc),
                                message=str(exc),
                            )
                        )
            conn.execute(
                "UPDATE dataset_versions SET example_count=("
                "SELECT COUNT(*) FROM dataset_examples WHERE dataset_id=?"
                ") WHERE id=?",
                (dataset_id, dataset_id),
            )

        import_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        error_report_digest: str | None = None
        if rejected:
            report = {
                "format_version": 1,
                "import_id": import_id,
                "dataset_id": dataset_id,
                "source_name": source_name,
                "imported": imported,
                "rejected": rejected,
                "errors_truncated": rejected > len(errors),
                "errors": [error.to_dict() for error in errors],
            }
            ref = self.artifacts.commit_bytes(
                (canonical_json(report) + "\n").encode("utf-8"),
                media_type="application/vnd.ml-lab.import-errors+json",
                metadata={"dataset_id": dataset_id, "import_id": import_id},
            )
            error_report_digest = ref.digest
        with self.database.transaction() as conn:
            conn.execute(
                "INSERT INTO dataset_imports"
                "(id,dataset_id,source_name,imported,rejected,"
                "error_report_artifact_digest,created_at) VALUES(?,?,?,?,?,?,?)",
                (
                    import_id,
                    dataset_id,
                    source_name,
                    imported,
                    rejected,
                    error_report_digest,
                    created_at,
                ),
            )
        return ImportSummary(
            imported=imported,
            rejected=rejected,
            errors=tuple(errors),
            import_id=import_id,
            error_report_artifact_digest=error_report_digest,
        )

    def commit_prepared_import(
        self,
        dataset_id: str,
        staged_database: Path,
        prepared_summary: Mapping[str, object],
        *,
        max_reported_errors: int = 1000,
    ) -> ImportSummary:
        """Commit a child-prepared import into authoritative Lab metadata.

        The child process may parse, validate, fingerprint, and stage rows, but it never
        mutates the Lab workspace database. This parent-side method performs the only
        authoritative dataset mutation.
        """
        dataset = self.get(dataset_id)
        if dataset.state is not DatasetState.DRAFT:
            raise RuntimeError("Only DRAFT datasets can accept imported examples.")
        if not staged_database.is_file():
            raise FileNotFoundError(staged_database)

        source_name_raw = prepared_summary.get("source_name")
        if not isinstance(source_name_raw, str) or not source_name_raw:
            raise ValueError("Prepared import summary is missing source_name.")
        source_name = source_name_raw
        rejected_raw = prepared_summary.get("rejected")
        if isinstance(rejected_raw, bool) or not isinstance(rejected_raw, int):
            raise ValueError("Prepared import summary rejected count is invalid.")
        prepared_rejected = rejected_raw
        errors = _prepared_import_errors(
            prepared_summary.get("errors"),
            limit=max_reported_errors,
        )

        duplicate_count = 0
        candidate_count = 0
        with self.database.transaction() as conn:
            conn.execute("ATTACH DATABASE ? AS prepared", (str(staged_database),))
            candidate_row = conn.execute(
                "SELECT COUNT(*) FROM prepared.prepared_examples"
            ).fetchone()
            candidate_count = int(candidate_row[0]) if candidate_row else 0
            duplicate_row = conn.execute(
                "SELECT COUNT(*) FROM prepared.prepared_examples p "
                "WHERE EXISTS ("
                "SELECT 1 FROM dataset_examples d "
                "WHERE d.dataset_id=? AND d.example_id=p.example_id"
                ")",
                (dataset_id,),
            ).fetchone()
            duplicate_count = int(duplicate_row[0]) if duplicate_row else 0

            remaining = max(0, max_reported_errors - len(errors))
            if remaining:
                duplicate_rows = conn.execute(
                    "SELECT p.line_number,p.example_id "
                    "FROM prepared.prepared_examples p "
                    "WHERE EXISTS ("
                    "SELECT 1 FROM dataset_examples d "
                    "WHERE d.dataset_id=? AND d.example_id=p.example_id"
                    ") ORDER BY p.line_number LIMIT ?",
                    (dataset_id, remaining),
                ).fetchall()
                errors.extend(
                    ImportErrorRecord(
                        line_number=int(row["line_number"]),
                        code="DUPLICATE_OR_CONSTRAINT",
                        message=(
                            f"example_id {row['example_id']!r} already exists "
                            "in this dataset"
                        ),
                    )
                    for row in duplicate_rows
                )

            conn.execute(
                "INSERT INTO dataset_examples("
                "dataset_id,example_id,split,source_id,lineage_group,fingerprint,"
                "normalized_fingerprint,near_signature,payload_json,label_json,"
                "tags_json,created_at"
                ") SELECT ?,p.example_id,p.split,p.source_id,p.lineage_group,"
                "p.fingerprint,p.normalized_fingerprint,p.near_signature,"
                "p.payload_json,p.label_json,p.tags_json,p.created_at "
                "FROM prepared.prepared_examples p "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM dataset_examples d "
                "WHERE d.dataset_id=? AND d.example_id=p.example_id"
                ") ORDER BY p.line_number",
                (dataset_id, dataset_id),
            )
            conn.execute(
                "UPDATE dataset_versions SET example_count=("
                "SELECT COUNT(*) FROM dataset_examples WHERE dataset_id=?"
                ") WHERE id=?",
                (dataset_id, dataset_id),
            )

        imported = candidate_count - duplicate_count
        rejected = prepared_rejected + duplicate_count
        errors.sort(key=lambda item: item.line_number)
        if len(errors) > max_reported_errors:
            errors = errors[:max_reported_errors]

        import_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        error_report_digest: str | None = None
        if rejected:
            report = {
                "format_version": 1,
                "import_id": import_id,
                "dataset_id": dataset_id,
                "source_name": source_name,
                "imported": imported,
                "rejected": rejected,
                "errors_truncated": rejected > len(errors),
                "errors": [error.to_dict() for error in errors],
            }
            ref = self.artifacts.commit_bytes(
                (canonical_json(report) + "\n").encode("utf-8"),
                media_type="application/vnd.ml-lab.import-errors+json",
                metadata={"dataset_id": dataset_id, "import_id": import_id},
            )
            error_report_digest = ref.digest

        with self.database.transaction() as conn:
            conn.execute(
                "INSERT INTO dataset_imports"
                "(id,dataset_id,source_name,imported,rejected,"
                "error_report_artifact_digest,created_at) VALUES(?,?,?,?,?,?,?)",
                (
                    import_id,
                    dataset_id,
                    source_name,
                    imported,
                    rejected,
                    error_report_digest,
                    created_at,
                ),
            )
        return ImportSummary(
            imported=imported,
            rejected=rejected,
            errors=tuple(errors),
            import_id=import_id,
            error_report_artifact_digest=error_report_digest,
        )

    def add_example(self, dataset_id: str, item: ValidatedExampleInput) -> None:
        dataset = self.get(dataset_id)
        if dataset.state is not DatasetState.DRAFT:
            raise RuntimeError("Only DRAFT datasets can be edited.")
        with self.database.transaction() as conn:
            _insert_example(conn, dataset_id, item)
            conn.execute(
                "UPDATE dataset_versions SET example_count=example_count+1 WHERE id=?",
                (dataset_id,),
            )

    def page_examples(
        self,
        dataset_id: str,
        *,
        split: DatasetSplit | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> list[DatasetExample]:
        if offset < 0:
            raise ValueError("offset must be >= 0")
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        where = "dataset_id=?"
        params: list[object] = [dataset_id]
        if split is not None:
            where += " AND split=?"
            params.append(split.value)
        params.extend((limit, offset))
        with self.database.connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM dataset_examples WHERE {where} "
                "ORDER BY example_id LIMIT ? OFFSET ?",
                tuple(params),
            ).fetchall()
        return [_example_from_row(row) for row in rows]

    def scan_leakage(self, dataset_id: str) -> LeakageReport:
        with self.database.connection() as conn:
            cursor = conn.execute(
                "SELECT example_id,split,lineage_group,fingerprint,"
                "normalized_fingerprint,near_signature FROM dataset_examples "
                "WHERE dataset_id=? ORDER BY example_id",
                (dataset_id,),
            )

            def examples() -> Iterator[LeakageExample]:
                for row in cursor:
                    yield LeakageExample(
                        example_id=row["example_id"],
                        split=DatasetSplit(row["split"]),
                        lineage_group=row["lineage_group"],
                        fingerprint=row["fingerprint"],
                        normalized_fingerprint=row["normalized_fingerprint"],
                        near_signature=row["near_signature"],
                    )

            return scan_leakage(examples())

    def freeze(self, dataset_id: str) -> DatasetVersion:
        dataset = self.get(dataset_id)
        if dataset.state is not DatasetState.DRAFT:
            raise RuntimeError("Only DRAFT datasets can be frozen.")
        if dataset.example_count == 0:
            raise ValueError("Cannot freeze an empty dataset.")

        report = self.scan_leakage(dataset_id)
        report_ref = self.artifacts.commit_bytes(
            (canonical_json(report.to_dict()) + "\n").encode("utf-8"),
            media_type="application/vnd.ml-lab.leakage-report+json",
            metadata={
                "dataset_id": dataset_id,
                "blocking_count": report.blocking_count,
            },
        )
        with self.database.transaction() as conn:
            conn.execute(
                "UPDATE dataset_versions "
                "SET leakage_report_artifact_digest=? WHERE id=?",
                (report_ref.digest, dataset_id),
            )
        if report.has_blockers:
            raise DatasetLeakageError(report_ref.digest, report.blocking_count)

        partition_refs: dict[DatasetSplit, tuple[str, int]] = {}
        with TemporaryDirectory(dir=self.workspace.root / "cache") as temp_dir:
            temp_root = Path(temp_dir)
            for split in DatasetSplit:
                path = temp_root / f"{split.value.lower()}.jsonl"
                count = self._write_partition(dataset_id, split, path)
                ref = self.artifacts.commit_file(
                    path,
                    media_type="application/x-ndjson",
                    metadata={
                        "dataset_id": dataset_id,
                        "split": split.value,
                        "example_count": count,
                    },
                )
                partition_refs[split] = (ref.digest, count)

        frozen_at = utc_now_iso()
        manifest = {
            "format_version": 1,
            "dataset_id": dataset.id,
            "project_id": dataset.project_id,
            "name": dataset.name,
            "contract_snapshot_id": dataset.contract_snapshot_id,
            "example_count": dataset.example_count,
            "frozen_at": frozen_at,
            "leakage_report_sha256": report_ref.digest,
            "partitions": {
                split.value: {"sha256": digest, "example_count": count}
                for split, (digest, count) in partition_refs.items()
            },
        }
        manifest_ref = self.artifacts.commit_bytes(
            (canonical_json(manifest) + "\n").encode("utf-8"),
            media_type="application/vnd.ml-lab.dataset-manifest+json",
            metadata={"dataset_id": dataset_id},
        )

        with self.database.transaction() as conn:
            for split, (digest, count) in partition_refs.items():
                conn.execute(
                    "INSERT INTO dataset_partitions"
                    "(dataset_id,split,artifact_digest,example_count,"
                    "partition_sha256,created_at) VALUES(?,?,?,?,?,?)",
                    (dataset_id, split.value, digest, count, digest, frozen_at),
                )
            cursor = conn.execute(
                "UPDATE dataset_versions SET state=?,manifest_artifact_digest=?,"
                "leakage_report_artifact_digest=?,frozen_at=? "
                "WHERE id=? AND state=?",
                (
                    DatasetState.FROZEN.value,
                    manifest_ref.digest,
                    report_ref.digest,
                    frozen_at,
                    dataset_id,
                    DatasetState.DRAFT.value,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Dataset state changed while freezing.")
        return self.get(dataset_id)

    def partitions(self, dataset_id: str) -> list[DatasetPartition]:
        with self.database.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM dataset_partitions "
                "WHERE dataset_id=? ORDER BY split",
                (dataset_id,),
            ).fetchall()
        return [
            DatasetPartition(
                dataset_id=row["dataset_id"],
                split=DatasetSplit(row["split"]),
                artifact_digest=row["artifact_digest"],
                example_count=row["example_count"],
                partition_sha256=row["partition_sha256"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def trainer_partition_handles(
        self,
        dataset_id: str,
        *,
        include_dev: bool = True,
    ) -> dict[str, str]:
        dataset = self.get(dataset_id)
        if dataset.state is not DatasetState.FROZEN:
            raise RuntimeError("Trainer jobs require a FROZEN dataset.")
        allowed = {DatasetSplit.TRAIN}
        if include_dev:
            allowed.add(DatasetSplit.DEV)
        result = {
            partition.split.value: partition.artifact_digest
            for partition in self.partitions(dataset_id)
            if partition.split in allowed
        }
        if DatasetSplit.TRAIN.value not in result:
            raise RuntimeError("Frozen dataset is missing a TRAIN partition.")
        visible = {split.value for split in TRAINER_VISIBLE_SPLITS}
        if not set(result).issubset(visible):
            raise AssertionError("Protected partition escaped into trainer handles.")
        return result

    def evaluation_partition_handles(self, dataset_id: str) -> dict[str, str]:
        dataset = self.get(dataset_id)
        if dataset.state is not DatasetState.FROZEN:
            raise RuntimeError("Evaluation requires a FROZEN dataset.")
        return {
            partition.split.value: partition.artifact_digest
            for partition in self.partitions(dataset_id)
        }

    def _write_partition(
        self,
        dataset_id: str,
        split: DatasetSplit,
        path: Path,
    ) -> int:
        count = 0
        with (
            self.database.connection() as conn,
            path.open("w", encoding="utf-8", newline="\n") as output,
        ):
            cursor = conn.execute(
                "SELECT example_id,split,source_id,lineage_group,payload_json,"
                "label_json,tags_json FROM dataset_examples "
                "WHERE dataset_id=? AND split=? ORDER BY example_id",
                (dataset_id, split.value),
            )
            for row in cursor:
                record = {
                    "example_id": row["example_id"],
                    "split": row["split"],
                    "source_id": row["source_id"],
                    "lineage_group": row["lineage_group"],
                    "payload": json.loads(row["payload_json"]),
                    "label": json.loads(row["label_json"]),
                    "tags": json.loads(row["tags_json"]),
                }
                output.write(canonical_json(record))
                output.write("\n")
                count += 1
        return count


def _insert_example(
    conn: sqlite3.Connection,
    dataset_id: str,
    item: ValidatedExampleInput,
) -> None:
    record = prepared_example_record(item)
    conn.execute(
        "INSERT INTO dataset_examples"
        "(dataset_id,example_id,split,source_id,lineage_group,fingerprint,"
        "normalized_fingerprint,near_signature,payload_json,label_json,"
        "tags_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            dataset_id,
            record["example_id"],
            record["split"],
            record["source_id"],
            record["lineage_group"],
            record["fingerprint"],
            record["normalized_fingerprint"],
            record["near_signature"],
            record["payload_json"],
            record["label_json"],
            record["tags_json"],
            record["created_at"],
        ),
    )


def validate_example_row(
    row: dict[str, object],
    line_number: int,
    source_name: str,
    *,
    validator: ExampleValidator | None = None,
) -> ValidatedExampleInput:
    return (validator or _generic_validator)(row, line_number, source_name)


def prepared_example_record(item: ValidatedExampleInput) -> dict[str, str]:
    return {
        "example_id": item.example_id,
        "split": item.split.value,
        "source_id": item.source_id,
        "lineage_group": item.lineage_group,
        "fingerprint": content_fingerprint(item.payload),
        "normalized_fingerprint": normalized_fingerprint(item.payload),
        "near_signature": near_signature(item.payload),
        "payload_json": canonical_json(item.payload),
        "label_json": canonical_json(item.label),
        "tags_json": canonical_json(list(item.tags)),
        "created_at": utc_now_iso(),
    }


def _prepared_import_errors(value: object, *, limit: int) -> list[ImportErrorRecord]:
    if not isinstance(value, list):
        raise ValueError("Prepared import summary errors must be a list.")
    result: list[ImportErrorRecord] = []
    for raw in value[:limit]:
        if not isinstance(raw, dict):
            raise ValueError("Prepared import error entry must be an object.")
        line_number = raw.get("line_number")
        code = raw.get("code")
        message = raw.get("message")
        if (
            isinstance(line_number, bool)
            or not isinstance(line_number, int)
            or line_number < 1
            or not isinstance(code, str)
            or not isinstance(message, str)
        ):
            raise ValueError("Prepared import error entry is invalid.")
        result.append(
            ImportErrorRecord(
                line_number=line_number,
                code=code,
                message=message,
            )
        )
    return result


def _generic_validator(
    row: dict[str, object],
    line_number: int,
    source_name: str,
) -> ValidatedExampleInput:
    example_id = row.get("example_id")
    if not isinstance(example_id, str) or not example_id.strip():
        raise ValueError("example_id must be a non-empty string")
    raw_split = row.get("split")
    if not isinstance(raw_split, str):
        raise ValueError("split must be a string")
    try:
        split = DatasetSplit(raw_split.upper())
    except ValueError as exc:
        raise ValueError(f"unsupported split {raw_split!r}") from exc
    if "payload" not in row:
        raise ValueError("payload is required")
    if "label" not in row:
        raise ValueError("label is required")
    source_id = row.get("source_id", f"{source_name}:{line_number}")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("source_id must be a non-empty string")
    lineage = row.get("lineage_group", example_id)
    if not isinstance(lineage, str) or not lineage.strip():
        raise ValueError("lineage_group must be a non-empty string")
    raw_tags = row.get("tags", [])
    if not isinstance(raw_tags, list) or not all(isinstance(tag, str) for tag in raw_tags):
        raise ValueError("tags must be a list of strings")
    return ValidatedExampleInput(
        example_id=example_id.strip(),
        split=split,
        source_id=source_id.strip(),
        lineage_group=lineage.strip(),
        payload=row["payload"],
        label=row["label"],
        tags=tuple(raw_tags),
    )


def _import_error_code(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return "INVALID_JSON"
    if isinstance(exc, sqlite3.IntegrityError):
        return "DUPLICATE_OR_CONSTRAINT"
    return "INVALID_EXAMPLE"


def _dataset_from_row(row: sqlite3.Row) -> DatasetVersion:
    return DatasetVersion(
        id=row["id"],
        project_id=row["project_id"],
        name=row["name"],
        state=DatasetState(row["state"]),
        contract_snapshot_id=row["contract_snapshot_id"],
        example_count=int(row["example_count"]),
        manifest_artifact_digest=row["manifest_artifact_digest"],
        leakage_report_artifact_digest=row["leakage_report_artifact_digest"],
        created_at=row["created_at"],
        frozen_at=row["frozen_at"],
    )


def _example_from_row(row: sqlite3.Row) -> DatasetExample:
    return DatasetExample(
        dataset_id=row["dataset_id"],
        example_id=row["example_id"],
        split=DatasetSplit(row["split"]),
        source_id=row["source_id"],
        lineage_group=row["lineage_group"],
        fingerprint=row["fingerprint"],
        normalized_fingerprint=row["normalized_fingerprint"],
        near_signature=row["near_signature"],
        payload_json=row["payload_json"],
        label_json=row["label_json"],
        tags_json=row["tags_json"],
        created_at=row["created_at"],
    )
