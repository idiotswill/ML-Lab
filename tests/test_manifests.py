import pytest

from ml_lab.adapters.phase_a import PHASE_A_ADAPTER_ID
from ml_lab.extensions.manifests import ExtensionManifest, ManifestError
from ml_lab.trainers.service import SPARSE_RUNTIME_PACK_ID, SPARSE_TRAINER_ID


def test_manifest_accepts_current_protocol() -> None:
    manifest = ExtensionManifest.from_dict(
        {
            "id": "sparse-v1",
            "version": "1.0.0",
            "protocol_version": 1,
            "display_name": "Sparse Runtime",
            "capabilities": ["train", "evaluate"],
            "config_schema": {"type": "object"},
        },
        expected_kind="runtime",
    )
    assert manifest.extension_id == "sparse-v1"


def test_manifest_rejects_incompatible_protocol() -> None:
    with pytest.raises(ManifestError, match="Incompatible protocol"):
        ExtensionManifest.from_dict(
            {
                "id": "future",
                "version": "9",
                "protocol_version": 9,
                "display_name": "Future",
            },
            expected_kind="adapter",
        )


def test_catalog_reports_bad_manifest_without_importing_code(tmp_path) -> None:
    from ml_lab.extensions.catalog import ExtensionCatalog

    bad_dir = tmp_path / "extensions" / "adapters"
    bad_dir.mkdir(parents=True)
    (bad_dir / "future.json").write_text(
        '{"id":"future","version":"1","protocol_version":99,"display_name":"Future"}',
        encoding="utf-8",
    )
    catalog = ExtensionCatalog(tmp_path)
    catalog.load()
    adapter_ids = {item["id"] for item in catalog.adapter_options()}
    assert adapter_ids == {"generic", PHASE_A_ADAPTER_ID}
    runtime_by_id = {
        item.extension_id: item for item in catalog.runtimes.list()
    }
    assert SPARSE_RUNTIME_PACK_ID in runtime_by_id
    assert SPARSE_TRAINER_ID in runtime_by_id[SPARSE_RUNTIME_PACK_ID].capabilities
    assert len(catalog.errors) == 1
