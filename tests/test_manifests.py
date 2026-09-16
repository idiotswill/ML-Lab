import pytest

from ml_lab.extensions.manifests import ExtensionManifest, ManifestError


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
            {"id": "future", "version": "9", "protocol_version": 9, "display_name": "Future"},
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
    assert catalog.adapter_options()[0]["id"] == "generic"
    assert len(catalog.errors) == 1
