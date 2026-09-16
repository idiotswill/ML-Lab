from __future__ import annotations

from dataclasses import dataclass
from typing import Any

HOST_PROTOCOL_VERSION = 1


class ManifestError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ExtensionManifest:
    kind: str
    extension_id: str
    version: str
    protocol_version: int
    display_name: str
    capabilities: tuple[str, ...]
    config_schema: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, expected_kind: str) -> "ExtensionManifest":
        required = ("id", "version", "protocol_version", "display_name")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ManifestError(f"Manifest missing required fields: {', '.join(missing)}")
        protocol = payload["protocol_version"]
        if not isinstance(protocol, int) or protocol != HOST_PROTOCOL_VERSION:
            raise ManifestError(
                f"Incompatible protocol {protocol!r}; host requires {HOST_PROTOCOL_VERSION}."
            )
        extension_id = payload["id"]
        if not isinstance(extension_id, str) or not extension_id.strip():
            raise ManifestError("Manifest id must be a non-empty string.")
        capabilities_raw = payload.get("capabilities", [])
        if not isinstance(capabilities_raw, list) or not all(
            isinstance(item, str) for item in capabilities_raw
        ):
            raise ManifestError("capabilities must be a string list.")
        schema = payload.get("config_schema", {})
        if not isinstance(schema, dict):
            raise ManifestError("config_schema must be an object.")
        return cls(
            kind=expected_kind,
            extension_id=extension_id.strip(),
            version=str(payload["version"]),
            protocol_version=protocol,
            display_name=str(payload["display_name"]),
            capabilities=tuple(capabilities_raw),
            config_schema=schema,
        )
