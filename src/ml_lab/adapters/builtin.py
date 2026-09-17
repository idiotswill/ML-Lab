from __future__ import annotations

from dataclasses import dataclass

from ml_lab.adapters.phase_a import PHASE_A_ADAPTER_ID, PhaseAResidualAdapter
from ml_lab.contracts.snapshot import ContractFileSpec
from ml_lab.datasets.service import ExampleValidator


@dataclass(frozen=True, slots=True)
class ContractCaptureDescriptor:
    adapter_id: str
    adapter_version: str
    contract_version: str
    files: tuple[ContractFileSpec, ...]


def dataset_validator_for(adapter_id: str) -> ExampleValidator | None:
    """Return the trusted built-in dataset validator for an adapter.

    The host remains domain-neutral: adapter-specific validation is resolved here,
    not embedded in Data Studio or the generic dataset service.
    """
    if adapter_id == PHASE_A_ADAPTER_ID:
        return PhaseAResidualAdapter().validate_dataset_row
    if adapter_id == "generic":
        return None
    raise KeyError(f"No built-in dataset validator for adapter {adapter_id!r}.")


def contract_capture_descriptor_for(adapter_id: str) -> ContractCaptureDescriptor | None:
    """Return the immutable contract declaration owned by a built-in adapter.

    A generic Lab project intentionally has no host-invented contract file list. Adapters
    must declare the exact committed files that define their compatibility boundary.
    """
    if adapter_id == PHASE_A_ADAPTER_ID:
        adapter = PhaseAResidualAdapter()
        return ContractCaptureDescriptor(
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.version,
            contract_version=adapter.contract_version,
            files=adapter.contract_files(),
        )
    if adapter_id == "generic":
        return None
    raise KeyError(f"No built-in contract descriptor for adapter {adapter_id!r}.")
