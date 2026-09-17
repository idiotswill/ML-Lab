from __future__ import annotations

from ml_lab.adapters.phase_a import PHASE_A_ADAPTER_ID, PhaseAResidualAdapter
from ml_lab.datasets.service import ExampleValidator


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
