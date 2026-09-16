from __future__ import annotations

import sys
from pathlib import Path


def compiled_application_executable() -> Path | None:
    """Return the running application binary when executing compiled code."""
    if "__compiled__" not in globals():
        return None
    candidate = Path(sys.argv[0]).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return None
    return resolved if resolved.is_file() else None


def application_command(*args: str) -> list[str]:
    compiled = compiled_application_executable()
    if compiled is not None:
        return [str(compiled), *args]
    return [sys.executable, "-m", "ml_lab", *args]


def worker_command(spec_path: Path) -> list[str]:
    compiled = compiled_application_executable()
    if compiled is not None:
        return [str(compiled), "--worker", str(spec_path)]
    return [sys.executable, "-m", "ml_lab.jobs.worker", str(spec_path)]
