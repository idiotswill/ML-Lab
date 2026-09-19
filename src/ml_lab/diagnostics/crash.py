from __future__ import annotations

import json
import logging
import os
import sys
import traceback
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import TracebackType
from typing import Any

from ml_lab.core.models import utc_now_iso

LOGGER = logging.getLogger(__name__)

# Crash handling is deliberately local-only. Diagnostics are exported only through
# the explicit user action in the desktop UI.
AUTOMATIC_CRASH_UPLOADS = False


def write_local_crash_report(
    log_dir: Path,
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 1,
        "created_at": utc_now_iso(),
        "automatic_upload": AUTOMATIC_CRASH_UPLOADS,
        "exception": {
            "type": exc_type.__name__,
            "message": str(exc_value),
            "traceback": "".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            ),
        },
    }
    target = log_dir / f"crash-{os.getpid()}-{uuid.uuid4().hex[:12]}.json"
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=log_dir,
            prefix=".crash-",
            suffix=".tmp",
            delete=False,
            mode="w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return target


def install_local_crash_handler(log_dir: Path) -> None:
    previous_hook = sys.excepthook

    def local_hook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
    ) -> None:
        try:
            report = write_local_crash_report(
                log_dir,
                exc_type,
                exc_value,
                exc_traceback,
            )
            LOGGER.critical(
                "Unhandled exception recorded locally path=%s automatic_upload=false",
                report,
            )
        except Exception:
            LOGGER.exception("Local crash report could not be written")
        previous_hook(exc_type, exc_value, exc_traceback)

    sys.excepthook = local_hook
