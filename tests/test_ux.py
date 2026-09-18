from datetime import datetime, timedelta, timezone

from ml_lab.ui.controller import _elapsed_text


def test_elapsed_text_for_terminal_job_uses_updated_time() -> None:
    started = datetime(2026, 9, 18, 20, 0, 0, tzinfo=timezone.utc)
    ended = started + timedelta(minutes=2, seconds=7)
    assert _elapsed_text(started.isoformat(), ended.isoformat(), "COMPLETED") == "2m 07s"


def test_elapsed_text_handles_invalid_timestamp() -> None:
    assert _elapsed_text("bad", "also-bad", "FAILED") == "—"
