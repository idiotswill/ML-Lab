$ErrorActionPreference = "Stop"
uv run ruff check .
uv run mypy src/ml_lab
uv run pytest
uv run python -m ml_lab --smoke-test
uv run python -m ml_lab --job-smoke-test
$env:QT_QPA_PLATFORM = "offscreen"
$env:QSG_RHI_BACKEND = "software"
uv run python -m ml_lab --qml-smoke-test
