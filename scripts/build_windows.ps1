$ErrorActionPreference = "Stop"
uv sync --extra dev
uv run pyside6-deploy main.py --name MLLab --mode standalone -f
Write-Host "Development standalone build complete. Phase 4 owns installer/productization."
