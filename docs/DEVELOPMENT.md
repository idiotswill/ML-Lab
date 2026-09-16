# Development

Phase 2 branch: `phase/2-foundation`

## Requirements

- Windows 10/11 x64 or a supported development OS
- Python 3.12
- `uv`

The normal desktop process intentionally has only PySide6 as a runtime dependency. Heavy ML frameworks belong in later runtime packs, not in the shell.

## Run

```powershell
uv sync --extra dev
uv run python -m ml_lab
```

On first launch choose or create a workspace. The workspace contains Lab metadata/artifacts only and must not be pointed at Frankenhomie's Production database directory as a substitute for a Lab workspace.

## Test

```powershell
./scripts/test.ps1
```

The checks include lint/static analysis, core tests, storage smoke, child-worker smoke and a QML load smoke.

## Development standalone build

```powershell
./scripts/build_windows.ps1
```

This uses Qt for Python's `pyside6-deploy` in `standalone` mode. Phase 4 owns the installer, portable archive and the first build offered for normal user testing.

## Architecture rules

- Never put Phase-A vocabulary into the generic host core.
- Never add Frankenhomie game-state mutation APIs to this application.
- Never run long ML/data work on the Qt GUI thread.
- Worker task types are explicit capabilities, not arbitrary shell commands.
- Worker results enter the immutable artifact store only after completion and hash verification.
- Do not add Torch/Transformers to base application startup dependencies.
