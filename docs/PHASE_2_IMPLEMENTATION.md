# Phase 2 — Product Foundation Implementation

Status: **IN PROGRESS — CODE SLICE IMPLEMENTED, EXTERNAL GATES PENDING**

This branch implements the product foundation from `docs/DELIVERY_PLAN.md`. It is deliberately not the first user-testing release.

## Implemented

- src-layout Python application and Qt Quick/QML desktop bootstrap;
- first-run workspace create/open flow;
- modern compact project/job/diagnostics/settings shell;
- system/dark/light theme modes;
- project create/open/archive lifecycle;
- manifest-only adapter/runtime catalog with protocol rejection;
- Lab-only SQLite metadata store with WAL, foreign keys and sequential transactional migrations;
- pre-migration backup path;
- immutable SHA-256 artifact store with atomic write and post-copy verification;
- child-process job protocol with structured progress, cancellation, timed termination fallback and restart reconciliation;
- worker event/result finalization into the immutable artifact store;
- compiled-application worker command path distinct from Python development execution;
- CPU/RAM/disk/NVIDIA diagnostics without a heavyweight ML import;
- rotating application logs and explicit diagnostics bundle export;
- non-GUI storage and packaged-worker smoke commands;
- headless QML-load smoke command with load timing;
- Windows/Linux test CI plus Windows standalone deployment and compiled smoke tests;
- unit/integration coverage for workspace, transactional migration rollback, artifact corruption, extension compatibility and worker lifecycle.

## Local evidence in this implementation session

- Python core/integration suite: 16 tests passed.
- `python -m ml_lab --job-smoke-test`: passed with an immutable result artifact.
- source compile pass: passed.

PySide6 is not installed in the local execution container used for this implementation, so the QML runtime and Windows standalone evidence must come from GitHub CI rather than being claimed locally.

## Deliberate boundaries

- no Phase-A-specific concepts in host code;
- no game-state database connector;
- no arbitrary worker command execution;
- no Torch/Transformers dependency;
- no trainer/protected split interfaces yet (Phase 3);
- no installer/release claim yet (Phase 4).

## Remaining Phase 2 closure evidence

Phase 2 stays open until all applicable gates in `QUALITY_GATES.md` are evidenced. In particular:

1. GitHub CI must pass lint, type checks, tests, QML load and Windows standalone/worker smoke.
2. The UI needs a real Windows visual/runtime review at normal and high DPI.
3. Startup time and idle working-set budgets need measurements on representative Windows hardware.
4. A forced worker kill/relaunch trial must confirm the persisted `INTERRUPTED` state in the actual packaged application.

No item above will be inferred from source code alone.
