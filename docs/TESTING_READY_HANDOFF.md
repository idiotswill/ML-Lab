# v0.1.0-testing.5 Candidate / Testing Guide

This guide belongs to the complete Phase 4 Windows candidate. The candidate is not `TESTING_READY` until the representative-machine gate passes and a testing-ready release manifest is produced.

## Candidate contents

The retained Phase 4 candidate artifact contains:

- `MLLab-Setup-v0.1.0-testing.5-x64.exe`;
- `MLLab-v0.1.0-testing.5-windows-x64-portable.zip`;
- candidate build/provenance manifest;
- SHA-256 evidence for the standalone tree, installer and portable archive;
- compiled performance-smoke evidence;
- UX evidence;
- `Run-Representative-Performance.cmd`.

The candidate manifest must say:

- `testing_ready: false`;
- `integration_gate: NO_GO`.

## Final representative-machine gate

Use the installer from the complete candidate artifact and keep the artifact folder available.

Install the application normally. The default installation is per-user under LocalAppData and does not require a separately installed Python.

After installation, run:

`Run-Representative-Performance.cmd`

The helper runs the full physical-machine gate against the installed `MLLab.exe` and writes:

`MLLab-v0.1.0-testing.5-representative-performance.json`

next to the helper.

A passing receipt is evidence only. It does not integrate anything into Frankenhomie and does not self-authorize the release.

Preserve the JSON receipt unchanged for testing-ready promotion/review.

Full physical mode runs three independent ordinary-navigation rounds. The 100 ms GUI-thread threshold is unchanged; the written "repeatable" gate fails when an over-threshold GUI stall recurs in at least two rounds. Scheduler delays are recorded separately and cannot be mislabeled as GUI-thread work.

## After testing-ready promotion

Normal user testing should exercise the full Lab workflow:

1. create/open a workspace;
2. create a project;
3. capture/freeze the intended real contract snapshot where required;
4. import or create a dataset;
5. inspect leakage and freeze TRAIN/DEV/TEST/REDTEAM partitions;
6. train a candidate using TRAIN/permitted DEV only;
7. run protected evaluation;
8. compare deterministic, lexical/provider and candidate evidence as applicable;
9. inspect failures and regressions;
10. register/promote a model only through `RELEASE_CANDIDATE`;
11. export and fresh-verify a model bundle.

Do not treat a `RELEASE_CANDIDATE` model as integrated into Frankenhomie. The Lab never installs or activates it.

## Diagnostics

The Diagnostics page can refresh hardware information, open local logs, and export a diagnostics ZIP. Crash reports are local-only and are not automatically uploaded.

Job rows expose progress, elapsed time, correlation ID, logs and cancellation.

## What to preserve when reporting a problem

Preserve, where relevant:

- the testing-ready release manifest;
- representative-performance receipt;
- diagnostics export;
- job correlation ID;
- immutable failure/regression IDs;
- bundle verification receipt;
- exact candidate/release artifact hash.

Avoid sharing Production/HomieDM campaign material unless it has been deliberately sanitized and is safe to disclose.
