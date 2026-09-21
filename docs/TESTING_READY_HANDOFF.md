# v0.1.0-testing.5 Testing-Ready Guide

This guide belongs to the Phase 4 Windows build that passed representative physical-Windows performance and fail-closed promotion. `v0.1.0-testing.5` is `TESTING_READY` for product testing only. `INTEGRATION_GATE = NO_GO` remains binding.

## Testing-ready contents

The testing-ready handoff reuses the exact bytes from the retained Phase 4 candidate artifact:

- `MLLab-Setup-v0.1.0-testing.5-x64.exe`;
- `MLLab-v0.1.0-testing.5-windows-x64-portable.zip`;
- candidate build/provenance manifest;
- SHA-256 evidence for the standalone tree, installer and portable archive;
- compiled performance-smoke evidence;
- UX evidence;
- `Run-Representative-Performance.cmd`.

The original candidate manifest correctly remains historical evidence with `testing_ready: false` and `integration_gate: NO_GO`. The separate promotion manifest records `testing_ready: true`, `promotion_scope: PRODUCT_TESTING_ONLY`, `integration_gate: NO_GO`, and `model_integration_approved: false`.

## Representative-machine gate — completed

The full testing.5 physical receipt passed on Windows with the exact candidate executable SHA-256:

`4af03345b9450a70429d68ceffce5f484bfba582043cc1d6afefe5fee2f89a9e`

Key results:

- startup: 1932.10 ms;
- idle RAM: 125.93 MB;
- 100k dataset paging remained bounded to 100 examples;
- three navigation rounds: zero >100 ms GUI stalls;
- 20k background import: zero >100 ms GUI stalls, max 92.85 ms;
- worker-load navigation: zero >100 ms GUI stalls;
- cancellation acknowledgement: 37.43 ms.

Receipt SHA-256:

`63fb4aeef5b9191aaa5e6bcf84191d9b394aab6cf1fdf3124f89058786cd0f72`

Testing-ready promotion manifest SHA-256:

`02192d63c258daca77e98c79d1e0c96a789e0ccbab16fb7012b3a141b4f211d9`

The promoter reused the exact tested installer and portable bytes; it did not rebuild them.

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
