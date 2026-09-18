# Delivery Plan

Status: Living delivery plan — Phase 3 complete / exit gates verified

Phases 1, 2, and 3 are implemented. Phase 3 exit gates are verified against `QUALITY_GATES.md`; Phase 4 has not started. This plan intentionally delays the first user-facing test build until the application has a complete, coherent vertical workflow and clean Windows packaging.

## Phase 1 — Architecture and delivery freeze

Purpose: make the expensive structural decisions before implementation makes them costly to change.

Deliverables:

- product charter and authority boundary;
- desktop-stack decision;
- host/adapter/runtime-pack separation;
- storage and immutable artifact model;
- worker-process protocol design;
- dataset freeze/provenance model;
- experiment/failure/model-registry model;
- Phase A adapter boundary;
- performance/quality gates;
- Windows packaging/release strategy.

No testing-ready executable is promised in this phase.

Exit condition:

- architecture documents are internally consistent;
- no host-core dependency on Phase-A-specific semantics;
- no live Frankenhomie authority in the Lab;
- Phases 2–4 have explicit gates.

## Phase 2 — Product foundation

Purpose: build the application as a product before loading it with research features.

Main implementation areas:

### Desktop shell

- PySide6/Qt Quick application bootstrap;
- modern ML Lab design system;
- project-aware navigation;
- dark/light/system theme;
- high-DPI and minimum-window handling;
- command palette / shortcuts where valuable;
- notification center and actionable error surfaces;
- Settings and Diagnostics.

### Workspace/project infrastructure

- first-run workspace wizard;
- recent workspaces/projects;
- local metadata SQLite schema and migrations;
- project lifecycle;
- immutable artifact service;
- content hashing;
- application configuration.

### Job engine

- subprocess launcher;
- job-spec protocol;
- structured progress/events;
- cancellation/interruption;
- restart reconciliation;
- log capture;
- staging -> validated artifact commit.

### Extensibility foundations

- built-in adapter registry;
- runtime-pack registry;
- capability/version negotiation;
- configuration-schema rendering primitives.

### Engineering infrastructure

- typed Python core;
- unit/integration/UI smoke tests;
- lint/type/static checks;
- Windows CI;
- development standalone packaging;
- performance instrumentation.

Phase 2 is the implemented product foundation, but it is not the build handed to the user as the intended first testing iteration.

## Phase 3 — Full lab workflow + first real project

Purpose: make the host useful end-to-end and prove its abstraction with a real Frankenhomie adapter.

### Data Studio

- streaming JSONL/JSON/CSV/import adapters as appropriate;
- schema validation and error inspector;
- provenance/lineage editing and filtering;
- TRAIN/DEV/TEST/REDTEAM assignment;
- exact/normalized duplicate scan;
- grouped leakage detection;
- scalable near-duplicate candidate scan;
- immutable freeze workflow;
- manifest/hash inspector.

### Training / experiments

- trainer configuration UI generated from runtime-pack schema;
- job queue/run history;
- experiment manifests;
- hardware/resource capture;
- reproducibility specs;
- deterministic seed management;
- immutable experiment completion.

### Baselines / compare

- deterministic baseline runner contract;
- bounded lexical/reference baseline contract;
- localhost external-provider runner/baseline evidence;
- trainable sparse reference runtime pack;
- side-by-side metric comparison;
- metric drilldown and disagreement explorer.

### Red team / failure library

- seeded mutator framework;
- standard/custom suites;
- immutable failure records;
- regression promotion;
- rerun history;
- veto-first reporting.

### Model registry / packaging

- stage history;
- compatibility status;
- release-candidate bundle builder;
- SHA-256 manifest;
- known-failure report;
- break-it guide generation;
- fresh-load verifier.

### Phase A adapter

- freeze real residual-semantic contract from an explicit Frankenhomie commit;
- import/adapt existing Phase A benchmark evidence without rewriting history;
- preserve deterministic no-model commitment routes;
- expose only bounded residual inputs/candidates;
- run candidate proposals through the real/pinned contract validation harness;
- surface Phase A veto metrics prominently;
- preserve current Qwen/provider baseline as a baseline, not host authority;
- add the first trainable bounded local scorer as an experiment method, not an assumed winner.

Phase 3 exit is the first time the complete intended workflow exists. The Phase 3 gates and exact-head implementation CI evidence have passed; `docs/PHASE3_EXIT_AUDIT.md` records that evidence.

## Phase 4 — Testing-ready productization

Purpose: turn the complete workflow into a release you can reasonably install and test on your PC.

### UX/product polish

- remove development-only placeholder surfaces;
- visual consistency pass on every primary screen;
- keyboard/navigation pass;
- accessibility contrast/focus pass;
- 1366x768 and high-DPI layouts;
- designed loading/empty/error states;
- task progress/cancel/log affordances;
- first-run guidance that explains Lab vs Frankenhomie authority.

### Performance

- startup profile and dependency audit;
- lazy-load nonessential services;
- table/model paging profiles at scale;
- background hashing/import benchmarking;
- job priority so CPU-saturating training does not starve the desktop shell where Windows controls permit;
- packaged-build memory/startup measurements.

### Reliability

- migration upgrade/downgrade recovery tests;
- worker kill/app kill/restart tests;
- partial artifact/disk failure tests;
- corrupted manifest/model tests;
- path/unicode tests;
- read-only/permission failure tests;
- diagnostics bundle export.

### Windows release engineering

- clean CI checkout -> locked environment;
- standalone `MLLab.exe` deployment directory through Qt/Nuitka tooling;
- x64 Inno Setup installer;
- portable ZIP;
- version metadata/icon/uninstall metadata;
- clean-machine installer smoke;
- SHA-256 build manifest;
- GitHub pre-release with artifacts.

### Testing-ready handoff

The handoff includes:

- `MLLab-Setup-v0.1.0-testing.1-x64.exe`;
- portable ZIP;
- SHA-256 hashes;
- release notes;
- known limitations;
- exact build/source provenance;
- short user testing guide;
- diagnostics/export instructions.

Only at this point is the first iteration labeled **TESTING_READY**.

## What is deliberately deferred beyond first testing-ready release

The first testing-ready release should have architectural hooks but does not need every future trainer/backend:

- heavyweight CUDA/Torch runtime packs;
- Hugging Face model-hub management;
- remote/cloud training;
- distributed training;
- arbitrary third-party Python adapters;
- live Frankenhomie deployment/integration (Lab output is handed off for a separate integration decision);
- Phase C/D/E/F/G project adapters before their contracts stabilize.

Deferring these avoids bloating the first release while preserving the seams needed to add them correctly later.

## Repository strategy

Implementation work should land in phase branches/PRs rather than one opaque mega-commit:

- `phase/2-foundation`
- `phase/3-lab-workflow`
- `phase/4-testing-ready`

Each phase PR must contain its own test evidence, performance observations, screenshots where useful, migration notes, and explicit remaining holes.

The default branch remains releasable/documented rather than a dumping ground for half-connected code.
