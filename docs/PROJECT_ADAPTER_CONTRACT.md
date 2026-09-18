# Project Adapter Contract

Status: Implemented contract baseline — Phase 3 exit audit in progress

The ML Lab host is generic. A **project adapter** translates one completed/stable Frankenhomie seam into the host's generic experiment model.

## 1. Adapter responsibilities

An adapter owns only development-time contract interpretation. It does not gain Frankenhomie runtime authority.

Required responsibilities:

- identify the project kind and adapter version;
- identify compatible Frankenhomie contract versions/commits;
- declare which repository files define the frozen contract snapshot;
- validate imported dataset payloads and labels;
- declare task types and bounded candidate structure;
- declare metrics and veto metrics;
- define baseline runners or baseline-import formats;
- define red-team mutation families;
- define display metadata for inspectors/tables;
- validate candidate outputs before they can be scored as accepted;
- emit compatibility and packaging metadata.

## 2. Host-facing interface

The concrete Python API has evolved through Phases 2 and 3, but this conceptual authority boundary remains frozen:

```text
AdapterDescriptor
  id
  version
  display_name
  task_kinds[]
  protocol_version

ContractSnapshotSpec
  source_repository
  source_commit
  source_files[]
  semantic_versions[]
  compatibility_signature

DatasetSchema
  payload_schema
  label_schema
  provenance_schema

TaskDefinition
  task_kind
  candidate_schema
  prediction_schema
  metric_definitions[]
  veto_definitions[]

BaselineDefinition
  id
  runner_kind
  configuration_schema

RedTeamDefinition
  family_id
  parameter_schema
  deterministic_seeded_mutator

PackageCompatibility
  target_contract
  required_features
  forbidden_authority
```

Adapters are versioned and immutable after experiments reference them.

## 3. Generic host vocabulary

The host may use these concepts:

- sample/example
- input/payload
- label/expected result
- candidate
- score/rank/class
- prediction/proposal
- accepted/rejected evaluation
- metric
- veto
- baseline
- trainer
- model
- experiment
- failure/regression
- dataset split
- contract snapshot
- artifact
- package

The host must not hard-code project vocabulary such as `HARM_TARGET`, `TARGET_COMBATANT`, initiative actions, concessions, or NPC intentions.

## 4. Adapter trust boundary

v0.1 adapters are built-in and signed/versioned with the application release. The app does **not** import arbitrary Python adapters from untrusted folders.

Later third-party adapter support, if ever added, requires an explicit trust/sandbox design rather than `importlib` discovery of arbitrary code.

## 5. Reference validation

When a project depends on a Frankenhomie validator, the adapter may invoke a **read-only reference-validation harness** pinned to the captured contract.

Rules:

- the harness receives immutable test payloads only;
- it has no Production credentials or write access;
- it cannot dispatch game actions;
- it must record exact Frankenhomie commit/contract identity;
- a model's raw proposal is never treated as valid merely because its schema parses;
- authoritative contract validation is preserved in evaluation wherever available.

## 6. Phase A adapter

The first real adapter targets the completed Phase A residual-semantic boundary.

The host will see a bounded interpretation/ranking task. The adapter will know about the actual Phase A contract and map host predictions into the real residual-semantic validator.

It must preserve:

- deterministic action-commitment gates upstream;
- exact/contextual deterministic routes upstream;
- visibility-safe, bounded candidates only;
- permitted decisions from the request;
- allowed family/slot/value envelopes;
- prebound slots;
- fact/candidate envelopes;
- Frankenhomie revalidation before an ML proposal counts as accepted.

The ML project cannot create a new action family, hidden entity, mechanics decision, die result, or state mutation.

Phase A vetoes include at minimum:

- false action commitment;
- accepted hidden/out-of-envelope selection;
- accepted contract violation;
- unsupported mechanics authority;
- zero-model-route violation;
- caller-overridable safety threshold.

Future Phase A experiments may change the trainer/model method without changing this adapter authority boundary.

## 7. Later adapters

Later project adapters may be added only after their Frankenhomie contracts stabilize. Examples include:

- combat candidate ranking;
- social response/concession ranking;
- generation/template ranking;
- NPC noncombat intention ranking;
- situation salience ranking.

The adapters should prefer the common pattern:

```text
authoritative state
 -> deterministic eligibility
 -> bounded candidates
 -> ML scoring/ranking
 -> Frankenhomie revalidation
 -> ordinary resolver
 -> ordinary receipt
```

The Lab stops before the ordinary resolver. It evaluates proposals and packages candidate models/evidence; it does not become the resolver, deploy into Frankenhomie, or grant itself integration approval.
