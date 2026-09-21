# Phase A Player Input Audit v1

Status: proposed protected-evaluation evidence. **Not TRAIN. Not integration approval.**

## Anchors

- Frankenhomie: `1ff3e2155a3c2d2b316023e9933edf3a6d53697f`
- contract: `semantic-residual-v2`
- roadmap at that commit: `IN_PROGRESS_PHASE_B2`
- ML Lab base: `550b8bdf42689b00bd3caa0b406a844bfc9e65a5`
- integration gate: `NO_GO`
- private external sources: S03 and S04 transcript exports
- S03 SHA-256: `b24dbd8a6452b1fc67243430b6c3d07537a927c36f815d2073a9354f414b7ac9`
- S04 SHA-256: `c7882074a059a33fbaecd1e1bd98e5dc290b8c1fa5825e4d848e1a65d55e8aa7`

Raw campaign transcripts are deliberately not checked into ML Lab. The corpus
`benchmarks/phase_a/player_input_audit_v1.jsonl.gz` contains sanitized player-language
excerpts, source hashes/line references, lineage, and audit labels. Every row has
`training_allowed=false`.

## Current contract finding

Phase A begins only after Frankenhomie has produced a `SEMANTIC_REQUIRED`
`semantic-residual-v2` request. The residual scorer may choose only within server-supplied
families, slots, candidate values and `permitted_decisions`. It does not own action
commitment, legality, mechanics, state mutation, or segmentation.

Current `intent_assessment.py` detects multiple committed clauses deterministically.
Current `semantic_routing.py` makes `COMMITMENT:MULTIPLE_COMMITTED_ACTIONS`,
`COMMITMENT:AMBIGUOUS_COMMITMENT`, and mixed committed/unresolved declarations ASK-only.
A clarification can select a stored deterministic primary-action branch; the original
multi-action request is not silently segmented or partially executed.

`input_preparation.py` preserves words and names while normalizing transport noise and
gating low-confidence/conflicting ASR. It does not rewrite named-PC declarations into first
person. Current Cortex semantic routing passes authenticated `actor_id` plus the submitted
declaration; it does not pass actor aliases to `assess_player_input`.

That makes the real table transcripts useful primarily as pre-residual and protected
evaluation evidence. Wake-word variants, named-PC phrasing, collective party declarations,
roll totals, rules questions, and multi-PC declarations must not be learned as model-owned
commitment logic.

## Corpus

55 high-value player turns:

- `ASK_PLAYER`: 23
- `DETERMINISTIC_PREPROCESSING_REQUIRED`: 15
- `RULES_META_OUTSIDE_RESIDUAL`: 10
- `MISSING_OR_UNCLEAR_CONTRACT`: 6
- `CLEAN_CURRENT_CONTRACT`: 1
- TEST: 15
- REDTEAM: 40
- TRAIN: 0

Exact duplicate transcript occurrences are recorded in `duplicate_source_lines` and share
one lineage instead of becoming independent evidence.

## Baseline evidence

A registry family-signal pass using the exact current deterministic aliases was run over the
sanitized slice. This is deliberately not reported as a protected Phase A baseline because
no authoritative residual request/candidate envelope exists for the raw transcript rows:

- zero family signals: 32/55
- exactly one family signal: 16/55
- multiple family signals: 7/55

That is already enough to reject a naive bag-of-words assumption for this hard slice.

The full Lab `phase-a-deterministic-abstention-v2` and
`phase-a-bounded-lexical-v2` runners were not run against raw rows. They correctly require
an authoritative exported `semantic-residual-v2` request including visible candidates and
family-slot envelopes. Inventing those from campaign prose would create parallel authority.

The local Qwen provider was not run here because no loopback provider was available; no
substitute model was used. Historical V7 reference remains `qwen3.5:4b-q4_K_M`:
91.176% decision accuracy, 82.353% family/slot accuracy, 100% accepted-resolution precision,
zero false commitments, one fact-envelope violation, median provider latency 5.718 s, and
no independent-QA claim.

## Cases exposed

1. Multi-PC/multi-action declarations are ASK-only; never pick a convenient clause.
2. `Homie`/`Homey`/`Humi`/etc. commitment handling belongs before Phase A.
3. Named-PC third-person phrasing needs an explicit deterministic input-surface contract
   before it can become a residual example.
4. Rules/meta questions and reported roll totals stay outside residual action resolution.
5. Corrections are zero-model veto material. The “Forget the thing ... Everything stays
   where it was.” pattern must never be turned back into an action.
6. Broad continuation such as “keep going until something significant happens” does not fit
   the current `MOVE_TRAVEL.DESTINATION` slot faithfully.
7. Contingencies/Ready-like declarations and simultaneous Help/support do not authorize Phase
   A to invent segmentation or mechanics.
8. Over-requested conclusions can preserve a search intent while evidence sufficiency remains
   authoritative.

## Next gate

Do not train yet. First obtain or generate authoritative exported residual requests for a
sanitized subset of these lineages from the pinned Frankenhomie contract. Only those derived
rows may enter Phase A TEST/REDTEAM baseline comparison. Synthetic TRAIN descendants must
carry the same lineage and must not cross protected partitions.

Then run deterministic abstention, bounded lexical, pinned local Qwen/provider, the existing
sparse candidate, and any later model through the existing reference preflight/validator and
unchanged veto metrics.
