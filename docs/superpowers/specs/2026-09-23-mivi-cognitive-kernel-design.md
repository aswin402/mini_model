# MIVI Cognitive Kernel Design

**Status:** Draft for review  
**Date:** 2026-09-23  
**Scope:** Architecture redesign before implementation

## Goal

Build a CPU-first neuro-symbolic cognitive kernel that can interpret natural language, grow a persistent 360-degree concept web, reason exactly when a verifier exists, admit UNKNOWN when evidence is insufficient, and improve its perception and routing models without retraining memory into weights.

MIVI is a cognitive architecture, not a single monolithic language model. Neural components propose interpretations and plans; the persistent evidence graph, tools, and verifiers determine what is accepted as knowledge.

## Repository reality and integration boundary

The repository currently contains the beginnings of this architecture, but it does
not yet execute it as one pipeline. The normal CLI path still routes through
heuristic question detection and `LearningEngine`, while the `LayaSystem1Gatekeeper`,
`DualSpeedInfillingEngine`, and parts of the spider-web prototype are exercised
primarily through isolated tests. The file named `laya_gatekeeper.py` is a
deterministic heuristic fallback; it is not the Laya checkpoint runtime.

The first implementation target is therefore integration, not model training:

```text
all CLI/chat/learning input
  → one cognitive-kernel entrypoint
  → perception adapter
  → candidate frame
  → grounding and evidence ledger
  → cognitive controller
  → verifier-backed result
```

No neural adapter, parser fallback, or direct learning path may bypass the
candidate-to-commit boundary.

## Research-grounded corrections

The architecture uses ideas from the referenced systems, but does not claim that those systems already implement MIVI’s symbolic behavior.

- Laya is a non-autoregressive typed-decision model. Its official repository describes `choice`, `score`, and `noul` outputs, no generated text, and a single forward pass. It also documents zero-shot limitations, high-cardinality degradation, and the need for routing and fine-tuning. Laya is therefore a perception and routing component, not a truth oracle.
- DeepSeek-V4.1-Flash focuses on long-context and serving efficiency, including CSA2, FP4 KV caching, and SWA bounded replay. These ideas motivate selective context and state compression, but they do not replace MIVI’s persistent graph or proof system.
- DeepSeek-R1 is the relevant reasoning reference. Its paper uses verifiable rewards, cold-start reasoning data, reinforcement learning, and distillation. MIVI will apply the verifiable-reward principle to structured plans and proofs rather than train free-form chain-of-thought as the source of truth.
- GLM-5.3-Flash provides a reference for efficient hybrid attention, multimodal processing, and controllable reasoning effort. Its official model card does not establish that its internals perform MIVI’s proposed bidirectional graph search. Graph infilling is an MIVI algorithmic design that must be benchmarked independently.

Primary references:

- [Laya official repository](https://github.com/NandhaKishorM/laya/blob/main/README.md)
- [DeepSeek-V4.1-Flash technical report](https://arxiv.org/abs/2609.19969)
- [DeepSeek-R1 technical report](https://arxiv.org/abs/2501.12948)
- [GLM-5 technical report](https://arxiv.org/abs/2602.15763)
- [GLM-5.3-Flash model card](https://huggingface.co/zai-org/GLM-5.3-Flash)

## Design principles

1. **Neural proposal, symbolic commitment.** A model may propose a frame, relation, action, or plan. Only a verifier-backed commit can change durable knowledge.
2. **Evidence before belief.** Every accepted claim has provenance, time, source, update type, and support or opposition counts.
3. **Open-world uncertainty.** Missing evidence is UNKNOWN, not false and not an invitation to guess.
4. **Data-driven domain behavior.** Domain knowledge belongs in schemas, profiles, and versioned knowledge packs. Python control flow must not contain object-specific behavior.
5. **Thinking is executable.** A thought is a typed operation with inputs, outputs, preconditions, evidence, and a verifier result.
6. **Memory is separate from model weights.** New facts are stored immediately. Neural model updates are periodic, versioned, replay-tested improvements to perception or routing.
7. **Inspectability is a product requirement.** Every answer must be explainable as direct evidence, a proof path, a tool result, a clarification request, or UNKNOWN.
8. **CPU-first, accelerator-optional.** The cognitive kernel, graph, verifiers, and deterministic tools must work without a GPU. Neural adapters may use GPU, ONNX, or a remote service when available.

## No-hardcoding contract

“No hardcoding” means that domain vocabulary, domain facts, grammar forms,
relation semantics, action effects, physical profiles, output templates, and
routing thresholds must not be embedded as object-specific Python branches.
General algorithms remain code. They consume versioned registries and packs.

The implementation must enforce these boundaries:

- domain behavior is loaded through schema/knowledge registries;
- learned or model-proposed content is represented as candidates first;
- confidence and abstention policies are versioned data, not unexplained literals;
- new concepts, materials, relations, and actions require data additions and
  tests, not new `if concept == ...` branches;
- model upgrades and schema upgrades are replay-tested before activation;
- compatibility fallbacks are temporary, named, measured, and removed when the
  registry has equivalent coverage.

## Target architecture

```text
                    sensory/text input
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Perception adapters  │
                 │ Laya / parser / VLM  │
                 └──────────┬───────────┘
                            │ CandidateFrame
                            ▼
                 ┌──────────────────────┐
                 │ Grounding and schema │
                 │ resolution           │
                 └──────────┬───────────┘
                            │ EvidenceRecord
                            ▼
                 ┌──────────────────────┐
                 │ Persistent evidence  │
                 │ graph + event log    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Cognitive controller │
                 │ fast / think / ask   │
                 └──────┬─────┬─────┬───┘
                        │     │     │
                 direct │ graph│ tool│ clarification
                 lookup │ plan │ call│
                        ▼     ▼     ▼
                 ┌──────────────────────┐
                 │ Verification stack   │
                 │ type/proof/tool/time │
                 └──────────┬───────────┘
                            │
                 commit / reject / UNKNOWN
                            │
                            ▼
                 answer composer + proof trace
```

## Core interfaces

These are conceptual contracts. Their names should become stable boundaries before implementation begins.

### CandidateFrame

Represents what a perception model believes the input may contain.

```text
CandidateFrame:
  frame_id
  source_text or sensory_reference
  intent: statement | question | action | correction | observation
  entities: [{surface, candidate_id, type_hint, span, probability}]
  claims: [{subject, predicate, object, attributes, probability}]
  actions: [{name, arguments, probability}]
  uncertainty: calibrated score and reasons
  model_id and model_version
```

CandidateFrame is never inserted directly into durable memory.

### EvidenceRecord

Represents a grounded observation or derived result.

```text
EvidenceRecord:
  evidence_id
  subject_id, predicate, object_id or value
  source_type: user | document | sensor | tool | inference | hypothesis
  source_reference
  observed_at and valid_from/valid_to
  support_weight and opposition_weight
  extraction_confidence
  derivation_proof_id, if derived
  status: candidate | accepted | rejected | superseded
```

### QueryPlan

Represents deliberate thinking as executable steps.

```text
QueryPlan:
  plan_id
  goal
  steps: [retrieve, unify, infer, calculate, simulate, ask, verify]
  expected_cost
  expected_information_gain
  risk_level
  stopping_condition
```

### ProofTrace

Represents why a result is admissible.

```text
ProofTrace:
  proof_id
  conclusion
  premises
  operations
  tool_outputs
  verifier_results
  rejected_alternatives
  final_status: proven | disproven | unknown | inconsistent
```

## Persistent memory model

The current SQLite store in `src/little/memory/store.py` is a useful starting point, but the schema must distinguish observations, beliefs, derivations, and hypotheses.

The target logical model is:

```text
Concept / Entity
    └── Relation edge
          ├── evidence records
          ├── temporal validity
          ├── support/opposition weights
          ├── provenance
          └── derivation proof

Experience/Event
    ├── raw input
    ├── candidate frame
    ├── accepted updates
    └── rejected or unresolved hypotheses

Schema registry
    ├── relation types
    ├── action schemas
    ├── state variables
    ├── invariants
    └── verifier policies
```

The six Concept Knot dimensions remain useful as an inspection view, but they must not be the ontology’s hard limit. New relation types and state dimensions can be registered without changing Python classes.

## Data-driven domain behavior

The current apple-specific behavior in `src/little/dynamics/cfc.py`, `src/little/dynamics/transformations.py`, and `src/little/core/concept_knot.py` should be replaced with generic schemas.

### RelationType

```text
RelationType:
  name
  domain_constraint
  range_constraint
  inverse
  symmetric
  transitive
  temporal_mode
  quantitative_type
  evidence_policy
  contradiction_policy
```

### ActionSchema

```text
ActionSchema:
  name
  argument_types
  preconditions
  effects
  generated_entities
  conservation_rules
  state_transition_rules
  required_evidence
```

### StateVariableSchema

```text
StateVariableSchema:
  name
  value_type
  unit
  valid_range
  transition_model
  parameters
  observation_model
```

Apple, bread, cheese, metal, and other materials should be data records that select schemas and parameters. They should not require new branches in the engine.

## Perception and grounding

The first neural adapter will be optional Laya integration. It will answer a fixed set of typed questions:

- What is the input intent?
- Which spans are entities?
- Which relation/action schema is likely?
- Is the extraction sufficiently clear to continue?
- Does the input require clarification?

The adapter returns probabilities and alternatives. Grounding then resolves names against the graph, handles aliases and coreference, and records unresolved terms as candidates. A high Laya confidence may reduce work, but it cannot bypass grounding or verification.

The existing deterministic parser remains as a fallback and as a test oracle for simple sentences. It should gradually become a grammar/data provider rather than the place where every domain rule lives.

## Cognitive controller

The controller selects a route using observable properties:

```text
route_score = f(
  perception_uncertainty,
  graph_novelty,
  proof_depth,
  contradiction_count,
  action_risk,
  expected_information_gain,
  latency_budget,
)
```

Routes:

- **FAST:** indexed direct relation lookup and simple grounded response.
- **THINK:** bounded graph search, unification, tool execution, and proof construction.
- **ASK:** clarification when multiple groundings or interpretations remain plausible.
- **STUDY:** curiosity-driven question generation and external/local source inspection.
- **REFUSE/UNKNOWN:** no admissible proof or evidence.

The controller should log the selected route and later compare it with outcome quality, latency, and verification result. This gives MIVI a trainable routing problem without making the entire system end-to-end opaque.

## Verifier stack

The current invariant gates are a prototype. The target verifier stack is composable:

1. **Schema verifier:** domain, range, units, argument arity.
2. **Grounding verifier:** every symbol resolves to an entity, value, or explicitly unresolved term.
3. **Graph verifier:** cycle policy, transitivity policy, disjointness, and contradiction handling.
4. **Temporal verifier:** intervals, event ordering, and stale beliefs.
5. **Tool verifier:** exact arithmetic, CAS results, code tests, or simulation outputs.
6. **Physical verifier:** conservation and valid-state constraints declared by an action/state schema.
7. **Provenance verifier:** every accepted conclusion points to observations or validated derivations.

The result is not “always true.” It is “accepted under explicitly named verification rules.” Unsupported claims remain UNKNOWN.

## Learning and self-improvement loop

```text
input → candidate frame → grounding
      → evidence update or unresolved hypothesis
      → answer/clarification
      → feedback and verification outcome
      → replay dataset
      → model/router/schema improvement
```

Three update classes must remain separate:

- **Memory update:** immediate append/reinforce/contradict operation.
- **Schema update:** reviewed addition of a relation, action, state variable, or verifier policy.
- **Model update:** versioned fine-tuning or distillation of a perception/router model.

Autonomous self-study may create hypotheses and ask questions, but it may only promote a hypothesis to accepted knowledge after evidence and verification. Self-study must not directly write arbitrary relations with made-up confidence values.

## Implementation sequence

The implementation should be split into independently testable projects:

### Project A: Cognitive contracts and evidence ledger

Introduce the typed interfaces, provenance records, hypothesis status, and commit/reject flow. Fix the current self-study/store API mismatch as part of this boundary.

### Project B: Schema-driven graph

Move relation metadata, construction definitions, action schemas, and state-variable profiles out of hardcoded branches. Preserve the apple examples as data fixtures.

### Project C: Perception adapter

Add an optional Laya adapter and deterministic fallback. Measure extraction accuracy, calibration, grounding accuracy, and false commits separately.

### Project D: Thinking controller and verifiers

Implement route selection, bounded graph planning, exact tools, proof traces, and composable verification.

### Project E: Generic dynamics and transformations

Implement state-variable transitions and action schemas independent of material names. Add apple, bread, and metal as separate data packs.

### Project F: Curiosity and spider-web growth

Choose missing relations using uncertainty and information gain, run study actions, verify results, and update the graph through the evidence ledger.

### Project G: Model training and continual evaluation

Create reviewed datasets, distill or fine-tune perception models, maintain replay data, and compare neural-only, symbolic-only, and hybrid ablations.

## First implementation slice after approval

The first code slice is deliberately narrow and integration-focused:

1. Add a `PerceptionAdapter` contract with a deterministic fallback adapter.
2. Convert perception output into `CandidateFrame` objects without mutating
   memory.
3. Add one cognitive-kernel entrypoint and route the CLI question/statement
   path through it.
4. Send accepted candidates through grounding and the existing evidence ledger.
5. Preserve current parser behavior behind the fallback so the migration is
   reversible and regression-testable.
6. Add tests proving that uncertain or ungrounded candidates produce `UNKNOWN`
   and cannot create graph relations.

The real Laya adapter is a subsequent optional slice. It must be installed and
benchmarked separately; the fallback must never be mislabeled as Laya.

## Evaluation gates

Every project must report:

- direct-answer accuracy
- multi-hop proof accuracy
- UNKNOWN precision and recall
- unsupported-claim rate
- contradiction detection rate
- grounding accuracy
- calibration error
- forgetting after model updates
- memory growth per accepted observation
- latency for FAST, THINK, ASK, and STUDY routes
- percentage of behavior defined by data schemas rather than code branches

The first meaningful milestone is not a large benchmark score. It is demonstrating that the same engine can learn and reason about apple, bread, and metal without adding object-specific Python logic.

## Acceptance criteria for the architecture

The design is ready for implementation when:

- candidate perception cannot directly mutate durable memory;
- every accepted relation has provenance and a verifier result;
- UNKNOWN is a first-class result;
- relation/action/state behavior is registry-driven;
- the six-axis web is an inspection projection, not a fixed ontology limit;
- the controller can select FAST, THINK, ASK, STUDY, or UNKNOWN;
- apple behavior is represented by data fixtures rather than engine branches;
- neural models can be disabled while the deterministic kernel remains testable;
- model upgrades are replay-tested for forgetting and false commits.
