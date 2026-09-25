# Hybrid System 1/System 2 Perception Composition

## Goal

Add an explicit composition boundary that combines a typed System 1 decision
backend, such as the optional Laya SDK bridge, with a separate grounding
provider that produces LITTLE candidate claims, queries, and actions.

The composition must preserve the existing epistemic contract:

- System 1 may classify intent and express uncertainty.
- The grounding provider may extract candidate payloads.
- Only the existing verification and evidence paths may make durable graph
  changes.
- Any disagreement, unsupported route, or malformed payload must fail closed as
  `unknown`; missing providers are rejected during explicit construction.
- No runtime package, model checkpoint, or network download may be loaded
  implicitly.

## Non-goals

- Training or fine-tuning a Laya model.
- Making Laya produce graph claims or action effects.
- Replacing the parser or evidence ledger.
- Adding a statistical fallback when the optional runtime is unavailable.
- Introducing sentence-specific rules or model-specific behavior into the
  composition layer.

## Recommended architecture

Introduce an explicit `HybridPerceptionAdapter` with two injected
dependencies:

```text
input text
    │
    ├── System 1 decision backend ──> intent, probabilities, uncertainty,
    │                                  model identity/version
    │
    └── grounding provider ─────────> candidate frame payload and parser
                                       identity/version
                    │
                    ▼
          policy-driven route reconciliation
                    │
        ┌───────────┴───────────┐
        │                       │
   route agrees             route disagrees,
   and payload valid        unknown, or malformed
        │                       │
        ▼                       ▼
  combined CandidateFrame   UNKNOWN CandidateFrame
  for normal verification   with no claims/actions

CandidateFrame
    │
    ▼
CognitiveKernel
    ├── ThinkingController for questions
    └── LearningEngine -> grounding, invariant verification, evidence ledger
```

The adapter must not access memory to create or verify facts. It may pass the
existing `memory` and dialogue context to the injected grounding provider,
because that provider already owns parser-level resolution. The System 1
backend receives only the input text, matching the existing Laya backend
contract.

## Route reconciliation

The adapter will use `LayaAdapterPolicy`'s data-owned intent mapping rather than
duplicating label rules in Python. The canonical route comes from the mapped
System 1 intent. The grounding provider's frame intent is the independently
observed route.

The policy will declare:

- the canonical unknown route;
- the required route agreement setting (true; disagreement cannot be disabled);
- the model identity format for a combined frame;
- the uncertainty combination rule;
- the payload fields allowed for each canonical route;
- the policy version recorded in the resulting frame.

The initial policy will require exact canonical route agreement. A valid
agreement preserves only the payload fields allowed for that route. A mismatch
does not choose the more confident source and does not fall back to the
grounding provider alone; it returns an empty unknown frame.

The combined frame will carry:

- System 1 intent probabilities and alternatives;
- conservative uncertainty derived from both sources according to policy;
- both model identities and versions in configured metadata;
- only validated grounding claims, entities, queries, or actions.

The composition layer will not infer a claim from an intent label. For example,
an `assertion` decision without a grounded claim remains unknown rather than
becoming a statement that can reach the ledger.

## Failure behavior

The adapter will fail closed at the candidate boundary:

- Empty input: raise the same input validation error as the existing adapters.
- Missing or invalid injected providers: reject construction with `TypeError`;
  no adapter is created.
- Invalid System 1 decision returned at runtime: return an unknown frame before
  any candidate payload is returned; never commit a candidate payload.
- Invalid grounding frame: return an unknown frame with no payload.
- Unknown or unmapped System 1 intent: return an unknown frame with no payload.
- Route mismatch: return an unknown frame with no payload.
- Disallowed or mixed payload: return an unknown frame with no payload.
- Missing optional runtime: keep the caller's explicitly selected adapter; do
  not auto-select the deterministic provider as a hidden fallback.

`CognitiveKernel` remains unchanged in its default construction path. Callers
must explicitly inject `HybridPerceptionAdapter`; this makes production model
selection observable and keeps the current deterministic behavior compatible.

## Data and configuration

Add a versioned hybrid policy under `data/schemas/`. The policy owns route
reconciliation and frame metadata decisions. Python code will validate policy
shape, route names, payload channels, uncertainty strategy, and model metadata
format before constructing the adapter.

The existing Laya runtime policy remains responsible only for translating the
installed SDK's request and response schema. The existing Laya adapter policy
remains responsible for translating typed decisions and validating payload
channels. The new hybrid policy composes those contracts; it does not repeat
their field mappings.

## Testing strategy

Add focused unit tests before implementation for:

1. matching statement routes preserve grounded claims;
2. matching question routes preserve parsed query payloads;
3. matching action routes preserve grounded actions;
4. route mismatch produces unknown with no claims or actions;
5. unknown System 1 intent cannot pass grounded claims;
6. missing or invalid grounding payload cannot pass through;
7. model metadata and uncertainty are combined according to policy;
8. direct construction does not import or load Laya;
9. the existing deterministic adapter and full kernel regressions remain green.

The tests will use injected fakes for both providers. No model weights,
network access, or optional dependency installation is part of the test suite.
After implementation, run the focused tests, the complete suite, compileall,
and scoped lint where the tool is available.

## Delivery boundary

This slice ends when the explicit hybrid adapter, its versioned policy, focused
tests, and documentation are complete. A later slice may add a benchmark
runner that compares deterministic-only, hybrid-with-Laya, and System 2
verification latency and error behavior using an explicitly installed local
checkpoint.

## Implemented public boundary

The opt-in composition is exposed as
`HybridPerceptionAdapter(system1, grounding, *, policy=None,
laya_policy=None)` in `little.language.hybrid_perception`. `system1` implements
`decide(text) -> LayaDecision` and receives only the input text. `grounding`
implements `perceive(text, *, memory, context=None) -> CandidateFrame` and owns
parser-level resolution. Both dependencies must be injected; construction does
not load the optional runtime or select a hidden fallback. The optional
`HybridPerceptionPolicy` loads from `data/schemas/hybrid_perception_policy.json`
by default, while `LayaAdapterPolicy` owns System 1 label mapping and
route-specific payload requirements.

Route agreement is a non-optional safety invariant: an unmapped or unknown
System 1 route, a fallback route, or any disagreement with the grounding
frame returns an empty `unknown` frame. `require_route_match` remains in the
versioned policy for explicit configuration validation and must be `true`;
the loader rejects false, and exact agreement is enforced by the adapter even
if a policy object is constructed directly with that field set false. A caller
cannot disable route reconciliation. Valid agreement forwards only validated
payload fields permitted for that route.

Callers opt in at the kernel boundary, for example:

```python
perception = HybridPerceptionAdapter(system1_backend, grounding_provider)
kernel = CognitiveKernel(memory, perception=perception)
```

The kernel's ordinary default remains `DeterministicPerceptionAdapter`. The
optional Laya SDK is not installed or exercised by this slice, and no model
weights/checkpoint are included. Tests use injected providers; they do not
establish behavior of an installed Laya checkpoint.
