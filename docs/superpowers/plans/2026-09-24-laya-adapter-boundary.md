# Optional Laya Adapter Boundary

The repository now has a dependency-free boundary for a real Laya runtime. It
does not contain model weights, install Transformers or ONNX, call a network,
or label the deterministic parser as Laya.

## Contract

`LayaPerceptionAdapter` receives an injected `LayaDecisionBackend`. The backend
only receives the input text and returns a validated `LayaDecision` containing:

- model identity and version;
- calibrated uncertainty and intent probabilities;
- optional entity, claim, action, and query candidates.

The adapter accepts `memory` and dialogue context to satisfy the common
`PerceptionAdapter` boundary, but deliberately does not access either object.
All candidate output remains non-durable until grounding, verification, and the
evidence ledger accept it.

## Data-driven safety policy

`data/schemas/laya_adapter_policy.json` owns the translation from external
model labels to canonical routes and declares the payload channels allowed for
each route. An unmapped label, missing required payload, mixed payload, malformed
query, or forbidden model identity becomes an explicit `unknown` candidate or a
clear validation error. There is no implicit deterministic fallback.

The actual runtime can be added later as a separate package or local adapter:

```text
runtime.decide(text) -> LayaDecision
LayaPerceptionAdapter(runtime).perceive(text, memory=..., context=...)
```

That runtime must be installed and benchmarked separately. Until then,
`DeterministicPerceptionAdapter` remains the explicit, observable
`deterministic-parser` fallback used by the kernel.

## Optional SDK bridge

`LayaSDKBackend` in `src/little/language/laya_runtime.py` is the explicit
bridge for an installed Laya SDK. It loads the configured typed-choice schema
from `data/schemas/laya_runtime_policy.json`, calls the SDK's prediction method,
and converts the returned choice, probability distribution, confidence, and
routing metadata into `LayaDecision`.

Construction is lazy and explicit: direct construction accepts an injected
agent for tests or embedding, while `from_installed(...)` is the only path that
imports the optional SDK and loads either the configured agent or router. It
requires package-version metadata before loading and requires checkpoint/model
version metadata before returning a decision. It never downloads weights or
imports the SDK during normal kernel startup.

The bridge is intent-only because the Laya SDK does not produce LITTLE graph
claims or action effects. Passing it directly to `LayaPerceptionAdapter`
therefore remains fail-closed until a separate grounding provider supplies
validated candidate payloads. This keeps System 1 routing separate from
System 2 grounding and verification.

## Hybrid composition boundary

`HybridPerceptionAdapter(system1, grounding, *, policy=None,
laya_policy=None)` composes this typed decision boundary with an explicitly
injected `PerceptionAdapter` grounding provider. System 1 receives only text;
grounding receives text, memory, and optional dialogue context. The hybrid
policy lives at `data/schemas/hybrid_perception_policy.json`; this adapter
policy continues to own external-label mapping and route-specific payload
validation.

The hybrid adapter requires exact agreement between the mapped System 1 route
and the grounding frame route. Disagreement, an unmapped route, or an unknown
route yields an empty `unknown` frame. This agreement cannot be disabled:
policy loading rejects `require_route_match: false`, and the adapter applies
the exact-match check even when given a directly constructed policy object
whose field is false. Callers opt into the composition explicitly with
`CognitiveKernel(memory, perception=HybridPerceptionAdapter(system1,
grounding))`; kernel defaults remain deterministic. The optional Laya SDK is
still uninstalled and unexercised here, and the repository slice includes no
weights or checkpoint.
