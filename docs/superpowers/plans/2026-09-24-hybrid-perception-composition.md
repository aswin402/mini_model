# Hybrid Perception Composition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Combine an injected typed System 1 decision backend with an injected LITTLE grounding provider through a policy-driven, fail-closed perception adapter.

**Architecture:** Add a focused `HybridPerceptionAdapter` that calls System 1 and grounding independently, maps the System 1 label through the existing `LayaAdapterPolicy`, and only forwards grounded payloads when both canonical routes agree. Add a small versioned hybrid policy for unknown routing, uncertainty combination, and combined model metadata; leave the kernel's deterministic default unchanged.

**Tech Stack:** Python 3.12, dataclasses, JSON policy schemas, pytest, existing `CandidateFrame`, `LayaDecision`, `PerceptionAdapter`, and `CognitiveKernel` contracts.

## Global Constraints

- System 1 receives only input text and never reads memory or dialogue context.
- Grounding is injected explicitly; no hidden parser fallback is allowed.
- Route disagreement, unknown intent, missing required payload, or disallowed payload returns an empty `unknown` frame.
- Only existing verification, evidence, and learning paths may mutate persistent graph state.
- Do not add Laya, Transformers, ONNX, model weights, network calls, or checkpoint downloads.
- Route labels, payload channels, unknown route, uncertainty strategy, and metadata formats are policy data, not sentence-specific Python rules.
- Existing deterministic kernel construction remains compatible and observable as `deterministic-parser`.
- Use test-first development: every production behavior begins with a failing focused test.

---

### Task 1: Add and validate the hybrid policy

**Files:**
- Create: `data/schemas/hybrid_perception_policy.json`
- Create: `src/little/language/hybrid_perception.py`
- Test: `tests/unit/test_hybrid_perception.py`

**Interfaces:**
- Produces `HybridPerceptionPolicy.load(directory: Path) -> HybridPerceptionPolicy` and `HybridPerceptionPolicy.default() -> HybridPerceptionPolicy`.
- Policy fields are `version: str`, `unknown_intent: str`, `require_route_match: bool`, `uncertainty_strategy: str`, `model_id_format: str`, and `model_version_format: str`.
- The initial data values are `unknown_intent: "unknown"`, `require_route_match: true`, `uncertainty_strategy: "maximum"`, and templates with `{system1_model_id}`, `{grounding_model_id}`, `{system1_model_version}`, and `{grounding_model_version}` placeholders.

- [x] **Step 1: Write failing policy tests**

Add tests that load the repository policy, assert every configured field, and reject missing fields, non-boolean route matching, unsupported uncertainty strategies, and templates missing required placeholders.

```python
def test_hybrid_policy_loads_data_owned_reconciliation_settings():
    policy = HybridPerceptionPolicy.load(Path("data/schemas"))

    assert policy.unknown_intent == "unknown"
    assert policy.require_route_match is True
    assert policy.uncertainty_strategy == "maximum"
    assert "{system1_model_id}" in policy.model_id_format


def test_hybrid_policy_rejects_unsupported_uncertainty_strategy(tmp_path):
    (tmp_path / "hybrid_perception_policy.json").write_text(
        json.dumps({"hybrid_perception_policy": {
            "version": "1",
            "unknown_intent": "unknown",
            "require_route_match": True,
            "uncertainty_strategy": "guess",
            "model_id_format": "{system1_model_id}",
            "model_version_format": "{system1_model_version}",
        }})
    )

    with pytest.raises(ValueError, match="uncertainty_strategy"):
        HybridPerceptionPolicy.load(tmp_path)
```

- [x] **Step 2: Run the policy tests and verify the expected RED failure**

Run:

```bash
PYTHONPATH=src pytest -q tests/unit/test_hybrid_perception.py
```

Expected: collection or import failure because the new policy class and module do not exist yet.

- [x] **Step 3: Add the versioned JSON policy**

Create `data/schemas/hybrid_perception_policy.json` with the exact fields used by the tests:

```json
{
  "hybrid_perception_policy": {
    "version": "1",
    "unknown_intent": "unknown",
    "require_route_match": true,
    "uncertainty_strategy": "maximum",
    "model_id_format": "{system1_model_id}+{grounding_model_id}",
    "model_version_format": "{system1_model_version}+{grounding_model_version}"
  }
}
```

- [x] **Step 4: Implement strict policy loading**

Implement `HybridPerceptionPolicy` as a frozen dataclass. Load JSON, validate all strings as non-empty, require `require_route_match` to be exactly true, allow only the configured supported strategy `maximum`, and require every metadata placeholder before returning the policy. Raise `ValueError` naming the policy path and invalid field.

- [x] **Step 5: Run the policy tests and verify GREEN**

Run:

```bash
PYTHONPATH=src pytest -q tests/unit/test_hybrid_perception.py
```

Expected: policy tests pass.

### Task 2: Implement the explicit hybrid adapter

**Files:**
- Modify: `src/little/language/hybrid_perception.py`
- Test: `tests/unit/test_hybrid_perception.py`

**Interfaces:**
- Consumes `LayaDecisionBackend`, `LayaAdapterPolicy`, and `PerceptionAdapter`.
- Produces `HybridPerceptionAdapter(system1, grounding, *, policy=None, laya_policy=None)`.
- Produces `HybridPerceptionAdapter.perceive(text: str, *, memory: MemoryStore, context: DialogueContext | None = None) -> CandidateFrame`.

- [x] **Step 1: Write failing adapter tests**

Add injected fake providers and tests for the full contract:

```python
class FakeSystem1:
    def __init__(self, decision):
        self.decision = decision
        self.inputs = []

    def decide(self, text):
        self.inputs.append(text)
        return self.decision


class FakeGrounding:
    def __init__(self, frame):
        self.frame = frame
        self.inputs = []

    def perceive(self, text, *, memory, context=None):
        self.inputs.append((text, memory, context))
        return self.frame


def test_matching_statement_routes_preserve_grounded_claims():
    decision = LayaDecision(
        intent="assertion", uncertainty=0.2, model_id="laya",
        model_version="checkpoint", intent_probabilities={"assertion": 1.0},
    )
    grounding = CandidateFrame.from_text(
        "A falcon is a bird.",
        claims=[CandidateClaim("falcon", "is_a", "bird", probability=0.9)],
        intent="statement", uncertainty=0.1,
        model_id="grounder", model_version="policy-1",
    )

    frame = HybridPerceptionAdapter(
        FakeSystem1(decision), FakeGrounding(grounding)
    ).perceive("A falcon is a bird.", memory=MemoryStore(":memory:"))

    assert frame.intent == "statement"
    assert frame.claims == grounding.claims
    assert frame.uncertainty == pytest.approx(0.2)
    assert frame.model_id == "laya+grounder"
    assert frame.model_version == "checkpoint+policy-1"


def test_route_mismatch_returns_unknown_without_payload():
    decision = decision_for("assertion")
    grounding = frame_for("question", parsed_query=("falcon", "is_a", "bird"))

    frame = make_hybrid(decision, grounding).perceive(
        "Is a falcon a bird?", memory=MemoryStore(":memory:")
    )

    assert frame.intent == "unknown"
    assert frame.claims == []
    assert frame.actions == []
    assert frame.parsed_query is None
```

Also cover matching questions and actions, unknown System 1 labels, missing required grounding payload, disallowed mixed payloads, conservative uncertainty, backend/provider injection, and malformed provider return types becoming an empty unknown frame. Assert that the System 1 fake receives only text and the grounding fake receives memory/context.

- [x] **Step 2: Run the adapter tests and verify the expected RED failure**

Run:

```bash
PYTHONPATH=src pytest -q tests/unit/test_hybrid_perception.py
```

Expected: the new adapter import or constructor fails because implementation is not present.

- [x] **Step 3: Implement route and payload reconciliation**

Implement the minimal adapter behavior:

1. Validate non-empty text and both injected provider interfaces.
2. Call `system1.decide(text)` and require a `LayaDecision`.
3. Map `decision.intent` through `LayaAdapterPolicy.intent_mapping`.
4. Call `grounding.perceive(text, memory=memory, context=context)`; a non-`CandidateFrame` result becomes an empty unknown frame.
5. Require the mapped route and grounding frame intent to agree unconditionally; the policy loader requires `require_route_match` to be true and the adapter must fail closed even if a policy object is directly constructed with false.
6. Use `LayaAdapterPolicy.candidate_requirements` and `allowed_payloads` to validate required and disallowed frame payloads without copying route labels into Python.
7. Return an empty unknown frame for unknown route, route mismatch, missing required payload, or disallowed payload.
8. For a valid agreement, copy only the allowed grounding payload into a new frame, carry System 1 probabilities/alternatives, use `max(decision.uncertainty, grounding.uncertainty)` for the configured `maximum` strategy, and render model metadata from the policy templates.
9. Never mutate either provider's frame or read memory directly in the adapter.

Keep the unknown-frame constructor centralized so every fail-closed path has no claims, entities, actions, parsed query, parsed queries, or question parts.

- [x] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
PYTHONPATH=src pytest -q tests/unit/test_hybrid_perception.py
```

Expected: all hybrid policy and adapter tests pass.

### Task 3: Prove kernel injection and persistence safety

**Files:**
- Test: `tests/unit/test_hybrid_perception.py`
- Test: `tests/unit/test_cognitive_kernel.py`
- Modify: `.superpowers/sdd/progress.md`

**Interfaces:**
- Consumes the completed `HybridPerceptionAdapter` and existing `CognitiveKernel(memory, perception=...)` injection point.
- Produces regression evidence that the adapter is opt-in and that disagreements cannot reach `LearningEngine.commit_frame()`.

- [x] **Step 1: Write failing kernel boundary tests**

Add tests that inject a matching hybrid statement into `CognitiveKernel` and assert normal learning reaches the existing verification/ledger path. Add a mismatch test that snapshots concept/relation counts, processes the input, and asserts `UNKNOWN` plus unchanged persistence.

```python
def test_hybrid_mismatch_cannot_commit_graph_state():
    memory = MemoryStore(":memory:", seed_ontology=False)
    kernel = CognitiveKernel(
        memory,
        perception=make_hybrid(
            decision_for("assertion"),
            frame_for("question", parsed_query=("falcon", "is_a", "bird")),
        ),
    )
    before = (len(memory.list_concepts()), memory.count_relations())

    outcome = kernel.process("A falcon is a bird.")

    assert outcome.frame.intent == "unknown"
    assert (len(memory.list_concepts()), memory.count_relations()) == before
```

- [x] **Step 2: Run boundary tests and verify RED for the new behavior**

Run:

```bash
PYTHONPATH=src pytest -q tests/unit/test_hybrid_perception.py tests/unit/test_cognitive_kernel.py
```

Expected: the new hybrid boundary test fails before the adapter implementation exists and passes after Task 2; any failure in existing kernel tests must be diagnosed rather than weakened.

- [x] **Step 3: Keep kernel construction unchanged and add only injection coverage**

Do not change `CognitiveKernel`'s default `DeterministicPerceptionAdapter`. The tests must demonstrate that callers can pass the hybrid adapter explicitly and that the default model identity remains `deterministic-parser`.

- [x] **Step 4: Run the complete regression suite**

Run:

```bash
PYTHONPATH=src pytest -q
```

Expected: the complete suite passes with the new focused tests included.

### Task 4: Document, validate, and hand off the slice

**Files:**
- Modify: `docs/superpowers/specs/2026-09-24-hybrid-perception-composition-design.md`
- Modify: `docs/superpowers/plans/2026-09-24-hybrid-perception-composition.md`
- Modify: `docs/superpowers/plans/2026-09-24-laya-adapter-boundary.md`
- Modify: `.superpowers/sdd/progress.md`

- [x] **Step 1: Document the implemented public boundary**

Document the constructor, provider responsibilities, route mismatch behavior, policy path, and explicit injection example. State that the optional Laya SDK remains uninstalled and no weights are included.

- [x] **Step 2: Run static and structural checks**

Run:

```bash
PYTHONPATH=src python3 -m compileall -q src/little tests/unit/test_hybrid_perception.py
git diff --check
```

Run scoped Ruff when available:

```bash
ruff check src/little/language/hybrid_perception.py tests/unit/test_hybrid_perception.py
```

Expected: `compileall` exits 0. `git diff --check` retains the pre-existing `todo.md:459` blank-line finding outside Task 4 scope. Ruff may be unavailable; if so, report that explicitly. Final results: compileall exit 0, Ruff unavailable on `PATH`, and the diff check reports the retained blank line.

- [x] **Step 3: Update progress evidence**

Record the exact focused hybrid suite result (`97 passed`), full-suite result (`342 passed`), compileall exit 0, retained `todo.md:459` diff-check finding, and Ruff unavailability in `.superpowers/sdd/progress.md`. Do not claim an installed Laya checkpoint was exercised.

- [x] **Step 4: Final verification**

Run the focused hybrid suite and full suite one final time; inspect
`git diff --check` and `git status --short`. Results: focused hybrid suite
`97 passed`, full suite `342 passed`, and compileall exit 0. The diff check
retained the pre-existing `todo.md:459` blank-line finding; Ruff was unavailable
on `PATH`. `.git` is read-only, so no commit was created.
