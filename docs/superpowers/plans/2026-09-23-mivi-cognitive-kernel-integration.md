# MIVI Cognitive Kernel Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route every semantic CLI/chat input through one CPU-first cognitive kernel where perception produces candidates, grounding and verification control commits, and unsupported inputs remain UNKNOWN without creating graph relations.

**Architecture:** Add a deterministic `PerceptionAdapter` that consumes the existing parser and data-driven policy without mutating memory. Add a `CognitiveKernel` that sends question candidates to the existing thinking controller and statement/action candidates to a single candidate-to-commit path in `LearningEngine` and `EvidenceLedger`. Keep the parser fallback available as a named compatibility implementation, but make it an adapter behind the same boundary.

**Tech Stack:** Python 3.12+, SQLite, existing dataclasses/enums, pytest, Ruff, uv, and the current SymPy dependency. This slice adds no Laya, Transformers, ONNX, GPU, or network dependency.

## Global Constraints

- Preserve the existing public `MemoryStore`, `LearningEngine`, and CLI behavior unless a focused test covers the new kernel route.
- A `CandidateFrame` must never directly mutate durable concepts, relations, attributes, evidence status, or ledger decisions.
- Every accepted relation must pass through `EvidenceLedger` and retain source provenance and a commit decision.
- Missing, malformed, or ungrounded input must produce `UNKNOWN` or a clarification route and zero new graph relations.
- Domain vocabulary, grammar, relation semantics, action effects, templates, and routing thresholds belong in versioned data files or registries, not object-specific Python branches.
- The deterministic adapter may use general parsing algorithms already present in the repository, but it must not introduce new `if concept == ...` or domain-specific regex branches.
- The fallback adapter must be named and observable as `deterministic-parser`; it must never be labelled as Laya.
- Keep all migrations additive and run tests against temporary databases; do not modify `data/little.db` during tests.
- Preserve the existing interactive commands such as `help`, `memory`, `inspect`, and `skills`; only semantic user input moves through the kernel.

## Scope boundary

This plan implements only the first integration slice from the approved design:

1. perception adapter contract and deterministic fallback;
2. candidate-only perception with no memory mutation;
3. one `CognitiveKernel.process()` entrypoint;
4. evidence-ledger commits for accepted candidates;
5. CLI `learn`, `ask`, and chat semantic input routed through that entrypoint;
6. tests proving UNKNOWN and ungrounded candidates cannot create relations.

Do not implement the real Laya checkpoint, GLM-style model serving, DeepSeek training, bidirectional graph search, generic dynamics, curiosity growth, or model fine-tuning in this plan. Those remain follow-on work after this boundary is stable.

## File map

- `src/little/language/perception.py` — adapter protocol, policy loader, and deterministic parser adapter.
- `data/schemas/perception_policy.json` — versioned fallback confidence and uncertainty values.
- `src/little/core/contracts.py` — candidate metadata needed to preserve polarity and property/action provenance.
- `src/little/language/parser.py` — candidate-frame commit method and compatibility wrapper for `learn()`.
- `src/little/core/kernel.py` — unified semantic input entrypoint and outcome object.
- `src/little/inference/thinking_controller.py` — add an explicit UNKNOWN route while preserving FAST/THINK/ASK/STUDY behavior.
- `src/little/main.py` — route command and interactive semantic paths through `CognitiveKernel`.
- `tests/unit/test_perception_adapter.py` — adapter purity, intent, and policy tests.
- `tests/unit/test_cognitive_kernel.py` — kernel route and commit-boundary tests.
- `tests/unit/test_learning_ledger_integration.py` — candidate commit regression tests.
- `tests/integration/test_cognitive_kernel_cli.py` — command-level routing and database isolation tests.

## Interfaces fixed by this plan

```python
class PerceptionAdapter(Protocol):
    def perceive(
        self,
        text: str,
        *,
        memory: MemoryStore,
        context: DialogueContext | None = None,
    ) -> CandidateFrame:
        """Return a candidate frame without mutating memory."""


class DeterministicPerceptionAdapter:
    def perceive(
        self,
        text: str,
        *,
        memory: MemoryStore,
        context: DialogueContext | None = None,
    ) -> CandidateFrame:
        raise NotImplementedError


class CognitiveMode(str, Enum):
    FAST = "FAST"
    THINK = "THINK"
    ASK = "ASK"
    STUDY = "STUDY"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class KernelOutcome:
    frame: CandidateFrame
    mode: CognitiveMode
    inference: InferenceResult | None = None
    learning: LearningResult | None = None


class CognitiveKernel:
    def process(self, text: str) -> KernelOutcome:
        raise NotImplementedError
```

The existing `LearningEngine.learn(text) -> LearningResult` and `LearningEngine.ask(question) -> InferenceResult` remain public compatibility methods. Internally, `learn()` must create a candidate frame and call `commit_frame()`; callers must not need to know the adapter implementation.

---

### Task 1: Add the data-driven perception adapter

**Files:**
- Create: `src/little/language/perception.py`
- Create: `data/schemas/perception_policy.json`
- Modify: `src/little/core/contracts.py:34-92`
- Test: `tests/unit/test_perception_adapter.py`

**Interfaces:**
- `PerceptionPolicy.load(directory: Path) -> PerceptionPolicy`
- `PerceptionPolicy.default() -> PerceptionPolicy`
- `DeterministicPerceptionAdapter(policy: PerceptionPolicy | None = None)`
- `PerceptionAdapter.perceive(text, *, memory, context=None) -> CandidateFrame`
- `CandidateClaim.positive: bool` records negative statements without embedding polarity in parser control flow.
- `CandidateClaim.is_property: bool` preserves the current attribute-vs-edge distinction through the candidate boundary.

- [ ] **Step 1: Write the failing purity and intent tests**

```python
from little.language.perception import DeterministicPerceptionAdapter
from little.memory.store import MemoryStore


def test_perception_returns_statement_candidates_without_mutating_memory():
    memory = MemoryStore(":memory:", seed_ontology=False)
    adapter = DeterministicPerceptionAdapter()
    before = (len(memory.list_concepts()), memory.count_relations())

    frame = adapter.perceive("A falcon is a bird.", memory=memory)

    after = (len(memory.list_concepts()), memory.count_relations())
    assert frame.intent == "statement"
    assert [(c.subject, c.predicate, c.object) for c in frame.claims] == [
        ("falcon", "is_a", "bird")
    ]
    assert before == after
    assert frame.model_id == "deterministic-parser"


def test_perception_marks_questions_without_creating_claims():
    memory = MemoryStore(":memory:", seed_ontology=False)

    frame = DeterministicPerceptionAdapter().perceive(
        "Is a falcon a bird?", memory=memory
    )

    assert frame.intent == "question"
    assert frame.claims == []
    assert memory.count_relations() == 0


def test_perception_preserves_negative_and_property_metadata():
    memory = MemoryStore(":memory:", seed_ontology=False)

    negative = DeterministicPerceptionAdapter().perceive(
        "A bird is not a vehicle.", memory=memory
    )
    property_frame = DeterministicPerceptionAdapter().perceive(
        "The apple is red.", memory=memory
    )

    assert negative.claims[0].positive is False
    assert property_frame.claims[0].is_property is True


def test_unrecognized_input_is_unknown_and_pure():
    memory = MemoryStore(":memory:", seed_ontology=False)
    frame = DeterministicPerceptionAdapter().perceive("qwerty zorp", memory=memory)

    assert frame.intent == "unknown"
    assert frame.claims == []
    assert memory.count_relations() == 0
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/unit/test_perception_adapter.py -q`

Expected: FAIL because the adapter module and candidate metadata fields do not exist.

- [ ] **Step 3: Add the versioned policy file**

Create `data/schemas/perception_policy.json` with values owned by data, not Python literals:

```json
{
  "perception_policy": {
    "version": "1",
    "deterministic_claim_probability": 0.99,
    "deterministic_frame_uncertainty": 0.01,
    "question_frame_uncertainty": 0.25,
    "unknown_frame_uncertainty": 1.0,
    "action_probability": 0.95
  }
}
```

Load and validate all four values as numbers in `[0.0, 1.0]`. Do not add a second hardcoded fallback value in the adapter. A missing or malformed policy must raise a clear `ValueError` during `PerceptionPolicy.load()`.

- [ ] **Step 4: Extend candidate metadata without changing the commit API**

Add these fields to `CandidateClaim` in `src/little/core/contracts.py` after `attributes`:

```python
positive: bool = True
is_property: bool = False
```

Keep the existing positional constructor calls valid. Update `EvidenceRecord.from_claim()` so `positive=claim.positive` is the default when callers do not pass an explicit polarity. Existing tests that pass `positive=` explicitly must continue to work.

- [ ] **Step 5: Implement the adapter with read-only parser access**

Implement `src/little/language/perception.py` with lazy imports for parser classes to avoid a module cycle:

```python
class DeterministicPerceptionAdapter:
    def __init__(self, policy: PerceptionPolicy | None = None) -> None:
        self.policy = policy or PerceptionPolicy.default()

    def perceive(
        self,
        text: str,
        *,
        memory: MemoryStore,
        context: DialogueContext | None = None,
    ) -> CandidateFrame:
        if not text.strip():
            raise ValueError("perception input must be non-empty")

        from little.language.construction import ConstructionEngine
        from little.language.parser import SimpleParser

        resolved_text = context.resolve_anaphora_in_text(text) if context else text
        question = SimpleParser.parse_question(
            resolved_text,
            known_concepts={c.name for c in memory.list_concepts()},
        )
        if question is not None or resolved_text.strip().endswith("?"):
            return CandidateFrame.from_text(
                text,
                claims=[],
                intent="question",
                uncertainty=self.policy.question_frame_uncertainty,
                model_id="deterministic-parser",
                model_version=self.policy.version,
            )

        action = ConstructionEngine.parse_action_with_constructions(
            resolved_text, memory
        )
        if action is None:
            action = SimpleParser.parse_action(resolved_text)
        if action is not None:
            name, arguments = action
            return CandidateFrame.from_text(
                text,
                claims=[],
                actions=[CandidateAction(
                    name=name,
                    arguments={str(k): str(v) for k, v in arguments.items()},
                    probability=self.policy.action_probability,
                )],
                intent="action",
                uncertainty=self.policy.deterministic_frame_uncertainty,
                model_id="deterministic-parser",
                model_version=self.policy.version,
            )

        triples = ConstructionEngine.parse_with_constructions(resolved_text, memory)
        if not triples:
            triples = SimpleParser.parse_statement(resolved_text)
        if not triples:
            return CandidateFrame.from_text(
                text,
                claims=[],
                intent="unknown",
                uncertainty=self.policy.unknown_frame_uncertainty,
                model_id="deterministic-parser",
                model_version=self.policy.version,
            )

        claims = [
            CandidateClaim(
                triple.subject,
                triple.predicate,
                triple.object_,
                probability=self.policy.deterministic_claim_probability,
                positive=not triple.is_negative,
                is_property=triple.is_property,
            )
            for triple in triples
        ]
        return CandidateFrame.from_text(
            text,
            claims=claims,
            intent="statement",
            uncertainty=self.policy.deterministic_frame_uncertainty,
            model_id="deterministic-parser",
            model_version=self.policy.version,
        )
```

The adapter may read concepts, constructions, and schemas. It must not call `create_concept`, `add_relation`, `update_concept_attributes`, `ledger`, or any write method.

- [ ] **Step 6: Run adapter and contract tests**

Run: `uv run pytest tests/unit/test_perception_adapter.py tests/unit/test_cognitive_contracts.py -q`

Expected: PASS.

- [ ] **Step 7: Save the adapter boundary**

```bash
git add src/little/language/perception.py data/schemas/perception_policy.json src/little/core/contracts.py tests/unit/test_perception_adapter.py
git commit -m "feat: add candidate-only perception adapter"
```

### Task 2: Refactor learning behind candidate-to-commit

**Files:**
- Modify: `src/little/language/parser.py:1104-1390`
- Modify: `src/little/memory/ledger.py:22-95`
- Modify: `tests/unit/test_learning_ledger_integration.py`
- Test: `tests/unit/test_cognitive_kernel.py`

**Interfaces:**
- `LearningEngine.commit_frame(frame: CandidateFrame, *, source_type: SourceType = SourceType.USER) -> LearningResult`
- `LearningEngine.learn(text: str) -> LearningResult` becomes a compatibility wrapper: perceive, then `commit_frame`.
- `EvidenceLedger.commit_property(evidence, verification) -> CommitDecision` handles accepted attributes through the same evidence decision boundary.
- `EvidenceLedger.commit_action(frame, action) -> LearningResult` delegates only to registered action schemas and preserves the existing transformation result.

- [ ] **Step 1: Write the failing commit-boundary tests**

```python
from little.core.kernel import CognitiveKernel
from little.core.models import UpdateType
from little.language.perception import DeterministicPerceptionAdapter
from little.language.parser import LearningEngine
from little.memory.store import MemoryStore


def test_learning_commits_a_candidate_with_user_provenance():
    memory = MemoryStore(":memory:", seed_ontology=False)
    result = LearningEngine(memory).learn("A falcon is a bird.")

    assert result.update_type is UpdateType.NEW_RELATION
    records = memory.evidence.list_by_source("user")
    assert len(records) == 1
    assert records[0].status.value == "accepted"
    assert records[0].source_text == "A falcon is a bird."
    assert memory.find_relation_by_names("falcon", "is_a", "bird") is not None


def test_candidate_frame_alone_cannot_create_a_relation():
    memory = MemoryStore(":memory:", seed_ontology=False)
    frame = DeterministicPerceptionAdapter().perceive(
        "A falcon is a bird.", memory=memory
    )
    before = memory.count_relations()

    assert frame.accepted is False
    assert memory.count_relations() == before


def test_unrecognized_candidate_returns_no_op_and_zero_edges():
    memory = MemoryStore(":memory:", seed_ontology=False)
    result = LearningEngine(memory).learn("qwerty zorp")

    assert result.update_type is UpdateType.NO_OP
    assert memory.count_relations() == 0
```

- [ ] **Step 2: Run the focused tests to verify the boundary is missing**

Run: `uv run pytest tests/unit/test_learning_ledger_integration.py tests/unit/test_cognitive_kernel.py -q`

Expected: FAIL because `LearningEngine` has no candidate commit method and the kernel module is not present.

- [ ] **Step 3: Extract candidate construction from `LearningEngine.learn()`**

Inject a `PerceptionAdapter` into `LearningEngine` without changing existing callers:

```python
def __init__(
    self,
    memory: MemoryStore,
    perception: PerceptionAdapter | None = None,
) -> None:
    self.memory = memory
    self.perception = perception or DeterministicPerceptionAdapter()
    self.inference = InferenceEngine(memory)
    self.verifier = DeepSeekInvariantVerifier(memory)

def learn(self, text: str) -> LearningResult:
    frame = self.perception.perceive(
        text, memory=self.memory, context=self.dialogue
    )
    return self.commit_frame(frame, source_type=SourceType.USER)
```

Move the current relation/property/action mutation body into `commit_frame()`. The method must consume only candidate metadata and the configured registries; it must not call `SimpleParser.parse_statement()` or perform a second independent parse.

- [ ] **Step 4: Route each relational claim through the existing ledger**

For every non-property claim, preserve the current sequence but make the candidate boundary explicit:

```python
evidence = self.memory.ledger.propose_relation(
    frame,
    claim,
    source_type=source_type,
    source_reference=f"frame:{frame.frame_id}",
    positive=claim.positive,
)
verification = self.verifier.verify_relation(
    claim.subject,
    claim.predicate,
    claim.object,
).to_verification_result()
decision = self.memory.ledger.commit_relation(
    evidence,
    verification,
    positive=claim.positive,
)
```

Only `CommitStatus.ACCEPTED` may add a relation to `relations_created`. UNKNOWN and REJECTED must preserve evidence and decision records but must not add an edge. Do not set candidate probability or frame uncertainty to an unexplained literal; use the adapter policy values.

- [ ] **Step 5: Add property and action commit methods without bypassing the boundary**

Add ledger methods that first persist candidate evidence and then apply the existing mutation only after `VerificationStatus.PASSED`:

```python
def commit_property(
    self,
    evidence: EvidenceRecord,
    verification: VerificationResult,
    *,
    key: str,
    value: str,
) -> CommitDecision:
    if verification.status is VerificationStatus.PASSED:
        concept = self.memory.get_or_create_concept(evidence.subject_id)
        self.memory.update_concept_attributes(concept.id, {key: value})
        self.memory.evidence.update_status(evidence.evidence_id, EvidenceStatus.ACCEPTED)
    elif verification.status is VerificationStatus.UNKNOWN:
        self.memory.evidence.update_status(evidence.evidence_id, EvidenceStatus.CANDIDATE)
    else:
        self.memory.evidence.update_status(evidence.evidence_id, EvidenceStatus.REJECTED)
    decision = verification.to_commit_decision(evidence.evidence_id)
    self.memory.evidence.save_decision(decision)
    return decision
```

`commit_property()` must not call `commit_relation()`, because a property is stored as an attribute rather than a graph edge. Actions must use `SchemaRegistry.action()` and the existing `TransformationEngine`; an unregistered action must become UNKNOWN with zero graph mutations.

- [ ] **Step 6: Preserve dialogue and experience logging after commit**

Keep the current `DialogueContext`, `last_subject`, `last_object`, and `add_experience()` behavior in `commit_frame()`, but derive entities and extracted triples from the candidate frame. A failed/unknown frame may be logged as an experience; it must not be treated as an accepted fact.

- [ ] **Step 7: Run learning and legacy regression tests**

Run:

```bash
uv run pytest tests/unit/test_learning_ledger_integration.py tests/unit/test_cognitive_kernel.py tests/unit/test_conversational_chat.py tests/unit/test_procedural.py tests/unit/test_dynamics.py -q
```

Expected: all focused tests pass, accepted facts retain user provenance, and existing action/dynamics behavior remains available through the candidate commit path.

- [ ] **Step 8: Save the candidate commit refactor**

```bash
git add src/little/language/parser.py src/little/memory/ledger.py tests/unit/test_learning_ledger_integration.py tests/unit/test_cognitive_kernel.py
git commit -m "refactor: commit learning only from candidate frames"
```

### Task 3: Create the unified cognitive kernel

**Files:**
- Create: `src/little/core/kernel.py`
- Modify: `src/little/inference/thinking_controller.py:20-190`
- Test: `tests/unit/test_cognitive_kernel.py`

**Interfaces:**
- `CognitiveKernel(memory, perception=None, learner=None, controller=None)`
- `CognitiveKernel.process(text: str) -> KernelOutcome`
- `KernelOutcome.frame` is the exact candidate returned by the adapter.
- `KernelOutcome.learning` is present only for a STUDY/action route.
- `KernelOutcome.inference` is present only for a question/UNKNOWN route.

- [ ] **Step 1: Write failing route tests**

```python
from little.core.kernel import CognitiveKernel
from little.inference.thinking_controller import CognitiveMode
from little.memory.store import MemoryStore


def test_kernel_routes_statement_to_study_and_commits_once():
    memory = MemoryStore(":memory:", seed_ontology=False)
    outcome = CognitiveKernel(memory).process("A falcon is a bird.")

    assert outcome.mode is CognitiveMode.STUDY
    assert outcome.learning is not None
    assert outcome.inference is None
    assert memory.count_relations() == 1


def test_kernel_routes_known_question_without_learning():
    memory = MemoryStore(":memory:", seed_ontology=False)
    CognitiveKernel(memory).process("A falcon is a bird.")
    before = memory.count_relations()

    outcome = CognitiveKernel(memory).process("Is a falcon a bird?")

    assert outcome.mode is CognitiveMode.FAST
    assert outcome.inference is not None
    assert outcome.inference.answer is True
    assert outcome.learning is None
    assert memory.count_relations() == before


def test_kernel_routes_unknown_text_without_guessing_or_writing():
    memory = MemoryStore(":memory:", seed_ontology=False)
    outcome = CognitiveKernel(memory).process("qwerty zorp")

    assert outcome.mode is CognitiveMode.UNKNOWN
    assert outcome.inference is not None
    assert outcome.inference.status.value == "UNKNOWN"
    assert outcome.inference.answer is None
    assert memory.count_relations() == 0


def test_kernel_keeps_unproven_relation_unknown():
    memory = MemoryStore(":memory:", seed_ontology=False)
    outcome = CognitiveKernel(memory).process("Is a falcon a spaceship?")

    assert outcome.inference is not None
    assert outcome.inference.status.value == "UNKNOWN"
    assert memory.count_relations() == 0
```

- [ ] **Step 2: Run the route tests to verify they fail**

Run: `uv run pytest tests/unit/test_cognitive_kernel.py -q`

Expected: FAIL because `CognitiveKernel`, `KernelOutcome`, and `CognitiveMode.UNKNOWN` do not exist.

- [ ] **Step 3: Add the explicit UNKNOWN controller mode**

Extend `CognitiveMode` with `UNKNOWN`. Keep existing `plan()` decisions unchanged for recognized questions, direct relations, multi-hop relations, and declarative statements. Use UNKNOWN only when the perception frame has no recognized statement, question, or action candidate.

- [ ] **Step 4: Implement `KernelOutcome` and `CognitiveKernel.process()`**

Implement the entrypoint with this control flow:

```python
class CognitiveKernel:
    def __init__(
        self,
        memory: MemoryStore,
        perception: PerceptionAdapter | None = None,
        learner: LearningEngine | None = None,
        controller: ThinkingController | None = None,
    ) -> None:
        self.memory = memory
        self.perception = perception or DeterministicPerceptionAdapter()
        self.learner = learner or LearningEngine(memory, perception=self.perception)
        self.controller = controller or ThinkingController(
            memory,
            resolver=self.learner.ask,
        )

    def process(self, text: str) -> KernelOutcome:
        frame = self.perception.perceive(
            text, memory=self.memory, context=self.learner.dialogue
        )
        if frame.intent in {"question"}:
            thinking = self.controller.run(text)
            return KernelOutcome(
                frame=frame,
                mode=thinking.mode,
                inference=thinking.inference,
            )
        if frame.intent in {"statement", "action"}:
            learning = self.learner.commit_frame(frame, source_type=SourceType.USER)
            return KernelOutcome(
                frame=frame,
                mode=CognitiveMode.STUDY,
                learning=learning,
            )
        return KernelOutcome(
            frame=frame,
            mode=CognitiveMode.UNKNOWN,
            inference=InferenceResult(
                query=text,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=0.0,
                evidence=[],
                trace=["Perception produced no grounded candidate."],
            ),
        )
```

The kernel must not call `MemoryStore.add_relation()` directly. The only write path is `LearningEngine.commit_frame()` → `EvidenceLedger`.

- [ ] **Step 5: Run kernel, controller, and contract tests**

Run: `uv run pytest tests/unit/test_cognitive_kernel.py tests/unit/test_thinking_controller.py tests/unit/test_cognitive_contracts.py -q`

Expected: PASS.

- [ ] **Step 6: Save the unified kernel**

```bash
git add src/little/core/kernel.py src/little/inference/thinking_controller.py tests/unit/test_cognitive_kernel.py
git commit -m "feat: add unified cognitive kernel entrypoint"
```

### Task 4: Route CLI commands through the kernel

**Files:**
- Modify: `src/little/main.py:30-85,274-575`
- Create: `tests/integration/test_cognitive_kernel_cli.py`
- Modify: `tests/unit/test_cli_database_path.py`

**Interfaces:**
- `get_kernel(db_path: Path, seed_ontology: bool = True) -> tuple[MemoryStore, CognitiveKernel]`
- `cmd_learn` consumes `KernelOutcome.learning`.
- `cmd_ask` consumes `KernelOutcome.inference`.
- `cmd_interact` keeps command handling local but sends all semantic input to `CognitiveKernel.process()`.

- [ ] **Step 1: Write failing command-routing tests**

```python
from pathlib import Path

from little.core.kernel import CognitiveKernel
from little.main import get_kernel


def test_get_kernel_uses_the_requested_database(tmp_path: Path):
    db_path = tmp_path / "kernel.db"
    store, kernel = get_kernel(db_path, seed_ontology=False)
    try:
        result = kernel.process("A falcon is a bird.")
        assert result.learning is not None
        assert store.find_relation_by_names("falcon", "is_a", "bird") is not None
        assert db_path.exists()
    finally:
        store.close()


def test_kernel_command_path_does_not_write_for_questions(tmp_path: Path):
    db_path = tmp_path / "kernel.db"
    store, kernel = get_kernel(db_path, seed_ontology=False)
    try:
        kernel.process("A falcon is a bird.")
        before = store.count_relations()
        outcome = kernel.process("Is a falcon a bird?")
        assert outcome.inference is not None
        assert outcome.inference.answer is True
        assert store.count_relations() == before
    finally:
        store.close()
```

- [ ] **Step 2: Run the integration tests to verify they fail**

Run: `uv run pytest tests/integration/test_cognitive_kernel_cli.py -q`

Expected: FAIL because `get_kernel()` does not exist and command functions still construct independent engine paths.

- [ ] **Step 3: Add `get_kernel()` while preserving `get_engine()`**

Keep `get_engine()` unchanged for existing imports and tests. Add:

```python
def get_kernel(
    db_path: Path = DEFAULT_DB_PATH,
    seed_ontology: bool = True,
) -> tuple[MemoryStore, CognitiveKernel]:
    store = MemoryStore(db_path, seed_ontology=seed_ontology)
    return store, CognitiveKernel(store)
```

Use the same database-path handling already covered by `test_cli_database_path.py`.

- [ ] **Step 4: Replace command-local semantic execution**

Update `cmd_learn()` to call `kernel.process(statement)` and read `outcome.learning`. If it is absent, print the UNKNOWN/no-op message without attempting a second parse. Update `cmd_ask()` to call `kernel.process(question)` and read `outcome.inference`; preserve verbose trace output and the existing response formatting.

In `cmd_interact()`, preserve only shell-level commands (`help`, `memory`, `inspect`, `skills`, `clear`, and exit). Replace the heuristic `is_question` block with:

```python
outcome = kernel.process(user_input)
if outcome.inference is not None:
    res = outcome.inference
    # Existing response and inquisitor display logic remains read-only here.
elif outcome.learning is not None:
    res = outcome.learning
else:
    # Display the UNKNOWN result from outcome.inference.
```

The chat layer may still ask `SimpleParser.parse_question()` to locate the clarification target for `ActiveInquisitor`, but it must not call `engine.learn()` or write relations outside the kernel outcome.

- [ ] **Step 5: Run command and chat regressions**

Run:

```bash
uv run pytest tests/integration/test_cognitive_kernel_cli.py tests/unit/test_cli_database_path.py tests/unit/test_conversational_chat.py tests/integration/test_persistence_and_reasoning.py -q
```

Expected: PASS; custom database paths remain isolated, questions do not learn facts, and accepted statements retain existing output behavior.

- [ ] **Step 6: Save CLI integration**

```bash
git add src/little/main.py tests/integration/test_cognitive_kernel_cli.py tests/unit/test_cli_database_path.py
git commit -m "refactor: route semantic CLI input through kernel"
```

### Task 5: Enforce the no-hardcoding and UNKNOWN gates

**Files:**
- Create: `tests/evaluation/test_kernel_boundaries.py`
- Modify: `tests/unit/test_laya_gatekeeper.py` only to clarify it is a standalone heuristic fallback, not the kernel adapter.
- Modify: `docs/superpowers/specs/2026-09-23-mivi-cognitive-kernel-design.md` only if implementation details need a factual status note.

**Interfaces:**
- Boundary tests use `MemoryStore(":memory:")` and never open `data/little.db`.
- The test suite checks behavior rather than searching for forbidden names in every source line; the adapter and kernel contracts are the enforcement points.

- [ ] **Step 1: Write failing boundary tests**

```python
from little.core.kernel import CognitiveKernel
from little.language.perception import DeterministicPerceptionAdapter
from little.memory.store import MemoryStore


def test_perception_is_side_effect_free_for_new_domain_words():
    memory = MemoryStore(":memory:", seed_ontology=False)
    adapter = DeterministicPerceptionAdapter()
    before = (len(memory.list_concepts()), memory.count_relations())

    frame = adapter.perceive("A zephyr is a vehicle.", memory=memory)

    assert frame.claims[0].subject == "zephyr"
    assert (len(memory.list_concepts()), memory.count_relations()) == before


def test_unknown_and_unregistered_inputs_create_no_edges():
    memory = MemoryStore(":memory:", seed_ontology=False)
    kernel = CognitiveKernel(memory)

    unknown = kernel.process("qwerty zorp")
    assert unknown.inference is not None
    assert unknown.inference.status.value == "UNKNOWN"
    assert memory.count_relations() == 0

    unsupported = kernel.process("A zephyr invents a relation.")
    assert unsupported.learning is not None
    assert memory.count_relations() == 0


def test_same_parser_path_handles_multiple_unlisted_concepts():
    memory = MemoryStore(":memory:", seed_ontology=False)
    kernel = CognitiveKernel(memory)

    first = kernel.process("A zephyr is a vehicle.")
    second = kernel.process("A lantern is an object.")

    assert first.learning is not None
    assert second.learning is not None
    assert memory.find_relation_by_names("zephyr", "is_a", "vehicle") is not None
    assert memory.find_relation_by_names("lantern", "is_a", "object") is not None
```

- [ ] **Step 2: Run the boundary tests to verify missing enforcement**

Run: `uv run pytest tests/evaluation/test_kernel_boundaries.py -q`

Expected: FAIL until all semantic paths use the adapter and unknown/unregistered candidates are prevented from committing.

- [ ] **Step 3: Make unknown and schema-failed candidates non-committing**

Ensure `DeterministicPerceptionAdapter` returns `intent="unknown"` for text it cannot parse. Ensure `LearningEngine.commit_frame()` does not create concepts or relations for an unsupported predicate/action before verification. If the verifier reports an unregistered predicate, retain the evidence as `candidate` or `rejected` according to the existing verification policy, but never call `MemoryStore.add_relation()`.

- [ ] **Step 4: Add an explicit adapter identity regression**

Keep `LayaSystem1Gatekeeper` tests separate from `DeterministicPerceptionAdapter` tests. Add an assertion that the fallback frame carries `model_id == "deterministic-parser"` and that no code path reports it as Laya. Do not alter the Laya file to claim model execution.

- [ ] **Step 5: Run the complete verification gate**

Run:

```bash
uv run pytest -q
uv run ruff check src/little/core/kernel.py src/little/language/perception.py src/little/language/parser.py src/little/memory/ledger.py src/little/inference/thinking_controller.py src/little/main.py tests/unit/test_perception_adapter.py tests/unit/test_cognitive_kernel.py tests/unit/test_learning_ledger_integration.py tests/integration/test_cognitive_kernel_cli.py tests/evaluation/test_kernel_boundaries.py
uv run python -m compileall -q src tests
git diff --check
```

Expected: all tests pass, Ruff reports no errors in changed files, compilation succeeds, and whitespace validation is clean.

- [ ] **Step 6: Save the integration boundary**

```bash
git add src tests data/schemas/perception_policy.json docs/superpowers/specs/2026-09-23-mivi-cognitive-kernel-design.md
git commit -m "test: enforce unified kernel and no-hardcoding boundary"
```

## Plan self-review

- **Spec coverage:** The plan implements every item in the approved first slice: adapter contract, candidate-only perception, unified entrypoint, ledger-gated commits, CLI/chat routing, and UNKNOWN/no-edge tests. The six-axis web, real model adapters, graph search, dynamics, curiosity, and training are explicitly out of scope.
- **No-hardcoding coverage:** Confidence and uncertainty values live in `data/schemas/perception_policy.json`; parsing and relation/action behavior remain registry-driven. The plan adds no object-specific Python branch.
- **Type consistency:** `CandidateClaim` metadata is extended before the adapter creates claims. `PerceptionAdapter` produces `CandidateFrame`; `LearningEngine.commit_frame()` consumes it; `CognitiveKernel.process()` returns `KernelOutcome`; CLI consumes that outcome.
- **Purity:** Adapter tests compare concept/relation counts before and after perception. Kernel tests verify questions and UNKNOWN inputs do not mutate relations.
- **Compatibility:** Existing `get_engine()`, `LearningEngine.learn()`, `LearningEngine.ask()`, shell commands, and standalone Laya tests remain supported while the semantic path migrates.
- **Research boundary:** This plan does not pretend that the deterministic fallback is Laya, that GLM provides formal graph proofs, or that DeepSeek provides a persistent symbolic graph.
- **Placeholder scan:** No step uses TODO, TBD, or unnamed interfaces. Every implementation step names the file, public method, test command, and expected result.
