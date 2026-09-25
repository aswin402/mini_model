# MIVI Cognitive Kernel Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first CPU-first foundation of MIVI: typed cognitive contracts, an evidence ledger, schema-driven domain metadata, verifier-backed commits, and safe integration with the existing LITTLE graph.

**Architecture:** Neural or rule-based perception produces `CandidateFrame` objects. Grounding converts candidates into provenance-bearing evidence. `EvidenceLedger` controls all durable graph mutations and only commits claims that pass the current verification policy. Relation, action, and state behavior comes from registries and data files rather than object-specific branches. Laya integration, advanced thinking control, generic dynamics, and autonomous study remain later plans built on these stable boundaries.

**Tech Stack:** Python 3.12+, SQLite, existing `dataclasses` and `enum`, pytest, Ruff, uv, existing SymPy dependency. No new runtime dependency is required for this foundation.

## Global Constraints

- Preserve existing user changes and the existing SQLite database; migrations must be additive and idempotent.
- Keep the deterministic kernel runnable on CPU with no neural model installed.
- A candidate frame must never directly mutate durable memory.
- Every accepted relation must have provenance, update type, and a verification decision.
- Missing evidence must remain `UNKNOWN`; it must not be converted into a negative fact.
- Domain behavior must be represented by registry records or data fixtures, not object-name conditionals.
- Keep existing public `MemoryStore` and `LearningEngine` behavior compatible unless a test explicitly covers the new contract.
- Use test-first changes: write the failing test, run the focused test, implement the smallest change, run the focused test, then run the regression suite.
- Do not add Laya, Transformers, ONNX, or GPU dependencies in this foundation plan.

---

## Scope boundary

This plan intentionally covers the first independently testable slice only:

1. contracts and result statuses;
2. evidence/provenance persistence;
3. verifier-backed commit flow;
4. schema registries and data-driven metadata;
5. integration with the existing parser and self-study code;
6. CLI database-path safety and regression evaluation.

The following are separate follow-on plans after this foundation passes:

- `2026-09-23-mivi-perception-adapter.md`: optional Laya adapter and typed extraction dataset;
- `2026-09-23-mivi-thinking-controller.md`: FAST/THINK/ASK/STUDY route planning and proof search;
- `2026-09-23-mivi-generic-dynamics.md`: data-driven action schemas, state variables, and continuous dynamics;
- `2026-09-23-mivi-curiosity-growth.md`: information-gain study loop and 360-degree graph growth;
- `2026-09-23-mivi-model-training-evaluation.md`: distillation, calibration, replay, and continual-learning benchmarks.

## File map

The foundation changes are grouped by responsibility:

- `src/little/core/contracts.py` — candidate, evidence, verification, and commit boundary types.
- `src/little/memory/schema.py` — additive SQLite schema objects for evidence, proofs, and decisions.
- `src/little/memory/evidence.py` — persistence operations for evidence and proof records.
- `src/little/memory/ledger.py` — policy-controlled commit service.
- `src/little/knowledge/registry.py` — relation, action, and state schema registry.
- `data/schemas/*.json` — versioned data-driven metadata fixtures.
- `src/little/memory/store.py` — compatibility wrappers and existing graph mutation integration.
- `src/little/language/parser.py` — candidate creation and ledger-based commits.
- `src/little/active/self_study.py` — use the ledger and current store API.
- `src/little/main.py` — fix duplicate `--db` parser defaults without adding new commands.
- `tests/unit/test_cognitive_contracts.py` — pure contract tests.
- `tests/unit/test_evidence_ledger.py` — persistence and commit tests.
- `tests/unit/test_schema_registry.py` — data-driven registry tests.
- `tests/unit/test_learning_ledger_integration.py` — parser integration tests.
- `tests/unit/test_self_study_loop.py` — corrected self-study integration tests.
- `tests/unit/test_cli_database_path.py` — CLI path regression tests.

### Task 1: Define the cognitive contracts

**Files:**
- Create: `src/little/core/contracts.py`
- Test: `tests/unit/test_cognitive_contracts.py`
- Modify: `src/little/core/models.py:1-25` only if shared enums must be re-exported for compatibility

**Interfaces:**
- Produces `CandidateFrame`, `CandidateEntity`, `CandidateClaim`, `CandidateAction`, `EvidenceRecord`, `VerificationResult`, and `CommitDecision`.
- `CandidateEntity(name, type_hint, span, probability)` and `CandidateAction(name, arguments, probability)` are frozen dataclasses.
- `EvidenceRecord.from_claim(claim, source_type, source_reference, source_text="") -> EvidenceRecord` creates a `CANDIDATE` record with a generated evidence ID.
- `VerificationResult.to_commit_decision(evidence_id) -> CommitDecision` maps `PASSED` to `ACCEPTED`, `FAILED`/`INCONSISTENT` to `REJECTED`, and `UNKNOWN` to `UNKNOWN`.
- `CommitDecision.accepted/rejected/unknown(evidence_id, verification) -> CommitDecision` are named constructors.
- `ProofTrace` stores the conclusion, premises, operations, tool outputs, verifier results, and final status used by `EvidenceStore.save_proof`.
- Produces enums `EvidenceStatus`, `SourceType`, `VerificationStatus`, and `CommitStatus`.
- Consumes existing `current_iso_timestamp()` and `generate_id()` from `little.core.models`.

- [ ] **Step 1: Write the failing contract tests**

```python
from little.core.contracts import (
    CandidateClaim,
    CandidateFrame,
    CommitStatus,
    EvidenceRecord,
    EvidenceStatus,
    SourceType,
    VerificationResult,
    VerificationStatus,
)


def test_candidate_frame_is_not_an_accepted_evidence_record():
    frame = CandidateFrame.from_text(
        "A falcon is a bird.",
        claims=[CandidateClaim("falcon", "is_a", "bird", probability=0.98)],
    )

    assert frame.claims[0].predicate == "is_a"
    assert frame.claims[0].probability == 0.98
    assert frame.accepted is False


def test_evidence_record_requires_explicit_source_and_status():
    evidence = EvidenceRecord.from_claim(
        CandidateClaim("falcon", "is_a", "bird", probability=0.98),
        source_type=SourceType.USER,
        source_reference="conversation:test-1",
    )

    assert evidence.status is EvidenceStatus.CANDIDATE
    assert evidence.source_type is SourceType.USER
    assert evidence.source_reference == "conversation:test-1"


def test_commit_decision_serializes_verification_result():
    verification = VerificationResult(
        status=VerificationStatus.PASSED,
        checks={"grounded": True, "schema": True},
        reasons=["All terms resolved"],
    )

    decision = verification.to_commit_decision("evidence_1")

    assert decision.status is CommitStatus.ACCEPTED
    assert decision.evidence_id == "evidence_1"
    assert decision.checks == {"grounded": True, "schema": True}
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `uv run pytest tests/unit/test_cognitive_contracts.py -q`

Expected: FAIL because `little.core.contracts` does not exist.

- [ ] **Step 3: Implement the smallest contract module**

Implement `src/little/core/contracts.py` with these exact public shapes:

```python
class SourceType(str, Enum):
    USER = "user"
    DOCUMENT = "document"
    SENSOR = "sensor"
    TOOL = "tool"
    INFERENCE = "inference"
    HYPOTHESIS = "hypothesis"


class EvidenceStatus(str, Enum):
    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class VerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNKNOWN = "unknown"
    INCONSISTENT = "inconsistent"


class CommitStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CandidateClaim:
    subject: str
    predicate: str
    object: str
    probability: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateFrame:
    frame_id: str
    source_text: str
    intent: str
    claims: list[CandidateClaim]
    entities: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    uncertainty: float = 1.0
    model_id: str = "deterministic-parser"
    model_version: str = "builtin"
    accepted: bool = False

    @classmethod
    def from_text(cls, source_text: str, claims: list[CandidateClaim], **kwargs: Any) -> "CandidateFrame":
        return cls(
            frame_id=generate_id("frame"),
            source_text=source_text,
            intent=kwargs.pop("intent", "statement"),
            claims=claims,
            **kwargs,
        )
```

Also define the remaining public records with these fields:

```python
@dataclass(frozen=True)
class CandidateEntity:
    name: str
    type_hint: str | None = None
    span: tuple[int, int] | None = None
    probability: float = 0.0


@dataclass(frozen=True)
class CandidateAction:
    name: str
    arguments: dict[str, str]
    probability: float = 0.0


@dataclass
class EvidenceRecord:
    evidence_id: str
    subject_id: str
    predicate: str
    object_id: str | None
    source_type: SourceType
    source_reference: str
    source_text: str
    extraction_confidence: float
    status: EvidenceStatus = EvidenceStatus.CANDIDATE
    derivation_proof_id: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    checks: dict[str, bool]
    reasons: list[str]


@dataclass(frozen=True)
class CommitDecision:
    decision_id: str
    evidence_id: str
    status: CommitStatus
    checks: dict[str, bool]
    reasons: list[str]


@dataclass(frozen=True)
class ProofTrace:
    proof_id: str
    conclusion: dict[str, Any]
    premises: list[str]
    operations: list[dict[str, Any]]
    tool_outputs: list[dict[str, Any]]
    verifier_results: list[dict[str, Any]]
    status: VerificationStatus
```

Add matching `from_claim`, `to_dict`, and `to_commit_decision` methods. Validate probability and uncertainty in `[0.0, 1.0]`; raise `ValueError` for invalid values.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `uv run pytest tests/unit/test_cognitive_contracts.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the contract boundary**

```bash
git add src/little/core/contracts.py tests/unit/test_cognitive_contracts.py
git commit -m "feat: add cognitive evidence contracts"
```

### Task 2: Add additive evidence and proof persistence

**Files:**
- Modify: `src/little/memory/schema.py:1-100`
- Create: `src/little/memory/evidence.py`
- Modify: `src/little/memory/store.py:30-65`
- Test: `tests/unit/test_evidence_ledger.py`

**Interfaces:**
- `EvidenceStore.append(record: EvidenceRecord) -> EvidenceRecord`
- `EvidenceStore.get(evidence_id: str) -> EvidenceRecord | None`
- `EvidenceStore.list_for_claim(subject_id: str, predicate: str, object_id: str) -> list[EvidenceRecord]`
- `EvidenceStore.list_by_source(source_type: str) -> list[EvidenceRecord]`
- `EvidenceStore.update_status(evidence_id: str, status: EvidenceStatus) -> None`
- `EvidenceStore.save_proof(proof: ProofTrace) -> str`
- `EvidenceStore.save_decision(decision: CommitDecision) -> str`
- `MemoryStore.evidence` exposes the `EvidenceStore` instance.
- `MemoryStore.has_evidence_tables() -> bool` and `MemoryStore.count_relations() -> int` are testable inspection helpers.

- [ ] **Step 1: Write failing migration and persistence tests**

```python
from little.core.contracts import CandidateClaim, EvidenceRecord, SourceType
from little.memory.store import MemoryStore


def test_memory_store_creates_evidence_tables_without_deleting_graph_data():
    store = MemoryStore(":memory:", seed_ontology=False)
    subject = store.create_concept("falcon")
    object_ = store.create_concept("bird")
    store.add_relation(subject.id, "is_a", object_.id)

    assert store.count_relations() == 1
    assert store.has_evidence_tables()


def test_evidence_round_trip_preserves_provenance():
    store = MemoryStore(":memory:", seed_ontology=False)
    evidence = EvidenceRecord.from_claim(
        CandidateClaim("falcon", "is_a", "bird", probability=0.98),
        source_type=SourceType.USER,
        source_reference="conversation:test-2",
    )

    saved = store.evidence.append(evidence)
    loaded = store.evidence.get(saved.evidence_id)

    assert loaded is not None
    assert loaded.source_reference == "conversation:test-2"
    assert loaded.status.value == "candidate"
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `uv run pytest tests/unit/test_evidence_ledger.py -q`

Expected: FAIL because evidence tables and `MemoryStore.evidence` do not exist.

- [ ] **Step 3: Add idempotent schema objects**

Add these tables to `SCHEMA_V1` using `CREATE TABLE IF NOT EXISTS`:

```sql
CREATE TABLE IF NOT EXISTS evidence_records (
    evidence_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_id TEXT,
    value_json TEXT,
    source_type TEXT NOT NULL,
    source_reference TEXT NOT NULL,
    source_text TEXT,
    observed_at TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    support_weight INTEGER NOT NULL DEFAULT 0,
    opposition_weight INTEGER NOT NULL DEFAULT 0,
    extraction_confidence REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'candidate',
    derivation_proof_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS proof_traces (
    proof_id TEXT PRIMARY KEY,
    conclusion_json TEXT NOT NULL,
    premises_json TEXT NOT NULL DEFAULT '[]',
    operations_json TEXT NOT NULL DEFAULT '[]',
    tool_outputs_json TEXT NOT NULL DEFAULT '[]',
    verifier_results_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS commit_decisions (
    decision_id TEXT PRIMARY KEY,
    evidence_id TEXT NOT NULL,
    status TEXT NOT NULL,
    checks_json TEXT NOT NULL DEFAULT '{}',
    reasons_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
```

Add indexes on `(subject_id, predicate, object_id)`, `source_type`, `status`, and `observed_at`. Initialize `EvidenceStore` after schema creation in `MemoryStore.__init__`. Do not rebuild or clear existing tables.

- [ ] **Step 4: Implement `EvidenceStore` serialization and queries**

Serialize enums using `.value` and JSON fields using `json.dumps`. Normalize predicates to lowercase. Reject blank subject, predicate, object/value, source type, or source reference with `ValueError`. Preserve `candidate` status on append; status changes happen through the ledger.

- [ ] **Step 5: Run focused and existing memory tests**

Run: `uv run pytest tests/unit/test_evidence_ledger.py tests/unit/test_memory.py -q`

Expected: PASS with existing relation behavior unchanged.

- [ ] **Step 6: Commit additive evidence persistence**

```bash
git add src/little/memory/schema.py src/little/memory/evidence.py src/little/memory/store.py tests/unit/test_evidence_ledger.py
git commit -m "feat: persist evidence and proof records"
```

### Task 3: Implement verifier-backed relation commits

**Files:**
- Create: `src/little/memory/ledger.py`
- Modify: `src/little/memory/store.py:240-330`
- Modify: `src/little/inference/invariant_gates.py:1-220`
- Test: `tests/unit/test_evidence_ledger.py`

**Interfaces:**
- `EvidenceLedger.propose_relation(frame, claim, source_type, source_reference) -> EvidenceRecord`
- `EvidenceLedger.commit_relation(evidence, verification) -> CommitDecision`
- `EvidenceLedger.reject(evidence, reasons) -> CommitDecision`
- `EvidenceLedger.commit_relation` is the only new path allowed to mutate a durable relation.
- `MemoryStore.ledger` exposes the `EvidenceLedger` instance.
- `MemoryStore.find_relation_by_names(subject, predicate, object) -> Relation | None` is a read-only test helper that resolves names before querying the graph.

- [ ] **Step 1: Write failing commit-policy tests**

```python
def test_unverified_candidate_does_not_create_relation():
    store = MemoryStore(":memory:", seed_ontology=False)
    ledger = store.ledger
    frame = CandidateFrame.from_text(
        "A falcon is a bird.",
        claims=[CandidateClaim("falcon", "is_a", "bird", probability=0.98)],
    )

    evidence = ledger.propose_relation(
        frame,
        frame.claims[0],
        source_type=SourceType.USER,
        source_reference="conversation:test-3",
    )
    decision = ledger.commit_relation(
        evidence,
        VerificationResult(
            status=VerificationStatus.UNKNOWN,
            checks={"grounded": False},
            reasons=["object is unresolved"],
        ),
    )

    assert decision.status is CommitStatus.UNKNOWN
    assert store.find_relation_by_names("falcon", "is_a", "bird") is None


def test_verified_candidate_creates_relation_and_decision():
    store = MemoryStore(":memory:", seed_ontology=False)
    ledger = store.ledger
    frame = CandidateFrame.from_text(
        "A falcon is a bird.",
        claims=[CandidateClaim("falcon", "is_a", "bird", probability=0.98)],
    )
    evidence = ledger.propose_relation(
        frame,
        frame.claims[0],
        source_type=SourceType.USER,
        source_reference="conversation:test-4",
    )
    decision = ledger.commit_relation(
        evidence,
        VerificationResult(
            status=VerificationStatus.PASSED,
            checks={"grounded": True, "schema": True},
            reasons=["terms resolved"],
        ),
    )

    assert decision.status is CommitStatus.ACCEPTED
    assert store.find_relation_by_names("falcon", "is_a", "bird") is not None
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/unit/test_evidence_ledger.py -q`

Expected: FAIL because `MemoryStore.ledger`, `VerificationResult`, and the commit path do not exist.

- [ ] **Step 3: Add explicit commit policy**

Implement this policy:

```python
class EvidenceLedger:
    def commit_relation(
        self,
        evidence: EvidenceRecord,
        verification: VerificationResult,
    ) -> CommitDecision:
        if verification.status is VerificationStatus.PASSED:
            subject = self.memory.get_or_create_concept(evidence.subject_id)
            object_ = self.memory.get_or_create_concept(evidence.object_id or "")
            self.memory.add_relation(
                subject.id,
                evidence.predicate,
                object_.id,
                positive=True,
                source_experience_id=evidence.source_reference,
            )
            evidence.status = EvidenceStatus.ACCEPTED
            self.memory.evidence.update_status(evidence.evidence_id, evidence.status)
            decision = CommitDecision.accepted(evidence.evidence_id, verification)
        elif verification.status is VerificationStatus.UNKNOWN:
            decision = CommitDecision.unknown(evidence.evidence_id, verification)
        else:
            decision = CommitDecision.rejected(evidence.evidence_id, verification)
        self.memory.evidence.save_decision(decision)
        return decision
```

The final implementation must resolve names to concept IDs before creating `EvidenceRecord`; the snippet’s names illustrate control flow, not permission to store unresolved IDs. Rejected and unknown evidence remain queryable but do not create graph edges.

- [ ] **Step 4: Make invariant verification return one stable result type**

Replace tuple-shape ambiguity in `DeepSeekInvariantVerifier` and `AutonomousSelfStudyEngine.verify_invariant_gates` with `VerificationResult`. Preserve a compatibility helper only if existing callers require it:

```python
def verify_invariant_gates_legacy(hypothesis: dict[str, str]) -> tuple[bool, str]:
    result = verify_invariant_gates(hypothesis)
    return result.status is VerificationStatus.PASSED, "; ".join(result.reasons)
```

Ensure `I_sort` validates registered domain/range types and `I_ground` checks actual evidence, not merely non-empty strings.

- [ ] **Step 5: Run all inference and self-study tests**

Run: `uv run pytest tests/unit/test_evidence_ledger.py tests/unit/test_invariant_gates.py tests/unit/test_self_study_loop.py -q`

Expected: PASS with no `get_recent_experiences` or unsupported `add_relation` argument errors.

- [ ] **Step 6: Commit verifier-backed commits**

```bash
git add src/little/memory/ledger.py src/little/memory/store.py src/little/inference/invariant_gates.py tests/unit/test_evidence_ledger.py tests/unit/test_self_study_loop.py
git commit -m "feat: gate durable relations through evidence verification"
```

### Task 4: Add data-driven relation, action, and state registries

**Files:**
- Create: `src/little/knowledge/registry.py`
- Create: `data/schemas/relation_types.json`
- Create: `data/schemas/action_schemas.json`
- Create: `data/schemas/state_variables.json`
- Test: `tests/unit/test_schema_registry.py`
- Modify: `src/little/memory/ontology.py:1-380` only to load registry metadata instead of embedding engine behavior
- Modify: `src/little/language/construction.py:1-824` only to accept registry-backed construction metadata

**Interfaces:**
- `SchemaRegistry.load(path: Path) -> SchemaRegistry`
- `SchemaRegistry.relation(name: str) -> RelationSchema | None`
- `SchemaRegistry.action(name: str) -> ActionSchema | None`
- `SchemaRegistry.state_variable(name: str) -> StateVariableSchema | None`
- `SchemaRegistry.validate_relation(predicate, subject_type, object_type) -> ValidationResult`
- `SchemaRegistry.validate_action(name, arguments) -> ValidationResult`

- [ ] **Step 1: Write failing registry tests**

```python
from pathlib import Path
from little.knowledge.registry import SchemaRegistry


def test_registry_loads_relation_and_action_metadata():
    registry = SchemaRegistry.load(Path("data/schemas"))

    part_of = registry.relation("part_of")
    split = registry.action("split")

    assert part_of is not None
    assert part_of.inverse == "has_part"
    assert split is not None
    assert "conservation" in split.invariants


def test_registry_rejects_invalid_relation_types():
    registry = SchemaRegistry.load(Path("data/schemas"))

    result = registry.validate_relation("is_a", "quantity", "animal")

    assert result.valid is False
    assert "domain" in result.reason.lower() or "range" in result.reason.lower()
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/unit/test_schema_registry.py -q`

Expected: FAIL because the registry and schema files do not exist.

- [ ] **Step 3: Add the minimal schema data**

Create JSON records with this shape:

```json
{
  "relation_types": [
    {
      "name": "is_a",
      "domain": ["concept", "entity"],
      "range": ["concept"],
      "inverse": null,
      "symmetric": false,
      "transitive": true,
      "evidence_policy": "observed_or_derived"
    },
    {
      "name": "part_of",
      "domain": ["entity", "concept"],
      "range": ["entity", "concept"],
      "inverse": "has_part",
      "symmetric": false,
      "transitive": false,
      "evidence_policy": "observed_or_derived"
    }
  ]
}
```

Add an action schema for generic `split` with typed arguments, preconditions, generated parts, and a named `mass_conservation` invariant. The schema must not mention apple, bread, cheese, or another concrete object.

- [ ] **Step 4: Implement registry validation**

Use frozen dataclasses for `RelationSchema`, `ActionSchema`, `StateVariableSchema`, and `ValidationResult`. Load all JSON files in deterministic filename order. Reject duplicate names, missing required fields, unknown inverse references, and malformed invariant lists with `ValueError` during load.

- [ ] **Step 5: Pass registry metadata into the verifier**

Make the invariant verifier accept `SchemaRegistry` and use it for sort validation. If a predicate is not registered, return `UNKNOWN` with reason `unregistered predicate`; do not silently accept it as a generic relation.

- [ ] **Step 6: Run focused tests and commit**

Run: `uv run pytest tests/unit/test_schema_registry.py tests/unit/test_invariant_gates.py -q`

Expected: PASS.

```bash
git add src/little/knowledge/registry.py data/schemas tests/unit/test_schema_registry.py src/little/memory/ontology.py src/little/language/construction.py src/little/inference/invariant_gates.py
git commit -m "feat: move domain behavior into schema registries"
```

### Task 5: Route parser and self-study updates through the ledger

**Files:**
- Modify: `src/little/language/parser.py:1665-1875`
- Modify: `src/little/active/self_study.py:1-360`
- Modify: `src/little/active/inquisitor.py:130-240`
- Create: `tests/unit/test_learning_ledger_integration.py`
- Modify: `tests/unit/test_self_study_loop.py`

**Interfaces:**
- `LearningEngine.learn` continues returning `LearningResult`.
- Internally it creates `CandidateFrame` and calls `EvidenceLedger.propose_relation` and `commit_relation`.
- Self-study uses the same ledger and records rejected/unknown hypotheses without direct `add_relation` calls.

- [ ] **Step 1: Write failing integration tests**

```python
def test_learning_records_user_provenance_and_commits_verified_fact():
    store = MemoryStore(":memory:", seed_ontology=False)
    engine = LearningEngine(store)

    result = engine.learn("A falcon is a bird.")

    assert result.relations_created
    records = store.evidence.list_by_source("user")
    assert len(records) == 1
    assert records[0].status.value == "accepted"
    assert records[0].source_text == "A falcon is a bird."


def test_unknown_candidate_is_not_written_as_a_relation():
    store = MemoryStore(":memory:", seed_ontology=False)
    engine = LearningEngine(store)

    result = engine.learn("Something uncertain relates to an unresolved thing.")

    assert result.update_type.value in {"NO_OP", "EVIDENCE_ADDITION"}
    assert store.count_relations() == 0
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/unit/test_learning_ledger_integration.py -q`

Expected: FAIL because parser writes relations directly and the evidence query is absent.

- [ ] **Step 3: Convert parsed triples into candidate frames**

After `SimpleParser` produces triples, construct one `CandidateFrame` with the original input text, parser model metadata, extracted claims, and the parser’s confidence. Do not create concepts or relations before grounding.

- [ ] **Step 4: Add a grounding helper**

Implement `LearningEngine._ground_claim(claim)` to resolve or create candidate concepts using normalized names. It may create candidate concepts, but it must not create accepted relations. Return the resolved IDs and a `VerificationResult` that records unresolved names.

- [ ] **Step 5: Replace direct relation writes**

For each grounded claim, call the ledger. Map accepted decisions to the existing `LearningResult` values. Map rejected and unknown decisions to `CONTRADICTION`, `EVIDENCE_ADDITION`, or `NO_OP` according to existing behavior, while preserving the decision and reason in the evidence log.

- [ ] **Step 6: Repair self-study against the new boundary**

Replace `get_recent_experiences` with the existing experience listing API or add a correctly tested `list_experiences(limit=...)` method. Remove unsupported `weight` and `confidence` arguments from `MemoryStore.add_relation`; the ledger controls evidence accumulation. Make `study_step` call the same grounding, verification, and commit path as user learning.

- [ ] **Step 7: Run parser, dialogue, self-study, and memory regression tests**

Run: `uv run pytest tests/unit/test_learning_ledger_integration.py tests/unit/test_self_study_loop.py tests/unit/test_conversational_chat.py tests/unit/test_memory.py -q`

Expected: PASS with existing accepted facts and UNKNOWN behavior preserved.

- [ ] **Step 8: Commit ledger integration**

```bash
git add src/little/language/parser.py src/little/active/self_study.py src/little/active/inquisitor.py tests/unit/test_learning_ledger_integration.py tests/unit/test_self_study_loop.py
git commit -m "refactor: route learning through evidence ledger"
```

### Task 6: Fix CLI database-path handling

**Files:**
- Modify: `src/little/main.py:625-760`
- Create: `tests/unit/test_cli_database_path.py`

**Interfaces:**
- Every subcommand accepts one `--db` option.
- `little --db PATH init` and `little init --db PATH` resolve to the same path.
- No subparser default may overwrite a value supplied by the parent parser.

- [ ] **Step 1: Write the failing CLI test**

```python
from pathlib import Path
from little.main import build_parser


def test_global_db_path_is_not_overwritten_by_subparser_default(tmp_path: Path):
    path = tmp_path / "custom.db"
    parser = build_parser()

    args = parser.parse_args(["--db", str(path), "init"])

    assert args.db == str(path)


def test_subcommand_db_path_is_supported(tmp_path: Path):
    path = tmp_path / "custom.db"
    parser = build_parser()

    args = parser.parse_args(["init", "--db", str(path)])

    assert args.db == str(path)
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `uv run pytest tests/unit/test_cli_database_path.py -q`

Expected: FAIL because the current parser defines the same default on parent and child parsers, overwriting the global value.

- [ ] **Step 3: Extract `build_parser()` and use a suppressed child default**

Move parser construction out of `main()` into `build_parser()`. Define the shared option once on the root parser with `default=str(DEFAULT_DB_PATH)`. For subparsers, either omit `--db` and rely on the root option or add it with `default=argparse.SUPPRESS` so it cannot overwrite a root value. Preserve both argument positions through tests.

- [ ] **Step 4: Run CLI and full regression tests**

Run: `uv run pytest tests/unit/test_cli_database_path.py tests/unit -q`

Expected: all existing tests pass and the custom database path remains isolated from `data/little.db`.

- [ ] **Step 5: Commit CLI safety fix**

```bash
git add src/little/main.py tests/unit/test_cli_database_path.py
git commit -m "fix: preserve explicit CLI database paths"
```

### Task 7: Add the foundation evaluation gate

**Files:**
- Create: `tests/evaluation/test_foundation_invariants.py`
- Create: `experiments/006_mivi_foundation/README.md`
- Create: `experiments/006_mivi_foundation/run.py`
- Create: `experiments/006_mivi_foundation/analysis.md`

**Interfaces:**
- The evaluator runs against a temporary SQLite database.
- It reports accepted claims, rejected claims, UNKNOWN results, provenance completeness, and domain-branch violations.
- It must not use or modify `data/little.db`.

- [ ] **Step 1: Write the failing invariant tests**

```python
def test_same_split_action_schema_works_for_multiple_material_records():
    result = run_action_fixture(
        "split",
        [
            {"name": "apple", "material": "organic"},
            {"name": "bread", "material": "organic"},
            {"name": "iron_bar", "material": "metal"},
        ],
    )

    assert all(item.status == "accepted" for item in result)
    assert all(item.invariant_results["mass_conservation"] for item in result)


def test_unknown_relation_never_becomes_an_accepted_edge():
    result = learn_unregistered_relation("falcon", "invented_predicate", "bird")

    assert result.status == "unknown"
    assert result.accepted_edges == 0
```

- [ ] **Step 2: Run the evaluator to verify it fails**

Run: `uv run pytest tests/evaluation/test_foundation_invariants.py -q`

Expected: FAIL because generic action fixtures and registry-backed commits are not complete.

- [ ] **Step 3: Implement the temporary-database evaluation harness**

Use `tempfile.TemporaryDirectory` and `MemoryStore(path, seed_ontology=False)`. Load only `data/schemas`. Store each run’s JSON result under the experiment directory. Do not import or open `data/little.db`.

- [ ] **Step 4: Add measurable acceptance thresholds**

The first gate passes only when:

- all existing unit tests pass;
- all foundation invariant tests pass;
- every accepted relation has a source and decision record;
- no unregistered predicate is accepted;
- no object-name conditional is needed for the split fixtures;
- UNKNOWN cases produce zero accepted edges;
- the deterministic kernel works without a neural dependency.

- [ ] **Step 5: Run the complete verification suite**

Run:

```bash
uv run pytest -q
uv run ruff check src tests experiments/006_mivi_foundation
uv run python -m compileall -q src tests experiments
git diff --check
```

Expected: all tests pass; Ruff reports no errors in changed files; compilation succeeds; diff check is clean.

- [ ] **Step 6: Commit the foundation evaluation**

```bash
git add tests/evaluation/test_foundation_invariants.py experiments/006_mivi_foundation
git commit -m "test: add MIVI foundation evaluation gate"
```

## Plan self-review

- **Spec coverage:** The plan covers the first implementation slice from the design: contracts, evidence ledger, verification, registries, parser/self-study integration, CLI safety, and evaluation. Neural perception, advanced thinking, dynamics, curiosity, and model training are explicitly isolated into later plans.
- **Placeholder scan:** No step depends on an unspecified file, unnamed interface, or undefined future function. Later plans are named deliverables, not tasks in this plan.
- **Type consistency:** `CandidateClaim`, `CandidateFrame`, `EvidenceRecord`, `VerificationResult`, and `CommitDecision` are introduced in Task 1 and reused by Tasks 2–5. Registry types are introduced in Task 4 and consumed by verification in Tasks 4–7.
- **Safety:** Existing databases are migrated additively. Evaluation uses temporary databases. The CLI regression test prevents another accidental write to `data/little.db`.
