# LITTLE — Technical Specification

**Version:** 0.1  
**Status:** Initial technical specification

---

## 1. Runtime target

### Primary development environment

- Python 3.12+ recommended.
- Rust stable toolchain.
- Linux preferred for development, with Windows support where practical.
- 16 GB RAM target for the CPU-first prototype.
- GPU optional.

The exact minimum versions should be pinned in the project configuration once the initial dependency set is selected.

---

## 2. Language strategy

### Python

Python is the primary implementation language because LITTLE is primarily a research system.

Python will own:

- orchestration;
- learning experiments;
- neural models;
- evaluation;
- CLI;
- data pipelines;
- high-level reasoning;
- storage interfaces.

### Rust

Rust will be used selectively for performance-sensitive code.

We should not prematurely rewrite the whole project in Rust.

The desired architecture is:

```text
Python
  |
  +-- research / orchestration
  |
  +-- neural components
  |
  +-- experiment framework
  |
  +-- public API
  |
  +-- memory semantics
  |
  +-- reasoning orchestration
  |
  v
Rust extensions
  |
  +-- hot loops
  +-- indexing
  +-- graph algorithms
  +-- serialization
  +-- high-throughput data operations
```

PyO3 is the bridge between Python and Rust. Maturin is the preferred packaging/build path for Rust-backed Python extensions.

---

## 3. Modern Python toolchain

### 3.1 uv

Use `uv` as the primary Python project/dependency/environment tool.

Responsibilities:

- Python version management where supported;
- virtual environments;
- dependency resolution;
- lockfile;
- running commands;
- reproducible development setup.

The project should commit `uv.lock`.

### 3.2 Ruff

Use Ruff for:

- linting;
- formatting;
- import organization.

Do not add Black/isort unless a specific compatibility requirement appears.

### 3.3 Pyrefly

Use Pyrefly as the primary static type checker.

All new core modules should have meaningful type annotations.

### 3.4 pytest

Use pytest for unit and integration testing.

### 3.5 pre-commit / CI

Add automated checks once the repository has a stable baseline.

Minimum CI:

```text
ruff check
ruff format --check
pyrefly check
pytest
cargo check
cargo test
```

---

## 4. Data model

### 4.1 Identifier

All persistent entities require stable IDs.

Recommended representation:

```text
UUID or compact string ID
```

IDs must not depend on display names.

---

### 4.2 Concept

```python
Concept(
    id: ConceptId,
    name: str,
    aliases: list[str],
    status: ConceptStatus,
    confidence: float,
)
```

Future fields:

```text
embedding
parents
children
attributes
examples
evidence
```

---

### 4.3 Entity

An entity represents a specific instance.

Example:

```text
Alice
my_dog
this_apple
```

Concept:

```text
DOG
```

Entity:

```text
DOG_INSTANCE_001
```

This distinction is important.

---

### 4.4 Relation

```python
Relation(
    subject_id,
    predicate,
    object_id,
    confidence,
    source_experience_id,
)
```

Examples:

```text
DOG --is_a--> ANIMAL
ALICE --owns--> DOG_001
APPLE_001 --instance_of--> APPLE
```

---

### 4.5 Observation

An observation is an input-derived claim.

```python
Observation(
    id,
    modality,
    raw_reference,
    extracted_structure,
    timestamp,
)
```

---

### 4.6 Experience

```python
Experience(
    id,
    observations,
    actions,
    interpretations,
    updates,
    feedback,
    timestamp,
)
```

---

### 4.7 Belief

```python
Belief(
    proposition,
    confidence,
    evidence_ids,
    status,
)
```

---

## 5. Confidence specification

Confidence must not pretend to be mathematically calibrated probability in v0.1.

Use a bounded score:

```text
0.0 <= confidence <= 1.0
```

Interpretation:

```text
0.00–0.20  very weak
0.20–0.40  weak
0.40–0.60  uncertain
0.60–0.80  supported
0.80–1.00  strongly supported
```

These bands are engineering semantics, not statistical guarantees.

Later, calibration experiments can replace this with better uncertainty estimation.

---

## 6. Inference API

Conceptual API:

```python
result = little.infer(query="Is a dog an animal?")
```

Possible result:

```python
InferenceResult(
    status="SUPPORTED",
    answer=True,
    confidence=0.96,
    evidence=["DOG is_a ANIMAL"],
    trace=[
        "retrieve DOG",
        "retrieve is_a relation",
        "match ANIMAL",
        "return supported",
    ],
)
```

Unknown:

```python
InferenceResult(
    status="UNKNOWN",
    answer=None,
    confidence=0.12,
    evidence=[],
    trace=[
        "retrieve DOG",
        "no DOG -> VEHICLE evidence",
        "insufficient evidence",
    ],
)
```

---

## 7. Learning API

Conceptual API:

```python
result = little.learn("A dog is an animal.")
```

The learning pipeline:

```text
input
 ↓
parse
 ↓
normalize
 ↓
identify entities/concepts
 ↓
construct candidate relations
 ↓
compare with memory
 ↓
classify update
 ↓
commit experience
 ↓
update semantic memory
 ↓
return LearningResult
```

---

## 8. Update types

The learner should classify updates as:

```text
NEW_CONCEPT
NEW_ENTITY
NEW_RELATION
PROPERTY_UPDATE
EVIDENCE_ADDITION
DUPLICATE
CONTRADICTION
REVISION
NO_OP
```

---

## 9. Contradiction policy

Never silently overwrite a conflicting belief.

Example:

Existing:

```text
APPLE color RED
```

New:

```text
APPLE color GREEN
```

The system should determine whether:

1. color is multi-valued;
2. observations refer to different entities;
3. the new statement contradicts the old one;
4. context/time differs.

The storage model should retain evidence until the contradiction can be resolved.

---

## 10. Memory storage

### MVP

SQLite.

Reasons:

- local;
- zero configuration;
- transactional;
- inspectable;
- portable;
- sufficient for early experiments.

### Future

Potential additions:

- vector index;
- graph-specific index;
- columnar experiment storage;
- remote database.

Do not introduce these until needed.

---

## 11. Serialization

Primary interchange:

```text
JSON
```

For internal Python data structures:

- dataclasses or typed models;
- `msgspec` may be evaluated for high-performance serialization once profiling justifies it.

Do not create a dependency on many serialization libraries in v0.1.

---

## 12. Numerical stack

Potential stack:

```text
NumPy
PyTorch
scikit-learn
```

Use PyTorch only when neural computation is needed.

Use NumPy/scikit-learn for small numerical experiments where they are simpler.

---

## 13. Data processing

Polars is an optional candidate for larger experiment datasets. Its core is written in Rust and it exposes both Python and Rust APIs.

For the MVP, ordinary Python/SQLite may be enough.

---

## 14. Rust crate strategy

Potential crates:

```text
pyo3
serde
serde_json
thiserror
anyhow
petgraph
tracing
criterion
```

These are candidates, not mandatory dependencies.

Add a crate only when it solves a concrete problem.

---

## 15. Rust extension boundary

Python:

```python
from little._core import GraphIndex
```

Rust:

```text
GraphIndex
  add_node()
  add_edge()
  neighbors()
  shortest_path()
```

The public Python API should hide implementation details.

This allows Rust internals to change without breaking experiments.

---

## 16. CLI

The initial CLI should support:

```text
little init
little learn "A dog is an animal."
little ask "Is a dog an animal?"
little inspect concept DOG
little memory list
little export
little experiment run 001
```

The CLI is an interface, not the core.

---

## 17. Logging

Use structured logs.

Important events:

```text
observation_created
concept_created
relation_added
belief_updated
inference_started
inference_finished
unknown_returned
contradiction_detected
learning_completed
```

Later, tracing can be unified across Python and Rust.

---

## 18. Configuration

Use a single configuration system.

Configuration should include:

```text
database path
logging level
random seed
model paths
embedding settings
experiment name
```

Do not scatter configuration across source files.

---

## 19. Testing specification

### Unit tests

Test:

- concept creation;
- relation creation;
- retrieval;
- inference;
- unknown detection;
- persistence;
- contradiction handling.

### Integration tests

Example:

```text
learn("A dog is an animal.")
restart
ask("Is a dog an animal?")
```

Expected:

```text
SUPPORTED
```

### Regression tests

Every discovered bug gets a test.

---

## 20. First benchmark

Dataset:

```text
50–500 manually authored facts
```

Categories:

- animals;
- objects;
- people;
- properties;
- categories;
- simple relationships.

Metrics:

```text
fact retention
query accuracy
unknown precision
unknown recall
reasoning depth
learning latency
memory size
```

---

## 21. Core constraint

The first benchmark must be solvable without a pretrained LLM.

If an LLM is later introduced, it should be explicitly identified as an experimental component and compared against the non-LLM architecture.

## Official tooling references used for the stack decision

- uv — modern Python project/dependency/environment workflow.
- Pyrefly — Python type checker and language server.
- Ruff — Python linter and formatter.
- PyO3 — Rust bindings for Python and native Python extension modules.
- Maturin — build/package workflow for Rust-based Python packages.
- SQLite — embedded, serverless, transactional local database.
- Polars — optional later data-processing layer with a Rust core.

The architecture deliberately avoids pinning fast-moving tool versions in design documents. Exact versions belong in `pyproject.toml`, `Cargo.toml`, and lockfiles.
