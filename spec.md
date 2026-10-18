# LITTLE — Technical Specification

**Version:** 0.2  
**Status:** Implemented technical specification

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

`ctypes` over a zero-dependency C-ABI `cdylib` (`rust/little_core` $\to$ `liblittle_core.so`, built via `little_build_backend.py` wrapping `uv_build`) is the bridge between Python and Rust, with an automatic pure-Python fallback in `src/little/rust/engine.py`.

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

`rust/little_core` is implemented as a zero-external-crate Rust `cdylib` (`edition = "2021"`, `crate-type = ["cdylib", "rlib"]`) so `cargo build --release` compiles offline with zero transitive crate overhead.

---

## 15. Rust extension boundary

Python (`src/little/rust/engine.py`):

```python
from little.rust.engine import RustCoreEngine
```

Rust (`rust/little_core/src/lib.rs` & `turbo.rs`):

```text
RustCoreEngine
  bidirectional_bfs()
  verify_dag_invariant()
  verify_mutex_invariant()
  fwht_transform()
  turbo_quant_search()
```

The public Python wrapper (`RustCoreEngine`) hides FFI pointer management and automatically falls back to pure Python if the compiled `cdylib` is unavailable.

---

## 16. CLI

The `little` CLI (`src/little/main.py`) supports:

```text
little init
little learn "A dog is an animal."
little ask "Is a dog an animal?" [-v] [--thinking]
little inspect concept DOG
little memory list
little memory prune [--dry-run]
little skills list
little chat [--thinking] [--auto-study]
little interact [--thinking] [--auto-study]
little curious [--top 10] [--concept <name>]
little study [--steps 5] [--target <name>] [-v]
little export [-o <path>]
little import [--dataset commonsense|world|large] [--file <path>]
little tool list
little tool exec <name> [--args '{}']
little status [--json]
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
- contradiction handling;
- dirty-only paged memory invalidation & multi-gram voting;
- continual plasticity EWLS trend estimation & Trunk/Leaf split;
- 5-brake speculative growth gate & survival-window pruning;
- PonderNet cumulative halting & held-out canary verification.

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

- uv (`uv_build` + `little_build_backend.py`) — modern Python project/dependency/environment workflow and PEP 517 build hook.
- Pyrefly — Python type checker and language server.
- Ruff — Python linter and formatter.
- Rust `cdylib` + Python `ctypes` — zero-dependency native FFI bridge with pure-Python fallback.
- SQLite — embedded, serverless, transactional local database.

The architecture deliberately avoids pinning fast-moving tool versions in design documents. Exact versions belong in `pyproject.toml`, `Cargo.toml`, and lockfiles.

---

## 22. Continual Plasticity, Paging, Growth Brakes & Adaptive Halting Specification

### 22.1 Dirty-Only Paged Concept Store (`PagedConceptStore`)
- `MemoryStore.concept_mutation_epoch: int` increments monotonically only on mutations to `concepts` or `relations`.
- `MemoryStore.dirty_concepts_since(epoch: int) -> set[str] | None` returns exact mutated concept keys or `None` if a full cache reset is required.
- `PagedConceptStore.vote_from_tokens(tokens: list[str])` casts weighted activation votes for 1-grams (`1.0x`), 2-grams (`1.5x`), and 3-grams (`2.0x`), pinning Resident concepts (`slot_reuses`).

### 22.2 Evidence-Governed Continual Plasticity (`PlasticityController`)
- `EWLSFit` tracks exponentially weighted least squares with slow ($\lambda=0.97$) and fast ($\lambda_{\text{fast}}=0.85$) forgetting factors.
- Scale-free effect size: $e = -\text{slope} / \sigma$.
- Spike holdback filter suppresses single-step spikes (`loss - mean > 2 * sigma`) unless corroborated on the next observation.
- Trunk concepts (`is_trunk_concept`) receive `trunk_lr_mult = 0.1`; leaf concepts start at `1.0` and decay toward `scale` with half-life `rate_half = 50.0`.

### 22.3 5-Brake Speculative Growth Gate & Survival-Window Pruning (`LatentKnotInducer`)
- `evaluate_growth_brakes()` enforces `allowed = room and used and earning and fits and honest`.
- `prune_unaddressed_latent_knots(survival_window_cycles=3)` removes `LATENT_*` concepts that remain unaddressed (`address_hits == 0` and zero non-taxonomic outgoing relations) past the survival window.

### 22.4 Load-Balanced Curiosity & Held-Out Canary (`AutonomousSelfStudyEngine`)
- `effective_study_priority(concept, raw_entropy)` penalizes concepts whose recent study share exceeds uniform share $1/N$.
- `evaluate_heldout_canary()` verifies `I_DAG` and `I_MUTEX` invariants across top-confidence relations during `consolidate_sleep_cycle()`.

### 22.5 PonderNet Adaptive Halting (`DualSpeedInfillingEngine`)
- Tracks conditional halting hazard $\lambda_d$, cumulative halting probability $P(\text{halt} \le d) = 1 - \prod_{k=1}^{d} (1 - \lambda_k)$, and `ponder_cost = ponder_beta * depth`, halting early when `halt_freeze=True` and $P(\text{halt} \le d) \ge \text{halt\_threshold}$.

