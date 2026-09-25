# Autonomous Closed-Loop Curiosity & English/Math Mastery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully generalized, non-hardcoded English Conversational Dialogue & Multi-Step Math Reasoning Core, paired with an Autonomous Closed-Loop Curiosity & Self-Study Engine (Milestone 19) that continuously scans 360° Concept Knots for missing radial spokes, deduces hypotheses via hypernym DAG traversal, validates them with DeepSeek Invariant Gates, and resolves contradictions live in SQLite.

**Architecture:** 
- `src/little/language/dialogue.py`: Dynamic multi-turn `DialogueContext` with salience tracking, pronoun resolution (`it`, `they`, `that`), and conversation history.
- `src/little/language/parser.py`: Generalized syntactic parser for coordinate compound clauses, restrictive relative clauses, and indirect queries.
- `src/little/procedural/math_cas.py`: General Symbolic CAS engine supporting arbitrary precision rational fractions, $N \times N$ / $2 \times 2$ linear systems via Cramer's rule, quadratic roots, dynamic unit conversion graphs, and geometry.
- `src/little/procedural/math_story.py`: Generalized word problem parser that maps entities, quantities, and sequential/proportional operations into mathematical graphs producing inspectable `<math_trace>` traces.
- `src/little/active/self_study.py`: Autonomous closed-loop curiosity engine scanning spoke entropy $\max H(\text{Axis}_j)$, discovering missing spokes via analogical hypernym DAG ascent, verifying through DeepSeek 4 gates, and resolving contradictions in SQLite.
- `src/little/main.py`: CLI exposure for `little study`, `little curious`, and conversational chat integration.

**Tech Stack:** Python 3.12, SQLite 3 (WAL mode), pure neuro-symbolic algorithms, zero external LLM dependencies, 100% deterministic execution on pure CPU.

## Global Constraints
- **NO HARDCODING:** Never hardcode specific test sentences, specific story problems, specific numbers, or specific concepts. All parsers, solvers, and curiosity traversals must be fully generic algorithms operating over arbitrary inputs.
- **0% Hallucination:** Maintain Open-World Assumption (OWA) — unobserved facts must strictly resolve to `UNKNOWN` or be verified by invariant gates.
- **Backward Compatibility:** All existing 74 unit and integration tests must pass 100% at all times.

---

### Task 1: Generalized Dialogue Context & Anaphora Resolution Engine

**Files:**
- Create: `src/little/language/dialogue.py`
- Test: `tests/unit/test_dialogue_context.py`

**Interfaces:**
- Consumes: `MemoryStore` concept lookups.
- Produces: `DialogueContext`, `SalientEntity`, `resolve_pronoun(pronoun, context)`, `resolve_anaphora_in_text(text, context)`.

- [ ] **Step 1: Write the failing test for `DialogueContext`**

```python
# tests/unit/test_dialogue_context.py
import pytest
from little.language.dialogue import DialogueContext, SalientEntity

def test_dialogue_salience_and_pronoun_resolution():
    ctx = DialogueContext()
    ctx.record_utterance("user", "The tiger is a ferocious predator.")
    ctx.register_entity("tiger", category="animal", is_animate=True, is_plural=False)
    
    # Resolving 'it' should dynamically bind to the most recent singular concept
    resolved = ctx.resolve_pronoun("it")
    assert resolved == "tiger"

    # Resolving 'they' when plural entity is introduced
    ctx.register_entity("lions", category="animal", is_animate=True, is_plural=True)
    assert ctx.resolve_pronoun("they") == "lions"
    assert ctx.resolve_pronoun("it") == "tiger"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_dialogue_context.py
```

- [ ] **Step 3: Implement `DialogueContext` in `src/little/language/dialogue.py`**

Implement `SalientEntity` and `DialogueContext` with:
- Animacy, plurality, recency stack, semantic category filtering.
- Generic `resolve_pronoun(pronoun)` handling `it`, `they`, `he`, `she`, `this`, `that`.
- Generic `resolve_anaphora(text)` substituting pronouns with salient nouns based on syntax position.

- [ ] **Step 4: Run tests and ensure they pass**

```bash
uv run pytest tests/unit/test_dialogue_context.py
```

- [ ] **Step 5: Commit changes**

```bash
git add src/little/language/dialogue.py tests/unit/test_dialogue_context.py
git commit -m "feat(dialogue): implement generalized dialogue context and pronoun resolution"
```

---

### Task 2: Generalized Complex English Parsing & Relative Clauses

**Files:**
- Modify: `src/little/language/parser.py`
- Test: `tests/unit/test_complex_english_parser.py`

**Interfaces:**
- Consumes: `DialogueContext` from Task 1.
- Produces: `SimpleParser.parse_statement` and `SimpleParser.parse_question` supporting compound coordinate clauses, restrictive relative clauses, and indirect questions.

- [ ] **Step 1: Write failing tests for complex sentences**

```python
# tests/unit/test_complex_english_parser.py
import pytest
from little.language.parser import SimpleParser
from little.language.dialogue import DialogueContext

def test_parse_coordinate_compound_clause():
    ctx = DialogueContext()
    text = "A falcon is a bird and it hunts rodents."
    triples = SimpleParser.parse_complex_statement(text, dialogue_context=ctx)
    assert len(triples) >= 2
    assert any(t.subject == "falcon" and t.predicate == "is_a" and t.object_ == "bird" for t in triples)
    assert any(t.subject == "falcon" and t.predicate == "hunts" and t.object_ == "rodents" for t in triples)

def test_parse_restrictive_relative_clause():
    text = "The animal that has gills and swims in water is a fish."
    triples = SimpleParser.parse_complex_statement(text)
    assert any(t.predicate == "is_a" and t.object_ == "fish" for t in triples)
```

- [ ] **Step 2: Run test to verify failure**

```bash
uv run pytest tests/unit/test_complex_english_parser.py
```

- [ ] **Step 3: Implement complex sentence decomposition in `src/little/language/parser.py`**

- Add `parse_complex_statement(text, dialogue_context)` to handle arbitrary coordinate conjunctions with pronoun back-references.
- Add restrictive relative clause matcher decomposing defining traits and linking to the target concept.
- Integrate `DialogueContext` into `LearningEngine` so conversation turns automatically update the dialogue state.

- [ ] **Step 4: Run tests and ensure they pass**

```bash
uv run pytest tests/unit/test_complex_english_parser.py
```

- [ ] **Step 5: Commit changes**

```bash
git add src/little/language/parser.py tests/unit/test_complex_english_parser.py
git commit -m "feat(parser): add generalized complex sentence and relative clause parsing"
```

---

### Task 3: Generalized Symbolic CAS Engine (`math_cas.py`)

**Files:**
- Create: `src/little/procedural/math_cas.py`
- Test: `tests/unit/test_math_cas.py`

**Interfaces:**
- Produces: `SymbolicCAS.evaluate_rational_expression(expr)`, `SymbolicCAS.solve_linear_system(eqs)`, `SymbolicCAS.solve_quadratic(a, b, c)`, `SymbolicCAS.convert_units(value, from_u, to_u)`, `SymbolicCAS.compute_geometry(shape, **kwargs)`.

- [ ] **Step 1: Write failing tests for Symbolic CAS**

```python
# tests/unit/test_math_cas.py
import pytest
from fractions import Fraction
from little.procedural/math_cas import SymbolicCAS

def test_rational_fractions_and_precision():
    cas = SymbolicCAS()
    res = cas.evaluate_rational("3/4 + 5/6 - 1/3")
    assert res == Fraction(5, 4)

def test_solve_linear_system():
    cas = SymbolicCAS()
    # 2x + 3y = 13, x - y = 4 => x=5, y=1
    sol = cas.solve_linear_system(["2x + 3y = 13", "x - y = 4"])
    assert sol == {"x": 5.0, "y": 1.0}

def test_unit_conversion_graph():
    cas = SymbolicCAS()
    # 100 km/h to m/s = 27.7778
    res = cas.convert_units(100.0, from_unit="km/h", to_unit="m/s")
    assert round(res, 3) == 27.778
```

- [ ] **Step 2: Run test to verify failure**

```bash
uv run pytest tests/unit/test_math_cas.py
```

- [ ] **Step 3: Implement `SymbolicCAS` in `src/little/procedural/math_cas.py`**

- Implement exact rational arithmetic with `fractions.Fraction`.
- Implement Cramer's rule / Gaussian elimination for linear equations without hardcoded variable names.
- Implement closed-form quadratic formula with real/complex detection.
- Implement topological unit conversion registry (length, mass, time, speed, temperature, area, volume).
- Implement geometric area/volume formulas.

- [ ] **Step 4: Run tests and ensure they pass**

```bash
uv run pytest tests/unit/test_math_cas.py
```

- [ ] **Step 5: Commit changes**

```bash
git add src/little/procedural/math_cas.py tests/unit/test_math_cas.py
git commit -m "feat(math): implement generalized Symbolic CAS engine"
```

---

### Task 4: Generalized Math Word Problem Solver (`math_story.py`)

**Files:**
- Create: `src/little/procedural/math_story.py`
- Test: `tests/unit/test_math_story.py`

**Interfaces:**
- Consumes: `SymbolicCAS` from Task 3.
- Produces: `MathStorySolver.solve_story_problem(text) -> MathStoryResult(answer, trace, steps)`.

- [ ] **Step 1: Write failing tests for story problems**

```python
# tests/unit/test_math_story.py
import pytest
from little.procedural/math_story import MathStorySolver

def test_sequential_change_word_problem():
    solver = MathStorySolver()
    # Any names, any quantities, any nouns
    text = "David has 28 books. He donates 9 books to a library and buys 14 more. How many books does David have?"
    res = solver.solve_story_problem(text)
    assert res.success
    assert res.numeric_answer == 33
    assert "<math_trace>" in res.trace

def test_proportional_scaling_word_problem():
    solver = MathStorySolver()
    text = "Sam has 16 pencils. Maya has half as many pencils as Sam. Leo has 7 more pencils than Maya. How many pencils does Leo have?"
    res = solver.solve_story_problem(text)
    assert res.success
    assert res.numeric_answer == 15
```

- [ ] **Step 2: Run test to verify failure**

```bash
uv run pytest tests/unit/test_math_story.py
```

- [ ] **Step 3: Implement `MathStorySolver` in `src/little/procedural/math_story.py`**

- Generic entity and number extractor (identifies entities, assigned initial quantities, units/nouns).
- Generic action transition parser (`buys`, `gives`, `donates`, `receives`, `loses`, `eats`, `finds`).
- Proportional constraint extractor (`twice as many`, `half as many`, `N times as many`, `fraction of`).
- Formulates equation graph and computes exact numerical answer with intermediate step explanations in `<math_trace>`.

- [ ] **Step 4: Run tests and ensure they pass**

```bash
uv run pytest tests/unit/test_math_story.py
```

- [ ] **Step 5: Commit changes**

```bash
git add src/little/procedural/math_story.py tests/unit/test_math_story.py
git commit -m "feat(math): implement generalized math word problem solver"
```

---

### Task 5: Autonomous 360° Closed-Loop Curiosity & Self-Study Engine (Milestone 19)

**Files:**
- Create: `src/little/active/self_study.py`
- Test: `tests/unit/test_self_study_loop.py`

**Interfaces:**
- Consumes: `MemoryStore`, `ConceptKnot`, `DeepSeekInvariantVerifier`.
- Produces: `ClosedLoopCuriosityEngine`, `compute_spoke_entropy(knot)`, `self_study_concept(concept_id)`, `run_study_cycle(steps)`.

- [ ] **Step 1: Write failing tests for Curiosity & Self-Study**

```python
# tests/unit/test_self_study_loop.py
import pytest
from little.memory.store import MemoryStore
from little.active.self_study import ClosedLoopCuriosityEngine
from little.inference.invariant_gates import DeepSeekInvariantVerifier

def test_spoke_entropy_and_autonomous_self_study():
    memory = MemoryStore(":memory:", seed_ontology=True)
    verifier = DeepSeekInvariantVerifier(memory)
    curiosity = ClosedLoopCuriosityEngine(memory, verifier)

    # Teach partial concept: Titanium is_a Metal, but has 0 physical properties, 0 parts, 0 procedural skills
    memory.create_concept("titanium")
    memory.add_relation("titanium", "is_a", "metal")

    # Spoke entropy for titanium should be high due to missing spokes
    entropy_before = curiosity.compute_spoke_entropy("titanium")
    assert entropy_before > 2.0

    # Run closed-loop self study
    results = curiosity.self_study_concept("titanium")
    assert len(results) > 0

    # Hypotheses inherited from Metal (e.g. conductive, solid, ductile) should be validated through invariant gates
    # and persisted into memory
    entropy_after = curiosity.compute_spoke_entropy("titanium")
    assert entropy_after < entropy_before
```

- [ ] **Step 2: Run test to verify failure**

```bash
uv run pytest tests/unit/test_self_study_loop.py
```

- [ ] **Step 3: Implement `ClosedLoopCuriosityEngine` in `src/little/active/self_study.py`**

- Generic spoke entropy computation across the 6 axes:
  $$H(K) = \sum_{j=1}^6 w_j \Psi_j(K)$$
- Generic hypernym DAG ascent: inspects parent knots dynamically, gathers candidate property/part/skill spokes.
- Dynamic hypothesis validation through `DeepSeekInvariantVerifier` ($\mathcal{I}_{\text{DAG}}, \mathcal{I}_{\text{mutex}}, \mathcal{I}_{\text{sort}}, \mathcal{I}_{\text{ground}}$).
- Automatic contradiction resolution via NARS evidence weights without user input.
- SQLite live update and entropy logging.

- [ ] **Step 4: Run tests and ensure they pass**

```bash
uv run pytest tests/unit/test_self_study_loop.py
```

- [ ] **Step 5: Commit changes**

```bash
git add src/little/active/self_study.py tests/unit/test_self_study_loop.py
git commit -m "feat(curiosity): implement autonomous closed-loop self-study engine"
```

---

### Task 6: CLI & Conversational Integration

**Files:**
- Modify: `src/little/main.py`
- Test: `tests/integration/test_curiosity_chat_integration.py`

**Interfaces:**
- Consumes: `DialogueContext`, `MathStorySolver`, `ClosedLoopCuriosityEngine`.
- Produces: CLI commands `little study`, `little curious`, updated `little chat` and `little ask`.

- [ ] **Step 1: Write integration tests for interactive chat and self-study**

```python
# tests/integration/test_curiosity_chat_integration.py
import pytest
from little.memory.store import MemoryStore
from little.language.parser import LearningEngine

def test_chat_dialogue_math_and_curiosity_flow(tmp_path):
    db_file = tmp_path / "chat_test.db"
    store = MemoryStore(db_file, seed_ontology=True)
    engine = LearningEngine(store)

    # 1. Multi-turn pronoun continuity
    engine.learn("The falcon is a bird.")
    res = engine.ask("What is it?")
    assert "falcon" in res.verbalize().lower() or "bird" in res.verbalize().lower()

    # 2. Math story problem in chat
    res_math = engine.ask("Bob has 10 apples and gives 3 to Alice. How many does Bob have?")
    assert res_math.answer == 7 or "7" in res_math.verbalize()
```

- [ ] **Step 2: Run test to verify failure**

```bash
uv run pytest tests/integration/test_curiosity_chat_integration.py
```

- [ ] **Step 3: Update `src/little/main.py` and `LearningEngine`**

- Connect `DialogueContext` to `LearningEngine` across consecutive calls.
- Dispatch mathematical story questions to `MathStorySolver` within `LearningEngine.ask()`.
- Add subcommands `little study` (with `--concept`, `--steps`, `--verbose`) and `little curious`.
- Add `--auto-study` option to `little chat`.

- [ ] **Step 4: Run all unit and integration tests**

```bash
uv run pytest
```

- [ ] **Step 5: Commit changes**

```bash
git add src/little/main.py tests/integration/test_curiosity_chat_integration.py
git commit -m "feat(cli): integrate dialogue context, math story solver, and self-study CLI"
```

---

### Task 7: Full Regression & Milestone Verification

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest -v
```

- [ ] **Step 2: Re-run empirical benchmark suite to ensure no regressions**

```bash
uv run python experiments/005_slm_comparison_benchmark/experiment.py
```

- [ ] **Step 3: Update `todo.md` marking Milestone 19 complete**

- [ ] **Step 4: Final commit and summary report to user**
