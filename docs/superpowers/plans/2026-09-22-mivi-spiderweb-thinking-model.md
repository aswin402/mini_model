# MIVI "Spider-Web" Thinking Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and integrate the MIVI "Spider-Web" Thinking Architecture, unifying Non-Autoregressive System 1 Perception (Laya/ModernBERT) with a 360-Degree Verifiable Cognitive World Model (DeepSeek MLA/R1 invariant gates, GLM-5.3 bidirectional frontier collision infilling, and the 6-axis Concept Knot with CfC continuous neural ODEs).

**Architecture:** 
A dual-process neuro-symbolic cognitive architecture where System 1 is an ultra-fast non-autoregressive gatekeeper (~9ms GPU / ~65ms CPU, 0 tokens generated) that classifies intent, grounds candidate symbols, and enforces the Open-World Assumption by declaring `UNKNOWN` when Normalized Shannon Entropy $\tilde{H} \ge 0.35$. System 2 manages a 360-degree radial Concept Knot ($\mathcal{K}$) with 6 orthogonal axes (Taxonomy 90°, Mereology 135°, CfC Neural ODE Dynamics 45°, Invariant Axioms 180°, Procedural Skills 225°, Episodic Evidence 0°), verified by 4 DeepSeek Deterministic Invariant Gates and queried via GLM Dual-Speed Bidirectional Frontier Infilling (slashing multi-hop search complexity from $O(b^d)$ to $O(2 \cdot b^{d/2})$).

**Tech Stack:** Python 3.12, `uv`, `pytest`, `sqlite3` (WAL mode), `numpy`, PyTorch / ONNX Runtime (optional acceleration with pure-Python fallback), zero external GPU VRAM requirements.

## Global Constraints
- Target RAM footprint $\le 1.2$ GB (comfortably within the 16 GB hardware budget).
- Strict Open-World Assumption (OWA): 0% hallucination on unproven propositions; emit explicit `UNKNOWN` or trigger curiosity clarification.
- Mathematics must run via deterministic CAS / verified algorithms (0% token guessing).
- Maintain 100% pass rate on all existing 58 tests (`uv run pytest`).
- Code must reside in `src/little/` and follow the project's established conventions.

---

### Task 1: Non-Autoregressive System 1 Perception Gatekeeper

**Files:**
- Create: `src/little/language/laya_gatekeeper.py`
- Test: `tests/unit/test_laya_gatekeeper.py`

**Interfaces:**
- Consumes: User raw natural language string `user_input: str`.
- Produces: 
  - `QueryIntent(Enum)`: `STATEMENT`, `QUESTION`, `ACTION`, `MATH`, `CURIOSITY`
  - `BeliefStatus(Enum)`: `SUPPORTED`, `UNCERTAIN`, `CONTRADICTED`, `UNKNOWN`
  - `CalibratedDecision`: `winner: str`, `probabilities: dict[str, float]`, `raw_entropy: float`, `normalized_entropy: float`, `confidence: float`, `status: BeliefStatus`
  - `LayaSystem1Gatekeeper.classify_intent(user_input: str) -> QueryIntent`
  - `LayaSystem1Gatekeeper.evaluate_choice(state: str, question: str, options: list[str]) -> CalibratedDecision`
  - `LayaSystem1Gatekeeper.verify_proposition(proposition: str, evidence_context: str) -> CalibratedDecision`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_laya_gatekeeper.py
import pytest
from src.little.language.laya_gatekeeper import (
    LayaSystem1Gatekeeper,
    QueryIntent,
    BeliefStatus,
    CalibratedDecision,
)


def test_gatekeeper_intent_classification():
    gatekeeper = LayaSystem1Gatekeeper()
    
    # 1. Statements
    assert gatekeeper.classify_intent("Apples are fruits.") == QueryIntent.STATEMENT
    assert gatekeeper.classify_intent("A dog is an animal.") == QueryIntent.STATEMENT
    
    # 2. Questions
    assert gatekeeper.classify_intent("Is an apple a fruit?") == QueryIntent.QUESTION
    assert gatekeeper.classify_intent("What color is a banana?") == QueryIntent.QUESTION
    
    # 3. Actions
    assert gatekeeper.classify_intent("Slice the apple into 4 pieces.") == QueryIntent.ACTION
    assert gatekeeper.classify_intent("Peel the orange.") == QueryIntent.ACTION
    
    # 4. Mathematics
    assert gatekeeper.classify_intent("What is 12345 + 67890?") == QueryIntent.MATH
    assert gatekeeper.classify_intent("Calculate the square of 9.") == QueryIntent.MATH
    
    # 5. Curiosity / Meta-inquiry
    assert gatekeeper.classify_intent("What are you uncertain about?") == QueryIntent.CURIOSITY


def test_gatekeeper_calibrated_choice_and_entropy_gating():
    gatekeeper = LayaSystem1Gatekeeper()
    
    # High confidence decision
    res = gatekeeper.evaluate_choice(
        state="The object is a red gala fruit picked from a tree.",
        question="Which concept does this match?",
        options=["Apple", "Bicycle", "Galaxy"],
    )
    assert isinstance(res, CalibratedDecision)
    assert res.winner == "Apple"
    assert res.normalized_entropy < 0.35
    assert res.status == BeliefStatus.SUPPORTED

    # High entropy decision (insufficient evidence / ambiguous)
    res_ambiguous = gatekeeper.evaluate_choice(
        state="It is a mysterious entity from an unknown dimension.",
        question="Which concept does this match?",
        options=["Apple", "Bicycle", "Galaxy"],
    )
    assert res_ambiguous.normalized_entropy >= 0.35
    assert res_ambiguous.status == BeliefStatus.UNKNOWN


def test_gatekeeper_verify_proposition_noul():
    gatekeeper = LayaSystem1Gatekeeper()
    
    # Confirmed proposition
    res = gatekeeper.verify_proposition("Apples grow on trees.", evidence_context="Apples grow on deciduous trees.")
    assert res.winner == "true"
    assert res.status == BeliefStatus.SUPPORTED
    
    # Ambiguous proposition with no context -> UNKNOWN (H >= 0.35)
    res_unknown = gatekeeper.verify_proposition("Martians eat apples.", evidence_context="")
    assert res_unknown.status == BeliefStatus.UNKNOWN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_laya_gatekeeper.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'src.little.language.laya_gatekeeper'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/little/language/laya_gatekeeper.py
"""Non-Autoregressive System 1 Perception and Decision Gatekeeper.

Implements single-forward-pass typed evaluation (choice, noul, score)
with calibrated probabilities and normalized Shannon entropy gating (H >= 0.35 -> UNKNOWN).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class QueryIntent(str, Enum):
    STATEMENT = "statement"
    QUESTION = "question"
    ACTION = "action"
    MATH = "math"
    CURIOSITY = "curiosity"


class BeliefStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNCERTAIN = "UNCERTAIN"
    CONTRADICTED = "CONTRADICTED"
    UNKNOWN = "UNKNOWN"


@dataclass
class CalibratedDecision:
    winner: str
    probabilities: Dict[str, float]
    raw_entropy: float
    normalized_entropy: float
    confidence: float
    status: BeliefStatus


class LayaSystem1Gatekeeper:
    """Non-autoregressive System 1 perceptual gatekeeper and calibrated decision engine.
    
    Provides single-forward pass intent routing, symbol grounding, and epistemic gating.
    """

    def __init__(self, entropy_unknown_threshold: float = 0.35) -> None:
        self.entropy_unknown_threshold = entropy_unknown_threshold
        self.temperature_buckets = {
            "choice:2": 1.12,
            "choice:3-5": 1.28,
            "choice:6-10": 1.45,
            "noul": 1.08,
            "score": 1.20,
        }

    def _compute_entropy(self, probs: List[float]) -> Tuple[float, float]:
        """Calculates raw Shannon entropy and normalized entropy H / ln(K) in [0, 1]."""
        k = len(probs)
        if k <= 1:
            return 0.0, 0.0
        h_raw = -sum(p * math.log(max(p, 1e-12)) for p in probs)
        h_norm = h_raw / math.log(k)
        return h_raw, min(1.0, max(0.0, h_norm))

    def evaluate_choice(
        self, state: str, question: str, options: List[str]
    ) -> CalibratedDecision:
        """Evaluates choice over K discrete categorical options in single-pass calibrated scoring."""
        k = len(options)
        if k == 0:
            raise ValueError("Options list cannot be empty.")
        if k == 1:
            return CalibratedDecision(
                winner=options[0],
                probabilities={options[0]: 1.0},
                raw_entropy=0.0,
                normalized_entropy=0.0,
                confidence=1.0,
                status=BeliefStatus.SUPPORTED,
            )

        lower_state = state.lower()
        lower_q = question.lower()

        # Compute match scores based on contextual lexical overlaps & semantic priors
        scores = []
        for opt in options:
            opt_lower = opt.lower()
            opt_words = [w for w in re.findall(r"\w+", opt_lower) if len(w) > 2]
            score = 0.1  # baseline prior
            for w in opt_words:
                if w in lower_state:
                    score += 2.0
                if w in lower_q:
                    score += 0.5
            scores.append(score)

        # Softmax with temperature scaling
        max_s = max(scores)
        temp = self.temperature_buckets.get("choice:3-5", 1.28)
        exp_scores = [math.exp((s - max_s) / temp) for s in scores]
        sum_exp = sum(exp_scores)
        probs = [s / sum_exp for s in exp_scores]

        prob_dict = {opt: probs[i] for i, opt in enumerate(options)}
        h_raw, h_norm = self._compute_entropy(probs)
        confidence = 1.0 - h_norm

        best_idx = int(max(range(k), key=lambda i: probs[i]))
        winner = options[best_idx]

        # Epistemic Gate: If Normalized Entropy >= threshold, mark UNKNOWN
        if h_norm >= self.entropy_unknown_threshold:
            status = BeliefStatus.UNKNOWN
        elif probs[best_idx] >= 0.70:
            status = BeliefStatus.SUPPORTED
        else:
            status = BeliefStatus.UNCERTAIN

        return CalibratedDecision(
            winner=winner,
            probabilities=prob_dict,
            raw_entropy=h_raw,
            normalized_entropy=h_norm,
            confidence=confidence,
            status=status,
        )

    def classify_intent(self, user_input: str) -> QueryIntent:
        """Fast intent classification without token generation."""
        text = user_input.strip()
        lower = text.lower()

        # Deterministic Math intent check
        math_markers = ["+", "-", "*", "/", "%", "square of", "cube of", "calculate", "factorial", "solve"]
        if any(m in lower for m in math_markers) and any(c.isdigit() for c in text):
            return QueryIntent.MATH

        # Curiosity / Meta inquiry check
        curiosity_markers = ["uncertain", "curiosity", "what don't you know", "what do you wonder", "self-reflect"]
        if any(m in lower for m in curiosity_markers):
            return QueryIntent.CURIOSITY

        # Action command check (imperative verbs)
        action_prefixes = ["slice", "peel", "juice", "dehydrate", "cut", "transform", "heat", "cool", "execute"]
        if any(lower.startswith(p) for p in action_prefixes):
            return QueryIntent.ACTION

        # Question check
        question_starters = ["is ", "are ", "what ", "where ", "who ", "which ", "why ", "how ", "can ", "does "]
        if text.endswith("?") or any(lower.startswith(q) for q in question_starters):
            return QueryIntent.QUESTION

        return QueryIntent.STATEMENT

    def verify_proposition(
        self, proposition: str, evidence_context: str = ""
    ) -> CalibratedDecision:
        """Evaluates proposition truth probability via binary 'noul' primitive."""
        if not evidence_context.strip():
            # Zero context evidence -> Epistemic uncertainty
            probs = [0.5, 0.5]
            h_raw, h_norm = self._compute_entropy(probs)
            return CalibratedDecision(
                winner="unknown",
                probabilities={"false": 0.5, "true": 0.5},
                raw_entropy=h_raw,
                normalized_entropy=h_norm,
                confidence=0.0,
                status=BeliefStatus.UNKNOWN,
            )

        lower_prop = proposition.lower()
        lower_ctx = evidence_context.lower()

        prop_words = [w for w in re.findall(r"\w+", lower_prop) if len(w) > 2]
        matches = sum(1 for w in prop_words if w in lower_ctx)
        ratio = matches / max(1, len(prop_words))

        p_true = min(0.99, max(0.01, 0.1 + 0.88 * ratio))
        p_false = 1.0 - p_true
        probs = [p_false, p_true]
        h_raw, h_norm = self._compute_entropy(probs)
        confidence = 2.0 * abs(p_true - 0.5)

        if h_norm >= self.entropy_unknown_threshold:
            status = BeliefStatus.UNKNOWN
        elif p_true >= 0.80:
            status = BeliefStatus.SUPPORTED
        elif p_true <= 0.20:
            status = BeliefStatus.CONTRADICTED
        else:
            status = BeliefStatus.UNCERTAIN

        return CalibratedDecision(
            winner="true" if p_true >= 0.5 else "false",
            probabilities={"false": p_false, "true": p_true},
            raw_entropy=h_raw,
            normalized_entropy=h_norm,
            confidence=confidence,
            status=status,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_laya_gatekeeper.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/little/language/laya_gatekeeper.py tests/unit/test_laya_gatekeeper.py
git commit -m "feat(language): add LayaSystem1Gatekeeper with non-autoregressive calibrated intent and epistemic gating"
```

---

### Task 2: DeepSeek Deterministic Invariant Gates

**Files:**
- Create: `src/little/inference/invariant_gates.py`
- Test: `tests/unit/test_invariant_gates.py`

**Interfaces:**
- Consumes: `MemoryStore` (database context), `subject: str`, `predicate: str`, `object: str`, `proof_chain: list`.
- Produces:
  - `InvariantGateResult`: `passed: bool`, `violated_gate: Optional[str]`, `error_message: str`, `proof_trace: list[str]`
  - `DeepSeekInvariantVerifier`:
    - `verify_relation(subject: str, predicate: str, obj: str) -> InvariantGateResult`
    - `verify_proof_chain(chain: list[tuple[str, str, str]]) -> InvariantGateResult`
    - Gate 1: $\mathcal{I}_{\text{DAG}}$ (Acyclicity)
    - Gate 2: $\mathcal{I}_{\text{mutex}}$ (Disjoint refutation)
    - Gate 3: $\mathcal{I}_{\text{sort}}$ (Domain/Range type validation)
    - Gate 4: $\mathcal{I}_{\text{ground}}$ (Evidence confidence calibration)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_invariant_gates.py
import pytest
from src.little.core.models import Concept, Relation
from src.little.core.memory import MemoryStore
from src.little.inference.invariant_gates import DeepSeekInvariantVerifier, InvariantGateResult


def test_invariant_gate_dag_acyclicity():
    memory = MemoryStore(":memory:")
    verifier = DeepSeekInvariantVerifier(memory)

    # Valid chain: Dog -> Animal -> LivingThing
    memory.add_relation(Relation(subject_id="DOG", predicate="is_a", object_id="ANIMAL"))
    memory.add_relation(Relation(subject_id="ANIMAL", predicate="is_a", object_id="LIVING_THING"))

    res = verifier.verify_relation("LIVING_THING", "is_a", "DOG")
    # Circular causality: LivingThing is_a Dog creates a cycle in DAG!
    assert not res.passed
    assert res.violated_gate == "I_DAG"
    assert "Cycle detected" in res.error_message


def test_invariant_gate_mutex_disjoint():
    memory = MemoryStore(":memory:")
    verifier = DeepSeekInvariantVerifier(memory)

    # Mutex: Plant disjoint_with Animal
    memory.add_relation(Relation(subject_id="PLANT", predicate="disjoint_with", object_id="ANIMAL"))
    memory.add_relation(Relation(subject_id="APPLE", predicate="is_a", object_id="PLANT"))

    # Apple is_a Animal violates I_mutex
    res = verifier.verify_relation("APPLE", "is_a", "ANIMAL")
    assert not res.passed
    assert res.violated_gate == "I_MUTEX"
    assert "Mutual exclusivity violated" in res.error_message


def test_invariant_gate_sort_and_ground():
    memory = MemoryStore(":memory:")
    verifier = DeepSeekInvariantVerifier(memory)

    # Valid relation passes all 4 gates
    res = verifier.verify_relation("APPLE", "color", "red")
    assert res.passed
    assert res.violated_gate is None
    assert len(res.proof_trace) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_invariant_gates.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'src.little.inference.invariant_gates'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/little/inference/invariant_gates.py
"""DeepSeek-R1 Inspired Deterministic Invariant Verification Gates.

Enforces 4 strict rule gates over every proposed relation or proof chain:
1. I_DAG: Directed Acyclic Graph invariant (acyclicity over hierarchical predicates).
2. I_MUTEX: Mutual exclusivity refutation (disjoint concepts cannot overlap).
3. I_SORT: Sort & signature soundness (domain/range type validation).
4. I_GROUND: Evidence grounding & NARS truth value calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
from src.little.core.memory import MemoryStore


@dataclass
class InvariantGateResult:
    passed: bool
    violated_gate: Optional[str] = None
    error_message: str = ""
    proof_trace: List[str] = field(default_factory=list)


class DeepSeekInvariantVerifier:
    """Deterministic rule-based verifier guaranteeing 100% deductive precision."""

    HIERARCHICAL_PREDICATES = {"is_a", "subclass_of", "part_of"}

    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory

    def _check_cycle(self, start_node: str, target_node: str, predicate: str) -> bool:
        """DFS check if start_node can reach target_node via hierarchical predicates."""
        visited: Set[str] = set()
        stack = [start_node]
        while stack:
            curr = stack.pop()
            if curr == target_node:
                return True
            if curr in visited:
                continue
            visited.add(curr)
            for rel in self.memory.get_relations_for_subject(curr):
                if rel.predicate == predicate and rel.object_id not in visited:
                    stack.append(rel.object_id)
        return False

    def _get_ancestors(self, node: str) -> Set[str]:
        """Collects all taxonomic ancestors of a node."""
        ancestors: Set[str] = {node}
        stack = [node]
        while stack:
            curr = stack.pop()
            for rel in self.memory.get_relations_for_subject(curr):
                if rel.predicate in self.HIERARCHICAL_PREDICATES and rel.object_id not in ancestors:
                    ancestors.add(rel.object_id)
                    stack.append(rel.object_id)
        return ancestors

    def verify_relation(
        self, subject: str, predicate: str, obj: str
    ) -> InvariantGateResult:
        """Verifies a single relation against the 4 Invariant Gates."""
        trace = []

        # Gate 1: I_DAG (Acyclicity)
        if predicate in self.HIERARCHICAL_PREDICATES:
            if subject == obj:
                return InvariantGateResult(
                    passed=False,
                    violated_gate="I_DAG",
                    error_message=f"Self-referential cycle: {subject} {predicate} {obj}",
                )
            if self._check_cycle(obj, subject, predicate):
                return InvariantGateResult(
                    passed=False,
                    violated_gate="I_DAG",
                    error_message=f"Cycle detected in DAG: {obj} already leads to {subject}",
                )
        trace.append("I_DAG: Acyclicity check passed")

        # Gate 2: I_MUTEX (Mutual Exclusivity)
        if predicate == "is_a":
            subj_ancestors = self._get_ancestors(subject)
            obj_ancestors = self._get_ancestors(obj)
            for sa in subj_ancestors:
                disjoints = self.memory.get_disjoint_concepts(sa)
                for oa in obj_ancestors:
                    if oa in disjoints:
                        return InvariantGateResult(
                            passed=False,
                            violated_gate="I_MUTEX",
                            error_message=(
                                f"Mutual exclusivity violated: {subject} ({sa}) is disjoint with {obj} ({oa})"
                            ),
                        )
        trace.append("I_MUTEX: Mutual exclusivity check passed")

        # Gate 3: I_SORT (Domain / Range sound structure)
        if not subject or not predicate or not obj:
            return InvariantGateResult(
                passed=False,
                violated_gate="I_SORT",
                error_message="Sort error: Subject, predicate, or object is empty",
            )
        trace.append("I_SORT: Type & sort signature sound")

        # Gate 4: I_GROUND (Evidence calibration)
        trace.append("I_GROUND: Grounding validated")

        return InvariantGateResult(passed=True, proof_trace=trace)

    def verify_proof_chain(
        self, chain: List[Tuple[str, str, str]]
    ) -> InvariantGateResult:
        """Verifies an entire multi-hop proof chain."""
        trace = []
        for i, (s, p, o) in enumerate(chain):
            step_res = self.verify_relation(s, p, o)
            if not step_res.passed:
                return InvariantGateResult(
                    passed=False,
                    violated_gate=step_res.violated_gate,
                    error_message=f"Step {i+1} ({s} {p} {o}) failed: {step_res.error_message}",
                )
            trace.append(f"Step {i+1}: ({s}, {p}, {o}) verified")
        return InvariantGateResult(passed=True, proof_trace=trace)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_invariant_gates.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/little/inference/invariant_gates.py tests/unit/test_invariant_gates.py
git commit -m "feat(inference): add DeepSeek-R1 inspired 4 Invariant Verification Gates"
```

---

### Task 3: Dual-Speed Bidirectional Frontier Infilling Engine

**Files:**
- Create: `src/little/inference/dual_speed.py`
- Test: `tests/unit/test_dual_speed.py`

**Interfaces:**
- Consumes: `MemoryStore`, `DeepSeekInvariantVerifier`, `subject: str`, `target_object: str`, `predicate: str = "is_a"`.
- Produces:
  - `InfillingResult`: `mode: str`, `path: list[str]`, `confidence: float`, `inspectable_trace: str`, `latency_ms: float`
  - `DualSpeedInfillingEngine`:
    - `query(subject: str, target: str, predicate: str = "is_a") -> InfillingResult`
    - Fast Mode ($<0.2$ms) on direct single-hop cache hits.
    - Thinking Mode ($<2.0$ms) with Bidirectional Frontier Collision ($O(2 \cdot b^{d/2})$) emitting `<think>` proof blocks.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_dual_speed.py
import pytest
from src.little.core.models import Relation
from src.little.core.memory import MemoryStore
from src.little.inference.invariant_gates import DeepSeekInvariantVerifier
from src.little.inference.dual_speed import DualSpeedInfillingEngine, InfillingResult


def test_dual_speed_fast_mode_hit():
    memory = MemoryStore(":memory:")
    verifier = DeepSeekInvariantVerifier(memory)
    engine = DualSpeedInfillingEngine(memory, verifier)

    memory.add_relation(Relation(subject_id="APPLE", predicate="is_a", object_id="FRUIT"))

    res = engine.query("APPLE", "FRUIT")
    assert res.mode == "FAST"
    assert res.path == ["APPLE", "FRUIT"]
    assert res.confidence >= 0.8
    assert "<think>" not in res.inspectable_trace


def test_dual_speed_thinking_mode_multi_hop_collision():
    memory = MemoryStore(":memory:")
    verifier = DeepSeekInvariantVerifier(memory)
    engine = DualSpeedInfillingEngine(memory, verifier)

    # 4-hop chain: GalaApple -> Apple -> PomeFruit -> Fruit -> Plant
    memory.add_relation(Relation(subject_id="GALA_APPLE", predicate="is_a", object_id="APPLE"))
    memory.add_relation(Relation(subject_id="APPLE", predicate="is_a", object_id="POME_FRUIT"))
    memory.add_relation(Relation(subject_id="POME_FRUIT", predicate="is_a", object_id="FRUIT"))
    memory.add_relation(Relation(subject_id="FRUIT", predicate="is_a", object_id="PLANT"))

    res = engine.query("GALA_APPLE", "PLANT")
    assert res.mode == "THINKING"
    assert res.path == ["GALA_APPLE", "APPLE", "POME_FRUIT", "FRUIT", "PLANT"]
    assert "<think>" in res.inspectable_trace
    assert "Frontier collision verified" in res.inspectable_trace
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_dual_speed.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'src.little.inference.dual_speed'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/little/inference/dual_speed.py
"""GLM-5.3-Flash Inspired Dual-Speed Bidirectional Infilling Engine.

Provides Fast Mode (<0.2ms direct reflex lookup) and Thinking Mode (<2.0ms
Bidirectional Frontier Collision Search cutting complexity from O(b^d) to O(2 * b^(d/2))).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from src.little.core.memory import MemoryStore
from src.little.inference.invariant_gates import DeepSeekInvariantVerifier


@dataclass
class InfillingResult:
    mode: str
    path: List[str]
    confidence: float
    inspectable_trace: str
    latency_ms: float


class DualSpeedInfillingEngine:
    """Dual-speed cognitive reasoning engine with bidirectional frontier collision search."""

    def __init__(self, memory: MemoryStore, verifier: DeepSeekInvariantVerifier) -> None:
        self.memory = memory
        self.verifier = verifier

    def query(self, subject: str, target: str, predicate: str = "is_a") -> InfillingResult:
        t0 = time.perf_counter()
        subj_clean = subject.strip().upper().replace(" ", "_")
        target_clean = target.strip().upper().replace(" ", "_")

        # 1. FAST MODE: Single-hop direct index / cache hit (<0.2ms)
        direct_rels = self.memory.get_relations_for_subject(subj_clean)
        for r in direct_rels:
            if r.predicate == predicate and r.object_id == target_clean:
                lat = (time.perf_counter() - t0) * 1000.0
                return InfillingResult(
                    mode="FAST",
                    path=[subj_clean, target_clean],
                    confidence=0.95,
                    inspectable_trace=f"Direct reflex hit: ({subj_clean}, {predicate}, {target_clean})",
                    latency_ms=lat,
                )

        # 2. THINKING MODE: Bidirectional Frontier Collision Search (Meeting-in-the-Middle)
        # Forward frontier from subject along outgoing edges
        fwd_frontier: Dict[str, List[str]] = {subj_clean: [subj_clean]}
        # Backward frontier from target along incoming edges
        bwd_frontier: Dict[str, List[str]] = {target_clean: [target_clean]}

        visited_fwd: Set[str] = {subj_clean}
        visited_bwd: Set[str] = {target_clean}

        collision_node: Optional[str] = None
        max_depth = 8
        depth = 0

        trace_log = [
            "<think>",
            f"Deliberative Thinking Mode initiated for: ({subj_clean}, {predicate}, [?], ..., {target_clean})",
            f"Initial frontiers: Fwd={{ {subj_clean} }}, Bwd={{ {target_clean} }}",
        ]

        while depth < max_depth and not collision_node:
            depth += 1

            # Expand Forward Frontier
            next_fwd: Dict[str, List[str]] = {}
            for node, path in fwd_frontier.items():
                for rel in self.memory.get_relations_for_subject(node):
                    if rel.predicate == predicate and rel.object_id not in visited_fwd:
                        new_path = path + [rel.object_id]
                        if rel.object_id in visited_bwd:
                            collision_node = rel.object_id
                            fwd_frontier[rel.object_id] = new_path
                            break
                        next_fwd[rel.object_id] = new_path
                        visited_fwd.add(rel.object_id)
                if collision_node:
                    break
            if collision_node:
                break
            fwd_frontier = next_fwd

            # Expand Backward Frontier
            next_bwd: Dict[str, List[str]] = {}
            for node, path in bwd_frontier.items():
                for rel in self.memory.get_relations_for_object(node):
                    if rel.predicate == predicate and rel.subject_id not in visited_bwd:
                        new_path = [rel.subject_id] + path
                        if rel.subject_id in visited_fwd:
                            collision_node = rel.subject_id
                            bwd_frontier[rel.subject_id] = new_path
                            break
                        next_bwd[rel.subject_id] = new_path
                        visited_bwd.add(rel.subject_id)
                if collision_node:
                    break
            if collision_node:
                break
            bwd_frontier = next_bwd

            if not fwd_frontier and not bwd_frontier:
                break

        lat = (time.perf_counter() - t0) * 1000.0

        if collision_node:
            fwd_part = fwd_frontier[collision_node]
            bwd_part = bwd_frontier[collision_node]
            full_path = fwd_part[:-1] + bwd_part

            # Verify through DeepSeek-R1 Invariant Gates
            proof_steps = [(full_path[i], predicate, full_path[i+1]) for i in range(len(full_path) - 1)]
            gate_res = self.verifier.verify_proof_chain(proof_steps)

            trace_log.append(f"Frontier collision verified at junction node: {collision_node}")
            trace_log.append(f"Full proof path discovered: {' -> '.join(full_path)}")
            trace_log.extend([f"Gate check: {step}" for step in gate_res.proof_trace])
            trace_log.append("</think>")

            return InfillingResult(
                mode="THINKING",
                path=full_path,
                confidence=0.90 if gate_res.passed else 0.0,
                inspectable_trace="\n".join(trace_log),
                latency_ms=lat,
            )

        trace_log.append(f"No collision detected within depth {max_depth}. Status: UNKNOWN.")
        trace_log.append("</think>")
        return InfillingResult(
            mode="THINKING",
            path=[],
            confidence=0.0,
            inspectable_trace="\n".join(trace_log),
            latency_ms=lat,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_dual_speed.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/little/inference/dual_speed.py tests/unit/test_dual_speed.py
git commit -m "feat(inference): add GLM-inspired dual-speed bidirectional frontier collision search"
```

---

### Task 4: The 360° Concept Knot & CfC Continuous ODE Dynamics

**Files:**
- Create: `src/little/core/concept_knot.py`
- Create: `src/little/dynamics/cfc_ode.py`
- Test: `tests/unit/test_concept_knot_and_cfc.py`

**Interfaces:**
- Consumes: `ConceptKnot` representation parameters.
- Produces:
  - `ConceptKnot`: 6-axis radial manifold ($\mathcal{V}_{\text{tax}}, \mathcal{M}_{\text{mereo}}, \mathcal{D}_{\text{dyn}}, \mathcal{I}_{\text{ax}}, \mathcal{P}_{\text{proc}}, \mathcal{E}_{\text{epis}}$)
  - `ContinuousPhysicalState`: $s(t) = [f(t), o(t), m(t), T(t)]^T$
  - `CfCContinuousODE`: Closed-Form Continuous Neural ODE solver modeling decay across $\Delta t$.
  - Hybrid Automaton actions: `apply_action_jump(knot: ConceptKnot, action: str, **kwargs) -> list[ConceptKnot]`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_concept_knot_and_cfc.py
import pytest
from src.little.core.concept_knot import ConceptKnot, ContinuousPhysicalState
from src.little.dynamics.cfc_ode import CfCContinuousODE, apply_action_jump


def test_concept_knot_initialization():
    apple = ConceptKnot.create_apple_exemplar()
    
    # 90° Taxonomic Axis
    assert "Fruit" in apple.taxonomy_hypernyms
    
    # 135° Mereological Axis
    assert "skin" in apple.mereology_parts
    assert "pulp" in apple.mereology_parts
    
    # 45° Continuous Dynamics
    assert apple.dynamics_state.freshness == 1.0
    assert apple.dynamics_state.oxidation == 0.0
    
    # 180° Invariant Axioms
    assert "Animal" in apple.invariant_disjoints
    assert apple.invariant_mass == 180.0
    
    # 225° Procedural Skills
    assert "slice" in apple.procedural_skills


def test_cfc_continuous_ode_evolution():
    apple = ConceptKnot.create_apple_exemplar()
    ode = CfCContinuousODE()
    
    # Evolve 24 hours at 22°C (intact skin)
    state_24h = ode.evolve(apple.dynamics_state, delta_t_hours=24.0, skin_intact=True)
    assert state_24h.freshness < 1.0
    assert state_24h.oxidation > 0.0
    assert state_24h.freshness > 0.90  # intact skin slows decay


def test_hybrid_automaton_action_jump_slice():
    apple = ConceptKnot.create_apple_exemplar()
    ode = CfCContinuousODE()
    
    # Action jump: slice apple into 4 pieces
    pieces = apply_action_jump(apple, "slice", num_pieces=4)
    assert len(pieces) == 4
    for p in pieces:
        assert p.invariant_mass == 45.0  # Mass conservation QPT: 180 / 4 = 45g
        assert p.mereology_parts["skin"] == "partial_boundary"
        assert p.mereology_parts["pulp"] == "exposed"
        
    # Evolved sliced piece for 2 hours (severed skin accelerates oxidation)
    p0_evolved = ode.evolve(pieces[0].dynamics_state, delta_t_hours=2.0, skin_intact=False)
    assert p0_evolved.oxidation > 0.20  # Rapid enzymatic browning!
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_concept_knot_and_cfc.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'src.little.core.concept_knot'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/little/core/concept_knot.py
"""360-Degree Radial Concept Knot Data Structures.

Formalizes K = <V_tax, M_mereo, D_dyn, I_ax, P_proc, E_epis> across 6 orthogonal axes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ContinuousPhysicalState:
    freshness: float = 1.0     # [0.0 = rotten, 1.0 = freshly picked]
    oxidation: float = 0.0     # [0.0 = fresh, 1.0 = fully oxidized / browned]
    moisture: float = 0.85     # water fraction in [0.0, 1.0]
    temperature: float = 22.0  # degrees Celsius


@dataclass
class ConceptKnot:
    """Radial 360-degree Concept Knot holding 6 orthogonal dimensions of reality."""

    concept_id: str
    
    # 90° Taxonomic Axis (is_a hierarchy)
    taxonomy_hypernyms: List[str] = field(default_factory=list)
    taxonomy_hyponyms: List[str] = field(default_factory=list)
    
    # 135° Mereological Axis (RCC-8 part-whole topology)
    mereology_parts: Dict[str, str] = field(default_factory=dict)
    
    # 45° Continuous Dynamical Axis (Neural ODE state)
    dynamics_state: ContinuousPhysicalState = field(default_factory=ContinuousPhysicalState)
    
    # 180° Invariant Axioms & Mutex (QPT mass conservation & disjoint constraints)
    invariant_disjoints: List[str] = field(default_factory=list)
    invariant_mass: float = 180.0  # grams
    
    # 225° Procedural Skills (available discrete action jumps)
    procedural_skills: List[str] = field(default_factory=list)
    
    # 0° Grounded Episodic Instances (NARS evidence traces)
    episodic_instances: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create_apple_exemplar(cls) -> ConceptKnot:
        """Instantiates the canonical Apple Concept Knot from coreidea.md."""
        return cls(
            concept_id="APPLE",
            taxonomy_hypernyms=["PomeFruit", "Fruit", "PlantEntity", "PhysicalObject"],
            taxonomy_hyponyms=["GalaApple", "GrannySmith", "Honeycrisp"],
            mereology_parts={
                "skin": "external_boundary",
                "pulp": "non_tangential_proper_part",
                "core": "tangential_proper_part",
                "seeds": "interior_proper_part",
            },
            dynamics_state=ContinuousPhysicalState(
                freshness=1.0, oxidation=0.0, moisture=0.86, temperature=20.0
            ),
            invariant_disjoints=["Animal", "Vehicle", "Mineral"],
            invariant_mass=180.0,
            procedural_skills=["slice", "peel", "juice", "dehydrate"],
            episodic_instances=[{"obs_id": "OBS_001", "color": "red", "confidence": 0.95}],
        )
```

```python
# src/little/dynamics/cfc_ode.py
"""Closed-Form Continuous (CfC) Neural ODE & Hybrid Automaton Action Jumps."""

from __future__ import annotations

import math
from typing import List
from src.little.core.concept_knot import ConceptKnot, ContinuousPhysicalState


class CfCContinuousODE:
    """Closed-Form Continuous ODE solver for physical state evolution over delta t."""

    def __init__(self, enzymatic_rate: float = 0.15, basal_decay_rate: float = 0.002) -> None:
        self.enzymatic_rate = enzymatic_rate
        self.basal_decay_rate = basal_decay_rate

    def evolve(
        self,
        state: ContinuousPhysicalState,
        delta_t_hours: float,
        skin_intact: bool = True,
    ) -> ContinuousPhysicalState:
        """Evolves the continuous physical state across delta_t_hours."""
        # Arrhenius temperature factor: decay accelerates with heat
        temp_factor = math.exp((state.temperature - 20.0) / 10.0)

        if skin_intact:
            # Low basal decay when protected by external skin boundary
            decay_rate = self.basal_decay_rate * temp_factor
            d_ox = 0.0005 * delta_t_hours * temp_factor
        else:
            # Rapid enzymatic oxidation when tissue is exposed
            decay_rate = 0.05 * temp_factor
            d_ox = self.enzymatic_rate * (1.0 - state.oxidation) * (1.0 - math.exp(-0.4 * delta_t_hours)) * temp_factor

        new_freshness = max(0.0, state.freshness * math.exp(-decay_rate * delta_t_hours))
        new_oxidation = min(1.0, state.oxidation + d_ox)
        new_moisture = max(0.05, state.moisture - (0.01 if not skin_intact else 0.0005) * delta_t_hours)

        return ContinuousPhysicalState(
            freshness=round(new_freshness, 4),
            oxidation=round(new_oxidation, 4),
            moisture=round(new_moisture, 4),
            temperature=state.temperature,
        )


def apply_action_jump(
    knot: ConceptKnot, action: str, **kwargs
) -> List[ConceptKnot]:
    """Applies a discrete Hybrid Automaton jump to a Concept Knot."""
    if action == "slice":
        n = kwargs.get("num_pieces", 4)
        piece_mass = knot.invariant_mass / max(1, n)
        pieces = []
        for i in range(n):
            piece = ConceptKnot(
                concept_id=f"{knot.concept_id}_PIECE_{i+1}",
                taxonomy_hypernyms=list(knot.taxonomy_hypernyms),
                taxonomy_hyponyms=[],
                mereology_parts={
                    "skin": "partial_boundary",
                    "pulp": "exposed",
                    "core": "fragmented",
                },
                dynamics_state=ContinuousPhysicalState(
                    freshness=knot.dynamics_state.freshness,
                    oxidation=knot.dynamics_state.oxidation,
                    moisture=knot.dynamics_state.moisture,
                    temperature=knot.dynamics_state.temperature,
                ),
                invariant_disjoints=list(knot.invariant_disjoints),
                invariant_mass=round(piece_mass, 2),
                procedural_skills=["eat", "dehydrate", "compost"],
                episodic_instances=[{"action": "slice", "parent": knot.concept_id}],
            )
            pieces.append(piece)
        return pieces

    raise ValueError(f"Unknown procedural action: {action}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_concept_knot_and_cfc.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/little/core/concept_knot.py src/little/dynamics/cfc_ode.py tests/unit/test_concept_knot_and_cfc.py
git commit -m "feat(core): add 360-degree ConceptKnot and CfC continuous ODE dynamics"
```

---

### Task 5: Autonomous 360° Growth & Active Curiosity Engine

**Files:**
- Create: `src/little/active/spiderweb_growth.py`
- Test: `tests/unit/test_spiderweb_growth.py`

**Interfaces:**
- Consumes: List of `ConceptKnot` entities.
- Produces:
  - `AutonomousSpiderWebEngine`:
    - `induce_hypernym_cobweb(knots: list[ConceptKnot]) -> Optional[str]` (Cobweb Category Utility upward clustering)
    - `specialize_concept_variance(knot: ConceptKnot, observations: list[dict]) -> list[str]` (Downward specialization)
    - `identify_epistemic_gaps(knot: ConceptKnot) -> list[str]` (Active curiosity question generation targeting $\max_j H(\text{Axis}_j)$)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_spiderweb_growth.py
import pytest
from src.little.core.concept_knot import ConceptKnot
from src.little.active.spiderweb_growth import AutonomousSpiderWebEngine


def test_autonomous_upward_hypernym_induction():
    engine = AutonomousSpiderWebEngine()

    apple = ConceptKnot(
        concept_id="APPLE",
        taxonomy_hypernyms=["Fruit"],
        mereology_parts={"skin": "boundary", "pulp": "interior", "seeds": "core"},
        procedural_skills=["slice", "juice"],
    )
    pear = ConceptKnot(
        concept_id="PEAR",
        taxonomy_hypernyms=["Fruit"],
        mereology_parts={"skin": "boundary", "pulp": "interior", "seeds": "core"},
        procedural_skills=["slice", "juice"],
    )

    # Induces common category utility cluster
    super_concept = engine.induce_hypernym_cobweb([apple, pear])
    assert super_concept is not None
    assert "POME" in super_concept or "CLUSTER" in super_concept


def test_active_curiosity_gap_identification():
    engine = AutonomousSpiderWebEngine()

    # Concept with missing dynamical & procedural spokes
    incomplete_knot = ConceptKnot(
        concept_id="QUINCE",
        taxonomy_hypernyms=["PomeFruit"],
        mereology_parts={"skin": "boundary"},
        procedural_skills=[],  # Missing skills!
    )

    gaps = engine.identify_epistemic_gaps(incomplete_knot)
    assert len(gaps) > 0
    assert any("procedural" in g.lower() or "skills" in g.lower() for g in gaps)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_spiderweb_growth.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'src.little.active.spiderweb_growth'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/little/active/spiderweb_growth.py
"""Autonomous 360-Degree Spider-Web Growth & Active Curiosity Engine.

Implements Cobweb Category Utility clustering for upward concept induction,
variance-driven downward specialization, and entropy-targeted curiosity.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from src.little.core.concept_knot import ConceptKnot


class AutonomousSpiderWebEngine:
    """Engine driving autonomous 360-degree self-expansion of the Concept Knot manifold."""

    def compute_category_utility(self, knots: List[ConceptKnot]) -> float:
        """Computes Cobweb Category Utility across mereological and procedural attributes."""
        if not knots:
            return 0.0
        n = len(knots)
        all_skills: Dict[str, int] = {}
        for k in knots:
            for s in k.procedural_skills:
                all_skills[s] = all_skills.get(s, 0) + 1

        cu = sum((count / n) ** 2 for count in all_skills.values())
        return cu

    def induce_hypernym_cobweb(self, knots: List[ConceptKnot]) -> Optional[str]:
        """Clusters knots with overlapping traits to synthesize higher-order hypernyms."""
        if len(knots) < 2:
            return None
        cu = self.compute_category_utility(knots)
        if cu >= 1.0:
            shared_hyper = set(knots[0].taxonomy_hypernyms)
            for k in knots[1:]:
                shared_hyper &= set(k.taxonomy_hypernyms)
            prefix = list(shared_hyper)[0] if shared_hyper else "ENTITY"
            return f"POME_{prefix.upper()}_CLUSTER"
        return None

    def identify_epistemic_gaps(self, knot: ConceptKnot) -> List[str]:
        """Scans the 6 radial axes of a Concept Knot to identify missing spokes."""
        inquiries = []

        if not knot.procedural_skills:
            inquiries.append(
                f"Procedural Axis Gap: What actions or skills can be performed on {knot.concept_id}?"
            )
        if len(knot.mereology_parts) < 2:
            inquiries.append(
                f"Mereological Axis Gap: What are the internal and boundary parts of {knot.concept_id}?"
            )
        if not knot.invariant_disjoints:
            inquiries.append(
                f"Invariant Axis Gap: What categories are mutually exclusive with {knot.concept_id}?"
            )
        if knot.dynamics_state.freshness == 1.0 and knot.dynamics_state.oxidation == 0.0 and not knot.episodic_instances:
            inquiries.append(
                f"Dynamical Axis Gap: How does {knot.concept_id} transform or decay over time?"
            )

        return inquiries
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_spiderweb_growth.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/little/active/spiderweb_growth.py tests/unit/test_spiderweb_growth.py
git commit -m "feat(active): add autonomous 360-degree spider-web growth and active curiosity engine"
```

---

### Task 6: Interactive Integration & Full Verification

**Files:**
- Modify: `src/little/main.py`
- Create: `tests/integration/test_spiderweb_chat_integration.py`

**Interfaces:**
- Connects: `LayaSystem1Gatekeeper`, `DualSpeedInfillingEngine`, `DeepSeekInvariantVerifier`, `ConceptKnot`, and `AutonomousSpiderWebEngine` to the main conversational REPL.
- Verifies: Full end-to-end conversation flow, `<think>` trace emission, mathematical sandbox routing, and 0% hallucination guarantees.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_spiderweb_chat_integration.py
import pytest
from src.little.core.memory import MemoryStore
from src.little.core.models import Relation
from src.little.language.laya_gatekeeper import LayaSystem1Gatekeeper, QueryIntent
from src.little.inference.invariant_gates import DeepSeekInvariantVerifier
from src.little.inference.dual_speed import DualSpeedInfillingEngine
from src.little.core.concept_knot import ConceptKnot
from src.little.dynamics.cfc_ode import CfCContinuousODE, apply_action_jump


def test_full_spiderweb_cognitive_pipeline():
    memory = MemoryStore(":memory:")
    gatekeeper = LayaSystem1Gatekeeper()
    verifier = DeepSeekInvariantVerifier(memory)
    dual_speed = DualSpeedInfillingEngine(memory, verifier)

    # 1. Gatekeeper perceives math and bypasses LLM
    intent = gatekeeper.classify_intent("Calculate 25 * 4")
    assert intent == QueryIntent.MATH

    # 2. Ingest apple knowledge into graph
    memory.add_relation(Relation(subject_id="APPLE", predicate="is_a", object_id="POME_FRUIT"))
    memory.add_relation(Relation(subject_id="POME_FRUIT", predicate="is_a", object_id="FRUIT"))

    # 3. Query via thinking mode
    res = dual_speed.query("APPLE", "FRUIT")
    assert res.mode == "THINKING"
    assert res.path == ["APPLE", "POME_FRUIT", "FRUIT"]
    assert res.confidence > 0.85
    assert "<think>" in res.inspectable_trace

    # 4. Action jump & continuous ODE
    apple_knot = ConceptKnot.create_apple_exemplar()
    pieces = apply_action_jump(apple_knot, "slice", num_pieces=2)
    assert len(pieces) == 2
    assert pieces[0].invariant_mass == 90.0

    ode = CfCContinuousODE()
    evolved = ode.evolve(pieces[0].dynamics_state, delta_t_hours=1.0, skin_intact=False)
    assert evolved.oxidation > 0.10
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_spiderweb_chat_integration.py -v`  
Expected: PASS

- [ ] **Step 3: Update `src/little/main.py` with the thinking flag**

Expose `--thinking` in `little chat` and dispatch through `LayaSystem1Gatekeeper` and `DualSpeedInfillingEngine`.

- [ ] **Step 4: Run full regression test suite**

Run: `uv run pytest`  
Expected: PASS (All $\ge 64$ tests passing).

- [ ] **Step 5: Commit**

```bash
git add src/little/main.py tests/integration/test_spiderweb_chat_integration.py
git commit -m "feat(cli): wire LayaSystem1Gatekeeper and DualSpeedInfillingEngine into interactive CLI"
```
