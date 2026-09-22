import pytest
from little.core.concept_knot import ConceptKnot
from little.dynamics.cfc_ode import CfCContinuousODE, apply_action_jump
from little.inference.dual_speed import DualSpeedInfillingEngine
from little.inference.invariant_gates import DeepSeekInvariantVerifier
from little.language.laya_gatekeeper import (
    BeliefStatus,
    LayaSystem1Gatekeeper,
    QueryIntent,
)
from little.memory.store import MemoryStore


def test_full_spiderweb_cognitive_pipeline():
    memory = MemoryStore(":memory:")
    gatekeeper = LayaSystem1Gatekeeper()
    verifier = DeepSeekInvariantVerifier(memory)
    dual_speed = DualSpeedInfillingEngine(memory, verifier)

    # 1. Gatekeeper perceives math and bypasses LLM
    intent = gatekeeper.classify_intent("Calculate 25 * 4")
    assert intent == QueryIntent.MATH

    # 2. Gatekeeper perceives question
    intent_q = gatekeeper.classify_intent("Is an apple a fruit?")
    assert intent_q == QueryIntent.QUESTION

    # 3. Ingest apple knowledge into graph
    memory.add_relation("APPLE", "is_a", "POME_FRUIT")
    memory.add_relation("POME_FRUIT", "is_a", "FRUIT")

    # 4. Query via thinking mode
    res = dual_speed.query("APPLE", "FRUIT")
    assert res.mode == "THINKING"
    assert res.path == ["APPLE", "POME_FRUIT", "FRUIT"]
    assert res.confidence > 0.85
    assert "<think>" in res.inspectable_trace

    # 5. Action jump & continuous ODE
    apple_knot = ConceptKnot.create_apple_exemplar()
    pieces = apply_action_jump(apple_knot, "slice", num_pieces=2)
    assert len(pieces) == 2
    assert pieces[0].invariant_mass == 90.0

    ode = CfCContinuousODE()
    evolved = ode.evolve(
        pieces[0].dynamics_state, delta_t_hours=1.0, skin_intact=False
    )
    assert evolved.oxidation > 0.10
