from little.core.models import BeliefStatus
from little.dynamics.counterfactual import CounterfactualDynamicsEngine
from little.language.parser import LearningEngine
from little.memory.store import MemoryStore


def test_counterfactual_temperature_and_duration_resolution():
    store = MemoryStore(":memory:", seed_ontology=True)
    engine = CounterfactualDynamicsEngine(store)
    try:
        assert engine.resolve_temperature("refrigerated at 4C") == 4.0
        assert engine.resolve_temperature("in the fridge") == 4.0
        assert engine.resolve_temperature("in the sun") == 35.0
        assert engine.resolve_temperature("ambient room temperature") == 20.0
        assert engine.resolve_temperature("kept at 30 degrees") == 30.0

        assert engine.resolve_duration_hours("for 5 hours") == 5.0
        assert engine.resolve_duration_hours("after 30 minutes") == 0.5
        assert engine.resolve_duration_hours("over 2 days") == 48.0
    finally:
        store.close()


def test_counterfactual_contrastive_comparison():
    store = MemoryStore(":memory:", seed_ontology=True)
    engine = CounterfactualDynamicsEngine(store)
    try:
        # Compare apple slice at 4°C vs 35°C after 6 hours
        res = engine.compare_scenarios(
            subject="apple slice",
            temp1=4.0,
            dur1=6.0,
            temp2=35.0,
            dur2=6.0,
            label1="Refrigerated",
            label2="In the Sun",
        )
        assert res.scenario_1.freshness > res.scenario_2.freshness
        assert res.scenario_1.oxidation < res.scenario_2.oxidation
        assert res.thermal_ratio > 3.0  # ~4.2x faster decay in the sun
        assert "Arrhenius" in res.verbalized
        assert "Refrigerated" in res.verbalized
    finally:
        store.close()


def test_counterfactual_query_end_to_end_ask():
    store = MemoryStore(":memory:", seed_ontology=True)
    learning_engine = LearningEngine(store)
    try:
        learning_engine.learn("Slice an apple into 4 pieces.")

        # Contrastive query
        res = learning_engine.ask(
            "Compare an apple slice at 4C vs 35C after 6 hours."
        )
        assert res.status == BeliefStatus.SUPPORTED
        assert "Counterfactual Analysis" in str(res.answer)
        assert "Arrhenius" in str(res.answer)

        # Hypothetical single world query
        res_single = learning_engine.ask(
            "What if an apple slice is refrigerated at 4C for 10 hours?"
        )
        assert res_single.status == BeliefStatus.SUPPORTED
        assert "Hypothetical Scenario" in str(res_single.answer)
        assert "Freshness=" in str(res_single.answer)
    finally:
        store.close()
