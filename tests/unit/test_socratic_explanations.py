from little.core.models import BeliefStatus
from little.language.parser import LearningEngine
from little.language.socratic import SocraticReasoner
from little.memory.store import MemoryStore


def test_socratic_syllogism_why_is_a():
    store = MemoryStore(":memory:", seed_ontology=True)
    engine = LearningEngine(store)
    try:
        engine.learn("A cheetah is a mammal.")
        engine.learn("A mammal is an animal.")

        socratic = SocraticReasoner(store, engine.inference)
        exp = socratic.explain_why_is_a("cheetah", "animal")
        assert exp.status == BeliefStatus.SUPPORTED
        assert len(exp.premises) == 2
        assert "A cheetah is a mammal" in exp.premises[0]
        assert "A mammal is an animal" in exp.premises[1]
        assert "Deductive Proof:" in exp.verbalized
        assert "by transitivity" in exp.verbalized

        # Verify via parser ask()
        res_why = engine.ask("Why is a cheetah an animal?")
        assert res_why.status == BeliefStatus.SUPPORTED
        assert "cheetah" in res_why.verbalize().lower()

        # Verify explicit formal explanation request
        res_explain = engine.ask("Explain why a cheetah is an animal")
        assert res_explain.status == BeliefStatus.SUPPORTED
        assert "Deductive Proof:" in res_explain.verbalize()
    finally:
        store.close()


def test_socratic_invariant_why_not():
    store = MemoryStore(":memory:", seed_ontology=True)
    engine = LearningEngine(store)
    try:
        engine.learn("A dog is an animal.")
        engine.learn("An animal is not a vehicle.")

        socratic = SocraticReasoner(store, engine.inference)
        exp = socratic.explain_why_not("dog", "vehicle")
        assert exp.status == BeliefStatus.REFUTED
        assert "Invariant Gate I_mutex" in exp.conclusion or "disjoint" in exp.conclusion

        # Verify via parser ask()
        res = engine.ask("Why is a dog not a vehicle?")
        assert res.status == BeliefStatus.SUPPORTED
        assert "disjoint" in res.verbalize().lower() or "vehicle" in res.verbalize().lower()
    finally:
        store.close()


def test_socratic_capability_inheritance():
    store = MemoryStore(":memory:", seed_ontology=True)
    engine = LearningEngine(store)
    try:
        engine.learn("A hawk is a bird.")
        engine.learn("A bird can fly.")

        socratic = SocraticReasoner(store, engine.inference)
        exp = socratic.explain_capability("hawk", "fly")
        assert exp.status == BeliefStatus.SUPPORTED
        assert "hawk is a bird" in exp.premises[0]
        assert "bird can fly" in exp.premises[1]
        assert "hawk is a bird, which can fly" in exp.conclusion

        # Verify via parser ask()
        res = engine.ask("Why can a hawk fly?")
        assert res.status == BeliefStatus.SUPPORTED
        assert "bird" in res.verbalize() and "fly" in res.verbalize()
    finally:
        store.close()
