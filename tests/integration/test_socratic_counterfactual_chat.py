from pathlib import Path

from little.core.kernel import CognitiveKernel
from little.core.models import BeliefStatus
from little.memory.store import MemoryStore


def test_socratic_and_counterfactual_chat_integration(tmp_path: Path):
    db_path = tmp_path / "socratic_test.db"
    store = MemoryStore(db_path, seed_ontology=True)
    kernel = CognitiveKernel(store)

    try:
        # 1. Multi-hop learning with novel taxonomy chain
        kernel.process("A thylacine is a dasyuromorph.")
        kernel.process("A dasyuromorph is a marsupial.")
        kernel.process("A marsupial is a mammal.")

        # 2. Socratic multi-hop syllogism
        out_socratic = kernel.process("Explain why a thylacine is a mammal")
        assert out_socratic.inference is not None
        assert out_socratic.inference.status == BeliefStatus.SUPPORTED
        ans_soc = str(out_socratic.inference.answer)
        assert "Deductive Proof:" in ans_soc
        assert "A thylacine is a dasyuromorph" in ans_soc
        assert "A dasyuromorph is a marsupial" in ans_soc
        assert "A marsupial is a mammal" in ans_soc
        assert "by transitivity" in ans_soc

        # 3. Disjoint invariant refutation explanation
        kernel.process("An animal is not a machine.")
        out_disj = kernel.process("Why is a cheetah not a machine?")
        assert out_disj.inference is not None
        assert out_disj.inference.status == BeliefStatus.SUPPORTED
        v_disj = out_disj.inference.verbalize()
        assert "disjoint" in v_disj.lower() or "machine" in v_disj.lower()

        # 4. Inherited capability why explanation
        kernel.process("A hawk is a bird.")
        kernel.process("A bird can fly.")
        out_cap = kernel.process("Why can a hawk fly?")
        assert out_cap.inference is not None
        assert out_cap.inference.status == BeliefStatus.SUPPORTED
        assert "bird" in str(out_cap.inference.answer)
        assert "fly" in str(out_cap.inference.answer)

        # 5. Continuous Counterfactual Simulation (CfC ODE + Arrhenius)
        kernel.process("Slice an apple into 4 pieces.")
        out_cf = kernel.process(
            "Compare an apple slice at 4C vs 35C after 6 hours."
        )
        assert out_cf.inference is not None
        assert out_cf.inference.status == BeliefStatus.SUPPORTED
        ans_cf = str(out_cf.inference.answer)
        assert "Counterfactual Analysis" in ans_cf
        assert "Condition 1" in ans_cf or "Refrigerated" in ans_cf
        assert "Arrhenius" in ans_cf
    finally:
        store.close()
