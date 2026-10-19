"""Unit and real-world curriculum tests for the 360° Spider-Web Mini Learning Model (MLM / MLU)."""

from __future__ import annotations

from little.memory.store import MemoryStore
from little.mlu import MiniLearningUnit, SpiderWebMLM, deterministic_seed_vector


def test_mlu_liquid_step_and_360_spoke_learning() -> None:
    """MiniLearningUnit updates LFM liquid state, 360° spoke projections, and token transitions."""
    orca_unit = MiniLearningUnit("orca", role="leaf", parent_hub="SUBCORE_TAXONOMY")
    mammal_vec = deterministic_seed_vector("mammal", orca_unit.dim)
    ocean_vec = deterministic_seed_vector("ocean", orca_unit.dim)

    state_after = orca_unit.forward_liquid_step([mammal_vec, ocean_vec])
    assert len(state_after) == orca_unit.dim
    assert orca_unit.activation_count == 1

    s1 = orca_unit.learn_spoke_relation("is_a", "mammal", mammal_vec, learning_rate=0.25, verified=True)
    s2 = orca_unit.learn_spoke_relation("located_in", "ocean", ocean_vec, learning_rate=0.25, verified=True)
    assert 0.0 <= s1 <= 1.0
    assert 0.0 <= s2 <= 1.0
    assert orca_unit.verified_hits == 2.0
    assert orca_unit.nars_confidence > 0.5

    tax_preds = orca_unit.predict_spoke_targets("taxonomy", {"MAMMAL": mammal_vec})
    assert len(tax_preds) == 1
    assert tax_preds[0].target_concept == "MAMMAL"
    assert tax_preds[0].combined_confidence > 0.4

    epi_preds = orca_unit.predict_spoke_targets("episodic", {"OCEAN": ocean_vec})
    assert len(epi_preds) == 1
    assert epi_preds[0].target_concept == "OCEAN"

    # Serialization round-trip
    restored = MiniLearningUnit.from_dict(orca_unit.to_dict())
    assert restored.concept_id == "ORCA"
    assert restored.verified_hits == 2.0
    assert "MAMMAL" in restored.spokes["taxonomy"].target_affinities


def test_spiderweb_mlm_real_world_curriculum_and_mitosis() -> None:
    """SpiderWebMLM learns a multi-domain curriculum, enforces I_DAG/I_MUTEX gates, promotes Sub-Cores via 5-brake mitosis, and answers 360° queries."""
    with MemoryStore(seed_ontology=False) as memory:
        mlm = SpiderWebMLM(
            memory=memory,
            mitosis_min_activations=3,
            mitosis_min_verified_spokes=2,
            survival_window_cycles=5,
        )

        # Feed a real-world multi-sentence science & biology stream
        corpus = (
            "An orca is a cetacean. "
            "A cetacean is a mammal. "
            "A mammal is an animal. "
            "An orca has a dorsal_fin. "
            "An orca can echolocate. "
            "An orca lives in the ocean. "
            "A mammal is not a reptile. "
            "A cobra is a reptile."
        )
        report = mlm.train_on_text(corpus)
        assert report.sentences_processed == 8
        assert report.relations_verified == 8
        assert report.relations_refuted == 0

        # Because ORCA has >= 3 activations across >= 2 spokes (taxonomy, mereology, procedural, episodic)
        # and passes all 5 brakes, it promotes via Fractal Mitosis into a 360° Sub-Core Hub!
        assert "ORCA" in report.mitosis_promotions
        assert mlm.nodes["ORCA"].role == "subcore"

        # Now feed a contradictory claim: "An animal is an orca." (Violates I_DAG cycle!)
        # and "An orca is a cobra." (Violates I_MUTEX via mammal disjoint_with reptile!)
        contradiction_report = mlm.train_on_text("An animal is an orca. An orca is a cobra.")
        assert contradiction_report.relations_refuted == 2
        assert contradiction_report.relations_verified == 0

        # Query 1: Multi-hop transitive 360° question ("Is an orca an animal?")
        q_trans = mlm.query_360("Is an orca an animal?")
        assert q_trans.status == "VERIFIED"
        assert "animal" in q_trans.answer.lower()
        assert q_trans.confidence > 0.4

        # Query 2: Disjointness refutation ("Is an orca a cobra?")
        q_ref = mlm.query_360("Is an orca a cobra?")
        assert q_ref.status == "REFUTED"
        assert "I_MUTEX" in q_ref.answer

        # Query 3: Open 360° concept synthesis ("Tell me about orca")
        q_open = mlm.query_360("Tell me about orca")
        assert q_open.status == "VERIFIED"
        assert "cetacean" in q_open.answer.lower()
        assert "dorsal fin" in q_open.answer.lower() or "echolocate" in q_open.answer.lower()

        # Query 4: Unknown concept abstention ("What is a Dyson_Sphere?")
        q_unk = mlm.query_360("What is a dyson_sphere?")
        assert q_unk.status == "UNKNOWN"
        assert q_unk.confidence == 0.0

        # Verify SQLite persistence across reload
        mlm_reloaded = SpiderWebMLM(memory=memory)
        assert "ORCA" in mlm_reloaded.nodes
        assert mlm_reloaded.nodes["ORCA"].role == "subcore"


def test_cognitive_kernel_mlm_integration() -> None:
    """CognitiveKernel integrates SpiderWebMLM into stream learning, cognitive commands, and telemetry snapshots."""
    from little.core.kernel import CognitiveKernel

    with MemoryStore(seed_ontology=False) as memory, CognitiveKernel(memory=memory) as kernel:
        train_out = kernel.command_dispatcher.dispatch(
            "mlm train: A falcon is a bird. A bird is an animal. A falcon can fly. A falcon has wings.",
            kernel,
        )
        assert train_out is not None
        assert "360° Spider-Web MLM Trained" in train_out
        assert "4 verified 360° spoke links" in train_out

        ask_out = kernel.command_dispatcher.dispatch("mlm ask: Is a falcon an animal?", kernel)
        assert ask_out is not None
        assert "Yes — falcon is a animal" in ask_out

        status_out = kernel.command_dispatcher.dispatch("show cognitive status", kernel)
        assert status_out is not None
        assert "360° Spider-Web MLM:" in status_out

        snap = kernel.get_telemetry_snapshot()
        assert "mlm_spiderweb" in snap
        assert snap["mlm_spiderweb"]["total_mlu_nodes"] >= 10

