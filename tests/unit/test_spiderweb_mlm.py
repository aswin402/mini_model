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
    assert any("MAMMAL" in s.target_affinities for s in restored.spokes.values())


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
        assert snap["mlm_spiderweb"]["rust_arena_native"] is True


def test_rust_mlu_arena_native_acceleration(tmp_path: object) -> None:
    """RustMluArena executes Wave 1 (LFM Liquid), Wave 2 (360° Spoke Prediction), Wave 3 (Local Predictive-Coding + Mitosis), and snapshot persistence in native Rust."""
    from pathlib import Path

    from little.rust import RustMluArena

    snap_file = str(Path(str(tmp_path)) / "arena_snapshot.json")
    with RustMluArena(mitosis_min_activations=3, mitosis_min_verified_spokes=2) as arena:
        assert arena.is_native is True
        # Initial arena has CORE_ROOT + 6 Sub-Core Hubs = 7 nodes
        assert arena.node_count == 7

        s1 = arena.learn_spoke("DOLPHIN", "is_a", "CETACEAN", learning_rate=0.25, verified=True)
        s2 = arena.learn_spoke("DOLPHIN", "has_part", "BLOWHOLE", learning_rate=0.25, verified=True)
        s3 = arena.learn_spoke("DOLPHIN", "can", "ECHOLOCATE", learning_rate=0.25, verified=True)
        assert s1 > 0.0 and s2 > 0.0 and s3 > 0.0
        assert arena.node_count == 11
        assert arena.mitosis_count == 1

        tax_preds = arena.predict_spoke("DOLPHIN", spoke_idx=0, top_k=5)
        assert len(tax_preds) >= 1
        assert any(
            p["target_concept"] == "CETACEAN" and p["combined_confidence"] > 0.5
            for p in tax_preds
        )

        stats_ep1 = arena.train_stream(
            "Dolphins are marine cetaceans that communicate with clicks and whistles. "
            "Dolphins swim rapidly through ocean currents.",
            learning_rate=0.25,
        )
        stats_ep2 = arena.train_stream(
            "Dolphins are marine cetaceans that communicate with clicks and whistles. "
            "Dolphins swim rapidly through ocean currents.",
            learning_rate=0.25,
        )
        assert stats_ep1["tokens_processed"] >= 16
        assert stats_ep1["spoke_updates"] > 0
        assert stats_ep2["average_surprise"] < stats_ep1["average_surprise"]

        generated = arena.generate("dolphins are", max_tokens=6)
        assert "marine" in generated or "cetaceans" in generated

        saved_nodes = arena.node_count
        assert arena.save_snapshot(snap_file) is True

    with RustMluArena() as restored_arena:
        assert restored_arena.load_snapshot(snap_file) is True
        assert restored_arena.node_count == saved_nodes
        assert restored_arena.mitosis_count >= 1
        restored_preds = restored_arena.predict_spoke("DOLPHIN", spoke_idx=0, top_k=5)
        assert len(restored_preds) >= 1
        assert any(p["target_concept"] == "CETACEAN" for p in restored_preds)


def test_spiderweb_mlm_corpus_ingestion_and_consolidation(tmp_path: object) -> None:
    """Phase 1-3: SpiderWebMLM ingests a multi-file directory across epochs, runs Rust sleep consolidation, and persists Rust arena state in SQLite."""
    from pathlib import Path

    corpus_dir = Path(str(tmp_path)) / "docs"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "biology.md").write_text(
        "An albatross is a seabird. A seabird is a bird. A bird is an animal. "
        "An albatross has wingspan. An albatross can glide. An albatross lives in the ocean.\n",
        encoding="utf-8",
    )
    (corpus_dir / "physics.txt").write_text(
        "A laser is an optical_device. A laser produces coherent_light. "
        "A laser requires a gain_medium.\n",
        encoding="utf-8",
    )

    with MemoryStore(seed_ontology=False) as memory:
        mlm = SpiderWebMLM(
            memory=memory,
            mitosis_min_activations=3,
            mitosis_min_verified_spokes=2,
        )
        summary = mlm.ingest_corpus_path(str(corpus_dir), epochs=2)
        assert summary["files_processed"] == 2
        assert summary["epochs"] == 2
        assert len(summary["epoch_surprises"]) == 2
        assert summary["epoch_surprises"][1] <= summary["epoch_surprises"][0]
        assert "ALBATROSS" in summary["mitosis_promotions"]
        assert summary["consolidation"]["rust_consolidation"]["replayed_edges"] > 0

        # Query multi-hop 360° beam ("Is an albatross an animal?")
        q_res = mlm.query_360("Is an albatross an animal?")
        assert q_res.status == "VERIFIED"
        assert "animal" in q_res.answer.lower()

        # Reload from SQLite and verify the Rust arena snapshot was restored automatically
        mlm_reloaded = SpiderWebMLM(memory=memory)
        assert mlm_reloaded.rust_arena.node_count == mlm.rust_arena.node_count
        beam_dto = mlm_reloaded.rust_arena.query_360_beam("ALBATROSS", max_hops=3, top_k=20)
        assert beam_dto["focal_concept"] == "ALBATROSS"
        assert any(p["target_concept"] == "ANIMAL" for p in beam_dto["predictions"])
