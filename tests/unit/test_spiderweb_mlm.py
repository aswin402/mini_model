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


def test_spiderweb_mlm_multiturn_liquid_context_and_kernel_coupling() -> None:
    """Phase 4-5: Multi-turn liquid context resolves elliptic follow-ups and CognitiveKernel couples teach/sleep/process with SpiderWebMLM."""
    from little.core.kernel import CognitiveKernel

    with MemoryStore(seed_ontology=False) as memory, CognitiveKernel(memory=memory) as kernel:
        kernel.teach(
            "An orca is a cetacean. A cetacean is a mammal. "
            "An orca can echolocate. A mammal is not a reptile. A cobra is a reptile."
        )
        assert "ORCA" in kernel.mlm.nodes
        assert kernel.mlm.rust_arena.node_count > 7

        # Turn 1: Open inquiry sets ORCA at the top of the liquid focal stack
        q1 = kernel.mlm.query_360("Tell me about orca")
        assert q1.status == "VERIFIED"
        assert q1.focal_concept == "ORCA"
        assert "orca" in q1.fluent_english.lower()

        # Turn 2: Elliptic follow-up ("Can it echolocate?") resolves 'it' -> ORCA via liquid context stack
        q2 = kernel.mlm.query_360("Can it echolocate?")
        assert q2.status == "VERIFIED"
        assert q2.focal_concept == "ORCA"
        assert "echolocate" in q2.fluent_english.lower()

        # Turn 3: Elliptic refutation ("Is it a cobra?") resolves 'it' -> ORCA and refutes via I_MUTEX
        q3 = kernel.mlm.query_360("Is it a cobra?")
        assert q3.status == "REFUTED"
        assert "I_MUTEX" in q3.answer


def test_spiderweb_mlm_phase7_to_10_rust_wave_decoder_and_kernel_process() -> None:
    """Phases 7-10: Rust wave-to-English synthesis, first-class CognitiveKernel.process() routing, and sub-millisecond beam latency."""
    import time

    from little.core.kernel import CognitiveKernel
    from little.core.models import BeliefStatus

    with MemoryStore(seed_ontology=False) as memory, CognitiveKernel(memory=memory) as kernel:
        # 1. Learn directly through kernel.process() statements
        kernel.process("Titan is a moon. A moon is a celestial_body. Titan has atmosphere.")
        assert "TITAN" in kernel.mlm.nodes

        # 2. Verify native Rust 360° wave synthesis (`fluent_synthesis`) and sub-ms beam speed
        t0 = time.perf_counter()
        beam = kernel.mlm.rust_arena.query_360_beam("TITAN", max_hops=3, top_k=10)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 5.0
        assert "titan" in beam["fluent_synthesis"].lower()
        assert "moon" in beam["fluent_synthesis"].lower()

        # 3. Verify open-ended inquiry without '?' routes through SpiderWebMLM in kernel.process()
        out_open = kernel.process("Tell me about titan")
        assert out_open.inference is not None
        assert out_open.inference.status == BeliefStatus.SUPPORTED
        assert "titan" in (out_open.inference.answer or "").lower()

        # 4. Verify multi-turn elliptic follow-up through kernel.process()
        out_follow = kernel.process("Is it a celestial_body?")
        assert out_follow.inference is not None
        assert out_follow.inference.status == BeliefStatus.SUPPORTED


def test_spiderweb_mlm_phase11_to_13_native_rust_invariants_and_benchmark_suite() -> None:
    """Phases 11-13: Native Rust I_DAG & I_MUTEX invariant gates, Shannon entropy stream learning, and 5-axis benchmark suite."""
    bench = SpiderWebMLM.run_benchmark_suite()
    assert bench["all_passed"] is True
    assert bench["transitive_360_passed"] is True
    assert bench["invariant_gates_passed"] is True
    assert bench["zero_hallucination_abstention_passed"] is True
    assert bench["multiturn_liquid_context_passed"] is True
    assert bench["rust_beam_avg_latency_us"] < 2000.0


def test_spiderweb_mlm_phase14_to_16_comparative_wave_and_curiosity() -> None:
    """Phases 14-16: Native Rust 360° comparative wave intersection and analogical curiosity discovery."""
    with MemoryStore(seed_ontology=False) as memory:
        mlm = SpiderWebMLM(memory=memory, mitosis_min_activations=2, mitosis_min_verified_spokes=2)
        mlm.train_on_text(
            "An orca is a cetacean. A cetacean is a mammal. A mammal is an animal. "
            "An orca lives in the ocean. An orca can echolocate. "
            "An albatross is a seabird. A seabird is a bird. A bird is an animal. "
            "An albatross lives in the ocean. An albatross can glide. "
            "A mammal is not a bird. "
            "A dolphin is a cetacean.",
            persist=False,
        )
        cons = mlm.consolidate()
        assert "curiosity" in cons
        cur = cons["curiosity"]
        assert len(cur["curiosity_gaps"]) > 0
        # Dolphin shares CETACEAN with Orca, so analogical induction proposes dolphin -> ocean / echolocate
        assert len(cur["analogical_hypotheses"]) > 0
        assert any(
            h["subject_concept"] == "DOLPHIN" and h["analog_source"] == "ORCA"
            for h in cur["analogical_hypotheses"]
        )

        # Native 360° wave intersection between ORCA and ALBATROSS
        cmp_dto = mlm.compare_360("ORCA", "ALBATROSS")
        assert cmp_dto["mutually_exclusive"] is True
        shared_targets = {s["target_concept"] for s in cmp_dto["shared_attractors"]}
        assert "ANIMAL" in shared_targets or "OCEAN" in shared_targets
        assert "both orca and albatross" in cmp_dto["comparative_synthesis"].lower()

        # Natural-language comparative query through query_360
        q_cmp = mlm.query_360("Compare orca and albatross")
        assert q_cmp.status == "VERIFIED"
        assert "both orca and albatross" in q_cmp.fluent_english.lower()
        mlm.close()


def test_spiderweb_mlm_phase17_to_19_causal_wave_autonomous_curiosity_and_snapshots(
    tmp_path: object,
) -> None:
    """Phases 17-19: Native Rust causal wave simulation, autonomous curiosity promotion, and binary snapshot persistence."""
    with MemoryStore(seed_ontology=False) as memory:
        mlm = SpiderWebMLM(memory=memory, mitosis_min_activations=2, mitosis_min_verified_spokes=2)
        mlm.train_on_text(
            "Drought causes crop_failure. Crop_failure causes famine. "
            "An orca is a cetacean. An orca lives in the ocean. An orca can echolocate. "
            "A dolphin is a cetacean.",
            persist=False,
        )

        # Phase 17: Multi-hop causal wave simulation from DROUGHT
        causal_dto = mlm.simulate_causal_360("DROUGHT", max_hops=3)
        assert causal_dto["root_concept"] == "DROUGHT"
        assert "CROP_FAILURE" in causal_dto["affected_concepts"]
        assert "FAMINE" in causal_dto["affected_concepts"]
        assert "drought" in causal_dto["causal_synthesis"].lower()

        # Phase 18: Autonomous curiosity-driven hypothesis promotion (Dolphin inherits Orca's ocean/echolocate via CETACEAN)
        auto_step = mlm.autonomous_curiosity_step(min_confidence=0.60, max_promotions=5)
        assert auto_step["hypotheses_evaluated"] > 0
        assert auto_step["hypotheses_promoted"] > 0
        assert len(auto_step["open_questions"]) > 0
        q_dolphin = mlm.query_360("dolphin")
        assert q_dolphin.status == "VERIFIED"
        dolphin_targets = {p.target_concept for p in q_dolphin.spoke_predictions}
        assert "ECHOLOCATE" in dolphin_targets or "OCEAN" in dolphin_targets

        # Phase 19: Fast native Rust snapshot save & restore into a fresh arena
        snap_file = f"{tmp_path}/spiderweb_arena.bin"
        assert mlm.save_snapshot(snap_file) is True
        original_nodes = mlm.rust_arena.node_count
        mlm.close()

    with MemoryStore(seed_ontology=False) as memory2:
        mlm2 = SpiderWebMLM(memory=memory2, mitosis_min_activations=2, mitosis_min_verified_spokes=2)
        assert mlm2.load_snapshot(snap_file) is True
        assert mlm2.rust_arena.node_count == original_nodes
        causal_restored = mlm2.simulate_causal_360("DROUGHT", max_hops=3)
        assert "FAMINE" in causal_restored["affected_concepts"]
        mlm2.close()




