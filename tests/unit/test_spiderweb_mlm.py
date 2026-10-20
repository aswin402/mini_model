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


def test_spiderweb_mlm_phase20_to_22_counterfactual_bridge_and_topology() -> None:
    """Phases 20-22: Native Rust counterfactual intervention, shortest relational bridge, and Mermaid topology export."""
    with MemoryStore(seed_ontology=False) as memory:
        mlm = SpiderWebMLM(memory=memory, mitosis_min_activations=2, mitosis_min_verified_spokes=2)
        mlm.train_on_text(
            "Drought causes crop_failure. Crop_failure causes famine. "
            "An orca is a cetacean. A dolphin is a cetacean.",
            persist=False,
        )

        # Phase 20: Counterfactual intervention suppressing CROP_FAILURE blocks FAMINE
        cf_dto = mlm.intervene_counterfactual_360("DROUGHT", "CROP_FAILURE", max_hops=3)
        assert cf_dto["root_concept"] == "DROUGHT"
        assert cf_dto["suppressed_concept"] == "CROP_FAILURE"
        assert "FAMINE" in cf_dto["invalidated_conclusions"]
        assert "invalidated" in cf_dto["counterfactual_synthesis"].lower()

        # Phase 21: Shortest bidirectional relational bridge between ORCA and DOLPHIN via CETACEAN
        bridge_dto = mlm.find_bridge_360("ORCA", "DOLPHIN", max_hops=4)
        assert bridge_dto["connected"] is True
        assert bridge_dto["hop_count"] == 2
        assert any(s["to_concept"] == "CETACEAN" for s in bridge_dto["bridge_steps"])
        assert "relational bridge" in bridge_dto["bridge_synthesis"].lower()

        # Phase 22: Export live 360° Spider-Web Arena topology and Mermaid graph
        topo_dto = mlm.export_topology_graph(max_edges=50)
        assert topo_dto["total_cells"] >= 9
        assert topo_dto["total_verified_edges"] >= 4
        assert "flowchart TD" in topo_dto["mermaid_diagram"]
        assert "CETACEAN" in topo_dto["mermaid_diagram"]
        mlm.close()


def test_spiderweb_mlm_phase23_to_25_latent_clusters_belief_revision_and_8_axis_benchmark() -> None:
    """Phases 23-25: Emergent latent cluster abstraction, non-monotonic belief revision, and 8-axis benchmark."""
    with MemoryStore(seed_ontology=False) as memory:
        mlm = SpiderWebMLM(memory=memory, mitosis_min_activations=2, mitosis_min_verified_spokes=2)
        mlm.train_on_text(
            "An orca is a cetacean. A dolphin is a cetacean. A whale is a cetacean. "
            "An albatross can glide.",
            persist=False,
        )

        # Phase 23: Discover emergent latent clusters sharing CETACEAN
        clusters = mlm.discover_clusters_360(min_members=2)
        assert len(clusters) > 0
        cetacean_cluster = next(
            (c for c in clusters if c["shared_attractor"] == "CETACEAN"), None
        )
        assert cetacean_cluster is not None
        assert cetacean_cluster["member_count"] >= 3
        assert cetacean_cluster["centroid_coherence"] > 0.0

        # Phase 24: Non-monotonic belief revision flipping ALBATROSS can GLIDE
        rev = mlm.revise_belief_360(
            "ALBATROSS", "can", "GLIDE", verified=False, evidence_weight=5.0
        )
        assert rev["subject_concept"] == "ALBATROSS"
        assert rev["old_polarity"] == 1
        assert rev["new_polarity"] == -1
        assert rev["polarity_flipped"] is True
        mlm.close()

    # Phase 25: Full 8-axis benchmark suite
    bench = SpiderWebMLM.run_benchmark_suite()
    assert bench["all_passed"] is True
    assert bench["comparative_360_passed"] is True
    assert bench["causal_counterfactual_passed"] is True
    assert bench["bridge_cluster_revision_passed"] is True


def test_spiderweb_mlm_phase26_to_28_unified_kernel_waves_and_lifecycle_persistence(tmp_path) -> None:
    """Verify Phases 26-28: Unified 360° wave routing (counterfactual, bridge, causal) and automatic close persistence."""
    db_file = str(tmp_path / "phase26_lifecycle.db")
    with MemoryStore(db_file) as mem:
        mlm = SpiderWebMLM(mem)
        mlm.train_on_text(
            "An orca is a mammal. A dolphin is a mammal. "
            "Drought causes crop_failure. Crop_failure causes famine. "
            "A bird can fly.",
            persist=True,
        )

        # Phase 26a: Counterfactual intervention via unified query_360 (negative polarity operator span)
        cf_res = mlm.query_360("What if drought is not crop_failure?")
        assert "[MLU-360-COUNTERFACTUAL]" in cf_res.inspectable_trace
        assert "FAMINE" in cf_res.fluent_english.upper()

        # Phase 26b: Comparative intersection via unified query_360 (shared attractor MAMMAL)
        cmp_res = mlm.query_360("orca and dolphin")
        assert "[MLU-360-COMPARE]" in cmp_res.inspectable_trace
        assert "mammal" in cmp_res.fluent_english.lower()

        # Phase 26c: Relational bridge discovery via unified query_360 (multi-hop DROUGHT -> CROP_FAILURE -> FAMINE)
        br_res = mlm.query_360("drought and famine")
        assert "[MLU-360-BRIDGE]" in br_res.inspectable_trace
        assert "crop failure" in br_res.fluent_english.lower()

        # Phase 26d: Causal wave synthesis enriched in single-concept query_360
        causal_res = mlm.query_360("drought")
        assert "[MLU-360-CAUSAL-WAVE]" in causal_res.inspectable_trace
        assert "Causal wave from drought:" in causal_res.fluent_english

        # Phase 27: Non-monotonic revision dirty flag auto-persisted on close()
        rev = mlm.revise_belief_360("BIRD", "can", "FLY", verified=False, evidence_weight=5.0)
        assert rev["polarity_flipped"] is True
        assert mlm._dirty_since_save is True
        mlm.close()

    # Re-open on same SQLite database and verify the revised polarity was persisted on close()
    with MemoryStore(db_file) as mem2:
        mlm2 = SpiderWebMLM(mem2)
        bird_res = mlm2.query_360("bird")
        fly_preds = [p for p in bird_res.spoke_predictions if p.target_concept == "FLY"]
        assert not fly_preds
        mlm2.close()


def test_spiderweb_mlm_phase29_to_31_multiclause_coreference_and_deliberation(tmp_path) -> None:
    """Verify Phases 29-31: Multi-clause sentence decomposition, cross-sentence anaphoric coreference, and 5-wave System-2 deliberation."""
    db_file = str(tmp_path / "phase29_deliberation.db")
    with MemoryStore(db_file) as mem:
        mlm = SpiderWebMLM(mem)
        # Phase 29: Single compound sentence with 3 operator bridges + second sentence with low-entropy anaphor 'It'
        rep = mlm.train_on_text(
            "An eagle is a bird, has talons, and can soar. "
            "It requires thermals. "
            "A falcon is a bird, and can dive.",
            persist=True,
        )
        assert rep.relations_verified >= 5

        # Verify all 3 clauses from sentence 1 + coreference clause from sentence 2 were learned on EAGLE
        q_bird = mlm.query_360("Is an eagle a bird?")
        q_talons = mlm.query_360("Does an eagle have talons?")
        q_soar = mlm.query_360("Can an eagle soar?")
        assert q_bird.status == "VERIFIED"
        assert q_talons.status == "VERIFIED"
        assert q_soar.status == "VERIFIED"

        eagle_open = mlm.query_360("eagle")
        eagle_targets = {p.target_concept for p in eagle_open.spoke_predictions}
        assert {"BIRD", "TALONS", "SOAR"}.issubset(eagle_targets)

        # Phase 30: 5-Wave System-2 Deliberation on FALCON (discovers cluster peer EAGLE & analogical insights)
        delib = mlm.deliberate_360("FALCON", max_hops=3)
        assert delib["focal_concept"] == "FALCON"
        assert delib["waves_executed"] == 5
        assert "EAGLE" in delib["cluster_peers"]
        assert delib["deliberative_synthesis"] != ""
        assert "[MLU-360-DELIBERATE]" in eagle_open.inspectable_trace
        mlm.close()

    # Phase 31: Verify 9-axis benchmark suite including comprehension_deliberation_passed
    bench = SpiderWebMLM.run_benchmark_suite()
    assert bench["all_passed"] is True
    assert bench["comprehension_deliberation_passed"] is True


def test_spiderweb_mlm_phase32_to_34_goal_planning_and_conversational_chat_turn(tmp_path) -> None:
    """Verify Phases 32-34: 360° backward-chaining goal planner, terminal goal query recovery, and conversational chat turn engine."""
    db_file = str(tmp_path / "phase32_planning.db")
    with MemoryStore(db_file) as mem:
        mlm = SpiderWebMLM(mem)
        mlm.train_on_text(
            "Sunlight causes evaporation. Evaporation causes clouds. "
            "Clouds cause rain. Rain causes harvest.",
            persist=True,
        )

        # Phase 32: Multi-step backward-chaining goal plan from HARVEST back to SUNLIGHT
        plan = mlm.plan_goal_360("HARVEST", max_steps=6)
        assert plan["achievable"] is True
        assert plan["total_steps"] == 4
        assert plan["start_concept"] == "SUNLIGHT"
        assert "SUNLIGHT" in plan["root_prerequisites"]
        assert "Step 1: sunlight causes evaporation" in plan["plan_synthesis"]
        assert "Step 4: rain causes harvest" in plan["plan_synthesis"]

        # Targeted start-to-goal plan from EVAPORATION to HARVEST
        sub_plan = mlm.plan_goal_360("HARVEST", start_concept="EVAPORATION", max_steps=6)
        assert sub_plan["achievable"] is True
        assert sub_plan["total_steps"] == 3
        assert sub_plan["start_concept"] == "EVAPORATION"

        # Phase 33: Querying terminal goal concept HARVEST recovers backward-chained plan via [MLU-360-PLAN]
        q_harvest = mlm.query_360("harvest")
        assert q_harvest.status == "VERIFIED"
        assert "[MLU-360-PLAN]" in q_harvest.inspectable_trace
        assert "sunlight causes evaporation" in q_harvest.fluent_english.lower()

        # Phase 34: Unified conversational turn engine (teaching + deliberation + multi-turn coreference inquiry)
        t1 = mlm.chat_turn_360("A cheetah is a mammal, has spots, and can sprint.", persist=True)
        assert t1["mode"] == "LEARN_AND_DELIBERATE"
        assert t1["status"] == "VERIFIED"
        assert t1["relations_verified"] >= 3

        t2 = mlm.chat_turn_360("Can it sprint?", persist=True)
        assert t2["mode"] == "INQUIRY"
        assert t2["status"] == "VERIFIED"
        assert t2["focal_concept"] == "CHEETAH"
        mlm.close()

    bench = SpiderWebMLM.run_benchmark_suite()
    assert bench["all_passed"] is True
    assert bench["goal_planning_passed"] is True


def test_spiderweb_mlm_phase35_to_37_order_poset_and_domain_summary(tmp_path) -> None:
    """Verify Phases 35-37: Native Rust 360° Transitive Poset Ordering (I_SORT), Domain Summarizer, and 11-axis benchmark."""
    db_file = str(tmp_path / "phase35_order_summary.db")
    with MemoryStore(db_file) as mem:
        mlm = SpiderWebMLM(mem, mitosis_min_activations=2, mitosis_min_verified_spokes=2)
        mlm.train_on_text(
            "Sunlight causes evaporation. Evaporation causes clouds. "
            "Clouds cause rain. Rain causes harvest. "
            "An orca is a cetacean. A dolphin is a cetacean. "
            "An orca lives in the ocean. An orca can echolocate.",
            persist=True,
        )

        # Phase 35a: Transitive poset ordering along 'causes' starting from SUNLIGHT
        order = mlm.evaluate_order_360("causes", start_concept="SUNLIGHT", max_hops=6)
        assert order["poset_consistent"] is True
        assert order["element_count"] == 5
        assert order["ordered_chain"] == [
            "SUNLIGHT",
            "EVAPORATION",
            "CLOUDS",
            "RAIN",
            "HARVEST",
        ]
        assert "SUNLIGHT -> HARVEST" in order["transitive_pairs"]
        assert "SUNLIGHT -> CLOUDS" in order["transitive_pairs"]
        assert "Transitive order" in order["order_synthesis"]

        # Phase 35b: I_SORT poset antisymmetry gate rejects cyclic ordering (HARVEST causes SUNLIGHT)
        inv_sort = mlm.rust_arena.verify_invariants("HARVEST", "causes", "SUNLIGHT")
        assert inv_sort["passed"] is False
        assert inv_sort["gate"] == "I_SORT"

        # Phase 36: Native Rust 360° Domain & Sub-Core Knowledge Summarizer
        summary = mlm.summarize_domain_360("", max_items=6)
        assert summary["domain_name"] == "ALL_DOMAINS"
        assert summary["active_concepts_count"] >= 8
        assert summary["verified_synapses_count"] >= 8
        assert len(summary["central_hubs"]) >= 2
        assert len(summary["key_principles"]) >= 3
        assert "Domain ALL_DOMAINS encompasses" in summary["domain_synthesis"]
        mlm.close()

    # Phase 37: Verify 11-axis benchmark suite including order_and_summary_passed
    bench = SpiderWebMLM.run_benchmark_suite()
    assert bench["all_passed"] is True
    assert bench["order_and_summary_passed"] is True


def test_spiderweb_mlm_phase38_to_40_abductive_diagnosis_and_discourse(tmp_path) -> None:
    """Verify Phases 38-40: Native Rust 360° Abductive Root-Cause Diagnosis, Multi-Paragraph Discourse Synthesis, and 12-axis benchmark."""
    db_file = str(tmp_path / "phase38_abductive_discourse.db")
    with MemoryStore(db_file) as mem:
        mlm = SpiderWebMLM(mem, mitosis_min_activations=2, mitosis_min_verified_spokes=2)
        mlm.train_on_text(
            "An orca is a cetacean. A cetacean is a mammal. A mammal has warm_blood. "
            "An orca can echolocate. An orca lives in the ocean. "
            "A dolphin is a cetacean. "
            "Drought causes crop_failure. Crop_failure causes famine.",
            persist=True,
        )

        # Phase 38a: Abductive diagnosis across taxonomic/capability/spatial spokes
        diag = mlm.diagnose_abductive_360(
            ["ECHOLOCATE", "OCEAN", "WARM_BLOOD"], max_hops=3, top_k=5
        )
        assert diag["best_explanation"] == "ORCA"
        assert diag["best_coverage_ratio"] == 1.0
        assert len(diag["candidates"]) >= 1
        assert diag["candidates"][0]["observations_explained"] == 3
        assert "Abductive 360° diagnosis" in diag["diagnostic_synthesis"]

        # Phase 38b: Abductive root-cause diagnosis along causal chains
        causal_diag = mlm.diagnose_abductive_360(
            ["CROP_FAILURE", "FAMINE"], max_hops=3, top_k=5
        )
        assert causal_diag["best_explanation"] == "DROUGHT"
        assert causal_diag["best_coverage_ratio"] == 1.0

        # Phase 39: Multi-paragraph 360° discourse composition
        essay = mlm.compose_discourse_360("ORCA", max_hops=3)
        assert essay["focal_concept"] == "ORCA"
        assert essay["paragraph_count"] == 3
        assert len(essay["paragraphs"]) == 3
        assert "cetacean" in essay["paragraphs"][0].lower()
        assert "dolphin" in essay["paragraphs"][2].lower()
        assert "\n\n" in essay["full_discourse"]
        mlm.close()

    # Phase 40: Verify 12-axis benchmark suite including abductive_and_discourse_passed
    bench = SpiderWebMLM.run_benchmark_suite()
    assert bench["all_passed"] is True
    assert bench["abductive_and_discourse_passed"] is True


def test_spiderweb_mlm_phase41_to_43_conversational_diagnosis_and_vs_llm_scorecard() -> None:
    """Verify Phases 41-43: Natural-language abductive diagnosis routing in query_360 and head-to-head vs-LLM scorecard."""
    scorecard = SpiderWebMLM.run_vs_llm_scorecard()
    assert scorecard["all_passed"] is True
    metrics = scorecard["mlu_360_metrics"]
    assert metrics["contradiction_trap_immunity"] is True
    assert metrics["six_step_goal_planning_and_counterfactual"] is True
    assert metrics["conversational_abductive_diagnosis"] is True
    assert metrics["zero_hallucination_abstention"] is True
    assert metrics["instant_non_monotonic_belief_revision"] is True
    assert len(scorecard["vs_standard_llm_comparison"]) == 5


def test_spiderweb_mlm_phase44_to_46_json_ontology_ingestion_and_everyday_english(tmp_path) -> None:
    """Verify Phases 44-46: Structured JSON ontology ingestion, everyday English curriculum, and full conversational reasoning."""
    db_file = str(tmp_path / "phase44_full_english.db")
    with MemoryStore(db_file, seed_ontology=False) as mem:
        mlm = SpiderWebMLM(mem, mitosis_min_activations=2, mitosis_min_verified_spokes=2)

        # Phase 44: Ingest structured JSON ontology slice
        ont_rep = mlm.ingest_json_ontology("data/knowledge/commonsense.json", max_triples=120)
        assert ont_rep["triples_processed"] == 120
        assert ont_rep["relations_verified"] >= 100

        # Phase 45: Train on everyday English & human life corpora
        mlm.ingest_corpus_path("data/knowledge/world_corpus/human_life_and_emotions.md", epochs=1)
        mlm.ingest_corpus_path("data/knowledge/world_corpus/everyday_objects_and_actions.md", epochs=1)

        # Phase 46: Verify conversational reasoning across taxonomy, goal-planning, and disjointness
        t_human = mlm.chat_turn_360("Tell me about human", persist=False)
        assert t_human["status"] == "VERIFIED"
        assert "human" in t_human["reply"].lower()

        t_wisdom = mlm.chat_turn_360("How to achieve wisdom?", persist=False)
        assert t_wisdom["status"] == "VERIFIED"
        assert "wisdom" in t_wisdom["reply"].lower()
        assert "[MLU-360-PLAN]" in t_wisdom["trace"]

        t_mutex = mlm.chat_turn_360("Is a tiger a vehicle?", persist=False)
        assert t_mutex["status"] == "REFUTED"
        assert "I_MUTEX" in t_mutex["trace"]
        mlm.close()




