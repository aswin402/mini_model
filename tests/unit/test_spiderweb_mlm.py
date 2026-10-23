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


def test_spiderweb_mlm_phase47_to_49_multiformat_ebooks_pdfs_csvs_and_hf_datasets(tmp_path) -> None:
    """Verify Phases 47-49: Multi-format document/dataset ingestion (.epub, .docx, .pdf, .csv, .jsonl) and cross-format reasoning."""
    db_file = str(tmp_path / "phase47_multiformat.db")
    with MemoryStore(db_file, seed_ontology=False) as mem:
        mlm = SpiderWebMLM(mem, mitosis_min_activations=2, mitosis_min_verified_spokes=2)

        # Phase 47a: Ingest EPUB, DOCX, PDF, and CSV in data/knowledge/ebooks
        eb_rep = mlm.ingest_corpus_path("data/knowledge/ebooks", epochs=1)
        assert eb_rep["files_processed"] == 4
        assert eb_rep["relations_verified"] >= 25

        # Phase 47b: Ingest HuggingFace JSONL science facts
        hf_rep = mlm.ingest_jsonl_dataset(
            "data/knowledge/hf_datasets/openbookqa_science_facts.jsonl", max_rows=20
        )
        assert hf_rep["rows_processed"] == 20
        assert hf_rep["relations_verified"] >= 10

        # Phase 49: Verify reasoning learned from EPUB, DOCX, PDF, and CSV
        q_epub = mlm.chat_turn_360("What explains dizziness and fatigue?", persist=False)
        assert q_epub["status"] == "VERIFIED"
        assert "dehydration" in q_epub["reply"].lower()

        q_docx = mlm.chat_turn_360("How to achieve happiness?", persist=False)
        assert q_docx["status"] == "VERIFIED"
        assert "empathy" in q_docx["reply"].lower()
        assert "[MLU-360-PLAN]" in q_docx["trace"]

        q_pdf = mlm.chat_turn_360("What explains power_outage and blown_fuse?", persist=False)
        assert q_pdf["status"] == "VERIFIED"
        assert "short circuit" in q_pdf["reply"].lower()

        q_csv_cmp = mlm.chat_turn_360("diamond and ruby", persist=False)
        assert q_csv_cmp["status"] == "VERIFIED"
        assert "mineral" in q_csv_cmp["reply"].lower()

        q_csv_mutex = mlm.chat_turn_360("Is a diamond a living_thing?", persist=False)
        assert q_csv_mutex["status"] == "REFUTED"
        assert "I_MUTEX" in q_csv_mutex["trace"]
        mlm.close()


def test_spiderweb_mlm_phase50_to_52_true_differentiable_neural_mlu_backprop(tmp_path) -> None:
    """Verify Phases 50-52: True end-to-end differentiable 360° Neural Spider-Web MLU trained via Backpropagation + AdamW."""
    db_file = str(tmp_path / "phase50_neural_mlu.db")
    corpus = (
        "is a tiger a mammal? yes a tiger is a warm blooded mammal.\n"
        "is a tiger a vehicle? no a tiger is a living animal not a vehicle.\n"
        "what causes rain? evaporation and clouds cause rain.\n"
        "how to achieve wisdom? curiosity and study lead to wisdom.\n"
    )
    with MemoryStore(db_file, seed_ontology=False) as mem:
        mlm = SpiderWebMLM(mem, mitosis_min_activations=2, mitosis_min_verified_spokes=2)
        assert mlm.neural_mlu.is_native is True

        rep = mlm.train_neural_mlu(corpus, epochs=45, learning_rate=0.04, persist=True)
        assert rep["sequences_trained"] == 180
        assert rep["final_loss"] < rep["initial_loss"] * 0.25
        assert rep["final_perplexity"] < rep["initial_perplexity"] * 0.35

        gen1 = mlm.generate_neural_mlu("is a tiger a mammal?", max_new_tokens=10)
        assert "yes" in gen1["generated_text"].lower()
        assert "mammal" in gen1["generated_text"].lower()
        assert len(gen1["active_spoke_weights"]) == 6
        assert len(gen1["dominant_logic_gates"]) == 6

        gen2 = mlm.generate_neural_mlu("is a tiger a vehicle?", max_new_tokens=10)
        assert "no" in gen2["generated_text"].lower()
        assert "vehicle" in gen2["generated_text"].lower()

        gen3 = mlm.generate_neural_mlu("what causes rain?", max_new_tokens=8)
        assert "rain" in gen3["generated_text"].lower()
        mlm.close()

    # Verify SQLite persistence and restoration of learned neural weights
    with MemoryStore(db_file, seed_ontology=False) as mem2:
        mlm2 = SpiderWebMLM(mem2, mitosis_min_activations=2, mitosis_min_verified_spokes=2)
        gen_restored = mlm2.generate_neural_mlu("how to achieve wisdom?", max_new_tokens=8)
        assert "wisdom" in gen_restored["generated_text"].lower()

        # Verify full English + Chain-of-Thought Thinking Curriculum training & chat_turn_360 integration
        cur_rep = mlm2.train_neural_curriculum(epochs=55, learning_rate=0.04, persist=False)
        assert cur_rep["final_perplexity"] < 1.25
        chat_hello = mlm2.chat_turn_360("hello how are you?", persist=False)
        assert chat_hello["status"] == "VERIFIED"
        assert "greeting" in chat_hello["neural_thinking"].lower()
        assert "hello" in chat_hello["neural_answer"].lower()
        assert "[NEURAL-THINK]" in chat_hello["trace"]
        mlm2.close()


def test_spiderweb_mlm_phase53_zero_regex_and_live_neural_distillation(tmp_path) -> None:
    """Verify Phase 53: Zero regex in MLU/Kernel pipeline, schema-driven operator codebook, and live Backprop distillation during chat."""
    from pathlib import Path

    # 1. Verify zero `import re` across active MLU + SemanticFramePolicy + CognitiveKernel modules
    for rel_path in (
        "src/little/mlu/spiderweb_net.py",
        "src/little/mlu/unit.py",
        "src/little/neural/semantic_frame_policy.py",
        "src/little/core/kernel.py",
    ):
        src = Path(rel_path).read_text(encoding="utf-8")
        assert "import re" not in src, f"Found forbidden regex import in {rel_path}"

    # 2. Verify live conversational learning distills verified 360° triples directly into NeuralSpiderWebMlu via Backprop + AdamW
    db_file = str(tmp_path / "phase53_live_distill.db")
    with MemoryStore(db_file, seed_ontology=False) as mem:
        mlm = SpiderWebMLM(mem, mitosis_min_activations=2, mitosis_min_verified_spokes=2)
        learn_turn = mlm.chat_turn_360("A peregrine is a bird, and can dive.", persist=False)
        assert learn_turn["mode"] == "LEARN_AND_DELIBERATE"
        assert learn_turn["status"] == "VERIFIED"
        assert learn_turn["relations_verified"] >= 2
        assert learn_turn["neural_sequences_distilled"] >= 84  # 3 synthesized CoT sequences * 28 epochs
        assert learn_turn["neural_final_perplexity"] < 1.5

        # Querying NeuralSpiderWebMlu directly should now produce both neural_thinking and neural_answer for peregrine!
        gen_dive = mlm.generate_neural_mlu("can a peregrine dive?", max_new_tokens=20)
        assert "procedural" in gen_dive["neural_thinking"].lower()
        assert "yes" in gen_dive["neural_answer"].lower()
        assert "dive" in gen_dive["neural_answer"].lower()

        ask_turn = mlm.chat_turn_360("Can a peregrine dive?", persist=False)
        assert ask_turn["status"] == "VERIFIED"
        assert "procedural" in ask_turn["neural_thinking"].lower()
        assert "[NEURAL-THINK]" in ask_turn["trace"]
        mlm.close()


def test_spiderweb_mlm_phase54_kaggle_and_huggingface_neural_training(tmp_path) -> None:
    """Verify Phase 54: Multi-dataset training across Kaggle CSVs (taxonomy, medical symptoms, DataScience Q&A) and HuggingFace JSONLs (SciQ, OpenBookQA) with holographic replay."""
    db_file = str(tmp_path / "phase54_kaggle_hf.db")
    with MemoryStore(db_file, seed_ontology=False) as mem:
        mlm = SpiderWebMLM(mem, mitosis_min_activations=2, mitosis_min_verified_spokes=2)

        # 1. Ingest Kaggle Animal Kingdom Taxonomy CSV
        rep_tax = mlm.ingest_csv_dataset(
            "data/knowledge/kaggle_datasets/hasanarcas_animal-kingdom-taxonomy/taxonomy.csv",
            max_rows=25,
            persist=False,
        )
        assert rep_tax["rows_processed"] == 25
        assert rep_tax["relations_verified"] >= 120

        # 2. Ingest Kaggle Clinical Disease Symptoms Profile CSV
        rep_med = mlm.ingest_csv_dataset(
            "data/knowledge/kaggle_datasets/uom190346a_disease-symptoms-and-patient-profile-dataset/Disease_symptom_and_patient_profile_dataset.csv",
            max_rows=25,
            persist=False,
        )
        assert rep_med["rows_processed"] == 25
        assert rep_med["relations_verified"] >= 40

        # 3. Ingest Kaggle DataScience QA CSV
        rep_ds = mlm.ingest_csv_dataset(
            "data/knowledge/kaggle_datasets/nvnikhil0001_data-science-questions-and-answers/DataScience QA.csv",
            max_rows=12,
            persist=False,
        )
        assert rep_ds["rows_processed"] == 12
        assert rep_ds["neural_sequences_trained"] >= 300

        # 4. Ingest HuggingFace SciQ JSONL
        rep_sciq = mlm.ingest_jsonl_dataset(
            "data/knowledge/hf_datasets/allenai_sciq_default.jsonl",
            max_rows=12,
            persist=False,
        )
        assert rep_sciq["rows_processed"] == 12
        assert rep_sciq["neural_sequences_trained"] >= 300

        # 5. Co-train English Thinking Curriculum with Holographic Replay
        cur_rep = mlm.train_neural_curriculum(epochs=35, learning_rate=0.04, persist=False)
        assert cur_rep["final_perplexity"] < 1.65

        # Verify 5-hop Kaggle taxonomic deduction
        q_felis = mlm.chat_turn_360("Is felis an animalia?", persist=False)
        assert q_felis["status"] == "VERIFIED"
        assert "yes" in q_felis["reply"].lower()

        # Verify Kaggle clinical abductive diagnosis
        q_diag = mlm.chat_turn_360(
            "What explains fever, fatigue, and difficulty_breathing?", persist=False
        )
        assert q_diag["status"] == "VERIFIED"
        assert "[MLU-360-DIAGNOSE]" in q_diag["trace"]

        # Verify HuggingFace SciQ neural Chain-of-Thought + answer
        q_decay = mlm.chat_turn_360(
            "what is the least dangerous radioactive decay?", persist=False
        )
        assert q_decay["status"] == "VERIFIED"
        assert "alpha" in q_decay["reply"].lower()
        assert "radioactive" in q_decay["neural_thinking"].lower()

        # Verify Kaggle DataScience QA neural Chain-of-Thought + answer
        q_nlp = mlm.chat_turn_360(
            "what is natural language processing nlp?", persist=False
        )
        assert q_nlp["status"] == "VERIFIED"
        assert "artificial intelligence" in q_nlp["reply"].lower()

        # Verify conversational greeting & zero-hallucination abstention
        q_hi = mlm.chat_turn_360("hii", persist=False)
        assert q_hi["status"] == "VERIFIED"
        assert "hello" in q_hi["reply"].lower()

        q_unk = mlm.chat_turn_360("What is a hyper_tachyon_quasar?", persist=False)
        assert q_unk["status"] == "UNKNOWN"
        assert q_unk["confidence"] == 0.0
        mlm.close()


def test_spiderweb_mlm_phase55_arena_conditioned_neural_fusion(tmp_path) -> None:
    """Verify Phase 55: Deep Neural <-> 360° Arena Fusion (spoke prior gating + multi-hop proof distillation into NeuralSpiderWebMlu)."""
    db_file = str(tmp_path / "phase55_arena_neural_fusion.db")
    with MemoryStore(db_file, seed_ontology=False) as mem:
        mlm = SpiderWebMLM(mem, mitosis_min_activations=2, mitosis_min_verified_spokes=2)

        # Teach 1-hop statements forming a 4-hop taxonomic & property inheritance chain + disjointness + causal symptoms
        mlm.train_on_text(
            "A felis is a felidae. "
            "A felidae is a carnivora. "
            "A carnivora is a mammalia. "
            "A mammalia is an animalia. "
            "A mammalia has warm_blood. "
            "An animalia is not a plant. "
            "A fern is a plant. "
            "Influenza causes fever. "
            "Influenza causes cough. "
            "Influenza causes fatigue.",
            persist=False,
        )

        # 1. Ask a 4-hop taxonomic question never seen as a direct 1-hop triple
        q_tax = mlm.chat_turn_360("Is a felis an animalia?", persist=False)
        assert q_tax["status"] == "VERIFIED"
        assert "[ARENA-NEURAL-FUSION]" in q_tax["trace"]
        assert "multi hop wave traverses" in q_tax["neural_thinking"].lower()
        assert "felis" in q_tax["neural_thinking"].lower()
        assert "yes" in q_tax["neural_answer"].lower()
        assert "animalia" in q_tax["neural_answer"].lower()
        assert q_tax["active_spoke_weights"][0][0] == "CELL_0_TAXONOMY"

        # 2. Ask a 4-hop property inheritance question never seen as a direct 1-hop triple
        q_prop = mlm.chat_turn_360("Does a felis have warm_blood?", persist=False)
        assert q_prop["status"] == "VERIFIED"
        assert "[ARENA-NEURAL-FUSION]" in q_prop["trace"]
        assert "multi hop wave" in q_prop["neural_thinking"].lower()
        assert "warm blood" in q_prop["neural_answer"].lower()
        assert q_prop["active_spoke_weights"][0][0] == "CELL_1_MEREOLOGY"

        # 3. Ask a multi-hop I_MUTEX refutation question ("Is a felis a fern?")
        q_ref = mlm.chat_turn_360("Is a felis a fern?", persist=False)
        assert q_ref["status"] == "REFUTED"
        assert "[ARENA-NEURAL-FUSION]" in q_ref["trace"]
        assert "mutual exclusivity" in q_ref["neural_thinking"].lower()
        assert "no" in q_ref["neural_answer"].lower()
        assert q_ref["active_spoke_weights"][0][0] == "CELL_0_TAXONOMY"

        # 4. Ask a 3-observation abductive root-cause diagnosis question
        q_abd = mlm.chat_turn_360("What explains fever, cough, and fatigue?", persist=False)
        assert q_abd["status"] == "VERIFIED"
        assert "[ARENA-NEURAL-FUSION]" in q_abd["trace"]
        assert "abductive wave" in q_abd["neural_thinking"].lower()
        assert "influenza" in q_abd["neural_answer"].lower()
        assert q_abd["active_spoke_weights"][0][0] == "CELL_2_CAUSALITY"

        # 4. Verify that NeuralSpiderWebMlu has consolidated both multi-hop proofs into its own neural weights
        pure_neural_tax = mlm.generate_neural_mlu("Is a felis an animalia?", max_new_tokens=36)
        assert "multi hop wave" in pure_neural_tax["neural_thinking"].lower()
        assert "yes" in pure_neural_tax["neural_answer"].lower()
        assert "animalia" in pure_neural_tax["neural_answer"].lower()

        pure_neural_prop = mlm.generate_neural_mlu("Does a felis have warm_blood?", max_new_tokens=36)
        assert "warm blood" in pure_neural_prop["neural_answer"].lower()
        assert pure_neural_prop["mean_token_confidence"] > 0.85
        mlm.close()


def test_spiderweb_mlm_phase56_sleep_autonomous_multihop_neural_consolidation(tmp_path) -> None:
    """Verify Phase 56: Sleep-Cycle Autonomous Multi-Hop Neural Consolidation (consolidate -> NeuralSpiderWebMlu)."""
    db_file = str(tmp_path / "phase56_sleep_neural_consolidation.db")
    with MemoryStore(db_file, seed_ontology=False) as mem:
        mlm = SpiderWebMLM(mem, mitosis_min_activations=2, mitosis_min_verified_spokes=2)

        # Teach 1-hop facts only (no multi-hop questions asked yet!)
        mlm.train_on_text(
            "An orca is a cetacean. "
            "A cetacean is a mammal. "
            "A mammal is an animal. "
            "An orca can echolocate. "
            "A dolphin is a cetacean. "
            "Drought causes crop_failure. "
            "Crop_failure causes famine.",
            persist=False,
        )

        # Run sleep-cycle consolidation (discovers analogies, multi-hop chains, and distills into NeuralSpiderWebMlu)
        cons = mlm.consolidate()
        sleep_neural = cons["neural_sleep_consolidation"]
        assert sleep_neural["synthesized_sequences"] >= 5
        assert sleep_neural["final_perplexity"] < 1.5

        # Now query NeuralSpiderWebMlu ZERO-SHOT (without calling query_360 or chat_turn_360!)
        # 1. Multi-hop taxonomy ("is an orca an animal?")
        n_tax = mlm.generate_neural_mlu("is an orca an animal?", max_new_tokens=36)
        assert "multi hop taxonomy" in n_tax["neural_thinking"].lower()
        assert "yes" in n_tax["neural_answer"].lower()
        assert "animal" in n_tax["neural_answer"].lower()
        assert n_tax["active_spoke_weights"][0][0] == "CELL_0_TAXONOMY"

        # 2. Analogical curiosity promotion ("can a dolphin echolocate?")
        n_ana = mlm.generate_neural_mlu("can a dolphin echolocate?", max_new_tokens=36)
        assert "analogical" in n_ana["neural_thinking"].lower()
        assert "yes" in n_ana["neural_answer"].lower()
        assert "echolocate" in n_ana["neural_answer"].lower()
        assert n_ana["active_spoke_weights"][0][0] == "CELL_4_CAPABILITY"

        # 3. Multi-hop causal cascade ("what causes famine?")
        n_cau = mlm.generate_neural_mlu("what causes famine?", max_new_tokens=36)
        assert "causal wave" in n_cau["neural_thinking"].lower()
        assert "drought" in n_cau["neural_answer"].lower()
        assert n_cau["active_spoke_weights"][0][0] == "CELL_2_CAUSALITY"
        mlm.close()


def test_spiderweb_mlm_phase57_multiturn_working_memory_comparative_and_proof_explanation(
    tmp_path,
) -> None:
    """Verify Phase 57: Multi-Turn Episodic Working Memory, Proof Explanation ('Why?' / 'Why not?'), Elliptic Coreference, and Comparative/Contrastive Reasoning."""
    db_file = str(tmp_path / "phase57_multiturn_working_memory.db")
    with MemoryStore(db_file, seed_ontology=False) as mem:
        mlm = SpiderWebMLM(mem, mitosis_min_activations=2, mitosis_min_verified_spokes=2)

        mlm.train_on_text(
            "A felis is a felidae. "
            "A felidae is a carnivora. "
            "A carnivora is a mammalia. "
            "A mammalia is an animalia. "
            "A mammalia has warm_blood. "
            "A panthera is a felidae. "
            "An animalia is not a plant. "
            "A fern is a plant.",
            persist=False,
        )

        # Turn 1: 4-hop taxonomic verification
        t1 = mlm.chat_turn_360("Is a felis an animalia?", persist=False)
        assert t1["status"] == "VERIFIED"
        assert len(mlm.episodic_working_memory) == 1
        assert len(mlm.episodic_working_memory[-1]["proof_steps"]) == 4

        # Turn 2: Follow-up proof explanation ("Why?")
        t2 = mlm.chat_turn_360("Why?", persist=False)
        assert t2["status"] == "VERIFIED"
        assert "[MLU-360-EXPLAIN]" in t2["trace"]
        assert "episodic working memory traces multi hop proof" in t2["neural_thinking"].lower()
        assert "because felis is a felidae" in t2["reply"].lower()
        assert "mammalia is a animalia" in t2["reply"].lower()
        assert "felidae" in t2["neural_answer"].lower()

        # Turn 3: Elliptic pronoun follow-up ("Does it have warm_blood?")
        t3 = mlm.chat_turn_360("Does it have warm_blood?", persist=False)
        assert t3["status"] == "VERIFIED"
        assert t3["focal_concept"] == "FELIS"
        assert "warm blood" in t3["neural_answer"].lower()

        # Turn 4: Elliptic pronoun refutation ("Is it a fern?")
        t4 = mlm.chat_turn_360("Is it a fern?", persist=False)
        assert t4["status"] == "REFUTED"
        assert t4["focal_concept"] == "FELIS"
        assert "mutual exclusivity" in t4["neural_thinking"].lower()

        # Turn 5: Follow-up refutation explanation ("Why not?")
        t5 = mlm.chat_turn_360("Why not?", persist=False)
        assert t5["status"] == "VERIFIED"
        assert "[MLU-360-EXPLAIN]" in t5["trace"]
        assert "animalia is mutually exclusive with plant" in t5["reply"].lower()
        assert "mutually exclusive" in t5["neural_answer"].lower()

        # Turn 6: Multi-entity similarity comparison ("How are a felis and a panthera similar?")
        t6 = mlm.chat_turn_360("How are a felis and a panthera similar?", persist=False)
        assert t6["status"] == "VERIFIED"
        assert "[MLU-360-COMPARE]" in t6["trace"]
        assert "shared attractors" in t6["neural_thinking"].lower()
        assert "both felis and panthera" in t6["reply"].lower()
        assert "felidae" in t6["neural_answer"].lower()

        # Turn 7: Multi-entity contrastive comparison ("What is the difference between a felis and a fern?")
        t7 = mlm.chat_turn_360("What is the difference between a felis and a fern?", persist=False)
        assert t7["status"] == "VERIFIED"
        assert "[MLU-360-COMPARE]" in t7["trace"]
        assert "mutex=True" in t7["trace"]
        assert "comparative wave contrasts felis and fern" in t7["neural_thinking"].lower()
        assert "whereas fern is a plant" in t7["reply"].lower()
        mlm.close()


def test_spiderweb_mlm_phase58_active_epistemic_self_questioning_and_curiosity_loop(
    tmp_path,
) -> None:
    """Verify Phase 58: Active Epistemic Self-Questioning (ask_curiosity_question_360) and Elliptic User Answer Binding in chat_turn_360."""
    db_file = str(tmp_path / "phase58_curiosity_loop.db")
    with MemoryStore(db_file, seed_ontology=False) as mem:
        mlm = SpiderWebMLM(mem, mitosis_min_activations=2, mitosis_min_verified_spokes=2)

        mlm.train_on_text(
            "An orca is a cetacean. "
            "An orca can echolocate. "
            "A dolphin is a cetacean. "
            "A pangolin is a mammal.",
            persist=False,
        )

        # 1. Analogical Hypothesis Self-Questioning for DOLPHIN (discovers ECHOLOCATE from ORCA via CETACEAN)
        cur1 = mlm.ask_curiosity_question_360("DOLPHIN", persist=False)
        assert cur1["mode"] == "CURIOSITY_INQUIRY"
        assert cur1["slot_type"] == "ANALOGICAL_HYPOTHESIS"
        assert cur1["focal_concept"] == "DOLPHIN"
        assert cur1["candidate_target"] == "ECHOLOCATE"
        assert "echolocate" in cur1["question"].lower()
        assert "epistemic curiosity" in cur1["neural_thinking"].lower()

        # User answers with a single word "Yes" -> binds to pending slot, verifies, and distills into NeuralSpiderWebMlu
        r1 = mlm.chat_turn_360("Yes", persist=False)
        assert r1["mode"] == "CURIOSITY_RESOLVED"
        assert r1["status"] == "VERIFIED"
        assert r1["focal_concept"] == "DOLPHIN"
        assert r1["target_concept"] == "ECHOLOCATE"
        n_dolphin = mlm.generate_neural_mlu("can a dolphin echolocate?", max_new_tokens=28)
        assert "yes" in n_dolphin["neural_answer"].lower()
        assert "echolocate" in n_dolphin["neural_answer"].lower()

        # 2. Missing Spoke Self-Questioning for PANGOLIN on mereology spoke
        cur2 = mlm.ask_curiosity_question_360("PANGOLIN", expected_spoke="mereology", persist=False)
        assert cur2["slot_type"] == "MISSING_SPOKE"
        assert cur2["focal_concept"] == "PANGOLIN"
        assert cur2["expected_spoke"] == "mereology"
        assert cur2["default_predicate"] == "has_part"
        assert "pangolin" in cur2["question"].lower()

        # User answers with bare concept phrase "keratin_scales" (no verb!) -> binds to PANGOLIN has_part KERATIN_SCALES
        r2 = mlm.chat_turn_360("keratin_scales", persist=False)
        assert r2["mode"] == "CURIOSITY_RESOLVED"
        assert r2["status"] == "VERIFIED"
        assert r2["focal_concept"] == "PANGOLIN"
        assert r2["target_concept"] == "KERATIN_SCALES"

        q_scales = mlm.chat_turn_360("Does a pangolin have keratin_scales?", persist=False)
        assert q_scales["status"] == "VERIFIED"
        assert "keratin scales" in q_scales["reply"].lower()

        # 3. Missing Spoke Self-Questioning for PANGOLIN on procedural spoke
        cur3 = mlm.ask_curiosity_question_360(
            "PANGOLIN", expected_spoke="procedural", persist=False
        )
        assert cur3["expected_spoke"] == "procedural"
        assert cur3["default_predicate"] == "can"
        r3 = mlm.chat_turn_360("curl_into_ball", persist=False)
        assert r3["status"] == "VERIFIED"
        n_curl = mlm.generate_neural_mlu("can a pangolin curl into ball?", max_new_tokens=28)
        assert "yes" in n_curl["neural_answer"].lower()
        assert "curl into ball" in n_curl["neural_answer"].lower()
        mlm.close()


def test_spiderweb_mlm_phase59_transitive_poset_dynamics_and_isort_neural_fusion(
    tmp_path,
) -> None:
    """Verify Phase 59: Transitive Poset Order (evaluate_order_360), I_SORT Antisymmetry Refutation, CELL_5_DYNAMICS Neural Fusion, and 'Why not?' Proof Explanation."""
    db_file = str(tmp_path / "phase59_poset_dynamics.db")
    with MemoryStore(db_file, seed_ontology=False) as mem:
        mlm = SpiderWebMLM(mem, mitosis_min_activations=2, mitosis_min_verified_spokes=2)

        mlm.chat_turn_360(
            "Jupiter is larger than Saturn. "
            "Saturn is larger than Earth. "
            "Earth is larger than Mars.",
            persist=False,
        )

        # 1. Multi-hop transitive poset verification (JUPITER > SATURN > EARTH > MARS)
        q1 = mlm.chat_turn_360("Is jupiter larger than mars?", persist=False)
        assert q1["status"] == "VERIFIED"
        assert "[MLU-360-ORDER]" in q1["trace"]
        assert "yes" in q1["reply"].lower()
        assert "mars" in q1["neural_answer"].lower()
        assert "poset" in q1["neural_thinking"].lower() or "dynamics" in q1["neural_thinking"].lower()
        top_cell_1, top_w_1 = q1["active_spoke_weights"][0]
        assert top_cell_1 == "CELL_5_DYNAMICS"
        assert top_w_1 > 0.25

        # 2. Multi-turn "Why?" proof explanation for transitive poset chain
        q2 = mlm.chat_turn_360("Why?", persist=False)
        assert q2["status"] == "VERIFIED"
        assert "[MLU-360-EXPLAIN]" in q2["trace"]
        assert "because jupiter larger than saturn" in q2["reply"].lower()
        assert "saturn larger than earth" in q2["reply"].lower()
        assert "earth larger than mars" in q2["reply"].lower()

        # 3. Multi-hop I_SORT antisymmetry refutation (MARS larger_than JUPITER is refuted by JUPITER > ... > MARS)
        q3 = mlm.chat_turn_360("Is mars larger than jupiter?", persist=False)
        assert q3["status"] == "REFUTED"
        assert "I_SORT" in q3["trace"]
        assert "mars cannot larger than jupiter" in q3["reply"].lower()
        assert "poset antisymmetry gate refutes" in q3["neural_thinking"].lower()
        assert "no" in q3["neural_answer"].lower()
        top_cell_3, top_w_3 = q3["active_spoke_weights"][0]
        assert top_cell_3 == "CELL_5_DYNAMICS"
        assert top_w_3 > 0.25

        # 4. Multi-turn "Why not?" proof explanation for I_SORT refutation
        q4 = mlm.chat_turn_360("Why not?", persist=False)
        assert q4["status"] == "VERIFIED"
        assert "[MLU-360-EXPLAIN]" in q4["trace"]
        assert "ref_status=REFUTED" in q4["trace"]
        assert q4["focal_concept"] == "MARS"
        assert "because jupiter larger than saturn" in q4["reply"].lower()
        assert "because jupiter larger than saturn" in q4["neural_answer"].lower()
        mlm.close()


def test_spiderweb_mlm_phase60_non_monotonic_exception_override_and_belief_revision(
    tmp_path,
) -> None:
    """Verify Phase 60: Default Taxonomic Inheritance, Non-Monotonic Exception Overriding ('A penguin cannot fly'), Live Neural Belief Revision, and 'Why not?' Exception Proof Explanation."""
    db_file = str(tmp_path / "phase60_non_monotonic_exception.db")
    with MemoryStore(db_file, seed_ontology=False) as mem:
        mlm = SpiderWebMLM(mem, mitosis_min_activations=2, mitosis_min_verified_spokes=2)

        # 1. Teach default rule on BIRD and two subclasses (EAGLE, PENGUIN)
        mlm.chat_turn_360(
            "A bird is an animal. "
            "A bird can fly. "
            "An eagle is a bird. "
            "A penguin is a bird.",
            persist=False,
        )

        # Before exception is taught, PENGUIN inherits 'can fly' from BIRD by default
        q_before = mlm.chat_turn_360("Can a penguin fly?", persist=False)
        assert q_before["status"] == "VERIFIED"
        assert "yes" in q_before["reply"].lower()

        # 2. User teaches explicit non-monotonic exception in natural language
        rev_turn = mlm.chat_turn_360("Actually, a penguin cannot fly.", persist=False)
        assert rev_turn["mode"] == "LEARN_AND_DELIBERATE"
        assert rev_turn["relations_refuted"] >= 1

        # 3. Sibling EAGLE still inherits 'can fly' from BIRD
        q_eagle = mlm.chat_turn_360("Can an eagle fly?", persist=False)
        assert q_eagle["status"] == "VERIFIED"
        assert "yes" in q_eagle["reply"].lower()

        # 4. PENGUIN now overrides the inherited BIRD default via I_EXCEPTION and updates NeuralSpiderWebMlu
        q_after = mlm.chat_turn_360("Can a penguin fly?", persist=False)
        assert q_after["status"] == "REFUTED"
        assert "I_EXCEPTION" in q_after["trace"]
        assert "penguin cannot fly" in q_after["reply"].lower()
        assert "non monotonic exception gate refutes" in q_after["neural_thinking"].lower()
        assert "no" in q_after["neural_answer"].lower()
        top_cell, top_w = q_after["active_spoke_weights"][0]
        assert top_cell == "CELL_4_CAPABILITY"
        assert top_w > 0.25

        # 5. Multi-turn "Why not?" explains both the default inheritance and the local exception override
        q_why_not = mlm.chat_turn_360("Why not?", persist=False)
        assert q_why_not["status"] == "VERIFIED"
        assert "[MLU-360-EXPLAIN]" in q_why_not["trace"]
        assert "penguin is a bird" in q_why_not["reply"].lower()
        assert "bird can fly" in q_why_not["reply"].lower()
        assert "penguin cannot fly" in q_why_not["reply"].lower()
        assert "penguin cannot fly" in q_why_not["neural_answer"].lower()
        mlm.close()


def test_spiderweb_mlm_phase61_cross_domain_proportional_analogy_and_neural_fusion(
    tmp_path,
) -> None:
    """Verify Phase 61: Cross-Domain Proportional Analogy Solving ('A : B :: C : ?'), Multi-Hop Inherited Spoke Alignment, 64-D Vector Offset Scoring, Neural Fusion, and 'Why?' Proof Explanation."""
    db_file = str(tmp_path / "phase61_proportional_analogy.db")
    with MemoryStore(db_file, seed_ontology=False) as mem:
        mlm = SpiderWebMLM(mem, mitosis_min_activations=2, mitosis_min_verified_spokes=2)

        # 1. Teach cross-domain facts across aerial, marine, and mereological spokes
        mlm.chat_turn_360(
            "An eagle lives in the sky. "
            "An eagle has wings. "
            "A cetacean lives in the ocean. "
            "A dolphin is a cetacean. "
            "A fish has fins.",
            persist=False,
        )

        # 2. Multi-hop cross-domain spatial analogy: EAGLE : SKY :: DOLPHIN : ? (DOLPHIN -> CETACEAN -> OCEAN)
        q_spatial = mlm.chat_turn_360(
            "Eagle is to sky as dolphin is to what?", persist=False
        )
        assert q_spatial["status"] == "VERIFIED"
        assert q_spatial["focal_concept"] == "OCEAN"
        assert "[MLU-360-ANALOGY]" in q_spatial["trace"]
        assert "EAGLE:SKY :: DOLPHIN:OCEAN" in q_spatial["trace"]
        assert "ocean" in q_spatial["reply"].lower()
        assert "proportional analogy maps" in q_spatial["neural_thinking"].lower()
        assert "ocean" in q_spatial["neural_answer"].lower()
        top_cell, top_w = q_spatial["active_spoke_weights"][0]
        assert top_cell == "CELL_3_SPATIAL"
        assert top_w > 0.25

        # 3. Multi-turn "Why?" explains the combined analogy proof chain (EAGLE->SKY and DOLPHIN->CETACEAN->OCEAN)
        q_why = mlm.chat_turn_360("Why?", persist=False)
        assert q_why["status"] == "VERIFIED"
        assert "[MLU-360-EXPLAIN]" in q_why["trace"]
        assert "eagle located in sky" in q_why["reply"].lower()
        assert "dolphin is a cetacean" in q_why["reply"].lower()
        assert "cetacean located in ocean" in q_why["reply"].lower()
        assert "cetacean located in ocean" in q_why["neural_answer"].lower()

        # 4. Mereological cross-domain analogy: EAGLE : WINGS :: FISH : ? (FISH -> FINS)
        q_mereo = mlm.chat_turn_360(
            "Eagle is to wings as fish is to what?", persist=False
        )
        assert q_mereo["status"] == "VERIFIED"
        assert q_mereo["focal_concept"] == "FINS"
        assert "[MLU-360-ANALOGY]" in q_mereo["trace"]
        assert "EAGLE:WINGS :: FISH:FINS" in q_mereo["trace"]
        assert "fins" in q_mereo["reply"].lower()
        assert "fins" in q_mereo["neural_answer"].lower()
        top_cell_m, top_w_m = q_mereo["active_spoke_weights"][0]
        assert top_cell_m == "CELL_1_MEREOLOGY"
        assert top_w_m > 0.25
        mlm.close()


def test_spiderweb_mlm_phase62_compositional_set_intersection_and_neural_fusion(
    tmp_path,
) -> None:
    """Verify Phase 62: Compositional Set & Conjunctive Constraint Intersection ('Which mammal can echolocate?', 'Which bird cannot fly?', 'Which bird has wings and lives in Antarctica?'), Neural Fusion, and 'Why?' Proof Explanation."""
    db_file = str(tmp_path / "phase62_compositional_intersection.db")
    with MemoryStore(db_file, seed_ontology=False) as mem:
        mlm = SpiderWebMLM(mem, mitosis_min_activations=2, mitosis_min_verified_spokes=2)

        # 1. Teach multi-hop taxonomy + positive & negated spoke constraints
        mlm.chat_turn_360(
            "A mammal is an animal. "
            "A cetacean is a mammal. "
            "A dolphin is a cetacean. "
            "A dolphin can echolocate. "
            "A bird is an animal. "
            "A bird has wings. "
            "A bird can fly. "
            "An eagle is a bird. "
            "A penguin is a bird. "
            "A penguin lives in Antarctica.",
            persist=False,
        )
        mlm.chat_turn_360("Actually, a penguin cannot fly.", persist=False)

        # 2. Multi-hop taxonomic + capability intersection: "Which mammal can echolocate?" -> DOLPHIN
        q_mammal = mlm.chat_turn_360("Which mammal can echolocate?", persist=False)
        assert q_mammal["status"] == "VERIFIED"
        assert q_mammal["focal_concept"] == "DOLPHIN"
        assert "[MLU-360-INTERSECT]" in q_mammal["trace"]
        assert "match=DOLPHIN" in q_mammal["trace"]
        assert "dolphin is a mammal that can echolocate" in q_mammal["reply"].lower()
        assert "compositional set intersection filters" in q_mammal["neural_thinking"].lower()
        assert "dolphin" in q_mammal["neural_answer"].lower()
        top_cell, top_w = q_mammal["active_spoke_weights"][0]
        assert top_cell == "CELL_4_CAPABILITY"
        assert top_w > 0.25

        # 3. Multi-turn "Why?" explains both the 2-hop taxonomic lineage and the capability spoke
        q_why_mammal = mlm.chat_turn_360("Why?", persist=False)
        assert q_why_mammal["status"] == "VERIFIED"
        assert "[MLU-360-EXPLAIN]" in q_why_mammal["trace"]
        assert "dolphin is a cetacean" in q_why_mammal["reply"].lower()
        assert "cetacean is a mammal" in q_why_mammal["reply"].lower()
        assert "dolphin can echolocate" in q_why_mammal["reply"].lower()

        # 4. Negated constraint intersection: "Which bird cannot fly?" -> PENGUIN
        q_neg = mlm.chat_turn_360("Which bird cannot fly?", persist=False)
        assert q_neg["status"] == "VERIFIED"
        assert q_neg["focal_concept"] == "PENGUIN"
        assert "[MLU-360-INTERSECT]" in q_neg["trace"]
        assert "penguin is a bird that cannot fly" in q_neg["reply"].lower()
        assert "penguin" in q_neg["neural_answer"].lower()

        # 5. Conjunctive multi-constraint intersection (inherited mereology + direct spatial):
        # "Which bird has wings and lives in Antarctica?" -> PENGUIN
        q_conj = mlm.chat_turn_360(
            "Which bird has wings and lives in Antarctica?", persist=False
        )
        assert q_conj["status"] == "VERIFIED"
        assert q_conj["focal_concept"] == "PENGUIN"
        assert "[MLU-360-INTERSECT]" in q_conj["trace"]
        assert "penguin is a bird that has wings and lives in antarctica" in q_conj["reply"].lower()
        assert "penguin" in q_conj["neural_answer"].lower()

        # 6. Multi-turn "Why?" on the conjunctive query explains taxonomy + inherited mereology + spatial spoke
        q_why_conj = mlm.chat_turn_360("Why?", persist=False)
        assert q_why_conj["status"] == "VERIFIED"
        assert "penguin is a bird" in q_why_conj["reply"].lower()
        assert "bird has part wings" in q_why_conj["reply"].lower()
        assert "penguin located in antarctica" in q_why_conj["reply"].lower()
        mlm.close()


def test_spiderweb_mlm_phase63_causal_cascade_counterfactual_intervention_and_neural_fusion(
    tmp_path,
) -> None:
    """Verify Phase 63: Multi-Hop Causal Cascade Simulation, Single- & Two-Concept Counterfactual Intervention ('What happens if sunlight is removed?'), CELL_2_CAUSALITY Neural Fusion, and 'Why?' Causal Proof Explanation."""
    db_file = str(tmp_path / "phase63_causal_counterfactual.db")
    with MemoryStore(db_file, seed_ontology=False) as mem:
        mlm = SpiderWebMLM(mem, mitosis_min_activations=2, mitosis_min_verified_spokes=2)

        # 1. Teach a 2-hop causal cascade: SUNLIGHT -(causes)-> PHOTOSYNTHESIS -(causes)-> OXYGEN
        mlm.chat_turn_360(
            "Sunlight causes photosynthesis. "
            "Photosynthesis causes oxygen.",
            persist=False,
        )

        # 2. Forward multi-hop causal wave inquiry
        q_wave = mlm.chat_turn_360("What does sunlight cause?", persist=False)
        assert q_wave["status"] == "VERIFIED"
        assert "[MLU-360-CAUSAL-WAVE]" in q_wave["trace"]
        assert "photosynthesis" in q_wave["reply"].lower()
        assert "oxygen" in q_wave["reply"].lower()
        assert "multi hop causal wave propagates" in q_wave["neural_thinking"].lower()
        top_cell_w, top_w_w = q_wave["active_spoke_weights"][0]
        assert top_cell_w == "CELL_2_CAUSALITY"
        assert top_w_w > 0.25

        # 3. Single-concept counterfactual root removal: "What happens if sunlight is removed?"
        q_cf = mlm.chat_turn_360(
            "What happens if sunlight is removed?", persist=False
        )
        assert q_cf["status"] == "VERIFIED"
        assert q_cf["focal_concept"] == "SUNLIGHT"
        assert "[MLU-360-COUNTERFACTUAL]" in q_cf["trace"]
        assert "photosynthesis" in q_cf["reply"].lower()
        assert "oxygen" in q_cf["reply"].lower()
        assert "invalidated" in q_cf["reply"].lower()
        assert "counterfactual intervention suppresses causal cascade" in q_cf["neural_thinking"].lower()
        assert "invalidated" in q_cf["neural_answer"].lower()
        top_cell_cf, top_w_cf = q_cf["active_spoke_weights"][0]
        assert top_cell_cf == "CELL_2_CAUSALITY"
        assert top_w_cf > 0.25

        # 4. Multi-turn "Why?" explains the full multi-hop causal cascade proof
        q_why = mlm.chat_turn_360("Why?", persist=False)
        assert q_why["status"] == "VERIFIED"
        assert "[MLU-360-EXPLAIN]" in q_why["trace"]
        assert "sunlight causes photosynthesis" in q_why["reply"].lower()
        assert "photosynthesis causes oxygen" in q_why["reply"].lower()
        assert "photosynthesis causes oxygen" in q_why["neural_answer"].lower()
        mlm.close()


def test_spiderweb_mlm_phase64_neural_embedding_sync_and_policy_driven_operators(
    tmp_path,
) -> None:
    """Verify Phase 64: Backprop-Learned Neural Embedding Synchronization into MiniLearningUnits, Policy-Driven Operator Polarity Resolution, and Zero Regex in MLU + Neural Modules."""
    from pathlib import Path

    db_file = str(tmp_path / "phase64_neural_sync.db")
    with MemoryStore(db_file, seed_ontology=False) as mem:
        mlm = SpiderWebMLM(mem, mitosis_min_activations=2, mitosis_min_verified_spokes=2)

        # 1. Verify chat_turn_360 distills into NeuralSpiderWebMlu and synchronizes learned embeddings back to Arena MiniLearningUnits
        mlm.chat_turn_360(
            "A reptile is an animal. "
            "A snake is a reptile. "
            "A reptile has legs.",
            persist=False,
        )
        assert "SNAKE" in mlm.nodes
        assert mlm.nodes["SNAKE"].neural_sync_steps >= 1
        assert len(mlm._cached_neural_vocab) > 0

        # 2. Policy-driven non-monotonic exception via 'without' (no hardcoded tuple in spiderweb_net.py)
        rev = mlm.chat_turn_360("However, a snake is without legs.", persist=False)
        assert rev["relations_refuted"] >= 1
        q_legs = mlm.chat_turn_360("Does a snake have legs?", persist=False)
        assert q_legs["status"] == "REFUTED"
        assert "I_EXCEPTION" in q_legs["trace"]
        assert "does not have legs" in q_legs["reply"].lower()

        # 3. Verify zero 'import re' in src/little/mlu/ and src/little/neural/
        repo_root = Path(__file__).resolve().parents[2]
        for pkg in ("mlu", "neural"):
            for py_file in (repo_root / "src" / "little" / pkg).glob("*.py"):
                lines = [ln.strip() for ln in py_file.read_text(encoding="utf-8").splitlines()]
                assert "import re" not in lines, f"Found 'import re' in {py_file}"

        mlm.close()


def test_spiderweb_mlm_phase65_dynamic_arena_conditioning_and_zero_regex_language_modules(
    tmp_path,
) -> None:
    """Verify Phase 65: Schema-Driven Spoke-to-Neural-Cell Conditioning & Zero Regex in Language Perception/Construction/Gatekeeper/Comparative Modules."""
    from pathlib import Path

    db_file = str(tmp_path / "phase65_arena_conditioning.db")
    with MemoryStore(db_file, seed_ontology=False) as mem:
        mlm = SpiderWebMLM(mem, mitosis_min_activations=2, mitosis_min_verified_spokes=2)

        # 1. Verify _predicate_to_neural_cell dynamically routes canonical & open predicates without hardcoded pred_to_cell dict
        assert mlm._predicate_to_neural_cell("is_a") == (0, "taxonomy")
        assert mlm._predicate_to_neural_cell("part_of") == (1, "mereology")
        assert mlm._predicate_to_neural_cell("causes") == (2, "causality")
        assert mlm._predicate_to_neural_cell("located_in") == (3, "spatial")
        assert mlm._predicate_to_neural_cell("can") == (4, "capability")
        assert mlm._predicate_to_neural_cell("before") == (5, "dynamics")

        # 2. Verify _build_arena_conditioning produces CoT conditioning prompt and 6-spoke neural cell prior
        mlm.chat_turn_360(
            "A bird is an animal. A bird has wings. A bird can fly.",
            persist=False,
        )
        q_res = mlm.query_360("Can a bird fly?")
        cot_prompt, spoke_prior = mlm._build_arena_conditioning("Can a bird fly?", q_res)
        assert isinstance(cot_prompt, str) and "think 360" in cot_prompt
        assert len(spoke_prior) == 6
        assert abs(sum(spoke_prior) - 1.0) < 1e-3
        assert spoke_prior[4] > spoke_prior[2]  # capability cell dominates causality cell

        # 3. Verify zero 'import re' in cleaned language modules
        repo_root = Path(__file__).resolve().parents[2]
        cleaned_language_files = (
            "laya_gatekeeper.py",
            "conjunction_policy.py",
            "number_policy.py",
            "question_boundary_policy.py",
            "perception.py",
            "comparative.py",
            "construction.py",
        )
        for fname in cleaned_language_files:
            py_file = repo_root / "src" / "little" / "language" / fname
            lines = [ln.strip() for ln in py_file.read_text(encoding="utf-8").splitlines()]
            assert "import re" not in lines, f"Found 'import re' in {py_file}"

        mlm.close()


def test_spiderweb_mlm_phase66_zero_regex_across_8_cognitive_packages(
    tmp_path,
) -> None:
    """Verify Phase 66: SchemaRegistry Axis-Driven Neural Cell Routing & Zero Regex Across 8 Full Packages (mlu, neural, reasoning, memory, perception, dynamics, knowledge, active) + core/models.py."""
    from pathlib import Path

    db_file = str(tmp_path / "phase66_zero_regex_packages.db")
    with MemoryStore(db_file, seed_ontology=False) as mem:
        mlm = SpiderWebMLM(mem, mitosis_min_activations=2, mitosis_min_verified_spokes=2)

        # 1. Verify SchemaRegistry axis-driven neural cell routing
        assert mlm._predicate_to_neural_cell("orbits") == (3, "spatial")
        assert mlm._predicate_to_neural_cell("cannot_be") == (0, "taxonomy")
        assert mlm._predicate_to_neural_cell("larger_than") == (5, "dynamics")

        # 2. Verify zero 'import re' across all 8 cognitive packages + core/models.py
        repo_root = Path(__file__).resolve().parents[2]
        zero_regex_packages = (
            "mlu",
            "neural",
            "reasoning",
            "memory",
            "perception",
            "dynamics",
            "knowledge",
            "active",
        )
        for pkg in zero_regex_packages:
            for py_file in (repo_root / "src" / "little" / pkg).glob("*.py"):
                lines = [ln.strip() for ln in py_file.read_text(encoding="utf-8").splitlines()]
                assert "import re" not in lines, f"Found 'import re' in {py_file}"

        models_py = repo_root / "src" / "little" / "core" / "models.py"
        models_lines = [ln.strip() for ln in models_py.read_text(encoding="utf-8").splitlines()]
        assert "import re" not in models_lines, f"Found 'import re' in {models_py}"

        mlm.close()


def test_spiderweb_mlm_phase67_zero_regex_across_12_cognitive_packages(
    tmp_path,
) -> None:
    """Verify Phase 67: Zero Regex Across 12 Full Packages (mlu, neural, reasoning, memory, perception, dynamics, knowledge, active, evaluation, core, actions, procedural)."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    zero_regex_packages = (
        "mlu",
        "neural",
        "reasoning",
        "memory",
        "perception",
        "dynamics",
        "knowledge",
        "active",
        "evaluation",
        "core",
        "actions",
        "procedural",
    )
    for pkg in zero_regex_packages:
        for py_file in (repo_root / "src" / "little" / pkg).glob("*.py"):
            lines = [ln.strip() for ln in py_file.read_text(encoding="utf-8").splitlines()]
            assert "import re" not in lines, f"Found 'import re' in {py_file}"

    # Verify CognitiveCommandRegistry, ActionRouter, AgenticPlanner, and MathStorySolver operate without regex
    db_file = str(tmp_path / "phase67_zero_regex.db")
    with MemoryStore(db_file, seed_ontology=True) as mem:
        from little.core.kernel import CognitiveKernel

        kernel = CognitiveKernel(mem, workspace_root=tmp_path)
        reply_calc = kernel.chat("calculate 250 * 4")
        assert "1000" in reply_calc

        reply_story = kernel.chat(
            "Sarah has 15 apples. She gives 4 apples to Bob, then buys 7 more apples. How many apples does Sarah have now?"
        )
        assert "18" in reply_story

        reply_status = kernel.chat("show cognitive status")
        assert "Cognitive Telemetry:" in reply_status
        kernel.mlm.close()


def test_spiderweb_mlm_phase68_zero_regex_across_entire_codebase(
    tmp_path,
) -> None:
    """Verify Phase 68: 100% Zero Regex Across the Entire src/little/ Codebase (All 13 Packages Including language/parser.py, dialogue_surface.py, dialogue_surface_policy.py, dialogue.py)."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    src_little = repo_root / "src" / "little"
    all_py_files = sorted(src_little.rglob("*.py"))
    assert len(all_py_files) >= 55

    for py_file in all_py_files:
        lines = [ln.strip() for ln in py_file.read_text(encoding="utf-8").splitlines()]
        assert "import re" not in lines, f"Found 'import re' in {py_file}"
        assert not any(
            ln.startswith("from re import") for ln in lines
        ), f"Found 'from re import' in {py_file}"

    # Verify end-to-end Language Parser, DialogueSurfaceEngine, DialogueContext, and SpiderWebMLM without regex
    db_file = str(tmp_path / "phase68_zero_regex_all.db")
    with MemoryStore(db_file, seed_ontology=True) as mem:
        from little.core.kernel import CognitiveKernel

        kernel = CognitiveKernel(mem, workspace_root=tmp_path)
        res_learn = kernel.chat("Hello! An axolotl is an amphibian, and can regenerate_limbs.")
        assert "Hello!" in res_learn
        assert "learned" in res_learn.lower()

        res_ask = kernel.chat("Is an axolotl an animal?")
        assert "yes" in res_ask.lower()
        assert "animal" in res_ask.lower()

        res_unk = kernel.chat("Is a hyper_quasar an animal?")
        assert "don't have that information" in res_unk.lower() or "unknown" in res_unk.lower()
        kernel.mlm.close()

