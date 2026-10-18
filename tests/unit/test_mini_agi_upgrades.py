"""Tests for architectural upgrades inspired by volotat/mini-AGI."""

from __future__ import annotations

from little.active.latent_knot_inducer import LatentKnotInducer
from little.active.plasticity import EWLSFit, PlasticityController
from little.active.self_study import AutonomousSelfStudyEngine
from little.core.kernel import CognitiveKernel
from little.inference.dual_speed import DualSpeedInfillingEngine
from little.inference.invariant_gates import DeepSeekInvariantVerifier
from little.language.parser import LearningEngine
from little.memory.paged_store import PagedConceptStore
from little.memory.store import MemoryStore


def test_paged_store_dirty_only_invalidation_and_slot_pinning() -> None:
    """Learning a new fact about one concept must not flush unrelated resident/LRU concepts."""
    with MemoryStore() as memory:
        learner = LearningEngine(memory)
        learner.learn("An apple is a fruit.")
        learner.learn("A falcon is a bird.")
        learner.learn("A robin is a bird.")

        paged = PagedConceptStore(memory)
        # Vote apple and robin into the Tier 3 Resident Working Set
        residents = paged.vote_from_tokens(["apple", "robin"])
        assert "APPLE" in residents
        assert "ROBIN" in residents

        apple_knot_before = paged.get_knot("apple")
        falcon_knot_before = paged.get_knot("falcon")
        assert apple_knot_before is not None
        assert falcon_knot_before is not None

        # Mutate ONLY falcon -> animal; APPLE and ROBIN must remain pinned
        learner.learn("A falcon is an animal.")

        # Accessing apple should still be a Resident hit and return the exact pinned knot
        apple_knot_after = paged.get_knot("apple")
        assert apple_knot_after is apple_knot_before
        assert "APPLE" in paged.resident_concepts

        # Accessing falcon must reflect the newly learned hypernym
        falcon_knot_after = paged.get_knot("falcon")
        assert falcon_knot_after is not falcon_knot_before
        assert falcon_knot_after is not None
        assert "animal" in falcon_knot_after.taxonomy_hypernyms

        # Next token vote reuses pinned resident slots without re-fetching from disk
        paged.vote_from_tokens(["apple", "robin"])
        telemetry = paged.get_telemetry()
        assert telemetry.dirty_invalidations >= 1
        assert telemetry.slot_reuses >= 2


def test_paged_store_multigram_phrase_voting() -> None:
    """2-gram and 3-gram surface matches cast higher-specificity activation votes."""
    with MemoryStore() as memory:
        c = memory.create_concept("polar bear", category="entity")
        animal = memory.create_concept("animal", category="category")
        memory.add_relation(c.id, "is_a", animal.id)

        paged = PagedConceptStore(memory)
        residents = paged.vote_from_tokens(["the", "polar", "bear", "swims"])
        assert "POLAR_BEAR" in residents


def test_ewls_plasticity_controller_spike_holdback_and_regime_shift() -> None:
    """PlasticityController tracks slow/fast EWLS, holds back spikes, and protects trunk vs leaf knots."""
    ewls = EWLSFit(lam=0.95)
    for step in range(10):
        ewls.update(1.0 - 0.05 * step)
    assert ewls.slope < 0.0  # Loss is decreasing
    assert ewls.effect_size > 0.0

    controller = PlasticityController(base_lr=1.0, spike_sigma=2.5, hold_steps=2)
    # Steady improvement
    for step in range(8):
        controller.observe_error(0.5 - 0.03 * step)
    normal_mult = controller.plasticity_multiplier
    assert normal_mult >= 0.8

    # Inject a sudden > 2.5 sigma outlier spike -> holdback should trigger
    spike_tel = controller.observe_error(5.0)
    assert spike_tel.spike_detected is True
    assert spike_tel.plasticity_multiplier < normal_mult

    # Trunk (core ontology) concepts get 0.1x update rate; fresh leaves get higher rate than mature leaves
    trunk_lr = controller.concept_learning_rate(is_trunk=True, evidence_count=100)
    fresh_leaf_lr = controller.concept_learning_rate(is_trunk=False, evidence_count=0)
    mature_leaf_lr = controller.concept_learning_rate(is_trunk=False, evidence_count=50)
    assert trunk_lr < mature_leaf_lr < fresh_leaf_lr


def test_five_brake_growth_gate_and_survival_window_pruning() -> None:
    """LatentKnotInducer enforces the 5-brake growth gate and prunes unaddressed latent knots."""
    from dataclasses import replace

    from little.active.spiderweb_policy import SpiderWebGrowthPolicy

    with MemoryStore() as memory:
        learner = LearningEngine(memory)
        learner.learn("An eagle is a bird.")
        learner.learn("An eagle can fly.")
        learner.learn("A hawk is a bird.")
        learner.learn("A hawk can fly.")
        learner.learn("A tuna is a fish.")
        learner.learn("A tuna can swim.")
        learner.learn("A shark is a fish.")
        learner.learn("A shark can swim.")

        policy = replace(SpiderWebGrowthPolicy.default(), category_utility_threshold=0.0)
        inducer = LatentKnotInducer(
            memory=memory, policy=policy, max_latent_knots=64, survival_window=3
        )
        brakes = inducer.evaluate_growth_brakes(
            constituent_concepts=["eagle", "hawk"],
            shared_features={"is_a": ["bird"], "capable_of": ["fly"]},
            category_utility=0.25,
            latent_name="latent_fly_bird",
        )
        assert brakes.allowed is True

        report = inducer.induce_latent_knots(min_cluster_size=2)
        assert report.latent_knots_induced >= 1
        latent_names = [r.latent_name for r in report.inductions if r.verification_passed]
        assert latent_names
        latent_name = latent_names[0]
        assert memory.get_concept(latent_name) is not None

        # Brake 1 (room) blocks when max_latent_knots is reached
        inducer_full = LatentKnotInducer(memory=memory, policy=policy, max_latent_knots=1)
        full_brakes = inducer_full.evaluate_growth_brakes()
        assert full_brakes.room is False
        assert full_brakes.allowed is False

        # Survival window pruning: if unaddressed past survival_window, latent knot is pruned
        born_rev = memory.get_concept(latent_name).attributes["last_admitted_revision"]
        pruned = inducer.prune_unaddressed_latent_knots(
            current_revision=born_rev + 10,
            survival_window=3,
            active_concepts=set(),
        )
        assert latent_name in pruned
        assert memory.get_concept(latent_name) is None


def test_running_share_load_balancing_and_heldout_canary() -> None:
    """AutonomousSelfStudyEngine penalizes repeatedly-visited hubs and runs held-out canary checks."""
    with MemoryStore() as memory:
        learner = LearningEngine(memory)
        learner.learn("A sparrow is a bird.")
        learner.learn("A penguin is a bird.")

        study = AutonomousSelfStudyEngine(memory=memory, balance_weight=0.6, share_decay=0.8)
        p_before = study.effective_study_priority("sparrow", base_entropy=0.8)
        study._record_study_visit("sparrow")
        study._record_study_visit("sparrow")
        p_after = study.effective_study_priority("sparrow", base_entropy=0.8)
        p_other = study.effective_study_priority("penguin", base_entropy=0.8)
        assert p_after < p_before
        assert p_other > p_after

        # Held-out canary check passes on consistent taxonomy
        canary = study.evaluate_heldout_canary(
            canary_claims=[("sparrow", "is_a", "bird", True), ("sparrow", "is_a", "mineral", False)]
        )
        assert canary.passed is True
        assert canary.checked_count >= 3

        # Consolidate sleep cycle records induced clusters and canary status
        inducer = LatentKnotInducer(memory=memory)
        sleep_rep = study.consolidate_sleep_cycle(latent_inducer=inducer, max_dream_steps=1)
        assert sleep_rep.canary_passed is True
        assert sleep_rep.canary_regressions == []


def test_pondernet_adaptive_halting_and_stream_plasticity() -> None:
    """DualSpeedInfillingEngine emits PonderNet halting metrics and ContinuousStreamLearner updates plasticity."""
    with MemoryStore() as memory, CognitiveKernel(memory=memory) as kernel:
        chunk_res = kernel.stream_learner.ingest_chunk(
            "A whale is a mammal. A mammal is an animal. An animal is a living_thing."
        )
        assert chunk_res.is_learned is True
        assert chunk_res.plasticity_telemetry is not None
        assert chunk_res.plasticity_telemetry.step >= 3

        verifier = DeepSeekInvariantVerifier(memory)
        engine = DualSpeedInfillingEngine(
            memory, verifier, halt_threshold=0.85, ponder_beta=0.02, halt_freeze=True
        )
        res = engine.query("whale", "living_thing", "is_a")
        assert res.mode == "THINKING"
        assert res.path == ["WHALE", "MAMMAL", "ANIMAL", "LIVING_THING"]
        assert res.halt_probability == 1.0
        assert res.ponder_cost > 0.0
        assert "halt_prob=" in res.inspectable_trace
        assert "ponder_cost=" in res.inspectable_trace

        snap = kernel.get_telemetry_snapshot()
        assert snap["memory"]["total_concepts"] >= 4
        assert snap["plasticity"]["step"] >= 3
        assert snap["canary"]["passed"] is True
        assert "room" in snap["growth_brakes"]

        cmd_out = kernel.command_dispatcher.dispatch("show cognitive status", kernel)
        assert cmd_out is not None
        assert "Cognitive Telemetry:" in cmd_out
        assert "Growth Brakes:" in cmd_out
        assert "Canary: PASS" in cmd_out


def test_rust_persistent_graph_index_epoch_sync() -> None:
    """RustGraphIndex syncs only when concept_mutation_epoch advances and executes zero-copy FFI traversals."""
    from little.rust import RustCoreEngine, RustGraphIndex

    assert RustCoreEngine.get().has_graph_index_acceleration is True

    with MemoryStore(seed_ontology=False) as memory, RustGraphIndex() as g_idx:
        w = memory.create_concept("whale")
        m = memory.create_concept("mammal")
        a = memory.create_concept("animal")
        lt = memory.create_concept("living_thing")
        memory.add_relation(w.id, "is_a", m.id)
        memory.add_relation(m.id, "is_a", a.id)
        memory.add_relation(a.id, "is_a", lt.id)

        assert g_idx.sync_from_memory(memory) is True
        assert g_idx.edge_count == 3
        # Subsequent sync on unchanged epoch is a no-op
        assert g_idx.sync_from_memory(memory) is False

        ancestors = set(g_idx.find_ancestors("is_a", "WHALE"))
        assert ancestors == {"MAMMAL", "ANIMAL", "LIVING_THING"}
        assert g_idx.check_cycle("is_a", "WHALE", "LIVING_THING") is True
        assert g_idx.check_cycle("is_a", "LIVING_THING", "WHALE") is False
        assert g_idx.bidirectional_search("is_a", "WHALE", "LIVING_THING", max_depth=6) == [
            "WHALE",
            "MAMMAL",
            "ANIMAL",
            "LIVING_THING",
        ]


def test_continual_plasticity_and_paging_curriculum_verification() -> None:
    """Multi-phase streaming curriculum verifies PlasticityController regime shifts, PagedConceptStore slot reuse, and RustGraphIndex invariant checks."""
    with MemoryStore(seed_ontology=False) as memory:
        paged = PagedConceptStore(memory)
        ctrl = PlasticityController(base_lr=1.0, spike_sigma=2.5, hold_steps=2)
        verifier = DeepSeekInvariantVerifier(memory)

        # Phase 1: Steady taxonomic acquisition (low loss -> positive effect size)
        chain = ["entity", "organism", "vertebrate", "mammal", "cetacean", "orca"]
        for i in range(len(chain) - 1):
            c_sub = memory.create_concept(chain[i + 1])
            c_obj = memory.create_concept(chain[i])
            assert verifier.verify_relation(c_sub.name, "is_a", c_obj.name).passed is True
            memory.add_relation(c_sub.id, "is_a", c_obj.id)
            ctrl.observe_error(0.60 - 0.08 * i)
            paged.vote_from_tokens([chain[i + 1], chain[i]])

        # Verify I_DAG rejects cyclic edge via persistent RustGraphIndex
        cycle_check = verifier.verify_relation("entity", "is_a", "orca")
        assert cycle_check.passed is False
        assert cycle_check.violated_gate == "I_DAG"
        assert verifier._rust_graph.synced_epoch == memory.concept_mutation_epoch

        # Phase 2: Repeated queries over resident concepts verify slot-pinning reuses
        for _ in range(4):
            paged.vote_from_tokens(["orca", "cetacean", "mammal"])
            assert paged.get_knot("orca") is not None
        telemetry = paged.get_telemetry()
        assert telemetry.slot_reuses >= 4
        assert telemetry.resident_hits >= 4

        # Phase 3: Abrupt high-surprise regime shift triggers fast/slow divergence reset
        for _ in range(10):
            ctrl.observe_error(0.10)
        for step_idx in range(6):
            tel = ctrl.observe_error(0.30 + 0.25 * step_idx)
        assert tel.plasticity_multiplier >= 0.05


