"""Integration tests for Autonomous Cobweb Concept Clustering and Latent Knot Induction."""

from __future__ import annotations

import pytest

from little.core.kernel import CognitiveKernel
from little.memory.store import MemoryStore


def test_latent_knot_induction_and_chat_integration() -> None:
    mem = MemoryStore(":memory:", seed_ontology=True)
    kernel = CognitiveKernel(memory=mem)

    # Ingest structured concepts
    c_eagle = mem.create_concept("eagle")
    c_falcon = mem.create_concept("falcon")
    c_hawk = mem.create_concept("hawk")
    c_bird = mem.get_concept("bird") or mem.create_concept("bird")
    c_fly = mem.get_concept("fly") or mem.create_concept("fly")
    c_hunt = mem.create_concept("hunt")

    for b in (c_eagle, c_falcon, c_hawk):
        mem.add_relation(b.id, "is_a", c_bird.id)
        mem.add_relation(b.id, "can", c_fly.id)
        mem.add_relation(b.id, "can", c_hunt.id)

    c_shark = mem.create_concept("shark")
    c_tuna = mem.create_concept("tuna")
    c_fish = mem.get_concept("fish") or mem.create_concept("fish")
    c_swim = mem.get_concept("swim") or mem.create_concept("swim")

    for f in (c_shark, c_tuna):
        mem.add_relation(f.id, "is_a", c_fish.id)
        mem.add_relation(f.id, "can", c_swim.id)

    # 1. Trigger autonomous concept clustering via chat
    reply_cluster = kernel.chat("cluster concepts")
    assert "Autonomous Cobweb clustering completed" in reply_cluster
    assert "Latent knots induced:" in reply_cluster

    # 2. Query discovered latent knots via chat
    reply_list = kernel.chat("what latent categories have you discovered?")
    assert "induced latent concept knot(s)" in reply_list
    assert "latent_" in reply_list

    # 3. Explain concept cluster membership via chat
    reply_explain = kernel.chat("explain concept cluster for eagle")
    assert "Eagle" in reply_explain
    assert "latent_" in reply_explain
    assert "Co-clustered peer concepts" in reply_explain

    # 4. Verify that the latent concept is now accessible for deductive reasoning
    all_concepts = mem.list_concepts()
    latent_knots = [c for c in all_concepts if c.name.startswith("latent_")]
    assert len(latent_knots) > 0
