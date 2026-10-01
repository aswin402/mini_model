"""Integration tests for Autonomous Self-Study & Curiosity Loop (The Cognitive Dreamer) in CognitiveKernel chat."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from little.core.kernel import CognitiveKernel
from little.core.models import BeliefStatus
from little.memory.store import MemoryStore


def test_cognitive_dreamer_chat_cycle_and_epistemic_gaps():
    """Verify autonomous curiosity dream cycle, knowledge gap discovery, and living vault sync."""
    with tempfile.TemporaryDirectory() as tmp_vault:
        mem = MemoryStore(":memory:", seed_ontology=True)
        kernel = CognitiveKernel(memory=mem)

        # Redirect living wiki to temp vault directory
        kernel.wiki.vault_dir = Path(tmp_vault)
        kernel.wiki.build_graph()

        # Seed knowledge: falcon can fly, hawk is a bird without procedural spoke
        kernel.chat("A falcon is a bird.")
        kernel.chat("A falcon can fly.")
        kernel.chat("A hawk is a bird.")

        # 1. Query epistemic knowledge gaps via conversational chat
        reply_gaps = kernel.chat("what are your knowledge gaps?")
        assert "knowledge gap" in reply_gaps.lower()
        assert "radial entropy" in reply_gaps.lower() or "missing axes" in reply_gaps.lower()

        # 2. Targeted study of concept hawk via chat
        reply_study = kernel.chat("study concept hawk")
        assert "Autonomous curiosity study for 'hawk'" in reply_study
        assert "fly" in reply_study or "can" in reply_study

        # Verify that hawk can fly is now directly deducible in memory
        ans = kernel.chat("Can a hawk fly?")
        assert "can indeed fly" in ans or "yes" in ans.lower() or "hawk can fly" in ans.lower()

        # 3. Trigger full autonomous dreaming cycle via conversational chat
        # Introduce sparse concepts
        kernel.chat("A trout is a fish.")
        kernel.chat("A salmon is a fish.")
        kernel.chat("A salmon lives in water.")

        reply_dream = kernel.chat("run dream cycle")
        assert "Autonomous cognitive dream cycle completed" in reply_dream
        assert "Assimilated" in reply_dream or "novel claim" in reply_dream

        # 4. Verify Living Wiki vault synchronization
        # An OKF file should have been exported for hawk into the vault
        hawk_okf = Path(tmp_vault) / "hawk.okf.md"
        assert hawk_okf.exists()
        content = hawk_okf.read_text(encoding="utf-8")
        assert "hawk" in content.lower()


def test_cognitive_dreamer_prevents_hallucinations_via_mutex_gate():
    """Verify that autonomous dreaming rejects contradictory hypotheses (e.g. penguin flying)."""
    mem = MemoryStore(":memory:", seed_ontology=True)
    kernel = CognitiveKernel(memory=mem)

    # Eagle can fly
    kernel.chat("An eagle is a bird.")
    kernel.chat("An eagle can fly.")

    # Penguin is a bird, but explicitly cannot fly
    kernel.chat("A penguin is a bird.")
    kernel.chat("A penguin cannot fly.")

    # Targeted study on penguin must defend against false analogical transfer
    reply = kernel.chat("study concept penguin")
    assert "rejected" in reply.lower() or "defense" in reply.lower() or "no active knowledge gaps" in reply.lower() or "mutex" in reply.lower()

    # Epistemic invariant check: penguin still cannot fly
    ans = kernel.chat("Can a penguin fly?")
    assert "cannot fly" in ans or "no" in ans.lower() or "negative" in ans.lower()
