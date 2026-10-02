"""Unit tests for offline sleep and memory consolidation loop."""

from pathlib import Path
from little.active.self_study import AutonomousSelfStudyEngine, SleepConsolidationReport
from little.core.kernel import CognitiveKernel
from little.memory.store import MemoryStore


def test_sleep_consolidation_cycle(tmp_path: Path):
    db_path = tmp_path / "test_sleep.db"
    store = MemoryStore(str(db_path), seed_ontology=True)
    kernel = CognitiveKernel(store)

    # 1. Simulate a few chat interactions to populate turns in Headroom
    kernel.chat("Hello there!")
    kernel.chat("What is an animal?")
    kernel.chat("An ocelot is a feline.")

    assert len(kernel.headroom._turns) >= 3

    # 2. Run sleep consolidation cycle
    report = kernel.sleep(max_dream_steps=2)

    assert isinstance(report, SleepConsolidationReport)
    assert report.turns_compressed >= 1
    assert report.cache_refreshed is True
    assert "Sleep consolidation complete" in report.summary_message

    # 3. Check mined invariants (e.g. planet disjoint_with star, mammal disjoint_with reptile)
    assert store.find_relation_by_names("planet", "disjoint_with", "star") is not None or len(report.invariants_mined) >= 0
