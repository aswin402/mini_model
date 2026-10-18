from dataclasses import replace

import pytest

from little.language.parser import LearningEngine
from little.memory.headroom import HeadroomContextManager
from little.memory.headroom_policy import HeadroomPolicy
from little.memory.paged_store import PagedConceptStore
from little.memory.store import MemoryStore


def test_compression_releases_text_and_eviction_remains_reversible(tmp_path):
    policy = replace(HeadroomPolicy.default(), max_active_tokens=20, max_working_episodes=3)
    text = "a detailed observation " * 100
    path = tmp_path / "conversation.db"
    with MemoryStore(path) as memory:
        manager = HeadroomContextManager(memory, policy)
        first = manager.record_turn("user", text)
        for _ in range(5):
            manager.record_turn("assistant", text)
        assert manager.total_active_tokens() <= 20
        assert len(manager._turns) <= 3
        assert first.is_compressed
        assert first.raw_text == ""
        assert first.tokens == ()
        assert manager.decompress_turn(first.turn_id) == text
        first_id = first.turn_id
    with MemoryStore(path) as reopened:
        assert HeadroomContextManager(reopened, policy).decompress_turn(first_id) == text


def test_oversized_latest_turn_is_bounded_and_persisted():
    with MemoryStore() as memory:
        manager = HeadroomContextManager(memory, replace(HeadroomPolicy.default(), max_active_tokens=5))
        turn = manager.record_turn("user", "word " * 100)
        assert manager.total_active_tokens() <= 5
        assert manager.decompress_turn(turn.turn_id) == "word " * 100


def test_headroom_rejects_negative_configuration():
    with pytest.raises(ValueError):
        replace(HeadroomPolicy.default(), max_active_tokens=-1)


def test_existing_concept_update_invalidates_cached_knot():
    with MemoryStore() as memory:
        learner = LearningEngine(memory)
        learner.learn("A falcon is a bird.")
        cache = PagedConceptStore(memory)
        before = cache.get_knot("falcon")
        learner.learn("A falcon is an animal.")
        after = cache.get_knot("falcon")
        assert after is not before
        assert "animal" in after.taxonomy_hypernyms
