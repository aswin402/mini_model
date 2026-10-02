"""Unit tests for ContinuousStreamLearner world corpus ingestion."""

from pathlib import Path
from little.active.stream_learner import StreamLearningSummary
from little.core.kernel import CognitiveKernel
from little.memory.store import MemoryStore


def test_world_corpus_ingestion(tmp_path: Path):
    db_path = tmp_path / "test_corpus.db"
    store = MemoryStore(str(db_path), seed_ontology=True)
    kernel = CognitiveKernel(store)

    # Ingest the world corpus
    summary = kernel.learn_world_corpus()

    assert isinstance(summary, StreamLearningSummary)
    assert summary.total_chunks >= 5
    assert summary.total_sentences >= 15
    assert summary.concepts_count >= 10
    assert summary.relations_count >= 15

    # Verify astronomy reasoning: Is Earth a celestial body? (Earth is planet -> planet is celestial body)
    earth_c = store.get_concept("earth")
    assert earth_c is not None

    reply = kernel.chat("Is Earth a celestial body?")
    assert "yes" in reply.lower() or "celestial" in reply.lower()
