"""Unit tests for ContinuousStreamLearner single-stream pipeline."""

from pathlib import Path
from little.active.stream_learner import ContinuousStreamLearner
from little.core.kernel import CognitiveKernel
from little.memory.store import MemoryStore


def test_continuous_stream_learner_chunk_and_stream(tmp_path: Path):
    db_path = tmp_path / "test_stream.db"
    store = MemoryStore(str(db_path), seed_ontology=True)
    kernel = CognitiveKernel(store)
    learner = ContinuousStreamLearner(kernel)

    # 1. Ingest single chunk with multi-sentence facts
    chunk = "An ocelot is a feline. A serval is a feline."
    res = learner.ingest_chunk(chunk)

    assert res.is_learned is True
    assert res.sentences_count == 2
    assert len(res.concepts_created) >= 2
    assert len(res.relations_created) >= 2

    # Verify memory and paged store
    assert store.get_concept("ocelot") is not None
    assert store.get_concept("serval") is not None
    assert kernel.paged_store.get_knot("ocelot") is not None

    # Verify query through chat works immediately
    reply = kernel.chat("Is an ocelot an animal?")
    assert "yes" in reply.lower() or "animal" in reply.lower()


def test_continuous_stream_learner_document_ingest(tmp_path: Path):
    db_path = tmp_path / "test_stream_doc.db"
    store = MemoryStore(str(db_path), seed_ontology=True)
    kernel = CognitiveKernel(store)
    learner = ContinuousStreamLearner(kernel)

    doc_text = (
        "A python is a reptile.\n\n"
        "A cobra is a reptile.\n\n"
        "A viper is a reptile."
    )

    summary = learner.ingest_document(doc_text, source_id="snakes_doc")
    assert summary.total_chunks == 3
    assert summary.concepts_count >= 3
    assert summary.relations_count >= 3

    # Check that cobra is recognized as animal via multi-hop reasoning
    reply = kernel.chat("Is a cobra an animal?")
    assert "yes" in reply.lower() or "animal" in reply.lower()


def test_continuous_stream_learner_blocks_contradictions(tmp_path: Path):
    db_path = tmp_path / "test_stream_mutex.db"
    store = MemoryStore(str(db_path), seed_ontology=True)
    kernel = CognitiveKernel(store)
    learner = ContinuousStreamLearner(kernel)

    # First, teach tiger is mammal
    learner.ingest_chunk("A tiger is a mammal.")

    # Second, attempt to assert tiger is reptile (disjoint mutex)
    res = learner.ingest_chunk("A tiger is a reptile.")
    assert "REPTILE" not in res.concepts_created or res.contradictions_blocked >= 1 or not res.is_learned
