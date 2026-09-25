"""Unit tests for SQLite persistent MemoryStore."""

from pathlib import Path

from little.memory.store import MemoryStore
from little.memory.export_policy import MemoryExportPolicy
from little.memory.memory_policy import MemoryPolicy


def test_memory_store_in_memory() -> None:
    store = MemoryStore(":memory:")
    c1 = store.create_concept("Dog", category="Animal")
    c2 = store.create_concept("Animal", category="Living Thing")

    assert c1.name == "dog"
    assert c2.name == "animal"

    # Verify duplicate create returns existing concept
    c1_dup = store.create_concept("dog")
    assert c1_dup.id == c1.id

    # Add relation
    rel = store.add_relation(c1.id, "is_a", c2.id)
    assert rel.subject_id == c1.id
    assert rel.object_id == c2.id
    assert rel.weight_positive == 1

    # Reinforce relation
    rel2 = store.add_relation(c1.id, "is_a", c2.id, positive=True)
    assert rel2.id == rel.id
    assert rel2.weight_positive == 2
    assert rel2.confidence > rel.confidence

    # Query relations
    rels = store.get_relations(subject_id=c1.id)
    assert len(rels) == 1
    assert rels[0].predicate == "is_a"

    store.close()


def test_memory_store_file_persistence(tmp_path: Path) -> None:
    db_file = tmp_path / "test_little.db"

    # Step 1: Open store, save data, and close
    with MemoryStore(db_file) as store1:
        store1.create_concept("Apple", category="Fruit", attributes={"edible": True})
        store1.add_experience(
            "The apple is red.",
            extracted_triples=[{"subject": "apple", "color": "red"}],
        )
        assert store1.get_concept("apple") is not None

    # Step 2: Open fresh store from the same file and verify persistence
    with MemoryStore(db_file) as store2:
        c_loaded = store2.get_concept("apple")
        assert c_loaded is not None
        assert c_loaded.name == "apple"
        assert c_loaded.category == "fruit"
        assert c_loaded.attributes["edible"] is True

        exps = store2.list_experiences()
        assert len(exps) == 1
        assert exps[0].input_text == "The apple is red."


def test_memory_export_uses_configurable_experience_limit():
    store = MemoryStore(":memory:")
    for index in range(3):
        store.add_experience(f"event {index}", [])

    policy = MemoryExportPolicy(version="test", experience_limit=1)
    store.export_policy = policy
    exported = store.export_state()

    assert len(exported["experiences"]) == 1


def test_memory_policy_controls_concept_and_query_defaults():
    policy = MemoryPolicy(
        version="test",
        concept_confidence=0.7,
        experience_query_limit=1,
        bulk_import_batch_size=2,
    )
    store = MemoryStore(":memory:", memory_policy=policy)

    concept = store.create_concept("policy-concept")
    for index in range(3):
        store.add_experience(f"event {index}", [])

    assert concept.confidence == 0.7
    assert len(store.list_experiences()) == 1
