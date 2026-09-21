"""Unit tests for SQLite persistent MemoryStore."""

from pathlib import Path

from little.memory.store import MemoryStore


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
