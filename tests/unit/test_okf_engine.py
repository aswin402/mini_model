"""Unit tests for the Google OKF (Open Knowledge Format) engine."""

import tempfile
from pathlib import Path
import pytest

from little.knowledge.okf import OKFEngine, parse_yaml_frontmatter, serialize_yaml_value
from little.memory.store import MemoryStore


def test_okf_yaml_roundtrip():
    """Verify clean parsing and serialization of nested frontmatter."""
    sample = """---
id: CONCEPT_APPLE
name: apple
aliases:
  - malus domestica
taxonomy:
  is_a:
    - fruit
    - plant
  disjoint_with:
    - animal
mereology:
  parts:
    - peel
    - pulp
attributes:
  freshness: 1.0
  edible: true
relations:
  - predicate: grows_on
    object: tree
    confidence: 1.0
---

# Apple

An **apple** is an edible [[fruit]] produced by an [[apple tree|Tree]].
"""
    fm, body = parse_yaml_frontmatter(sample)
    assert fm["id"] == "CONCEPT_APPLE"
    assert fm["name"] == "apple"
    assert "fruit" in fm["taxonomy"]["is_a"]
    assert fm["attributes"]["edible"] is True
    assert fm["relations"][0]["predicate"] == "grows_on"
    assert "[[fruit]]" in body


def test_okf_export_and_import():
    """Verify exporting a concept to OKF markdown and importing it into another store."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Create source store and populate
        store1 = MemoryStore()
        c_apple = store1.create_concept("apple", aliases=["orchard apple"], attributes={"color": "red"})
        c_fruit = store1.create_concept("fruit")
        c_tree = store1.create_concept("tree")
        store1.add_relation(c_apple.id, "is_a", c_fruit.id)
        store1.add_relation(c_apple.id, "grows_on", c_tree.id)

        engine1 = OKFEngine(store1)
        exported_path = engine1.export_concept("apple", output_dir=tmp_dir)

        assert exported_path.exists()
        content = exported_path.read_text(encoding="utf-8")
        assert "name: apple" in content
        assert "is_a" in content
        assert "[[fruit]]" in content

        # 2. Import into a fresh clean store
        store2 = MemoryStore()
        engine2 = OKFEngine(store2)
        import_result = engine2.import_okf_file(exported_path)

        assert import_result.passed_verification is True
        assert import_result.concept_name == "apple"
        assert import_result.relations_added >= 2
        assert import_result.relations_rejected == 0

        # Verify concept and relations exist in store2
        concept = store2.get_concept("apple")
        assert concept is not None
        assert concept.attributes.get("color") == "red"

        apple_rels = store2.get_relations(subject_id=concept.id, predicate="is_a")
        assert len(apple_rels) == 1
        fruit_concept = store2.get_concept(apple_rels[0].object_id)
        assert fruit_concept.name == "fruit"


def test_okf_invariant_gate_rejection():
    """Verify that an OKF file asserting an illegal cycle is rejected by I_DAG gate."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = MemoryStore()
        c_fruit = store.create_concept("fruit")
        c_plant = store.create_concept("plant")
        store.add_relation(c_fruit.id, "is_a", c_plant.id)

        # Now try to import an OKF file claiming plant is_a fruit (cycle!)
        bad_okf = """---
id: CONCEPT_PLANT
name: plant
taxonomy:
  is_a:
    - fruit
---

# Plant

A plant [[fruit]].
"""
        bad_file = Path(tmp_dir) / "plant.okf.md"
        bad_file.write_text(bad_okf, encoding="utf-8")

        engine = OKFEngine(store)
        res = engine.import_okf_file(bad_file)

        # Must reject the cycle!
        assert res.relations_rejected > 0
        assert any("cycle" in reason.lower() or "i_dag" in reason.lower() for reason in res.rejection_reasons)
