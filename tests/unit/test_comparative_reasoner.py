"""Unit tests for comparative order reasoner and strict partial orders."""

from little.language.comparative import ComparativeOrderReasoner
from little.memory.store import MemoryStore


def test_transitive_comparative_chain():
    reasoner = ComparativeOrderReasoner()
    
    # Assert A > B and B > C
    success1, _ = reasoner.assert_comparison("John", "Mary", "taller")
    assert success1 is True
    success2, _ = reasoner.assert_comparison("Mary", "Peter", "taller")
    assert success2 is True

    # Query transitive A > C
    ans = reasoner.query_comparison("John", "Peter", "taller")
    assert "Yes" in ans
    assert "John is taller than Mary" in ans
    assert "Mary is taller than Peter" in ans

    # Query refutation C > A
    ans_rev = reasoner.query_comparison("Peter", "John", "taller")
    assert "No" in ans_rev
    assert "John is shorter than Peter" in ans_rev or "Peter is shorter than John" in ans_rev or "In fact" in ans_rev

    # Query extremum
    extremum = reasoner.query_extremum("tallest")
    assert "John is the tallest" in extremum

    extremum_min = reasoner.query_extremum("shortest")
    assert "Peter is the shortest" in extremum_min


def test_comparative_contradiction_detection():
    reasoner = ComparativeOrderReasoner()
    reasoner.assert_comparison("Cheetah", "Lion", "faster")
    reasoner.assert_comparison("Lion", "Elephant", "faster")

    # Attempting to say Elephant is faster than Cheetah must be caught as a contradiction
    success, msg = reasoner.assert_comparison("Elephant", "Cheetah", "faster")
    assert success is False
    assert "Contradiction detected" in msg


def test_evaluate_compound_text():
    reasoner = ComparativeOrderReasoner()
    story = "Alice is older than Bob, and Bob is older than Charlie. Who is the oldest?"
    result = reasoner.evaluate_text(story)
    assert result is not None
    assert "Alice is the oldest" in result


def test_sqlite_persistence_of_comparisons(tmp_path):
    db_file = tmp_path / "comparative_test.db"
    mem1 = MemoryStore(str(db_file), seed_ontology=False)
    reasoner1 = ComparativeOrderReasoner(memory=mem1)
    reasoner1.assert_comparison("Falcon", "Hawk", "faster")
    reasoner1.assert_comparison("Hawk", "Pigeon", "faster")
    mem1.close()

    # Reopen in fresh instance
    mem2 = MemoryStore(str(db_file), seed_ontology=False)
    reasoner2 = ComparativeOrderReasoner(memory=mem2)
    ans = reasoner2.query_comparison("Falcon", "Pigeon", "faster")
    assert "Yes" in ans
    assert "Falcon is faster than Hawk" in ans
    mem2.close()
