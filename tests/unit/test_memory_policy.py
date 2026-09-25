from pathlib import Path

from little.memory.memory_policy import MemoryPolicy


def test_memory_policy_loads_persistence_defaults():
    policy = MemoryPolicy.load(Path("data/schemas"))

    assert policy.version == "1"
    assert policy.concept_confidence == 1.0
    assert policy.experience_query_limit == 50
    assert policy.bulk_import_batch_size == 5000
