from pathlib import Path

from little.memory.export_policy import MemoryExportPolicy


def test_memory_export_policy_loads_unbounded_default():
    policy = MemoryExportPolicy.load(Path("data/schemas"))

    assert policy.version == "1"
    assert policy.experience_limit is None
