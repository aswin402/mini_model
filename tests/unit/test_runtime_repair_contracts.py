from dataclasses import replace

from little.core.kernel import CognitiveKernel
from little.core.runtime_policy import RuntimePolicy
from little.memory.headroom_policy import HeadroomPolicy
from little.memory.memory_policy import MemoryPolicy
from little.memory.store import MemoryStore


def test_kernel_uses_store_memory_capacity():
    policy = MemoryPolicy.default()
    policy = replace(policy, paged_memory=replace(policy.paged_memory, lru_cache_capacity=2))
    with MemoryStore(memory_policy=policy) as store:
        kernel = CognitiveKernel(store)
        assert kernel.paged_store.paged_config.lru_cache_capacity == 2


def test_separate_workspaces_get_separate_vaults(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    with MemoryStore() as left, MemoryStore() as right:
        first = CognitiveKernel(left, workspace_root=a)
        second = CognitiveKernel(right, workspace_root=b)
        assert first.wiki.vault_dir != second.wiki.vault_dir
        assert first.wiki.vault_dir.is_relative_to(a)
        assert second.wiki.vault_dir.is_relative_to(b)


def test_headroom_policy_is_injectable():
    runtime = replace(RuntimePolicy.default(), headroom=replace(HeadroomPolicy.default(), max_active_tokens=5))
    with MemoryStore() as store:
        kernel = CognitiveKernel(store, runtime_policy=runtime)
        kernel.headroom.record_turn("user", "word " * 100)
        assert kernel.headroom.total_active_tokens() <= 5


def test_kernel_cleanup_does_not_close_borrowed_memory():
    with MemoryStore() as store:
        kernel = CognitiveKernel(store)
        kernel.close()
        assert store.count_relations() == 0
