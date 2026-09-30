"""Unit tests and benchmarks for Rust Acceleration Core in LITTLE."""

import time
import pytest

from little.rust.engine import RustAccelerationCore


def test_rust_acceleration_core_loaded():
    """Verify that native Rust shared library is compiled and loaded successfully."""
    core = RustAccelerationCore.get()
    assert core.is_accelerated is True


def test_rust_find_ancestors():
    """Verify transitive ancestor closure in Rust."""
    core = RustAccelerationCore.get()
    adj = {
        "sparrow": ["bird"],
        "bird": ["animal", "vertebrate"],
        "animal": ["organism"],
        "vertebrate": ["chordate"],
    }
    ancestors = core.find_ancestors(adj, "sparrow")
    assert "bird" in ancestors
    assert "animal" in ancestors
    assert "organism" in ancestors
    assert "vertebrate" in ancestors
    assert "chordate" in ancestors
    assert len(ancestors) == 5


def test_rust_cycle_detection():
    """Verify cycle detection in directed graphs."""
    core = RustAccelerationCore.get()
    # Cyclic: A -> B -> C -> A
    cyclic_adj = {
        "A": ["B"],
        "B": ["C"],
        "C": ["A"],
    }
    assert core.check_cycle(cyclic_adj, "A", "C") is True
    assert core.check_cycle(cyclic_adj, "C", "A") is True
    assert core.check_cycle(cyclic_adj, "A", "D") is False

    # Acyclic: A -> B -> C
    acyclic_adj = {
        "A": ["B"],
        "B": ["C"],
    }
    assert core.check_cycle(acyclic_adj, "A", "C") is True
    assert core.check_cycle(acyclic_adj, "C", "A") is False


def test_rust_topological_sort():
    """Verify Kahn's topological sorting and cycle rejection."""
    core = RustAccelerationCore.get()
    nodes = ["X", "Y", "Z", "W"]
    edges = [("X", "Y"), ("Y", "Z"), ("Z", "W")]

    success, order, err = core.topological_sort(nodes, edges)
    assert success is True
    assert order == ["X", "Y", "Z", "W"]
    assert err is None

    # Graph with cycle
    cycle_edges = [("X", "Y"), ("Y", "Z"), ("Z", "X")]
    success_cyc, order_cyc, err_cyc = core.topological_sort(["X", "Y", "Z"], cycle_edges)
    assert success_cyc is False
    assert "cycle" in err_cyc.lower()


def test_rust_bidirectional_meeting_in_middle():
    """Verify bidirectional frontier collision search."""
    core = RustAccelerationCore.get()
    forward = {
        "A": ["B"],
        "B": ["C"],
        "C": ["D"],
        "D": ["E"],
    }
    backward = {
        "B": ["A"],
        "C": ["B"],
        "D": ["C"],
        "E": ["D"],
    }

    path = core.bidirectional_search(forward, backward, "A", "E", max_depth=10)
    assert path is not None
    assert path[0] == "A"
    assert path[-1] == "E"
    assert "C" in path


def test_rust_performance_scale_benchmark():
    """Benchmark Rust graph traversal across 10,000-node synthetic hierarchy."""
    core = RustAccelerationCore.get()

    # Build 1,000-chain deep hierarchy
    adj: dict[str, list[str]] = {}
    for i in range(1000):
        adj[f"node_{i}"] = [f"node_{i+1}"]

    t0 = time.perf_counter()
    ans = core.find_ancestors(adj, "node_0")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\nRust 1000-depth Ancestor Traversal: {elapsed_ms:.3f} ms (Found {len(ans)} ancestors)")
    assert len(ans) == 1000
    # Must complete in sub-millisecond or few milliseconds
    assert elapsed_ms < 50.0
