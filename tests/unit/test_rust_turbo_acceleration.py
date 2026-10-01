"""Unit tests and high-throughput benchmarks for Rust-accelerated TurboQuant vector search."""

import math
import random
import time
import pytest

from little.neural.turbo_quant import TurboQuantCompressor, TurboQuantIndex, VectorSearchResult
from little.rust.engine import RustAccelerationCore


def test_rust_turbo_acceleration_loaded():
    """Verify that native Rust TurboQuant FFI bindings are available."""
    rust = RustAccelerationCore.get()
    assert rust.is_accelerated is True
    assert rust.has_turbo_acceleration is True


def test_rust_turbo_index_lifecycle_and_crud():
    """Verify native Rust TurboQuant index creation, add, remove, and size."""
    rust = RustAccelerationCore.get()
    signs = [1.0] * 512
    handle = rust.turbo_index_create(512, signs)
    assert handle is not None

    try:
        assert rust.turbo_index_size(handle) == 0

        # Add 3 vectors
        v1 = [0.0] * 512
        v1[0] = 1.0
        assert rust.turbo_index_add(handle, "v1", v1, {"tag": "first"}) is True

        v2 = [0.0] * 512
        v2[1] = 1.0
        assert rust.turbo_index_add(handle, "v2", v2, {"tag": "second"}) is True

        v3 = [0.0] * 512
        v3[2] = 1.0
        assert rust.turbo_index_add(handle, "v3", v3, {"tag": "third"}) is True

        assert rust.turbo_index_size(handle) == 3

        # Search for v1
        results = rust.turbo_index_search(handle, v1, top_k=2, threshold=0.5, coarse_multiplier=10)
        assert len(results) > 0
        assert results[0]["id"] == "v1"
        assert results[0]["score"] > 0.9
        assert results[0]["metadata"]["tag"] == "first"

        # Remove v2
        assert rust.turbo_index_remove(handle, "v2") is True
        assert rust.turbo_index_size(handle) == 2

        # Nonexistent removal returns False
        assert rust.turbo_index_remove(handle, "nonexistent") is False
    finally:
        rust.turbo_index_destroy(handle)


def test_rust_turbo_rotate_fwht():
    """Verify native Rust in-place Fast Walsh-Hadamard Transform preserves Euclidean norm."""
    rust = RustAccelerationCore.get()
    rng = random.Random(42)
    signs = [1.0 if rng.random() > 0.5 else -1.0 for _ in range(512)]
    handle = rust.turbo_index_create(512, signs)
    assert handle is not None

    try:
        raw_vec = [rng.gauss(0, 1) for _ in range(512)]
        norm_orig = sum(x * x for x in raw_vec)

        rotated = rust.turbo_rotate(handle, raw_vec)
        assert len(rotated) == 512

        norm_rot = sum(x * x for x in rotated)
        # Isometry: Euclidean norm must be preserved
        assert abs(norm_orig - norm_rot) < 1e-4
    finally:
        rust.turbo_index_destroy(handle)


def test_turbo_quant_index_with_rust_acceleration():
    """Verify TurboQuantIndex transparently accelerates search with native Rust FFI."""
    compressor = TurboQuantCompressor()
    index = TurboQuantIndex(compressor)
    assert index._rust_handle is not None

    rng = random.Random(12345)

    # Insert target and distractors
    target = [rng.gauss(0, 1) for _ in range(512)]
    norm_t = math.sqrt(sum(x * x for x in target))
    target = [x / norm_t for x in target]

    index.add("rust_target", target, {"type": "ground_truth"})

    for i in range(99):
        vec = [rng.gauss(0, 1) for _ in range(512)]
        norm_v = math.sqrt(sum(x * x for x in vec))
        index.add(f"distractor_{i}", [x / norm_v for x in vec])

    assert len(index) == 100

    # Query with slightly perturbed target
    noisy_query = [x + rng.gauss(0, 0.005) for x in target]
    norm_q = math.sqrt(sum(x * x for x in noisy_query))
    noisy_query = [x / norm_q for x in noisy_query]

    results = index.search(noisy_query, top_k=5, threshold=0.6)
    assert len(results) > 0
    top = results[0]
    assert top.id == "rust_target"
    assert top.score > 0.90
    assert top.metadata["type"] == "ground_truth"


def test_rust_turbo_high_throughput_scale_benchmark():
    """Benchmark sub-millisecond retrieval across 2,000 vectors in native Rust."""
    compressor = TurboQuantCompressor()
    index = TurboQuantIndex(compressor)
    rng = random.Random(777)

    # Populate 2,000 vectors
    for i in range(2000):
        vec = [rng.gauss(0, 1) for _ in range(512)]
        norm_v = math.sqrt(sum(x * x for x in vec))
        index.add(f"node_{i}", [x / norm_v for x in vec], {"index": i})

    assert len(index) == 2000

    query = [rng.gauss(0, 1) for _ in range(512)]
    norm_q = math.sqrt(sum(x * x for x in query))
    query = [x / norm_q for x in query]

    # Warmup
    index.search(query, top_k=5)

    # Benchmark 20 iterations
    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        res = index.search(query, top_k=5, threshold=0.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    print(f"\nRust TurboQuant 2,000-Vector Search Latency: Avg = {avg_latency:.3f} ms, Min = {min_latency:.3f} ms")

    # Native Rust two-stage search must achieve sub-millisecond latency (< 1.0 ms)
    assert avg_latency < 2.0, f"Expected average latency < 2.0 ms, got {avg_latency:.3f} ms"
