"""Unit tests for TurboQuant / TurboVec / TurboQdrant vector compression."""

import math
import random
import pytest

from little.neural.turbo_quant import (
    TurboQuantCompressor,
    TurboQuantIndex,
    QuantizedVector,
    VectorSearchResult,
)
from little.neural.turbo_quant_policy import TurboQuantPolicy


def test_turbo_quant_isometry():
    """Verify Fast Walsh-Hadamard with random signs is an exact isometry."""
    compressor = TurboQuantCompressor()
    rng = random.Random(123)

    v1 = [rng.gauss(0, 1) for _ in range(512)]
    v2 = [rng.gauss(0, 1) for _ in range(512)]

    r1 = compressor.rotate(v1)
    r2 = compressor.rotate(v2)

    norm_v1 = sum(x * x for x in v1)
    norm_r1 = sum(x * x for x in r1)
    assert abs(norm_v1 - norm_r1) < 1e-5

    dot_orig = sum(a * b for a, b in zip(v1, v2))
    dot_rot = sum(a * b for a, b in zip(r1, r2))
    assert abs(dot_orig - dot_rot) < 1e-5


def test_turbo_quant_compression_ratios():
    """Verify 4-bit (8x) and 1-bit (32x) memory footprints."""
    compressor = TurboQuantCompressor()
    rng = random.Random(456)
    vec = [rng.gauss(0, 1) for _ in range(512)]

    qv = compressor.compress("test_node", vec, metadata={"category": "concept"})

    assert isinstance(qv, QuantizedVector)
    assert len(qv.vector_1bit) == 8  # 8 uint64s = 64 bytes (32x compression)
    assert len(qv.vector_4bit) == 256  # 256 bytes = 512 nibbles (8x compression)
    assert qv.byte_size == 336  # 64 + 256 + 8 (scale) + 8 (norm) = 336 bytes total
    assert qv.metadata["category"] == "concept"


def test_turbo_quant_accuracy():
    """Verify 4-bit dot product error is small (<5%)."""
    compressor = TurboQuantCompressor()
    rng = random.Random(789)

    vecs = []
    for _ in range(20):
        v = [rng.gauss(0, 1) for _ in range(512)]
        norm = math.sqrt(sum(x * x for x in v))
        vecs.append([x / norm for x in v])

    query = vecs[0]
    q_rot = compressor.rotate(query)

    for v in vecs:
        exact_dot = sum(a * b for a, b in zip(query, v))
        qv = compressor.compress("item", v)
        approx_dot = compressor.approx_dot_product(qv, q_rot)
        err = abs(exact_dot - approx_dot)
        assert err < 0.03


def test_turbo_quant_index_search():
    """Verify coarse-to-fine index search retrieves the exact nearest neighbor."""
    compressor = TurboQuantCompressor()
    index = TurboQuantIndex(compressor)
    rng = random.Random(999)

    # Insert 50 random vectors
    target_vec = [rng.gauss(0, 1) for _ in range(512)]
    norm_target = math.sqrt(sum(x * x for x in target_vec))
    target_vec = [x / norm_target for x in target_vec]

    index.add("target", target_vec, {"name": "target_concept"})

    for i in range(49):
        v = [rng.gauss(0, 1) for _ in range(512)]
        norm_v = math.sqrt(sum(x * x for x in v))
        index.add(f"distractor_{i}", [x / norm_v for x in v])

    assert len(index) == 50
    assert index.memory_bytes() > 0

    # Query with slightly perturbed target (high cosine similarity)
    noisy_query = [x + rng.gauss(0, 0.01) for x in target_vec]
    norm_q = math.sqrt(sum(x * x for x in noisy_query))
    noisy_query = [x / norm_q for x in noisy_query]

    results = index.search(noisy_query, top_k=5, threshold=0.5)
    assert len(results) > 0
    top_hit = results[0]
    assert top_hit.id == "target"
    assert top_hit.score > 0.85
    assert top_hit.metadata["name"] == "target_concept"
