"""TurboQuant / TurboVec / TurboQdrant Vector Compression & Indexing Engine.

Uses deterministic Fast Walsh-Hadamard rotation, a 1-bit sign sketch and a
4-bit scalar encoding. Distortion depends on the input; no universal accuracy
bound is claimed. Payload sizes exclude Python objects, IDs, metadata and native copies.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from little.neural.turbo_quant_policy import TurboQuantPolicy
from little.rust.engine import RustAccelerationCore


@dataclass(frozen=True)
class QuantizedVector:
    """A compressed 512-dim continuous embedding vector."""

    id: str
    vector_1bit: tuple[int, ...]  # 8 x uint64 (64 bytes total)
    vector_4bit: bytes            # 256 bytes (512 x 4-bit nibbles)
    scale: float                  # coordinate dynamic range scale
    norm: float                   # L2 norm of original vector
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def byte_size(self) -> int:
        """Logical serialized payload bytes; excludes runtime object and metadata overhead."""
        return len(self.vector_1bit) * 8 + len(self.vector_4bit) + 8 + 8


@dataclass(frozen=True)
class VectorSearchResult:
    """Result of a nearest neighbor vector query."""

    id: str
    score: float
    hamming_distance: int
    metadata: dict[str, Any] = field(default_factory=dict)


class TurboQuantCompressor:
    """Data-oblivious orthogonal rotation and multi-rate scalar quantization."""

    def __init__(self, policy: TurboQuantPolicy | None = None) -> None:
        self.policy = policy or TurboQuantPolicy.default()
        self.dim = self.policy.dimension
        self._closed = False
        if (self.dim & (self.dim - 1)) != 0:
            raise ValueError(f"TurboQuant dimension must be a power of 2, got {self.dim}")

        # Deterministic random diagonal sign matrix D = diag(+-1)
        rng = random.Random(self.policy.rotation_seed)
        self._signs: tuple[float, ...] = tuple(
            1.0 if rng.random() > 0.5 else -1.0 for _ in range(self.dim)
        )
        self._scale: float = 1.0 / math.sqrt(self.dim)
        self.rust = RustAccelerationCore.get()
        self._rust_handle = None
        if self.rust.has_turbo_acceleration:
            self._rust_handle = self.rust.turbo_index_create(self.dim, self._signs)

    def close(self) -> None:
        """Release the native handle. Repeated close calls are harmless."""
        if getattr(self, "_rust_handle", None) and getattr(self, "rust", None):
            self.rust.turbo_index_destroy(self._rust_handle)
            self._rust_handle = None
        if getattr(self, '_owns_compressor', False) and getattr(self, 'compressor', None) is not None:
            self.compressor.close()
        if hasattr(self, "_vectors"):
            self._vectors.clear()
        self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("TurboQuant object is closed")

    def rotate(self, vector: Sequence[float]) -> list[float]:
        """Apply deterministic orthogonal rotation R = (1/sqrt(d)) * H * D.

        Time complexity: O(d log d) additions/subtractions (4,608 ops for d=512).
        Guarantees exact isometric energy dispersion: ||Rx|| = ||x|| and <Rx, Ry> = <x, y>.
        """
        self._ensure_open()
        if any(not math.isfinite(value) for value in vector):
            raise ValueError("Vectors must contain finite numbers")
        if len(vector) != self.dim:
            raise ValueError(f"Expected vector of dimension {self.dim}, got {len(vector)}")

        # Fast-Path: Native Rust FWHT
        if self._rust_handle is not None:
            rot = self.rust.turbo_rotate(self._rust_handle, vector)
            if rot:
                return rot

        # Pure Python Fallback
        # 1. Coordinate sign flip D * x
        a = [vector[i] * self._signs[i] for i in range(self.dim)]

        # 2. In-place Fast Walsh-Hadamard Transform H * (D * x)
        h = 1
        n = self.dim
        while h < n:
            step = h * 2
            for i in range(0, n, step):
                for j in range(i, i + h):
                    u = a[j]
                    v = a[j + h]
                    a[j] = u + v
                    a[j + h] = u - v
            h = step

        # 3. Orthogonal normalization (1 / sqrt(d))
        return [val * self._scale for val in a]

    def quantize_1bit(self, rotated: Sequence[float]) -> tuple[int, ...]:
        """Quantize into 1-bit binary sign hash packed into 64-bit unsigned integers.

        512 bits / 64 = 8 integers (64 bytes total -> 32x compression).
        """
        ints = []
        for chunk in range(0, self.dim, 64):
            val = 0
            for bit in range(64):
                if rotated[chunk + bit] >= 0.0:
                    val |= (1 << bit)
            ints.append(val)
        return tuple(ints)

    def quantize_4bit(self, rotated: Sequence[float]) -> tuple[bytes, float]:
        """Quantize into 4-bit scalar bins [-8, +7] packed 2 per byte.

        512 coordinates -> 256 bytes (8x compression).
        """
        alpha = max(abs(v) for v in rotated)
        if alpha < 1e-12:
            alpha = 1.0

        inv_scale = 7.0 / alpha
        nibbles = [max(-8, min(7, round(v * inv_scale))) + 8 for v in rotated]

        packed = bytearray(self.dim // 2)
        for i in range(0, self.dim, 2):
            packed[i // 2] = (nibbles[i] << 4) | nibbles[i + 1]

        return bytes(packed), alpha

    def compress(
        self,
        id: str,
        vector: Sequence[float],
        metadata: dict[str, Any] | None = None,
    ) -> QuantizedVector:
        """Full pipeline: rotate vector, extract 1-bit and 4-bit representations."""
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 1e-12:
            normalized_vec = [v / norm for v in vector]
        else:
            normalized_vec = list(vector)
            norm = 1.0

        rotated = self.rotate(normalized_vec)
        v1 = self.quantize_1bit(rotated)
        v4, scale = self.quantize_4bit(rotated)

        return QuantizedVector(
            id=id,
            vector_1bit=v1,
            vector_4bit=v4,
            scale=scale,
            norm=norm,
            metadata=dict(metadata or {}),
        )

    def hamming_distance(
        self,
        quantized_1bit: tuple[int, ...],
        query_1bit: tuple[int, ...],
    ) -> int:
        """Sub-microsecond CPU popcount Hamming distance."""
        return sum(
            (u1 ^ u2).bit_count()
            for u1, u2 in zip(quantized_1bit, query_1bit)
        )

    def cosine_from_hamming(self, hamming: int) -> float:
        """Estimate cosine similarity from Hamming distance via Charikar's formula."""
        ratio = min(1.0, max(0.0, hamming / self.dim))
        return math.cos(math.pi * ratio)

    def approx_dot_product(
        self,
        quantized: QuantizedVector,
        rotated_query: Sequence[float],
    ) -> float:
        """Evaluate dot product using 4-bit nibbles and pre-rotated query vector."""
        accum = 0.0
        packed = quantized.vector_4bit
        step = quantized.scale / 7.0

        for i in range(0, self.dim, 2):
            b = packed[i // 2]
            n0 = (b >> 4) - 8
            n1 = (b & 0x0F) - 8
            accum += n0 * rotated_query[i] + n1 * rotated_query[i + 1]

        return accum * step


class TurboQuantIndex:
    """In-memory vector database using two-stage coarse-to-fine search.

    Stage 1: 1-bit Hamming distance coarse filter (popcount in microseconds).
    Stage 2: 4-bit nibble dot product fine reranking on top candidates.
    Logical payload: 336 bytes per vector. Python objects, metadata, IDs and native
    copies consume additional memory.
    """

    def __init__(
        self,
        compressor: TurboQuantCompressor | None = None,
        policy: TurboQuantPolicy | None = None,
    ) -> None:
        self._closed = False
        self._owns_compressor = compressor is None
        self.compressor = compressor or TurboQuantCompressor(policy)
        self.policy = self.compressor.policy
        self._vectors: dict[str, QuantizedVector] = {}
        self.rust = RustAccelerationCore.get()
        self._rust_handle = None
        if self.rust.has_turbo_acceleration:
            self._rust_handle = self.rust.turbo_index_create(
                self.compressor.dim, self.compressor._signs
            )

    def close(self) -> None:
        """Release the native handle. Repeated close calls are harmless."""
        if getattr(self, "_rust_handle", None) and getattr(self, "rust", None):
            self.rust.turbo_index_destroy(self._rust_handle)
            self._rust_handle = None
        if getattr(self, '_owns_compressor', False) and getattr(self, 'compressor', None) is not None:
            self.compressor.close()
        if hasattr(self, "_vectors"):
            self._vectors.clear()
        self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("TurboQuant object is closed")

    def __len__(self) -> int:
        return len(self._vectors)

    def add(
        self,
        id: str,
        vector: Sequence[float],
        metadata: dict[str, Any] | None = None,
    ) -> QuantizedVector:
        """Compress and insert a continuous vector into the index."""
        self._ensure_open()
        if not isinstance(id, str) or not id or "\x00" in id:
            raise ValueError("Vector ID must be a nonempty string without NUL bytes")
        qv = self.compressor.compress(id, vector, metadata)
        self._vectors[id] = qv
        if self._rust_handle is not None:
            try:
                inserted = self.rust.turbo_index_add(self._rust_handle, id, vector, metadata)
            except (TypeError, ValueError, OverflowError):
                inserted = False
            if not inserted:
                self._disable_native()
        return qv

    def add_quantized(self, quantized: QuantizedVector) -> None:
        """Directly insert an already-quantized vector."""
        self._ensure_open()
        if not isinstance(quantized.id, str) or not quantized.id or "\x00" in quantized.id:
            raise ValueError("Vector ID must be a nonempty string without NUL bytes")
        if len(quantized.vector_1bit) != 8 or len(quantized.vector_4bit) != 256:
            raise ValueError("Quantized payload must encode 512 dimensions")
        if not math.isfinite(quantized.scale) or quantized.scale <= 0 or not math.isfinite(quantized.norm) or quantized.norm <= 0:
            raise ValueError("Quantized scale and norm must be finite and positive")
        if any(type(word) is not int or not 0 <= word < 2**64 for word in quantized.vector_1bit):
            raise ValueError("Quantized words must be uint64 values")
        # Native insertion accepts raw vectors only. Retain exact prequantized
        # bytes and switch this index to Python instead of silently losing rows.
        self._disable_native()
        self._vectors[quantized.id] = quantized

    def _disable_native(self) -> None:
        if self._rust_handle is not None:
            self.rust.turbo_index_destroy(self._rust_handle)
            self._rust_handle = None

    def get(self, id: str) -> QuantizedVector | None:
        return self._vectors.get(id)

    def remove(self, id: str) -> bool:
        self._ensure_open()
        if self._rust_handle is not None:
            self.rust.turbo_index_remove(self._rust_handle, id)
        return self._vectors.pop(id, None) is not None

    def memory_bytes(self) -> int:
        """Logical payload bytes only, excluding object/metadata/native-copy overhead."""
        return sum(qv.byte_size for qv in self._vectors.values())

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Execute coarse-to-fine nearest neighbor search.

        Returns top_k nearest neighbors meeting the similarity threshold.
        """
        self._ensure_open()
        if len(query_vector) != self.compressor.dim or any(not math.isfinite(v) for v in query_vector):
            raise ValueError("Query must contain 512 finite coordinates")
        k = top_k if top_k is not None else self.policy.top_k_default
        min_sim = threshold if threshold is not None else self.policy.similarity_threshold

        if type(k) is not int or k < 0 or not math.isfinite(min_sim) or not -1 <= min_sim <= 1:
            raise ValueError("Invalid search limits")
        if k == 0 or not self._vectors:
            return []

        # Fast-Path: Native Rust sub-millisecond search
        if self._rust_handle is not None:
            raw_results = self.rust.turbo_index_search(
                self._rust_handle,
                query_vector,
                top_k=k,
                threshold=min_sim,
                coarse_multiplier=self.policy.coarse_filter_multiplier,
            )
            return [
                VectorSearchResult(
                    id=r["id"],
                    score=r["score"],
                    hamming_distance=r["hamming_distance"],
                    metadata=r["metadata"],
                )
                for r in raw_results
            ]

        # Pure Python Fallback
        # Normalize query vector
        norm = math.sqrt(sum(v * v for v in query_vector))
        if norm > 1e-12:
            norm_query = [v / norm for v in query_vector]
        else:
            norm_query = list(query_vector)

        # 1. Rotate query once
        q_rot = self.compressor.rotate(norm_query)
        q_1bit = self.compressor.quantize_1bit(q_rot)

        # 2. Stage 1: Coarse 1-bit Hamming distance filter across all vectors
        coarse_pool_size = min(len(self._vectors), k * self.policy.coarse_filter_multiplier)
        scored_coarse: list[tuple[int, QuantizedVector]] = []

        for qv in self._vectors.values():
            dist = self.compressor.hamming_distance(qv.vector_1bit, q_1bit)
            scored_coarse.append((dist, qv))

        # Sort by smallest Hamming distance
        scored_coarse.sort(key=lambda item: item[0])
        candidates = scored_coarse[:coarse_pool_size]

        # 3. Stage 2: Fine 4-bit dot product rerank on top candidates
        results: list[VectorSearchResult] = []
        for dist, qv in candidates:
            score = self.compressor.approx_dot_product(qv, q_rot)
            if score >= min_sim:
                results.append(
                    VectorSearchResult(
                        id=qv.id,
                        score=round(score, 4),
                        hamming_distance=dist,
                        metadata=dict(qv.metadata),
                    )
                )

        # Sort by highest fine dot product score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]
