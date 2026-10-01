"""Python FFI bindings and fallback engine for Rust-accelerated graph algorithms."""

from __future__ import annotations

import ctypes
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any


class RustAccelerationCore:
    """Provides sub-microsecond graph traversal, cycle detection, and toposort via Rust cdylib."""

    _instance: RustAccelerationCore | None = None

    def __init__(self) -> None:
        self._lib: ctypes.CDLL | None = self._load_library()

    @classmethod
    def get(cls) -> RustAccelerationCore:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_accelerated(self) -> bool:
        """Return True if native Rust shared library is successfully loaded."""
        return self._lib is not None

    @property
    def has_turbo_acceleration(self) -> bool:
        """Return True if native Rust TurboQuant vector acceleration is available."""
        return self._lib is not None and hasattr(self._lib, "little_core_turbo_index_create")

    def _load_library(self) -> ctypes.CDLL | None:
        """Search workspace and system paths for liblittle_core shared library."""
        search_dirs = [
            Path(__file__).resolve().parent,
            Path(__file__).resolve().parents[2] / "rust" / "little_core" / "target" / "release",
            Path(__file__).resolve().parents[4] / "target" / "release",
            Path.cwd() / "target" / "release",
        ]

        lib_names = ["liblittle_core.so", "liblittle_core.dylib", "little_core.dll"]
        for d in search_dirs:
            for name in lib_names:
                cand = d / name
                if cand.exists():
                    try:
                        lib = ctypes.CDLL(str(cand))
                        self._bind_signatures(lib)
                        return lib
                    except Exception:
                        continue
        return None

    def _bind_signatures(self, lib: ctypes.CDLL) -> None:
        lib.little_core_find_ancestors.argtypes = [ctypes.c_char_p]
        lib.little_core_find_ancestors.restype = ctypes.c_void_p

        lib.little_core_check_cycle.argtypes = [ctypes.c_char_p]
        lib.little_core_check_cycle.restype = ctypes.c_int32

        lib.little_core_toposort.argtypes = [ctypes.c_char_p]
        lib.little_core_toposort.restype = ctypes.c_void_p

        lib.little_core_bidirectional_search.argtypes = [ctypes.c_char_p]
        lib.little_core_bidirectional_search.restype = ctypes.c_void_p

        lib.little_core_free_string.argtypes = [ctypes.c_void_p]
        lib.little_core_free_string.restype = None

        # TurboQuant vector index bindings
        if hasattr(lib, "little_core_turbo_index_create"):
            lib.little_core_turbo_index_create.argtypes = [
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
            ]
            lib.little_core_turbo_index_create.restype = ctypes.c_void_p

            lib.little_core_turbo_index_destroy.argtypes = [ctypes.c_void_p]
            lib.little_core_turbo_index_destroy.restype = None

            lib.little_core_turbo_index_add.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.c_char_p,
            ]
            lib.little_core_turbo_index_add.restype = ctypes.c_int32

            lib.little_core_turbo_index_remove.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            lib.little_core_turbo_index_remove.restype = ctypes.c_int32

            lib.little_core_turbo_index_size.argtypes = [ctypes.c_void_p]
            lib.little_core_turbo_index_size.restype = ctypes.c_size_t

            lib.little_core_turbo_index_search.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_float,
                ctypes.c_size_t,
            ]
            lib.little_core_turbo_index_search.restype = ctypes.c_void_p

            lib.little_core_turbo_rotate.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
            ]
            lib.little_core_turbo_rotate.restype = ctypes.c_int32

    # -------------------------------------------------------------------------
    # Public Accelerated Graph API
    # -------------------------------------------------------------------------

    def find_ancestors(
        self, adjacency: dict[str, list[str]], start_node: str
    ) -> list[str]:
        """Compute all transitive ancestors reachable from start_node."""
        if self._lib is not None:
            payload = json.dumps({"adjacency": adjacency, "start_node": start_node}).encode("utf-8")
            ptr = self._lib.little_core_find_ancestors(payload)
            if ptr:
                res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
                self._lib.little_core_free_string(ptr)
                data = json.loads(res_str)
                return data.get("ancestors", [])

        # Pure Python Fallback
        visited: set[str] = set()
        stack: list[str] = list(adjacency.get(start_node, []))
        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
                for n in adjacency.get(curr, []):
                    if n not in visited:
                        stack.append(n)
        return list(visited)

    def check_cycle(
        self, adjacency: dict[str, list[str]], start_node: str, target_node: str
    ) -> bool:
        """Check if directed path exists from start_node to target_node."""
        if self._lib is not None:
            payload = json.dumps({
                "adjacency": adjacency,
                "start_node": start_node,
                "target_node": target_node,
            }).encode("utf-8")
            res = self._lib.little_core_check_cycle(payload)
            return bool(res == 1)

        # Pure Python Fallback
        visited: set[str] = set()
        stack: list[str] = [start_node]
        while stack:
            curr = stack.pop()
            if curr == target_node:
                return True
            if curr not in visited:
                visited.add(curr)
                for n in adjacency.get(curr, []):
                    if n == target_node:
                        return True
                    if n not in visited:
                        stack.append(n)
        return False

    def topological_sort(
        self, nodes: list[str], edges: list[tuple[str, str]]
    ) -> tuple[bool, list[str], str | None]:
        """Kahn's topological sort for poset ordering."""
        if self._lib is not None:
            payload = json.dumps({"nodes": nodes, "edges": edges}).encode("utf-8")
            ptr = self._lib.little_core_toposort(payload)
            if ptr:
                res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
                self._lib.little_core_free_string(ptr)
                data = json.loads(res_str)
                return bool(data.get("success", False)), data.get("order", []), data.get("error")

        # Pure Python Fallback
        in_deg: dict[str, int] = {n: 0 for n in nodes}
        adj: dict[str, list[str]] = {n: [] for n in nodes}
        for u, v in edges:
            if u in adj and v in in_deg:
                adj[u].append(v)
                in_deg[v] += 1

        queue = deque([n for n, deg in in_deg.items() if deg == 0])
        order: list[str] = []
        while queue:
            curr = queue.popleft()
            order.append(curr)
            for neighbor in adj.get(curr, []):
                in_deg[neighbor] -= 1
                if in_deg[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) == len(nodes):
            return True, order, None
        return False, [], "Cycle detected: graph is not a Directed Acyclic Graph"

    def bidirectional_search(
        self,
        forward_adj: dict[str, list[str]],
        backward_adj: dict[str, list[str]],
        start: str,
        target: str,
        max_depth: int = 10,
    ) -> list[str] | None:
        """Bidirectional Meeting-in-the-Middle search."""
        if self._lib is not None:
            payload = json.dumps({
                "forward_adj": forward_adj,
                "backward_adj": backward_adj,
                "start": start,
                "target": target,
                "max_depth": max_depth,
            }).encode("utf-8")
            ptr = self._lib.little_core_bidirectional_search(payload)
            if ptr:
                res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
                self._lib.little_core_free_string(ptr)
                data = json.loads(res_str)
                return data.get("path") if data.get("found") else None

        # Pure Python Fallback
        if start == target:
            return [start]
        visited_fwd: dict[str, list[str]] = {start: [start]}
        visited_bwd: dict[str, list[str]] = {target: [target]}
        q_fwd: deque[str] = deque([start])
        q_bwd: deque[str] = deque([target])
        depth = 0
        while (q_fwd or q_bwd) and depth < max_depth:
            depth += 1
            for _ in range(len(q_fwd)):
                if not q_fwd:
                    break
                curr = q_fwd.popleft()
                path = visited_fwd[curr]
                for n in forward_adj.get(curr, []):
                    if n in visited_bwd:
                        return path + list(reversed(visited_bwd[n]))
                    if n not in visited_fwd:
                        visited_fwd[n] = path + [n]
                        q_fwd.append(n)
            for _ in range(len(q_bwd)):
                if not q_bwd:
                    break
                curr = q_bwd.popleft()
                path = visited_bwd[curr]
                for n in backward_adj.get(curr, []):
                    if n in visited_fwd:
                        return visited_fwd[n] + list(reversed(path))
                    if n not in visited_bwd:
                        visited_bwd[n] = path + [n]
                        q_bwd.append(n)
        return None

    # -------------------------------------------------------------------------
    # Public Accelerated TurboQuant Vector Index API
    # -------------------------------------------------------------------------

    def turbo_index_create(self, dimension: int, signs: Sequence[float]) -> Any:
        """Create a native Rust TurboQuant index handle."""
        if self._lib is None or not hasattr(self._lib, "little_core_turbo_index_create"):
            return None
        c_signs = (ctypes.c_float * len(signs))(*signs)
        return self._lib.little_core_turbo_index_create(dimension, c_signs, len(signs))

    def turbo_index_destroy(self, handle: Any) -> None:
        """Destroy native Rust TurboQuant index handle."""
        if self._lib is not None and handle and hasattr(self._lib, "little_core_turbo_index_destroy"):
            self._lib.little_core_turbo_index_destroy(handle)

    def turbo_index_add(
        self,
        handle: Any,
        id_str: str,
        vector: Sequence[float],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Add a continuous vector into native Rust TurboQuant index."""
        if self._lib is None or not handle or not hasattr(self._lib, "little_core_turbo_index_add"):
            return False
        c_vec = (ctypes.c_float * len(vector))(*vector)
        id_bytes = id_str.encode("utf-8")
        meta_bytes = json.dumps(metadata or {}).encode("utf-8")
        res = self._lib.little_core_turbo_index_add(
            handle, id_bytes, c_vec, len(vector), meta_bytes
        )
        return bool(res == 1)

    def turbo_index_remove(self, handle: Any, id_str: str) -> bool:
        """Remove a vector from native Rust TurboQuant index by ID."""
        if self._lib is None or not handle or not hasattr(self._lib, "little_core_turbo_index_remove"):
            return False
        id_bytes = id_str.encode("utf-8")
        res = self._lib.little_core_turbo_index_remove(handle, id_bytes)
        return bool(res == 1)

    def turbo_index_size(self, handle: Any) -> int:
        """Return total number of vectors stored in native Rust index."""
        if self._lib is None or not handle or not hasattr(self._lib, "little_core_turbo_index_size"):
            return 0
        return int(self._lib.little_core_turbo_index_size(handle))

    def turbo_index_search(
        self,
        handle: Any,
        query_vector: Sequence[float],
        top_k: int = 5,
        threshold: float = 0.08,
        coarse_multiplier: int = 10,
    ) -> list[dict[str, Any]]:
        """Execute sub-millisecond coarse-to-fine vector search in native Rust."""
        if self._lib is None or not handle or not hasattr(self._lib, "little_core_turbo_index_search"):
            return []
        c_query = (ctypes.c_float * len(query_vector))(*query_vector)
        ptr = self._lib.little_core_turbo_index_search(
            handle, c_query, len(query_vector), top_k, ctypes.c_float(threshold), coarse_multiplier
        )
        if not ptr:
            return []
        res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.little_core_free_string(ptr)
        return json.loads(res_str)

    def turbo_rotate(self, handle: Any, vector: Sequence[float]) -> list[float]:
        """Apply fast in-place FWHT rotation using native Rust."""
        if self._lib is None or not handle or not hasattr(self._lib, "little_core_turbo_rotate"):
            return []
        dim = len(vector)
        c_vec = (ctypes.c_float * dim)(*vector)
        c_out = (ctypes.c_float * dim)()
        res = self._lib.little_core_turbo_rotate(handle, c_vec, dim, c_out)
        if res == 1:
            return list(c_out)
        return []
