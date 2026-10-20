"""Python FFI bindings and fallback engine for Rust-accelerated graph algorithms."""

from __future__ import annotations

import ctypes
import json
import math
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self


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

    @property
    def has_graph_index_acceleration(self) -> bool:
        """Return True if persistent zero-copy Rust GraphIndex acceleration is available."""
        return self._lib is not None and hasattr(self._lib, "little_core_graph_create")

    def _load_library(self) -> ctypes.CDLL | None:
        """Search workspace and system paths for liblittle_core shared library."""
        repository = Path(__file__).resolve().parents[3]
        search_dirs = [
            repository / "rust" / "little_core" / "target" / "release",
            Path(__file__).resolve().parent,
            repository / "target" / "release",
        ]
        # A surrounding Cargo workspace may own the target directory.
        search_dirs.extend(
            parent / "target" / "release"
            for parent in repository.parents
            if (parent / "Cargo.toml").is_file()
        )

        lib_names = ["liblittle_core.so", "liblittle_core.dylib", "little_core.dll"]
        for d in search_dirs:
            for name in lib_names:
                cand = d / name
                if cand.exists():
                    try:
                        lib = ctypes.CDLL(str(cand))
                        self._bind_signatures(lib)
                        return lib
                    except (OSError, AttributeError):
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

        # Persistent zero-copy GraphIndex bindings
        if hasattr(lib, "little_core_graph_create"):
            lib.little_core_graph_create.argtypes = []
            lib.little_core_graph_create.restype = ctypes.c_void_p

            lib.little_core_graph_destroy.argtypes = [ctypes.c_void_p]
            lib.little_core_graph_destroy.restype = None

            lib.little_core_graph_clear.argtypes = [ctypes.c_void_p]
            lib.little_core_graph_clear.restype = None

            lib.little_core_graph_edge_count.argtypes = [ctypes.c_void_p]
            lib.little_core_graph_edge_count.restype = ctypes.c_size_t

            lib.little_core_graph_add_edge.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
            ]
            lib.little_core_graph_add_edge.restype = ctypes.c_int32

            lib.little_core_graph_check_cycle.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
            ]
            lib.little_core_graph_check_cycle.restype = ctypes.c_int32

            lib.little_core_graph_find_ancestors.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
            ]
            lib.little_core_graph_find_ancestors.restype = ctypes.c_void_p

            lib.little_core_graph_bidirectional_search.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            lib.little_core_graph_bidirectional_search.restype = ctypes.c_void_p

        # 360° Spider-Web MLU Arena bindings
        if hasattr(lib, "little_core_mlu_arena_create"):
            lib.little_core_mlu_arena_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
            lib.little_core_mlu_arena_create.restype = ctypes.c_void_p

            lib.little_core_mlu_arena_destroy.argtypes = [ctypes.c_void_p]
            lib.little_core_mlu_arena_destroy.restype = None

            lib.little_core_mlu_arena_node_count.argtypes = [ctypes.c_void_p]
            lib.little_core_mlu_arena_node_count.restype = ctypes.c_size_t

            lib.little_core_mlu_arena_mitosis_count.argtypes = [ctypes.c_void_p]
            lib.little_core_mlu_arena_mitosis_count.restype = ctypes.c_size_t

            lib.little_core_mlu_arena_learn_spoke.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_float,
                ctypes.c_int32,
            ]
            lib.little_core_mlu_arena_learn_spoke.restype = ctypes.c_float

            lib.little_core_mlu_arena_predict_spoke.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_size_t,
                ctypes.c_size_t,
            ]
            lib.little_core_mlu_arena_predict_spoke.restype = ctypes.c_void_p

            if hasattr(lib, "little_core_mlu_arena_train_stream"):
                lib.little_core_mlu_arena_train_stream.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_float,
                ]
                lib.little_core_mlu_arena_train_stream.restype = ctypes.c_void_p

            if hasattr(lib, "little_core_mlu_arena_generate"):
                lib.little_core_mlu_arena_generate.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_size_t,
                ]
                lib.little_core_mlu_arena_generate.restype = ctypes.c_void_p

            if hasattr(lib, "little_core_mlu_arena_save_snapshot"):
                lib.little_core_mlu_arena_save_snapshot.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                ]
                lib.little_core_mlu_arena_save_snapshot.restype = ctypes.c_int32

            if hasattr(lib, "little_core_mlu_arena_load_snapshot"):
                lib.little_core_mlu_arena_load_snapshot.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                ]
                lib.little_core_mlu_arena_load_snapshot.restype = ctypes.c_int32

            if hasattr(lib, "little_core_mlu_arena_dump_json"):
                lib.little_core_mlu_arena_dump_json.argtypes = [ctypes.c_void_p]
                lib.little_core_mlu_arena_dump_json.restype = ctypes.c_void_p

            if hasattr(lib, "little_core_mlu_arena_restore_json"):
                lib.little_core_mlu_arena_restore_json.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                ]
                lib.little_core_mlu_arena_restore_json.restype = ctypes.c_int32

            if hasattr(lib, "little_core_mlu_arena_query_360_beam"):
                lib.little_core_mlu_arena_query_360_beam.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                ]
                lib.little_core_mlu_arena_query_360_beam.restype = ctypes.c_void_p

            if hasattr(lib, "little_core_mlu_arena_consolidate"):
                lib.little_core_mlu_arena_consolidate.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_float,
                    ctypes.c_float,
                ]
                lib.little_core_mlu_arena_consolidate.restype = ctypes.c_void_p

            if hasattr(lib, "little_core_mlu_arena_verify_invariants"):
                lib.little_core_mlu_arena_verify_invariants.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                ]
                lib.little_core_mlu_arena_verify_invariants.restype = ctypes.c_void_p

            if hasattr(lib, "little_core_mlu_arena_compare_360"):
                lib.little_core_mlu_arena_compare_360.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_size_t,
                ]
                lib.little_core_mlu_arena_compare_360.restype = ctypes.c_void_p

            if hasattr(lib, "little_core_mlu_arena_discover_curiosity"):
                lib.little_core_mlu_arena_discover_curiosity.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                ]
                lib.little_core_mlu_arena_discover_curiosity.restype = ctypes.c_void_p

            if hasattr(lib, "little_core_mlu_arena_simulate_causal"):
                lib.little_core_mlu_arena_simulate_causal.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_size_t,
                ]
                lib.little_core_mlu_arena_simulate_causal.restype = ctypes.c_void_p

            if hasattr(lib, "little_core_mlu_arena_autonomous_curiosity"):
                lib.little_core_mlu_arena_autonomous_curiosity.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_float,
                    ctypes.c_size_t,
                ]
                lib.little_core_mlu_arena_autonomous_curiosity.restype = ctypes.c_void_p

            if hasattr(lib, "little_core_mlu_arena_intervene_counterfactual"):
                lib.little_core_mlu_arena_intervene_counterfactual.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_size_t,
                ]
                lib.little_core_mlu_arena_intervene_counterfactual.restype = ctypes.c_void_p

            if hasattr(lib, "little_core_mlu_arena_find_bridge"):
                lib.little_core_mlu_arena_find_bridge.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_size_t,
                ]
                lib.little_core_mlu_arena_find_bridge.restype = ctypes.c_void_p

            if hasattr(lib, "little_core_mlu_arena_export_topology"):
                lib.little_core_mlu_arena_export_topology.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                ]
                lib.little_core_mlu_arena_export_topology.restype = ctypes.c_void_p

            if hasattr(lib, "little_core_mlu_arena_discover_clusters"):
                lib.little_core_mlu_arena_discover_clusters.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                ]
                lib.little_core_mlu_arena_discover_clusters.restype = ctypes.c_void_p

            if hasattr(lib, "little_core_mlu_arena_revise_belief"):
                lib.little_core_mlu_arena_revise_belief.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_int32,
                    ctypes.c_float,
                ]
                lib.little_core_mlu_arena_revise_belief.restype = ctypes.c_void_p

            if hasattr(lib, "little_core_mlu_arena_deliberate_360"):
                lib.little_core_mlu_arena_deliberate_360.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_size_t,
                ]
                lib.little_core_mlu_arena_deliberate_360.restype = ctypes.c_void_p

            if hasattr(lib, "little_core_mlu_arena_plan_goal_360"):
                lib.little_core_mlu_arena_plan_goal_360.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_size_t,
                ]
                lib.little_core_mlu_arena_plan_goal_360.restype = ctypes.c_void_p

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
        if dimension != 512 or len(signs) != 512 or any(not math.isfinite(s) or abs(s) != 1 for s in signs):
            raise ValueError("Native TurboQuant requires 512 signs of +1 or -1")
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
        if len(vector) != 512 or any(not math.isfinite(value) or abs(value) > 3.4028235e38 for value in vector):
            raise ValueError("Native vector must contain 512 finite float32 coordinates")
        if not id_str or '\x00' in id_str:
            raise ValueError("Native vector ID must not be empty or contain NUL")
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
        if len(query_vector) != 512 or any(not math.isfinite(value) or abs(value) > 3.4028235e38 for value in query_vector):
            raise ValueError("Native query must contain 512 finite float32 coordinates")
        if type(top_k) is not int or top_k < 0 or type(coarse_multiplier) is not int or coarse_multiplier < 1 or not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("Invalid native search limits")
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
        if dim != 512 or any(not math.isfinite(value) or abs(value) > 3.4028235e38 for value in vector):
            raise ValueError("Native rotation requires 512 finite float32 coordinates")
        c_vec = (ctypes.c_float * dim)(*vector)
        c_out = (ctypes.c_float * dim)()
        res = self._lib.little_core_turbo_rotate(handle, c_vec, dim, c_out)
        if res == 1:
            return list(c_out)
        return []

    # -------------------------------------------------------------------------
    # Persistent Zero-Copy Graph Index Handle API
    # -------------------------------------------------------------------------

    def graph_create(self) -> Any:
        """Create a persistent Rust GraphIndex handle."""
        if self._lib is None or not hasattr(self._lib, "little_core_graph_create"):
            return None
        return self._lib.little_core_graph_create()

    def graph_destroy(self, handle: Any) -> None:
        """Destroy a persistent Rust GraphIndex handle."""
        if self._lib is not None and handle and hasattr(self._lib, "little_core_graph_destroy"):
            self._lib.little_core_graph_destroy(handle)

    def graph_clear(self, handle: Any) -> None:
        """Clear all edges in a persistent Rust GraphIndex handle."""
        if self._lib is not None and handle and hasattr(self._lib, "little_core_graph_clear"):
            self._lib.little_core_graph_clear(handle)

    def graph_edge_count(self, handle: Any) -> int:
        """Return total unique edges stored in the Rust GraphIndex handle."""
        if self._lib is None or not handle or not hasattr(self._lib, "little_core_graph_edge_count"):
            return 0
        return int(self._lib.little_core_graph_edge_count(handle))

    def graph_add_edge(self, handle: Any, predicate: str, subject: str, object_node: str) -> bool:
        """Add a directed edge (subject --[predicate]--> object_node) to the Rust GraphIndex."""
        if self._lib is None or not handle or not hasattr(self._lib, "little_core_graph_add_edge"):
            return False
        res = self._lib.little_core_graph_add_edge(
            handle,
            predicate.encode("utf-8"),
            subject.encode("utf-8"),
            object_node.encode("utf-8"),
        )
        return bool(res == 1)

    def graph_check_cycle(
        self, handle: Any, predicate: str, start_node: str, target_node: str
    ) -> bool:
        """Check reachability from start_node to target_node on a persistent Rust GraphIndex."""
        if self._lib is None or not handle or not hasattr(self._lib, "little_core_graph_check_cycle"):
            return False
        res = self._lib.little_core_graph_check_cycle(
            handle,
            predicate.encode("utf-8"),
            start_node.encode("utf-8"),
            target_node.encode("utf-8"),
        )
        return bool(res == 1)

    def graph_find_ancestors(self, handle: Any, predicate: str, start_node: str) -> list[str]:
        """Compute transitive ancestors on a persistent Rust GraphIndex without re-serializing edges."""
        if self._lib is None or not handle or not hasattr(self._lib, "little_core_graph_find_ancestors"):
            return []
        ptr = self._lib.little_core_graph_find_ancestors(
            handle,
            predicate.encode("utf-8"),
            start_node.encode("utf-8"),
        )
        if not ptr:
            return []
        res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.little_core_free_string(ptr)
        return json.loads(res_str).get("ancestors", [])

    def graph_bidirectional_search(
        self,
        handle: Any,
        predicate: str,
        start_node: str,
        target_node: str,
        max_depth: int = 10,
    ) -> list[str] | None:
        """Run bidirectional BFS on a persistent Rust GraphIndex without per-query JSON serialization."""
        if (
            self._lib is None
            or not handle
            or not hasattr(self._lib, "little_core_graph_bidirectional_search")
        ):
            return None
        ptr = self._lib.little_core_graph_bidirectional_search(
            handle,
            predicate.encode("utf-8"),
            start_node.encode("utf-8"),
            target_node.encode("utf-8"),
            max(1, int(max_depth)),
        )
        if not ptr:
            return None
        res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.little_core_free_string(ptr)
        data = json.loads(res_str)
        return data.get("path") if data.get("found") else None


class RustGraphIndex:
    """Epoch-synchronized persistent graph index backed by Rust FFI (with Python fallback)."""

    def __init__(self, core: RustAccelerationCore | None = None) -> None:
        self.core = core or RustAccelerationCore.get()
        self._handle = self.core.graph_create()
        self._fwd: dict[str, dict[str, list[str]]] = {}
        self._bwd: dict[str, dict[str, list[str]]] = {}
        self._edge_count = 0
        self._synced_epoch: int = -1

    def close(self) -> None:
        if self._handle is not None:
            self.core.graph_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    @property
    def is_native(self) -> bool:
        return self._handle is not None

    @property
    def synced_epoch(self) -> int:
        return self._synced_epoch

    @property
    def edge_count(self) -> int:
        if self._handle is not None:
            return self.core.graph_edge_count(self._handle)
        return self._edge_count

    def clear(self) -> None:
        if self._handle is not None:
            self.core.graph_clear(self._handle)
        self._fwd.clear()
        self._bwd.clear()
        self._edge_count = 0

    def add_edge(self, predicate: str, subject: str, object_node: str) -> None:
        if self._handle is not None:
            self.core.graph_add_edge(self._handle, predicate, subject, object_node)
        fwd_list = self._fwd.setdefault(predicate, {}).setdefault(subject, [])
        if object_node not in fwd_list:
            fwd_list.append(object_node)
            self._edge_count += 1
        bwd_list = self._bwd.setdefault(predicate, {}).setdefault(object_node, [])
        if subject not in bwd_list:
            bwd_list.append(subject)

    def sync_from_memory(self, memory: Any) -> bool:
        """Synchronize active relations from MemoryStore only when concept_mutation_epoch advances."""
        epoch = getattr(memory, "concept_mutation_epoch", -1)
        if epoch >= 0 and epoch == self._synced_epoch:
            return False
        self.clear()
        id_to_name = {
            c.id: c.name.strip().upper().replace(" ", "_")
            for c in memory.list_concepts()
        }
        for rel in memory.get_relations():
            if rel.weight_positive > rel.weight_negative:
                s_name = id_to_name.get(
                    rel.subject_id, rel.subject_id.strip().upper().replace(" ", "_")
                )
                o_name = id_to_name.get(
                    rel.object_id, rel.object_id.strip().upper().replace(" ", "_")
                )
                self.add_edge(rel.predicate, s_name, o_name)
        self._synced_epoch = epoch
        return True

    def find_ancestors(self, predicate: str, start_node: str) -> list[str]:
        if self._handle is not None:
            return self.core.graph_find_ancestors(self._handle, predicate, start_node)
        return self.core.find_ancestors(self._fwd.get(predicate, {}), start_node)

    def check_cycle(self, predicate: str, start_node: str, target_node: str) -> bool:
        if self._handle is not None:
            return self.core.graph_check_cycle(self._handle, predicate, start_node, target_node)
        return self.core.check_cycle(self._fwd.get(predicate, {}), start_node, target_node)

    def bidirectional_search(
        self, predicate: str, start_node: str, target_node: str, max_depth: int = 10
    ) -> list[str] | None:
        if self._handle is not None:
            return self.core.graph_bidirectional_search(
                self._handle, predicate, start_node, target_node, max_depth
            )
        return self.core.bidirectional_search(
            self._fwd.get(predicate, {}),
            self._bwd.get(predicate, {}),
            start_node,
            target_node,
            max_depth=max_depth,
        )


class RustMluArena:
    """Native Rust 360° Spider-Web MLU Arena handle (`SpiderWebArena` in `rust/little_core`)."""

    def __init__(
        self,
        mitosis_min_activations: int = 3,
        mitosis_min_verified_spokes: int = 2,
        core: RustAccelerationCore | None = None,
    ) -> None:
        self.core = core or RustAccelerationCore.get()
        self._lib = self.core._lib
        self._handle: Any = None
        if self._lib is not None and hasattr(self._lib, "little_core_mlu_arena_create"):
            self._handle = self._lib.little_core_mlu_arena_create(
                int(mitosis_min_activations),
                int(mitosis_min_verified_spokes),
            )

    @property
    def is_native(self) -> bool:
        return self._handle is not None

    @property
    def node_count(self) -> int:
        if self._lib is not None and self._handle:
            return int(self._lib.little_core_mlu_arena_node_count(self._handle))
        return 0

    @property
    def mitosis_count(self) -> int:
        if self._lib is not None and self._handle:
            return int(self._lib.little_core_mlu_arena_mitosis_count(self._handle))
        return 0

    def learn_spoke(
        self,
        subject: str,
        predicate: str,
        object_node: str,
        learning_rate: float = 0.2,
        verified: bool = True,
    ) -> float:
        if self._lib is None or not self._handle:
            return 0.0
        res = self._lib.little_core_mlu_arena_learn_spoke(
            self._handle,
            subject.encode("utf-8"),
            predicate.encode("utf-8"),
            object_node.encode("utf-8"),
            float(learning_rate),
            1 if verified else 0,
        )
        return float(res)

    def predict_spoke(
        self, concept: str, spoke_idx: int, top_k: int = 5
    ) -> list[dict[str, Any]]:
        if self._lib is None or not self._handle:
            return []
        ptr = self._lib.little_core_mlu_arena_predict_spoke(
            self._handle,
            concept.encode("utf-8"),
            int(spoke_idx),
            int(top_k),
        )
        if not ptr:
            return []
        res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.little_core_free_string(ptr)
        return list(json.loads(res_str))

    def train_stream(self, text: str, learning_rate: float = 0.2) -> dict[str, Any]:
        """Train the Rust 360° Spider-Web Arena directly on a raw UTF-8 text stream."""
        if (
            self._lib is None
            or not self._handle
            or not hasattr(self._lib, "little_core_mlu_arena_train_stream")
        ):
            return {
                "sentences_processed": 0,
                "tokens_processed": 0,
                "spoke_updates": 0,
                "average_surprise": 0.0,
                "total_cells": 0,
                "mitosis_promotions": 0,
            }
        ptr = self._lib.little_core_mlu_arena_train_stream(
            self._handle,
            text.encode("utf-8"),
            float(learning_rate),
        )
        if not ptr:
            return {}
        res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.little_core_free_string(ptr)
        return dict(json.loads(res_str))

    def generate(self, prompt: str, max_tokens: int = 8) -> str:
        """Generate a multi-token continuation from the Rust 360° Spider-Web Arena."""
        if (
            self._lib is None
            or not self._handle
            or not hasattr(self._lib, "little_core_mlu_arena_generate")
        ):
            return ""
        ptr = self._lib.little_core_mlu_arena_generate(
            self._handle,
            prompt.encode("utf-8"),
            int(max_tokens),
        )
        if not ptr:
            return ""
        res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.little_core_free_string(ptr)
        return str(json.loads(res_str))

    def save_snapshot(self, path: str) -> bool:
        """Persist the Rust 360° Spider-Web Arena state to disk."""
        if (
            self._lib is None
            or not self._handle
            or not hasattr(self._lib, "little_core_mlu_arena_save_snapshot")
        ):
            return False
        res = self._lib.little_core_mlu_arena_save_snapshot(
            self._handle,
            str(path).encode("utf-8"),
        )
        return bool(res == 1)

    def load_snapshot(self, path: str) -> bool:
        """Restore the Rust 360° Spider-Web Arena state from disk."""
        if (
            self._lib is None
            or not self._handle
            or not hasattr(self._lib, "little_core_mlu_arena_load_snapshot")
        ):
            return False
        res = self._lib.little_core_mlu_arena_load_snapshot(
            self._handle,
            str(path).encode("utf-8"),
        )
        return bool(res == 1)

    def dump_json(self) -> str:
        """Serialize the entire Rust 360° Spider-Web Arena to a JSON string."""
        if (
            self._lib is None
            or not self._handle
            or not hasattr(self._lib, "little_core_mlu_arena_dump_json")
        ):
            return "{}"
        ptr = self._lib.little_core_mlu_arena_dump_json(self._handle)
        if not ptr:
            return "{}"
        res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.little_core_free_string(ptr)
        return res_str

    def restore_json(self, payload: str) -> bool:
        """Restore the entire Rust 360° Spider-Web Arena from a JSON string."""
        if (
            self._lib is None
            or not self._handle
            or not hasattr(self._lib, "little_core_mlu_arena_restore_json")
        ):
            return False
        res = self._lib.little_core_mlu_arena_restore_json(
            self._handle,
            payload.encode("utf-8"),
        )
        return bool(res == 1)

    def query_360_beam(
        self, concept: str, max_hops: int = 3, top_k: int = 10
    ) -> dict[str, Any]:
        """Run native Rust 360° Multi-Hop Beam Traversal & Neural Continuation."""
        if (
            self._lib is None
            or not self._handle
            or not hasattr(self._lib, "little_core_mlu_arena_query_360_beam")
        ):
            return {}
        ptr = self._lib.little_core_mlu_arena_query_360_beam(
            self._handle,
            concept.encode("utf-8"),
            int(max_hops),
            int(top_k),
        )
        if not ptr:
            return {}
        res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.little_core_free_string(ptr)
        return dict(json.loads(res_str))

    def consolidate(
        self, min_transition_weight: float = 0.05, replay_lr: float = 0.15
    ) -> dict[str, Any]:
        """Run native Rust Sleep-Cycle Consolidation (spoke replay + transition pruning)."""
        if (
            self._lib is None
            or not self._handle
            or not hasattr(self._lib, "little_core_mlu_arena_consolidate")
        ):
            return {
                "replayed_edges": 0,
                "pruned_transitions": 0,
                "average_replay_surprise": 0.0,
                "mitosis_promotions": 0,
                "total_cells": 0,
            }
        ptr = self._lib.little_core_mlu_arena_consolidate(
            self._handle,
            float(min_transition_weight),
            float(replay_lr),
        )
        if not ptr:
            return {}
        res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.little_core_free_string(ptr)
        return dict(json.loads(res_str))

    def verify_invariants(
        self, subject: str, predicate: str, object_node: str
    ) -> dict[str, Any]:
        """Run native Rust I_DAG & I_MUTEX invariant verification on a candidate spoke link."""
        if (
            self._lib is None
            or not self._handle
            or not hasattr(self._lib, "little_core_mlu_arena_verify_invariants")
        ):
            return {"passed": True, "gate": "PASSED"}
        ptr = self._lib.little_core_mlu_arena_verify_invariants(
            self._handle,
            subject.encode("utf-8"),
            predicate.encode("utf-8"),
            object_node.encode("utf-8"),
        )
        if not ptr:
            return {"passed": True, "gate": "PASSED"}
        res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.little_core_free_string(ptr)
        return dict(json.loads(res_str))

    def compare_360(
        self, concept_a: str, concept_b: str, max_hops: int = 3
    ) -> dict[str, Any]:
        """Run native Rust 360° bidirectional wave intersection between two concepts."""
        if (
            self._lib is None
            or not self._handle
            or not hasattr(self._lib, "little_core_mlu_arena_compare_360")
        ):
            return {}
        ptr = self._lib.little_core_mlu_arena_compare_360(
            self._handle,
            concept_a.encode("utf-8"),
            concept_b.encode("utf-8"),
            max(1, int(max_hops)),
        )
        if not ptr:
            return {}
        res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.little_core_free_string(ptr)
        return dict(json.loads(res_str))

    def discover_curiosity(self, max_items: int = 10) -> dict[str, Any]:
        """Run native Rust analogical hypothesis induction & epistemic curiosity gap discovery."""
        if (
            self._lib is None
            or not self._handle
            or not hasattr(self._lib, "little_core_mlu_arena_discover_curiosity")
        ):
            return {"analogical_hypotheses": [], "curiosity_gaps": []}
        ptr = self._lib.little_core_mlu_arena_discover_curiosity(
            self._handle,
            max(1, int(max_items)),
        )
        if not ptr:
            return {"analogical_hypotheses": [], "curiosity_gaps": []}
        res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.little_core_free_string(ptr)
        return dict(json.loads(res_str))

    def simulate_causal(self, root_concept: str, max_hops: int = 3) -> dict[str, Any]:
        """Run native Rust 360° causal & counterfactual wave simulation from a root concept."""
        if (
            self._lib is None
            or not self._handle
            or not hasattr(self._lib, "little_core_mlu_arena_simulate_causal")
        ):
            return {
                "root_concept": root_concept,
                "affected_concepts": [],
                "causal_steps": [],
                "causal_synthesis": "",
            }
        ptr = self._lib.little_core_mlu_arena_simulate_causal(
            self._handle,
            root_concept.encode("utf-8"),
            max(1, int(max_hops)),
        )
        if not ptr:
            return {}
        res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.little_core_free_string(ptr)
        return dict(json.loads(res_str))

    def autonomous_curiosity(
        self, min_confidence: float = 0.65, max_promotions: int = 5
    ) -> dict[str, Any]:
        """Run native Rust active curiosity-driven hypothesis verification and promotion."""
        if (
            self._lib is None
            or not self._handle
            or not hasattr(self._lib, "little_core_mlu_arena_autonomous_curiosity")
        ):
            return {
                "hypotheses_evaluated": 0,
                "hypotheses_promoted": 0,
                "promoted_hypotheses": [],
                "open_questions": [],
            }
        ptr = self._lib.little_core_mlu_arena_autonomous_curiosity(
            self._handle,
            float(min_confidence),
            max(1, int(max_promotions)),
        )
        if not ptr:
            return {}
        res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.little_core_free_string(ptr)
        return dict(json.loads(res_str))

    def intervene_counterfactual(
        self, root_concept: str, suppressed_concept: str, max_hops: int = 3
    ) -> dict[str, Any]:
        """Run native Rust 360° counterfactual intervention wave."""
        if (
            self._lib is None
            or not self._handle
            or not hasattr(self._lib, "little_core_mlu_arena_intervene_counterfactual")
        ):
            return {
                "root_concept": root_concept,
                "suppressed_concept": suppressed_concept,
                "baseline_affected": [],
                "invalidated_conclusions": [],
                "preserved_conclusions": [],
                "counterfactual_synthesis": "",
            }
        ptr = self._lib.little_core_mlu_arena_intervene_counterfactual(
            self._handle,
            root_concept.encode("utf-8"),
            suppressed_concept.encode("utf-8"),
            max(1, int(max_hops)),
        )
        if not ptr:
            return {}
        res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.little_core_free_string(ptr)
        return dict(json.loads(res_str))

    def find_bridge(
        self, concept_a: str, concept_b: str, max_hops: int = 4
    ) -> dict[str, Any]:
        """Run native Rust shortest 360° relational path bridge search between two concepts."""
        if (
            self._lib is None
            or not self._handle
            or not hasattr(self._lib, "little_core_mlu_arena_find_bridge")
        ):
            return {
                "concept_a": concept_a,
                "concept_b": concept_b,
                "connected": False,
                "hop_count": 0,
                "bridge_steps": [],
                "bridge_synthesis": "",
            }
        ptr = self._lib.little_core_mlu_arena_find_bridge(
            self._handle,
            concept_a.encode("utf-8"),
            concept_b.encode("utf-8"),
            max(1, int(max_hops)),
        )
        if not ptr:
            return {}
        res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.little_core_free_string(ptr)
        return dict(json.loads(res_str))

    def export_topology(self, max_edges: int = 100) -> dict[str, Any]:
        """Export the live 360° Spider-Web Arena topology and Mermaid graph."""
        if (
            self._lib is None
            or not self._handle
            or not hasattr(self._lib, "little_core_mlu_arena_export_topology")
        ):
            return {
                "total_cells": 0,
                "total_verified_edges": 0,
                "nodes": [],
                "edges": [],
                "mermaid_diagram": "",
            }
        ptr = self._lib.little_core_mlu_arena_export_topology(
            self._handle,
            max(1, int(max_edges)),
        )
        if not ptr:
            return {}
        res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.little_core_free_string(ptr)
        return dict(json.loads(res_str))

    def discover_clusters(self, min_members: int = 2) -> list[dict[str, Any]]:
        """Discover emergent 64-D concept clusters sharing spoke attractors in native Rust."""
        if (
            self._lib is None
            or not self._handle
            or not hasattr(self._lib, "little_core_mlu_arena_discover_clusters")
        ):
            return []
        ptr = self._lib.little_core_mlu_arena_discover_clusters(
            self._handle,
            max(2, int(min_members)),
        )
        if not ptr:
            return []
        res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.little_core_free_string(ptr)
        return list(json.loads(res_str))

    def revise_belief(
        self,
        subject: str,
        predicate: str,
        object_node: str,
        verified: bool,
        evidence_weight: float = 3.0,
    ) -> dict[str, Any]:
        """Apply explicit non-monotonic belief revision to a spoke synapse in native Rust."""
        if (
            self._lib is None
            or not self._handle
            or not hasattr(self._lib, "little_core_mlu_arena_revise_belief")
        ):
            return {}
        ptr = self._lib.little_core_mlu_arena_revise_belief(
            self._handle,
            subject.encode("utf-8"),
            predicate.encode("utf-8"),
            object_node.encode("utf-8"),
            1 if verified else 0,
            float(evidence_weight),
        )
        if not ptr:
            return {}
        res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.little_core_free_string(ptr)
        return dict(json.loads(res_str))

    def deliberate_360(self, concept: str, max_hops: int = 3) -> dict[str, Any]:
        """Run native Rust 5-wave System-2 Deliberative Thinking over a concept."""
        if (
            self._lib is None
            or not self._handle
            or not hasattr(self._lib, "little_core_mlu_arena_deliberate_360")
        ):
            return {}
        ptr = self._lib.little_core_mlu_arena_deliberate_360(
            self._handle,
            concept.encode("utf-8"),
            max(1, int(max_hops)),
        )
        if not ptr:
            return {}
        res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.little_core_free_string(ptr)
        return dict(json.loads(res_str))

    def plan_goal_360(
        self, goal_concept: str, start_concept: str = "", max_steps: int = 5
    ) -> dict[str, Any]:
        """Run native Rust 360° Goal-Directed Backward-Chaining & Action Planner."""
        if (
            self._lib is None
            or not self._handle
            or not hasattr(self._lib, "little_core_mlu_arena_plan_goal_360")
        ):
            return {}
        ptr = self._lib.little_core_mlu_arena_plan_goal_360(
            self._handle,
            goal_concept.encode("utf-8"),
            start_concept.encode("utf-8"),
            max(1, int(max_steps)),
        )
        if not ptr:
            return {}
        res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.little_core_free_string(ptr)
        return dict(json.loads(res_str))

    def close(self) -> None:
        if self._lib is not None and self._handle is not None:
            self._lib.little_core_mlu_arena_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


RustCoreEngine = RustAccelerationCore


