"""Versioned defaults for semantic and episodic memory operations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from little.core.runtime_paths import RuntimePaths
from typing import Any


@dataclass(frozen=True)
class PagedMemoryConfig:
    """Configuration for 3-tier paged concept memory (Disk -> RAM LRU -> Resident)."""

    lru_cache_capacity: int = 512
    resident_capacity: int = 32
    activation_decay: float = 0.85
    activation_threshold: float = 0.1
    spreading_factor: float = 0.5

    def __post_init__(self) -> None:
        if self.lru_cache_capacity <= 0:
            raise ValueError("lru_cache_capacity must be positive")
        if self.resident_capacity <= 0:
            raise ValueError("resident_capacity must be positive")
        if not (0.0 < self.activation_decay <= 1.0):
            raise ValueError("activation_decay must be in (0, 1]")
        if not (0.0 <= self.activation_threshold < 1.0):
            raise ValueError("activation_threshold must be in [0, 1)")
        if not (0.0 <= self.spreading_factor <= 1.0):
            raise ValueError("spreading_factor must be in [0, 1]")


@dataclass(frozen=True)
class MemoryPolicy:
    """Data-owned persistence defaults used by MemoryStore and PagedConceptStore."""

    version: str
    concept_confidence: float
    experience_query_limit: int | None
    bulk_import_batch_size: int
    paged_memory: PagedMemoryConfig = field(default_factory=PagedMemoryConfig)

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("memory policy version must be non-empty")
        if (
            isinstance(self.concept_confidence, bool)
            or not isinstance(self.concept_confidence, (int, float))
            or not math.isfinite(float(self.concept_confidence))
            or not 0.0 <= self.concept_confidence <= 1.0
        ):
            raise ValueError("concept_confidence must be a finite probability")
        if self.experience_query_limit is not None and (
            isinstance(self.experience_query_limit, bool)
            or not isinstance(self.experience_query_limit, int)
            or self.experience_query_limit < 0
        ):
            raise ValueError(
                "experience_query_limit must be a non-negative integer or None"
            )
        if (
            isinstance(self.bulk_import_batch_size, bool)
            or not isinstance(self.bulk_import_batch_size, int)
            or self.bulk_import_batch_size <= 0
        ):
            raise ValueError("bulk_import_batch_size must be a positive integer")

    @classmethod
    def load(cls, directory: Path) -> MemoryPolicy:
        path = Path(directory) / "memory_policy.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            root: Any = payload["memory_policy"]
            if not isinstance(root, dict):
                raise TypeError("memory_policy must be an object")

            paged_raw = root.get("paged_memory", {})
            paged_cfg = (
                PagedMemoryConfig(
                    lru_cache_capacity=int(paged_raw.get("lru_cache_capacity", 512)),
                    resident_capacity=int(paged_raw.get("resident_capacity", 32)),
                    activation_decay=float(paged_raw.get("activation_decay", 0.85)),
                    activation_threshold=float(paged_raw.get("activation_threshold", 0.1)),
                    spreading_factor=float(paged_raw.get("spreading_factor", 0.5)),
                )
                if isinstance(paged_raw, dict)
                else PagedMemoryConfig()
            )

            return cls(
                version=root["version"],
                concept_confidence=root["concept_confidence"],
                experience_query_limit=root["experience_query_limit"],
                bulk_import_batch_size=root["bulk_import_batch_size"],
                paged_memory=paged_cfg,
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid memory policy at {path}: {exc}") from exc

    @classmethod
    def default(cls) -> MemoryPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
