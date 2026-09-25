"""Versioned defaults for semantic and episodic memory operations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from little.core.runtime_paths import RuntimePaths
from typing import Any


@dataclass(frozen=True)
class MemoryPolicy:
    """Data-owned persistence defaults used by MemoryStore."""

    version: str
    concept_confidence: float
    experience_query_limit: int | None
    bulk_import_batch_size: int

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
            return cls(
                version=root["version"],
                concept_confidence=root["concept_confidence"],
                experience_query_limit=root["experience_query_limit"],
                bulk_import_batch_size=root["bulk_import_batch_size"],
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid memory policy at {path}: {exc}") from exc

    @classmethod
    def default(cls) -> MemoryPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
