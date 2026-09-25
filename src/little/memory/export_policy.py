"""Versioned configuration for memory export boundaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from little.core.runtime_paths import RuntimePaths
from typing import Any


@dataclass(frozen=True)
class MemoryExportPolicy:
    """Control export volume without embedding a fixed history cutoff."""

    version: str
    experience_limit: int | None

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("memory export policy version must be non-empty")
        if self.experience_limit is not None and (
            isinstance(self.experience_limit, bool)
            or not isinstance(self.experience_limit, int)
            or self.experience_limit < 0
        ):
            raise ValueError(
                "experience_limit must be a non-negative integer or None"
            )

    @classmethod
    def load(cls, directory: Path) -> MemoryExportPolicy:
        path = Path(directory) / "memory_export_policy.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            root = payload["memory_export_policy"]
            if not isinstance(root, dict):
                raise TypeError("memory_export_policy must be an object")
            limit: Any = root["experience_limit"]
            return cls(version=root["version"], experience_limit=limit)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid memory export policy at {path}: {exc}") from exc

    @classmethod
    def default(cls) -> MemoryExportPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
