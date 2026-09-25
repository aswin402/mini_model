"""Versioned defaults for construction confidence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from little.core.runtime_paths import RuntimePaths
from typing import Any


@dataclass(frozen=True)
class ConstructionPolicy:
    """Data-owned calibration for grammar constructions without evidence."""

    version: str
    default_confidence: float
    learned_confidence: float

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("construction policy version must be non-empty")
        for field_name in ("default_confidence", "learned_confidence"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{field_name} must be a finite probability")

    @classmethod
    def load(cls, directory: Path) -> ConstructionPolicy:
        path = Path(directory) / "construction_policy.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            root: Any = payload["construction_policy"]
            if not isinstance(root, dict):
                raise TypeError("construction_policy must be an object")
            return cls(
                version=root["version"],
                default_confidence=root["default_confidence"],
                learned_confidence=root["learned_confidence"],
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid construction policy at {path}: {exc}") from exc

    @classmethod
    def default(cls) -> ConstructionPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
