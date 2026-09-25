"""Versioned configuration for bounded System 2 graph reasoning."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from little.core.runtime_paths import RuntimePaths
from typing import Any


def _probability(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite probability")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be a finite probability in [0, 1]")
    return value


@dataclass(frozen=True)
class ReasoningPolicy:
    """Data-owned limits and confidence calibration for infilling proofs."""

    version: str
    max_depth: int
    fast_confidence: float
    verified_confidence: float
    unknown_confidence: float
    inheritance_confidence_factor: float
    inference_unknown_confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("reasoning policy version must be non-empty")
        if (
            isinstance(self.max_depth, bool)
            or not isinstance(self.max_depth, int)
            or self.max_depth <= 0
        ):
            raise ValueError("reasoning policy max_depth must be a positive integer")
        for field_name in (
            "fast_confidence",
            "verified_confidence",
            "unknown_confidence",
            "inheritance_confidence_factor",
            "inference_unknown_confidence",
        ):
            _probability(getattr(self, field_name), field_name)

    @classmethod
    def load(cls, directory: Path) -> ReasoningPolicy:
        path = Path(directory) / "reasoning_policy.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            root = payload["reasoning_policy"]
            if not isinstance(root, dict):
                raise TypeError("reasoning_policy must be an object")
            version = root["version"]
            max_depth = root["max_depth"]
            if (
                isinstance(max_depth, bool)
                or not isinstance(max_depth, int)
                or max_depth <= 0
            ):
                raise ValueError("max_depth must be a positive integer")
            return cls(
                version=version,
                max_depth=max_depth,
                fast_confidence=_probability(
                    root["fast_confidence"], "fast_confidence"
                ),
                verified_confidence=_probability(
                    root["verified_confidence"], "verified_confidence"
                ),
                unknown_confidence=_probability(
                    root["unknown_confidence"], "unknown_confidence"
                ),
                inheritance_confidence_factor=_probability(
                    root["inheritance_confidence_factor"],
                    "inheritance_confidence_factor",
                ),
                inference_unknown_confidence=_probability(
                    root["inference_unknown_confidence"],
                    "inference_unknown_confidence",
                ),
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid reasoning policy at {path}: {exc}") from exc

    @classmethod
    def default(cls) -> ReasoningPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
