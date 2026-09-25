"""Versioned confidence calibration for deterministic parser results."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from little.core.runtime_paths import RuntimePaths
from typing import Any


@dataclass(frozen=True)
class ParserConfidencePolicy:
    """Named confidence levels used by parser and deterministic query branches."""

    version: str
    certain: float
    derived: float
    unknown: float
    low: float

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("parser confidence policy version must be non-empty")
        for field_name in ("certain", "derived", "unknown", "low"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{field_name} must be a finite probability")

    @classmethod
    def load(cls, directory: Path) -> ParserConfidencePolicy:
        path = Path(directory) / "parser_confidence_policy.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            root: Any = payload["parser_confidence_policy"]
            if not isinstance(root, dict):
                raise TypeError("parser_confidence_policy must be an object")
            return cls(
                version=root["version"],
                certain=root["certain"],
                derived=root["derived"],
                unknown=root["unknown"],
                low=root["low"],
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid parser confidence policy at {path}: {exc}"
            ) from exc

    @classmethod
    def default(cls) -> ParserConfidencePolicy:
        return cls.load(RuntimePaths.default().schema_directory)
