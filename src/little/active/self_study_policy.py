"""Versioned configuration for autonomous self-study and web growth."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from little.core.runtime_paths import RuntimePaths
from typing import Any


_COVERAGE_KINDS = frozenset({"relation", "attributes", "experience"})


def _probability(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite probability")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field_name} must be a finite probability in [0, 1]")
    return numeric


@dataclass(frozen=True)
class SelfStudyAxis:
    """One configurable spoke and the memory signal that covers it."""

    name: str
    coverage: str

    def __post_init__(self) -> None:
        clean_name = self.name.strip().lower()
        clean_coverage = self.coverage.strip().lower()
        if not clean_name:
            raise ValueError("self-study axis name must be non-empty")
        if clean_coverage not in _COVERAGE_KINDS:
            raise ValueError(
                f"unsupported self-study axis coverage: {self.coverage!r}"
            )
        object.__setattr__(self, "name", clean_name)
        object.__setattr__(self, "coverage", clean_coverage)


@dataclass(frozen=True)
class SelfStudyPolicy:
    """Data-owned axes, provenance calibration, and bounded study settings."""

    version: str
    axes: tuple[SelfStudyAxis, ...]
    hierarchy_axis: str
    default_axis: str
    no_analogy_axis: str
    hypothesis_probability: float
    hypothesis_uncertainty: float
    model_id: str
    model_version: str
    max_steps: int

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("self-study policy version must be non-empty")
        if not self.axes:
            raise ValueError("self-study policy must define at least one axis")
        names = [axis.name for axis in self.axes]
        if len(names) != len(set(names)):
            raise ValueError("duplicate self-study axis name")
        for field_name in ("hierarchy_axis", "default_axis"):
            value = getattr(self, field_name).strip().lower()
            if value not in names:
                raise ValueError(f"{field_name} must reference a configured axis")
            object.__setattr__(self, field_name, value)
        for field_name in ("no_analogy_axis", "model_id", "model_version"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps <= 0
        ):
            raise ValueError("self-study max_steps must be a positive integer")
        _probability(self.hypothesis_probability, "hypothesis_probability")
        _probability(self.hypothesis_uncertainty, "hypothesis_uncertainty")

    @property
    def axis_names(self) -> tuple[str, ...]:
        return tuple(axis.name for axis in self.axes)

    def axis(self, name: str) -> SelfStudyAxis | None:
        clean_name = name.strip().lower()
        return next((axis for axis in self.axes if axis.name == clean_name), None)

    @classmethod
    def load(cls, directory: Path) -> SelfStudyPolicy:
        path = Path(directory) / "self_study_policy.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            root = payload["self_study_policy"]
            if not isinstance(root, dict):
                raise TypeError("self_study_policy must be an object")
            raw_axes = root["axes"]
            if not isinstance(raw_axes, list):
                raise TypeError("axes must be a list")
            axes = tuple(
                SelfStudyAxis(name=item["name"], coverage=item["coverage"])
                for item in raw_axes
                if isinstance(item, dict)
            )
            if len(axes) != len(raw_axes):
                raise TypeError("each self-study axis must be an object")
            return cls(
                version=root["version"],
                axes=axes,
                hierarchy_axis=root["hierarchy_axis"],
                default_axis=root["default_axis"],
                no_analogy_axis=root["no_analogy_axis"],
                hypothesis_probability=_probability(
                    root["hypothesis_probability"], "hypothesis_probability"
                ),
                hypothesis_uncertainty=_probability(
                    root["hypothesis_uncertainty"], "hypothesis_uncertainty"
                ),
                model_id=root["model_id"],
                model_version=root["model_version"],
                max_steps=root["max_steps"],
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid self-study policy at {path}: {exc}") from exc

    @classmethod
    def default(cls) -> SelfStudyPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
