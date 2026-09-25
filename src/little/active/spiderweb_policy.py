"""Versioned configuration for autonomous spider-web growth."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from little.core.runtime_paths import RuntimePaths
from typing import Any


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be a finite number")
    return numeric


@dataclass(frozen=True)
class SpiderWebGrowthPolicy:
    """Data-owned thresholds, naming, and curiosity prompts for web growth."""

    version: str
    minimum_cluster_size: int
    category_utility_threshold: float
    cluster_prefix: str
    cluster_suffix: str
    fallback_hypernym: str
    minimum_mereology_parts: int
    uninitialized_freshness: float
    uninitialized_oxidation: float
    require_episodic_state_for_dynamics_gap: bool
    gap_templates: dict[str, str]

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("spider-web policy version must be non-empty")
        for field_name in ("minimum_cluster_size", "minimum_mereology_parts"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.category_utility_threshold < 0 or not math.isfinite(
            self.category_utility_threshold
        ):
            raise ValueError("category_utility_threshold must be non-negative")
        for field_name in ("cluster_prefix", "cluster_suffix", "fallback_hypernym"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        _finite_number(self.uninitialized_freshness, "uninitialized_freshness")
        _finite_number(self.uninitialized_oxidation, "uninitialized_oxidation")
        if not self.gap_templates or any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
            for key, value in self.gap_templates.items()
        ):
            raise ValueError("gap_templates must map non-empty names to strings")

    @classmethod
    def load(cls, directory: Path) -> SpiderWebGrowthPolicy:
        path = Path(directory) / "spiderweb_growth_policy.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            root = payload["spiderweb_growth_policy"]
            if not isinstance(root, dict):
                raise TypeError("spiderweb_growth_policy must be an object")
            templates = root["gap_templates"]
            if not isinstance(templates, dict):
                raise TypeError("gap_templates must be an object")
            return cls(
                version=root["version"],
                minimum_cluster_size=root["minimum_cluster_size"],
                category_utility_threshold=_finite_number(
                    root["category_utility_threshold"],
                    "category_utility_threshold",
                ),
                cluster_prefix=root["cluster_prefix"],
                cluster_suffix=root["cluster_suffix"],
                fallback_hypernym=root["fallback_hypernym"],
                minimum_mereology_parts=root["minimum_mereology_parts"],
                uninitialized_freshness=_finite_number(
                    root["uninitialized_freshness"], "uninitialized_freshness"
                ),
                uninitialized_oxidation=_finite_number(
                    root["uninitialized_oxidation"], "uninitialized_oxidation"
                ),
                require_episodic_state_for_dynamics_gap=root[
                    "require_episodic_state_for_dynamics_gap"
                ],
                gap_templates=dict(templates),
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid spider-web policy at {path}: {exc}") from exc

    @classmethod
    def default(cls) -> SpiderWebGrowthPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
