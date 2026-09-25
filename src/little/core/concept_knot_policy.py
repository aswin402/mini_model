"""Versioned hydration configuration for persistent Concept Knots."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from little.core.runtime_paths import RuntimePaths
from typing import Any


@dataclass(frozen=True)
class ConceptKnotPolicy:
    """Map registry axes and stored attributes into the six knot dimensions."""

    version: str
    taxonomy_axis: str
    mereology_axis: str
    invariant_axis: str
    procedural_axis: str
    defaults: dict[str, float]
    mereology_roles: dict[str, str]

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("concept-knot policy version must be non-empty")
        for field_name in (
            "taxonomy_axis",
            "mereology_axis",
            "invariant_axis",
            "procedural_axis",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        required_defaults = {
            "freshness",
            "oxidation",
            "moisture",
            "temperature",
            "mass",
            "record_mass",
            "constructor_mass",
        }
        if not required_defaults.issubset(self.defaults):
            raise ValueError("concept-knot defaults are missing a required state value")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in self.defaults.values()
        ):
            raise ValueError("concept-knot defaults must be finite numbers")
        if any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
            for key, value in self.mereology_roles.items()
        ):
            raise ValueError("mereology_roles must map predicate names to roles")

    @classmethod
    def load(cls, directory: Path) -> ConceptKnotPolicy:
        path = Path(directory) / "concept_knot_policy.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            root = payload["concept_knot_policy"]
            if not isinstance(root, dict):
                raise TypeError("concept_knot_policy must be an object")
            defaults = root["defaults"]
            roles = root["mereology_roles"]
            if not isinstance(defaults, dict) or not isinstance(roles, dict):
                raise TypeError("defaults and mereology_roles must be objects")
            return cls(
                version=root["version"],
                taxonomy_axis=root["taxonomy_axis"],
                mereology_axis=root["mereology_axis"],
                invariant_axis=root["invariant_axis"],
                procedural_axis=root["procedural_axis"],
                defaults={str(key).strip().lower(): float(value) for key, value in defaults.items()},
                mereology_roles={
                    str(key).strip().lower(): str(value).strip()
                    for key, value in roles.items()
                },
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid concept-knot policy at {path}: {exc}") from exc

    @classmethod
    def default(cls) -> ConceptKnotPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
