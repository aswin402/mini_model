"""Data-driven material and perception profiles for continuous dynamics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from little.core.runtime_paths import RuntimePaths
from typing import Any


@dataclass(frozen=True)
class DynamicsProfile:
    name: str
    tau_whole_seconds: float
    tau_exposed_seconds: float
    default_interior_color: str
    perishable: bool
    categories: frozenset[str]
    attributes: frozenset[str]


@dataclass(frozen=True)
class PerceptionThresholds:
    color: tuple[float, float]
    condition: tuple[float, float, float]
    color_labels: tuple[str, str, str]
    condition_labels: tuple[str, str, str, str]


@dataclass(frozen=True)
class ODEParameters:
    reference_temperature_c: float
    temperature_scale_c: float
    protected_decay_rate_per_hour: float
    protected_oxidation_rate_per_hour: float
    exposed_decay_rate_per_hour: float
    exposed_enzymatic_rate_per_hour: float
    protected_moisture_loss_per_hour: float
    exposed_moisture_loss_per_hour: float
    minimum_moisture: float


@dataclass(frozen=True)
class ContinuousParameters:
    """Data-owned defaults and thermal controls for the physical-state engine."""

    q10: float
    min_rate_factor: float
    max_rate_factor: float
    default_temperature_c: float
    initial_freshness: float
    initial_oxidation: float
    protected_color_oxidation_threshold: float


class DynamicsProfileRegistry:
    def __init__(
        self,
        profiles: dict[str, DynamicsProfile],
        defaults: dict[str, str],
        perception: PerceptionThresholds,
        ode: ODEParameters,
        continuous: ContinuousParameters,
    ) -> None:
        self._profiles = profiles
        self._defaults = defaults
        self.perception = perception
        self.ode = ode
        self.continuous = continuous

    @classmethod
    def load(cls, directory: Path) -> DynamicsProfileRegistry:
        path = Path(directory) / "dynamics_profiles.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_profiles = payload.get("dynamics_profiles")
        if not isinstance(raw_profiles, list):
            raise TypeError(f"{path} must contain a list under 'dynamics_profiles'")

        profiles: dict[str, DynamicsProfile] = {}
        for raw in raw_profiles:
            if not isinstance(raw, dict):
                raise TypeError(f"{path} profiles must be objects")
            name = str(raw["name"]).strip().lower()
            if name in profiles:
                raise ValueError(f"duplicate dynamics profile: {name}")
            profiles[name] = DynamicsProfile(
                name=name,
                tau_whole_seconds=float(raw["tau_whole_seconds"]),
                tau_exposed_seconds=float(raw["tau_exposed_seconds"]),
                default_interior_color=str(raw.get("default_interior_color", "white")),
                perishable=bool(raw.get("perishable", False)),
                categories=frozenset(
                    str(value).strip().lower() for value in raw.get("categories", [])
                ),
                attributes=frozenset(
                    str(value).strip().lower() for value in raw.get("attributes", [])
                ),
            )

        defaults = payload.get("defaults", {})
        if not isinstance(defaults, dict):
            raise TypeError(f"{path} field 'defaults' must be an object")
        for key in ("whole", "exposed", "fallback"):
            profile_name = str(defaults.get(key, "")).strip().lower()
            if profile_name not in profiles:
                raise ValueError(f"{path} default {key!r} references unknown profile")

        raw_perception = payload.get("perception", {})
        if not isinstance(raw_perception, dict):
            raise TypeError(f"{path} field 'perception' must be an object")
        color = tuple(float(value) for value in raw_perception["color_thresholds"])
        condition = tuple(
            float(value) for value in raw_perception["condition_thresholds"]
        )
        if len(color) != 2 or len(condition) != 3:
            raise ValueError(f"{path} perception thresholds have invalid lengths")
        color_labels = tuple(str(value) for value in raw_perception["color_labels"])
        condition_labels = tuple(
            str(value) for value in raw_perception["condition_labels"]
        )
        if len(color_labels) != 3 or len(condition_labels) != 4:
            raise ValueError(f"{path} perception labels have invalid lengths")

        raw_ode = payload.get("ode", {})
        if not isinstance(raw_ode, dict):
            raise TypeError(f"{path} field 'ode' must be an object")
        ode_fields = (
            "reference_temperature_c",
            "temperature_scale_c",
            "protected_decay_rate_per_hour",
            "protected_oxidation_rate_per_hour",
            "exposed_decay_rate_per_hour",
            "exposed_enzymatic_rate_per_hour",
            "protected_moisture_loss_per_hour",
            "exposed_moisture_loss_per_hour",
            "minimum_moisture",
        )
        if any(field not in raw_ode for field in ode_fields):
            missing = [field for field in ode_fields if field not in raw_ode]
            raise ValueError(f"{path} ode configuration is missing: {missing}")

        raw_continuous = payload.get("continuous")
        if not isinstance(raw_continuous, dict):
            raise TypeError(f"{path} field 'continuous' must be an object")
        continuous_fields = (
            "q10",
            "min_rate_factor",
            "max_rate_factor",
            "default_temperature_c",
            "initial_freshness",
            "initial_oxidation",
            "protected_color_oxidation_threshold",
        )
        if any(field not in raw_continuous for field in continuous_fields):
            missing = [
                field for field in continuous_fields if field not in raw_continuous
            ]
            raise ValueError(f"{path} continuous configuration is missing: {missing}")

        return cls(
            profiles=profiles,
            defaults={
                key: str(defaults[key]).strip().lower()
                for key in ("whole", "exposed", "fallback")
            },
            perception=PerceptionThresholds(
                color=color,
                condition=condition,
                color_labels=color_labels,
                condition_labels=condition_labels,
            ),
            ode=ODEParameters(
                **{
                    field: float(raw_ode[field])
                    for field in ode_fields
                }
            ),
            continuous=ContinuousParameters(
                **{
                    field: float(raw_continuous[field])
                    for field in continuous_fields
                }
            ),
        )

    @classmethod
    def default(cls) -> DynamicsProfileRegistry:
        return cls.load(RuntimePaths.default().schema_directory)

    def profile(self, name: str) -> DynamicsProfile:
        clean_name = name.strip().lower()
        try:
            return self._profiles[clean_name]
        except KeyError as exc:
            raise KeyError(f"unknown dynamics profile: {name}") from exc

    def default_for(self, exposed_to_air: bool) -> DynamicsProfile:
        key = "exposed" if exposed_to_air else "whole"
        return self.profile(self._defaults[key])

    def for_concept(
        self, category: str | None, attributes: dict[str, Any] | None = None
    ) -> DynamicsProfile:
        attrs = attributes or {}
        explicit = attrs.get("dynamics_profile")
        if explicit:
            return self.profile(str(explicit))

        clean_category = (category or "").strip().lower()
        for profile in self._profiles.values():
            if clean_category and clean_category in profile.categories:
                return profile
            if any(bool(attrs.get(attribute)) for attribute in profile.attributes):
                return profile

        return self.profile(self._defaults["fallback"])
