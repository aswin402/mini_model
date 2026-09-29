"""Versioned Continuous Counterfactual Dynamics Policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class CounterfactualPolicy:
    """Data-owned policy governing hypothetical world modeling and Arrhenius comparisons."""

    counterfactual_prefixes: tuple[str, ...]
    temperature_aliases: dict[str, float]
    default_baseline_temperature_c: float
    default_counterfactual_temperature_c: float
    default_duration_hours: float
    contrastive_template: str
    single_hypothetical_template: str
    freshness_thresholds: dict[str, float]

    @classmethod
    def load(cls, directory: str | Path) -> CounterfactualPolicy:
        path = Path(directory) / "counterfactual_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.counterfactual_policy.v1":
            raise ValueError(f"{path} has an unsupported counterfactual-policy format")

        data = payload.get("counterfactual_policy")
        if not isinstance(data, dict):
            raise TypeError(f"{path} must contain a 'counterfactual_policy' object")

        prefixes = data.get("counterfactual_prefixes", [])
        if not isinstance(prefixes, list) or not all(isinstance(v, str) for v in prefixes):
            raise TypeError(f"{path} field 'counterfactual_prefixes' must be a list of strings")

        temp_aliases = data.get("temperature_aliases", {})
        if not isinstance(temp_aliases, dict):
            raise TypeError(f"{path} field 'temperature_aliases' must be an object")
        parsed_aliases = {
            str(k).strip().lower(): float(v) for k, v in temp_aliases.items()
        }

        fresh_thresh = data.get("freshness_thresholds", {})
        if not isinstance(fresh_thresh, dict):
            raise TypeError(f"{path} field 'freshness_thresholds' must be an object")
        parsed_fresh = {
            str(k).strip().lower(): float(v) for k, v in fresh_thresh.items()
        }

        return cls(
            counterfactual_prefixes=tuple(v.lower() for v in prefixes if v.strip()),
            temperature_aliases=parsed_aliases,
            default_baseline_temperature_c=float(
                data.get("default_baseline_temperature_c", 20.0)
            ),
            default_counterfactual_temperature_c=float(
                data.get("default_counterfactual_temperature_c", 4.0)
            ),
            default_duration_hours=float(data.get("default_duration_hours", 2.0)),
            contrastive_template=str(data.get("contrastive_template", "")),
            single_hypothetical_template=str(data.get("single_hypothetical_template", "")),
            freshness_thresholds=parsed_fresh,
        )

    @classmethod
    def default(cls) -> CounterfactualPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
