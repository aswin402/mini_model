"""Versioned Socratic explanation and syllogism policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class SocraticPolicy:
    """Data-owned policy governing Socratic proofs and why-inquiry routing."""

    why_prefixes: tuple[str, ...]
    why_not_prefixes: tuple[str, ...]
    syllogism_header: str
    premise_template: str
    deduction_template: str
    refutation_disjoint_template: str
    refutation_negative_template: str
    capability_inheritance_template: str
    property_inheritance_template: str
    location_transitive_template: str
    unknown_explanation_template: str

    @classmethod
    def load(cls, directory: str | Path) -> SocraticPolicy:
        path = Path(directory) / "socratic_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.socratic_policy.v1":
            raise ValueError(f"{path} has an unsupported socratic-policy format")

        data = payload.get("socratic_policy")
        if not isinstance(data, dict):
            raise TypeError(f"{path} must contain a 'socratic_policy' object")

        def read_list(k: str) -> tuple[str, ...]:
            vals = data.get(k, [])
            if not isinstance(vals, list) or not all(isinstance(v, str) for v in vals):
                raise TypeError(f"{path} field {k!r} must be a list of strings")
            return tuple(v.lower() for v in vals if v.strip())

        def read_str(k: str, default: str = "") -> str:
            v = data.get(k, default)
            if not isinstance(v, str):
                raise TypeError(f"{path} field {k!r} must be a string")
            return v

        return cls(
            why_prefixes=read_list("why_prefixes"),
            why_not_prefixes=read_list("why_not_prefixes"),
            syllogism_header=read_str("syllogism_header", "Deductive Proof:"),
            premise_template=read_str("premise_template"),
            deduction_template=read_str("deduction_template"),
            refutation_disjoint_template=read_str("refutation_disjoint_template"),
            refutation_negative_template=read_str("refutation_negative_template"),
            capability_inheritance_template=read_str("capability_inheritance_template"),
            property_inheritance_template=read_str("property_inheritance_template"),
            location_transitive_template=read_str("location_transitive_template"),
            unknown_explanation_template=read_str("unknown_explanation_template"),
        )

    @classmethod
    def default(cls) -> SocraticPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
