"""Configuration for synthetic ontology expansion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class OntologyGenerationPolicy:
    root_category: str
    taxonomy_predicate: str
    composition_predicate: str
    location_predicate: str
    component_predicate: str

    @classmethod
    def load(cls, directory: Path) -> OntologyGenerationPolicy:
        path = Path(directory) / "ontology_generation_policy.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        data = payload.get("ontology_generation_policy")
        if not isinstance(data, dict):
            raise TypeError(
                f"{path} must contain 'ontology_generation_policy' object"
            )
        return cls.from_mapping(data, path=path)

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
        *,
        path: Path | None = None,
        base: OntologyGenerationPolicy | None = None,
    ) -> OntologyGenerationPolicy:
        values: dict[str, str] = {}
        for name in (
            "root_category",
            "taxonomy_predicate",
            "composition_predicate",
            "location_predicate",
            "component_predicate",
        ):
            if name in data:
                value = data[name]
            elif base is not None:
                value = getattr(base, name)
            else:
                source = path or Path("ontology generation configuration")
                raise TypeError(f"{source} field {name!r} is required")
            if not isinstance(value, str) or not value.strip():
                source = path or Path("ontology generation configuration")
                raise TypeError(f"{source} field {name!r} must be a non-empty string")
            values[name] = value.strip().lower()
        return cls(**values)

    @classmethod
    def default(cls) -> OntologyGenerationPolicy:
        path = RuntimePaths.default().schema_directory / "ontology_generation_policy.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        data = payload.get("ontology_generation_policy")
        if not isinstance(data, dict):
            raise TypeError(f"{path} must contain 'ontology_generation_policy' object")
        return cls.from_mapping(data, path=path)
