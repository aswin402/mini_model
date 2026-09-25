"""Data-owned semantic predicate roles used by the language and reasoner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class SemanticPredicatePolicy:
    """Names used by user-facing parser routes to select relation families."""

    taxonomy: str
    part: str
    whole: str
    disjoint: str
    location: str
    identity: tuple[str, ...]
    property: str
    color: str
    material: str
    capability: str
    habitat: str
    diet: str
    composition: str
    purpose: str
    causality: str

    @classmethod
    def load(cls, directory: Path) -> SemanticPredicatePolicy:
        path = Path(directory) / "semantic_predicate_policy.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        data = payload.get("semantic_predicate_policy")
        if not isinstance(data, dict):
            raise TypeError(f"{path} must contain 'semantic_predicate_policy' object")

        roles = data.get("roles")
        if not isinstance(roles, dict):
            raise TypeError(f"{path} field 'roles' must be an object")

        def role(name: str) -> str:
            value = roles.get(name)
            if not isinstance(value, str) or not value.strip():
                raise TypeError(f"{path} role {name!r} must be a non-empty string")
            return value.strip().lower()

        identity = data.get("identity_predicates")
        if not isinstance(identity, list) or not all(
            isinstance(value, str) and value.strip() for value in identity
        ):
            raise TypeError(
                f"{path} field 'identity_predicates' must be a list of strings"
            )

        return cls(
            taxonomy=role("taxonomy"),
            part=role("part"),
            whole=role("whole"),
            disjoint=role("disjoint"),
            location=role("location"),
            identity=tuple(value.strip().lower() for value in identity),
            property=role("property"),
            color=role("color"),
            material=role("material"),
            capability=role("capability"),
            habitat=role("habitat"),
            diet=role("diet"),
            composition=role("composition"),
            purpose=role("purpose"),
            causality=role("causality"),
        )

    @classmethod
    def default(cls) -> SemanticPredicatePolicy:
        return cls.load(RuntimePaths.default().schema_directory)
