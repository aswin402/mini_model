"""Versioned configuration for physical transformation semantics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from little.core.runtime_paths import RuntimePaths
from typing import Any


@dataclass(frozen=True)
class TransformationPolicy:
    default_piece_count: int
    predicates: dict[str, Any]
    attributes: dict[str, str]
    defaults: dict[str, str]

    @classmethod
    def load(cls, directory: Path) -> TransformationPolicy:
        path = Path(directory) / "transformation_policy.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        data = payload.get("transformation_policy")
        if not isinstance(data, dict):
            raise TypeError(f"{path} must contain 'transformation_policy' object")
        return cls._from_mapping(data, path)

    @classmethod
    def _from_mapping(
        cls, data: dict[str, Any], path: Path
    ) -> TransformationPolicy:
        count = data.get("default_piece_count")
        predicates = data.get("predicates")
        attributes = data.get("attributes")
        defaults = data.get("defaults")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise TypeError(f"{path} default_piece_count must be a positive integer")
        if not isinstance(predicates, dict):
            raise TypeError(f"{path} predicates must be an object")
        if not isinstance(attributes, dict) or not all(
            isinstance(key, str)
            and key.strip()
            and isinstance(value, str)
            and value.strip()
            for key, value in attributes.items()
        ):
            raise TypeError(f"{path} attributes must map strings to strings")
        if not isinstance(defaults, dict) or not all(
            isinstance(key, str)
            and key.strip()
            and isinstance(value, str)
            and value.strip()
            for key, value in defaults.items()
        ):
            raise TypeError(f"{path} defaults must map strings to strings")
        required_predicates = ("taxonomy", "part", "interior_color_relations")
        for key in required_predicates:
            value = predicates.get(key)
            if key == "interior_color_relations":
                if not isinstance(value, list) or not all(
                    isinstance(item, str) and item.strip() for item in value
                ):
                    raise TypeError(
                        f"{path} predicate {key!r} must be a list of strings"
                    )
            elif not isinstance(value, str) or not value.strip():
                raise TypeError(
                    f"{path} predicate {key!r} must be a non-empty string"
                )
        for key in ("category", "slice_suffix", "experience_source"):
            value = defaults.get(key)
            if not isinstance(value, str) or not value.strip():
                raise TypeError(f"{path} default {key!r} must be a non-empty string")
        return cls(
            default_piece_count=count,
            predicates={
                **predicates,
                "taxonomy": predicates["taxonomy"].strip().lower(),
                "part": predicates["part"].strip().lower(),
                "interior_color_relations": tuple(
                    item.strip().lower() for item in predicates["interior_color_relations"]
                ),
            },
            attributes={
                key.strip(): value.strip() for key, value in attributes.items()
            },
            defaults={key.strip(): value.strip() for key, value in defaults.items()},
        )

    @classmethod
    def default(cls) -> TransformationPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
