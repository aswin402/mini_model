"""Versioned construction-grammar pack loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from little.core.models import Construction
from little.core.construction_policy import ConstructionPolicy
from little.core.runtime_paths import RuntimePaths


class GrammarRegistry:
    """Load persisted grammar constructions without embedding domain phrases."""

    @classmethod
    def load(cls, path: str | Path) -> list[Construction]:
        pack_path = Path(path)
        payload = json.loads(pack_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{pack_path} must contain a JSON object")
        if payload.get("format") != "little.constructions.v1":
            raise ValueError(f"{pack_path} has an unsupported grammar format")
        records = payload.get("constructions")
        if not isinstance(records, list):
            raise TypeError(f"{pack_path} field 'constructions' must be a list")

        constructions: list[Construction] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise TypeError(f"{pack_path} construction {index} must be an object")
            constructions.append(cls._parse_record(record, pack_path, index))
        return constructions

    @classmethod
    def default(cls) -> list[Construction]:
        return cls.load(RuntimePaths.default().construction_pack)

    @staticmethod
    def _parse_record(
        record: dict[str, Any], path: Path, index: int
    ) -> Construction:
        required = ("name", "pattern_tokens", "slot_roles", "predicate_template")
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"{path} construction {index} is missing {missing}")

        name = record["name"]
        pattern_tokens = record["pattern_tokens"]
        slot_roles = record["slot_roles"]
        predicate_template = record["predicate_template"]
        if not isinstance(name, str) or not name.strip():
            raise TypeError(f"{path} construction {index} has an invalid name")
        if not isinstance(pattern_tokens, list) or not all(
            isinstance(token, str) and token.strip() for token in pattern_tokens
        ):
            raise TypeError(
                f"{path} construction {index} has invalid pattern_tokens"
            )
        if not isinstance(slot_roles, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in slot_roles.items()
        ):
            raise TypeError(f"{path} construction {index} has invalid slot_roles")
        if not isinstance(predicate_template, str) or not predicate_template.strip():
            raise TypeError(
                f"{path} construction {index} has an invalid predicate_template"
            )

        return Construction.create(
            name=name,
            pattern_tokens=pattern_tokens,
            slot_roles=slot_roles,
            predicate_template=predicate_template,
            construction_type=str(record.get("construction_type", "statement")),
            is_negative=bool(record.get("is_negative", False)),
            is_property=bool(record.get("is_property", False)),
            confidence=float(
                record.get(
                    "confidence", ConstructionPolicy.default().default_confidence
                )
            ),
        )
