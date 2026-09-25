"""Versioned natural-language response templates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class ResponsePolicy:
    """Data-owned templates used to verbalize structured results."""

    math_templates: dict[str, str]
    generic_templates: dict[str, str]

    @classmethod
    def load(cls, directory: str | Path) -> ResponsePolicy:
        directory = Path(directory)
        path = directory / "response_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.response_policy.v1":
            raise ValueError(f"{path} has an unsupported response-policy format")
        values = payload.get("response_policy")
        if not isinstance(values, dict):
            raise TypeError(f"{path} must contain a 'response_policy' object")

        def templates(key: str) -> dict[str, str]:
            raw = values.get(key)
            if not isinstance(raw, dict) or not raw:
                raise TypeError(f"{path} field {key!r} must be a non-empty object")
            if not all(
                isinstance(name, str)
                and name.strip()
                and isinstance(template, str)
                and template.strip()
                for name, template in raw.items()
            ):
                raise TypeError(f"{path} field {key!r} must map names to templates")
            return {name.strip(): template for name, template in raw.items()}

        return cls(
            math_templates=templates("math_templates"),
            generic_templates=templates("generic_templates"),
        )

    @classmethod
    def default(cls) -> ResponsePolicy:
        return cls.load(RuntimePaths.default().schema_directory)

    def math(self, name: str, **values: Any) -> str:
        return self._render(self.math_templates, name, values)

    def generic(self, name: str, **values: Any) -> str:
        return self._render(self.generic_templates, name, values)

    @staticmethod
    def _render(
        templates: dict[str, str], name: str, values: dict[str, Any]
    ) -> str:
        try:
            template = templates[name]
        except KeyError as exc:
            raise KeyError(f"Missing response template: {name}") from exc
        try:
            return template.format(**values)
        except KeyError as exc:
            raise ValueError(
                f"Response template {name!r} requires missing value {exc.args[0]!r}"
            ) from exc
