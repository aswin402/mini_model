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
    math_routes: dict[str, dict[str, Any]]

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

        raw_routes = values.get("math_routes")
        if raw_routes is None:
            routes = cls.default().math_routes
        elif not isinstance(raw_routes, dict) or not all(
            isinstance(name, str)
            and name.strip()
            and isinstance(route, dict)
            for name, route in raw_routes.items()
        ):
            raise TypeError(f"{path} field 'math_routes' must map names to objects")
        else:
            routes = {
                name.strip().upper(): dict(route)
                for name, route in raw_routes.items()
            }

        return cls(
            math_templates=templates("math_templates"),
            generic_templates=templates("generic_templates"),
            math_routes=routes,
        )

    @classmethod
    def default(cls) -> ResponsePolicy:
        return cls.load(RuntimePaths.default().schema_directory)

    def math(self, name: str, **values: Any) -> str:
        return self._render(self.math_templates, name, values)

    def generic(self, name: str, **values: Any) -> str:
        return self._render(self.generic_templates, name, values)

    def render_math(
        self,
        skill_name: str,
        arguments: dict[str, str],
        answer: Any,
    ) -> str | None:
        """Render the configured route for a procedural skill, if one exists."""
        route = self.math_routes.get(skill_name.strip().upper())
        if route is None:
            return None

        values = {
            output_name: arguments.get(argument_name, "")
            for output_name, argument_name in route.get("arguments", {}).items()
        }
        values["answer"] = self._format_answer(answer, route.get("answer", "raw"))

        kind = route.get("kind", "template")
        if kind == "boolean":
            template_name = route.get(
                "true_template" if answer is True else "false_template"
            )
        elif kind == "quadratic":
            if isinstance(answer, list):
                if len(answer) == 1:
                    template_name = route.get("single_template")
                    values["answer"] = answer[0]
                else:
                    template_name = route.get("multiple_template")
                    values["answer"] = ", ".join(str(value) for value in answer)
            else:
                template_name = route.get("value_template")
        else:
            template_name = route.get("template")

        if not isinstance(template_name, str) or not template_name.strip():
            raise ValueError(f"Invalid math response route for skill {skill_name!r}")
        return self.math(template_name, **values)

    @staticmethod
    def _format_answer(answer: Any, mode: Any) -> Any:
        if mode == "grouped" and isinstance(answer, int) and not isinstance(answer, bool):
            return f"{answer:,}"
        return answer

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
