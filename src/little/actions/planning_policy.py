"""Policy loader for multi-step agentic planning and needle action chains."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class AgenticPlanningPolicy:
    """Policy configuring multi-step goal decomposition, step budgets, and invariant gates."""

    max_steps: int = 8
    max_retries_per_step: int = 2
    step_timeout_seconds: float = 5.0
    enable_backtracking: bool = True
    chainable_tools: tuple[str, ...] = (
        "calculator",
        "python_repl",
        "system_time",
        "web_search",
        "file_reader",
        "file_writer",
        "bash_executor",
    )
    conjunction_delimiters: tuple[str, ...] = (
        "and then",
        "then",
        "after that",
        "next",
        "and finally",
        ";",
    )
    variable_reference_patterns: tuple[str, ...] = (
        r"\$(?:result|output|prev(?:ious)?|step_\d+)",
        r"\b(?:the\s+result|the\s+output|the\s+answer|that\s+result)\b",
    )
    invariant_gates: tuple[str, ...] = (
        "I_dag_acyclicity",
        "I_param_required",
        "I_sort_type",
        "I_security_sandbox",
        "I_mutex_resource",
        "I_ground_provenance",
    )

    @classmethod
    def load(cls, directory: str | Path) -> AgenticPlanningPolicy:
        path = Path(directory) / "agentic_planning_policy.json"
        if not path.exists():
            return cls()

        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.agentic_planning_policy.v1":
            raise ValueError(f"{path} has an unsupported agentic planning policy format")

        return cls(
            max_steps=int(payload.get("max_steps", 8)),
            max_retries_per_step=int(payload.get("max_retries_per_step", 2)),
            step_timeout_seconds=float(payload.get("step_timeout_seconds", 5.0)),
            enable_backtracking=bool(payload.get("enable_backtracking", True)),
            chainable_tools=tuple(str(t) for t in payload.get("chainable_tools", ())),
            conjunction_delimiters=tuple(str(d) for d in payload.get("conjunction_delimiters", ())),
            variable_reference_patterns=tuple(
                str(p) for p in payload.get("variable_reference_patterns", ())
            ),
            invariant_gates=tuple(str(g) for g in payload.get("invariant_gates", ())),
        )

    @classmethod
    def default(cls) -> AgenticPlanningPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
