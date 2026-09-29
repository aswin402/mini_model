"""Policy loaders for tool definitions and action routing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.actions.models import ToolDefinition, ToolParameter
from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class ToolDefinitionsPolicy:
    """Policy configuring available tools and their parameter schemas."""

    tools: dict[str, ToolDefinition]

    @classmethod
    def load(cls, directory: str | Path) -> ToolDefinitionsPolicy:
        path = Path(directory) / "tool_definitions_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.tool_definitions_policy.v1":
            raise ValueError(f"{path} has an unsupported tool-definitions format")

        raw_tools = payload.get("tools", [])
        tools_dict: dict[str, ToolDefinition] = {}

        for t in raw_tools:
            name = str(t.get("name", "")).strip()
            if not name:
                continue
            desc = str(t.get("description", "")).strip()
            ret_type = str(t.get("return_type", "string")).strip()
            sandboxed = bool(t.get("sandboxed", True))
            timeout = float(t.get("timeout_seconds", 5.0))
            req_confirm = bool(t.get("requires_confirmation", False))

            raw_params = t.get("parameters", {})
            params: dict[str, ToolParameter] = {}
            for p_name, p_spec in raw_params.items():
                p_type = str(p_spec.get("type", "string")).strip()
                p_req = bool(p_spec.get("required", True))
                p_desc = str(p_spec.get("description", "")).strip()
                p_def = p_spec.get("default", None)
                params[p_name] = ToolParameter(
                    name=p_name,
                    param_type=p_type,
                    required=p_req,
                    description=p_desc,
                    default=p_def,
                )

            tools_dict[name] = ToolDefinition(
                name=name,
                description=desc,
                parameters=params,
                return_type=ret_type,
                sandboxed=sandboxed,
                timeout_seconds=timeout,
                requires_confirmation=req_confirm,
            )

        return cls(tools=tools_dict)

    @classmethod
    def default(cls) -> ToolDefinitionsPolicy:
        return cls.load(RuntimePaths.default().schema_directory)


@dataclass(frozen=True)
class ActionPolicy:
    """Policy configuring intent matching and execution limits for tool routing."""

    intent_patterns: dict[str, tuple[str, ...]]
    allowed_bash_binaries: tuple[str, ...]
    workspace_root: Path
    max_execution_timeout_sec: float

    @classmethod
    def load(cls, directory: str | Path, workspace_root: Path | None = None) -> ActionPolicy:
        path = Path(directory) / "action_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.action_policy.v1":
            raise ValueError(f"{path} has an unsupported action-policy format")

        raw_intents = payload.get("intent_patterns", {})
        intents = {
            tool_name: tuple(str(pat) for pat in patterns)
            for tool_name, patterns in raw_intents.items()
        }

        allowed_binaries = tuple(
            str(b) for b in payload.get("allowed_bash_binaries", ())
        )
        ws_fallback = payload.get("workspace_root_fallback", ".")
        root = workspace_root or Path(ws_fallback).resolve()
        max_timeout = float(payload.get("max_execution_timeout_sec", 10.0))

        return cls(
            intent_patterns=intents,
            allowed_bash_binaries=allowed_binaries,
            workspace_root=root,
            max_execution_timeout_sec=max_timeout,
        )

    @classmethod
    def default(cls, workspace_root: Path | None = None) -> ActionPolicy:
        return cls.load(RuntimePaths.default().schema_directory, workspace_root=workspace_root)
