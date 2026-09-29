"""Typed models for tool definitions, action calls, and verifiable execution results."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolParameter:
    """Specification of a single parameter in a tool definition."""

    name: str
    param_type: str
    required: bool = True
    description: str = ""
    default: Any = None


@dataclass(frozen=True)
class ToolDefinition:
    """Specification of an executable tool."""

    name: str
    description: str
    parameters: dict[str, ToolParameter]
    return_type: str
    sandboxed: bool = True
    timeout_seconds: float = 5.0
    requires_confirmation: bool = False


@dataclass(frozen=True)
class ToolCall:
    """A concrete request to execute a tool with arguments."""

    tool_name: str
    arguments: dict[str, Any]
    call_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass(frozen=True)
class VerificationResult:
    """Pre-dispatch verification check result for a tool call."""

    valid: bool
    error: str | None = None
    checked_invariants: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolExecutionResult:
    """The outcome of executing a tool call."""

    tool_name: str
    arguments: dict[str, Any]
    success: bool
    output: Any = None
    error: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    execution_time_ms: float = 0.0
    verifiable_hash: str = ""

    @classmethod
    def compute_hash(cls, tool_name: str, arguments: dict[str, Any], output: Any, success: bool) -> str:
        """Compute SHA-256 verifiable hash of tool execution for provenance."""
        payload = {
            "tool": tool_name,
            "args": arguments,
            "success": success,
            "out": str(output),
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]
