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


class PlanStepStatus:
    """Status lifecycle of a multi-step agentic plan step."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    BACKTRACKED = "BACKTRACKED"
    SKIPPED = "SKIPPED"


@dataclass
class PlanStep:
    """An individual discrete step within an agentic action plan."""

    step_index: int
    description: str
    tool_name: str
    arguments_template: dict[str, Any]
    dependencies: tuple[int, ...] = ()
    status: str = PlanStepStatus.PENDING
    retries: int = 0
    result: ToolExecutionResult | None = None
    observation: str | None = None


@dataclass(frozen=True)
class Plan:
    """An ordered Directed Acyclic Graph (DAG) of actionable steps to fulfill a goal."""

    plan_id: str
    goal: str
    steps: tuple[PlanStep, ...]
    created_at: str = ""

    def is_acyclic(self) -> bool:
        """Verify I_DAG invariant: plan step dependencies must form a strictly acyclic DAG."""
        # Step indices are 1-based. A step i cannot depend on any j >= i.
        step_indices = {s.step_index for s in self.steps}
        for s in self.steps:
            for dep in s.dependencies:
                if dep not in step_indices or dep >= s.step_index:
                    return False
        return True


@dataclass(frozen=True)
class AgenticPlanExecutionResult:
    """Verifiable outcome of executing an autonomous multi-step plan."""

    plan_id: str
    goal: str
    steps: tuple[PlanStep, ...]
    success: bool
    final_output: Any = None
    total_execution_time_ms: float = 0.0
    trace_hash: str = ""
    invariants_checked: tuple[str, ...] = ()
    retries_used: int = 0
    backtracks_count: int = 0
    error: str | None = None

    @classmethod
    def compute_trace_hash(cls, plan_id: str, steps: tuple[PlanStep, ...], success: bool) -> str:
        """Compute SHA-256 verifiable hash over the entire execution trace DAG."""
        trace_payload = {
            "plan_id": plan_id,
            "success": success,
            "steps": [
                {
                    "step": s.step_index,
                    "tool": s.tool_name,
                    "status": s.status,
                    "hash": s.result.verifiable_hash if s.result else "none",
                }
                for s in steps
            ],
        }
        encoded = json.dumps(trace_payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

