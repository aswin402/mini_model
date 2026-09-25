"""Data-driven evaluation for verified System 2 graph reasoning."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from little.inference.dual_speed import DualSpeedInfillingEngine
from little.inference.invariant_gates import DeepSeekInvariantVerifier
from little.inference.reasoning_policy import ReasoningPolicy
from little.memory.store import MemoryStore


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class System2Case:
    case_id: str
    subject: str
    predicate: str
    target: str
    edges: tuple[tuple[str, str, str], ...]
    expected_mode: str
    expect_proof: bool
    expect_unknown: bool

    def __post_init__(self) -> None:
        for field_name in ("case_id", "subject", "predicate", "target", "expected_mode"):
            object.__setattr__(
                self,
                field_name,
                _text(getattr(self, field_name), field_name),
            )
        if not self.edges:
            raise ValueError("edges must be non-empty")
        if not all(
            isinstance(edge, tuple)
            and len(edge) == 3
            and all(isinstance(value, str) and value.strip() for value in edge)
            for edge in self.edges
        ):
            raise ValueError("edges must contain three non-empty strings")
        if not isinstance(self.expect_proof, bool):
            raise TypeError("expect_proof must be a boolean")
        if not isinstance(self.expect_unknown, bool):
            raise TypeError("expect_unknown must be a boolean")


@dataclass(frozen=True)
class System2Observation:
    case_id: str
    expected_mode: str
    actual_mode: str
    path: tuple[str, ...]
    expected_unknown: bool
    proof_valid: bool
    unknown_safe: bool
    mode_correct: bool
    latency_ms: float
    trace: str
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "expected_mode": self.expected_mode,
            "actual_mode": self.actual_mode,
            "path": list(self.path),
            "expected_unknown": self.expected_unknown,
            "proof_valid": self.proof_valid,
            "unknown_safe": self.unknown_safe,
            "mode_correct": self.mode_correct,
            "latency_ms": self.latency_ms,
            "trace": self.trace,
            "error": self.error,
        }


@dataclass(frozen=True)
class System2Report:
    observations: tuple[System2Observation, ...]

    @property
    def total(self) -> int:
        return len(self.observations)

    @property
    def errors(self) -> int:
        return sum(item.error is not None for item in self.observations)

    @property
    def mode_accuracy(self) -> float:
        return self._ratio(item.mode_correct for item in self.observations)

    @property
    def proof_accuracy(self) -> float:
        return self._ratio(item.proof_valid for item in self.observations)

    @property
    def unknown_safety(self) -> float:
        return self._ratio(
            item.unknown_safe
            for item in self.observations
            if item.error is None and item.expected_unknown
        )

    @property
    def latency_ms(self) -> float:
        values = [item.latency_ms for item in self.observations]
        return float(median(values)) if values else 0.0

    @property
    def p95_latency_ms(self) -> float:
        values = sorted(item.latency_ms for item in self.observations)
        if not values:
            return 0.0
        index = min(len(values) - 1, max(0, math.ceil(len(values) * 0.95) - 1))
        return float(values[index])

    def to_dict(self) -> dict[str, object]:
        return {
            "metrics": {
                "total": self.total,
                "errors": self.errors,
                "mode_accuracy": self.mode_accuracy,
                "proof_accuracy": self.proof_accuracy,
                "unknown_safety": self.unknown_safety,
                "latency_ms": self.latency_ms,
                "p95_latency_ms": self.p95_latency_ms,
            },
            "observations": [item.to_dict() for item in self.observations],
        }

    @staticmethod
    def _ratio(values: Any) -> float:
        materialized = list(values)
        return sum(materialized) / len(materialized) if materialized else 0.0


def load_cases(path: Path) -> tuple[str, tuple[System2Case, ...]]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        root = payload["system2_benchmark"]
        if not isinstance(root, dict):
            raise TypeError("system2_benchmark must be an object")
        version = _text(root["version"], "version")
        raw_cases = root["cases"]
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError("cases must be a non-empty array")
        cases: list[System2Case] = []
        seen: set[str] = set()
        for index, raw_case in enumerate(raw_cases):
            if not isinstance(raw_case, dict):
                raise TypeError(f"cases[{index}] must be an object")
            case_id = _text(raw_case["id"], f"cases[{index}].id")
            if case_id in seen:
                raise ValueError(f"duplicate system2 case id: {case_id}")
            seen.add(case_id)
            raw_edges = raw_case["edges"]
            if not isinstance(raw_edges, list):
                raise TypeError(f"cases[{index}].edges must be an array")
            edges: list[tuple[str, str, str]] = []
            for edge_index, raw_edge in enumerate(raw_edges):
                if (
                    not isinstance(raw_edge, list)
                    or len(raw_edge) != 3
                    or not all(isinstance(value, str) and value.strip() for value in raw_edge)
                ):
                    raise ValueError(
                        f"cases[{index}].edges[{edge_index}] must have three non-empty strings"
                    )
                edges.append(tuple(raw_edge))  # type: ignore[arg-type]
            cases.append(
                System2Case(
                    case_id=case_id,
                    subject=raw_case["subject"],
                    predicate=raw_case["predicate"],
                    target=raw_case["target"],
                    edges=tuple(edges),
                    expected_mode=raw_case["expected_mode"],
                    expect_proof=raw_case["expect_proof"],
                    expect_unknown=raw_case["expect_unknown"],
                )
            )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid System 2 benchmark cases at {path}: {exc}") from exc
    return version, tuple(cases)


def evaluate_cases(
    cases: tuple[System2Case, ...] | list[System2Case],
    *,
    policy: ReasoningPolicy | None = None,
) -> System2Report:
    observations: list[System2Observation] = []
    for case in cases:
        memory = MemoryStore(":memory:", seed_ontology=False)
        for subject, predicate, target in case.edges:
            memory.add_relation(subject, predicate, target)
        verifier = DeepSeekInvariantVerifier(memory)
        engine = DualSpeedInfillingEngine(memory, verifier, policy=policy)
        started = time.perf_counter()
        error: str | None = None
        actual_mode = "ERROR"
        path: tuple[str, ...] = ()
        trace = ""
        proof_valid = False
        unknown_safe = False
        try:
            result = engine.query(case.subject, case.target, predicate=case.predicate)
            actual_mode = result.mode
            path = tuple(result.path)
            trace = result.inspectable_trace
            proof_valid = not case.expect_proof or (
                bool(path)
                and all(
                    gate in trace
                    for gate in ("I_DAG", "I_MUTEX", "I_SORT", "I_GROUND")
                )
            )
            unknown_safe = not case.expect_unknown or not path
        except Exception as exc:  # benchmark records each failed case
            error = f"{type(exc).__name__}: {exc}"
        finally:
            latency_ms = (time.perf_counter() - started) * 1000.0
            memory.close()
        observations.append(
            System2Observation(
                case_id=case.case_id,
                expected_mode=case.expected_mode,
                actual_mode=actual_mode,
                path=path,
                expected_unknown=case.expect_unknown,
                proof_valid=error is None and proof_valid,
                unknown_safe=error is None and unknown_safe,
                mode_correct=error is None and actual_mode == case.expected_mode,
                latency_ms=latency_ms,
                trace=trace,
                error=error,
            )
        )
    return System2Report(observations=tuple(observations))
