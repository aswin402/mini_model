"""Data-driven evaluation for deterministic and hybrid perception adapters."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

from little.core.contracts import CandidateFrame
from little.language.hybrid_perception import HybridPerceptionAdapter
from little.language.laya_runtime import (
    LayaRuntimePolicy,
    LayaSDKBackend,
    Loader,
)
from little.language.perception import PerceptionAdapter

if TYPE_CHECKING:
    from little.language.dialogue import DialogueContext
    from little.memory.store import MemoryStore


def _non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class BenchmarkCase:
    """One externally defined perception expectation."""

    case_id: str
    text: str
    expected_intent: str
    expect_payload: bool
    expect_unknown: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _non_empty_string(self.case_id, "case_id"))
        object.__setattr__(self, "text", _non_empty_string(self.text, "text"))
        object.__setattr__(
            self,
            "expected_intent",
            _non_empty_string(self.expected_intent, "expected_intent"),
        )
        if not isinstance(self.expect_payload, bool):
            raise TypeError("expect_payload must be a boolean")
        if not isinstance(self.expect_unknown, bool):
            raise TypeError("expect_unknown must be a boolean")


@dataclass(frozen=True)
class BenchmarkObservation:
    case_id: str
    expected_intent: str
    actual_intent: str
    expected_payload: bool
    payload_present: bool
    expected_unknown: bool
    unknown_safe: bool
    route_correct: bool
    payload_correct: bool
    latency_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "expected_intent": self.expected_intent,
            "actual_intent": self.actual_intent,
            "expected_payload": self.expected_payload,
            "payload_present": self.payload_present,
            "expected_unknown": self.expected_unknown,
            "unknown_safe": self.unknown_safe,
            "route_correct": self.route_correct,
            "payload_correct": self.payload_correct,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass(frozen=True)
class BenchmarkReport:
    adapter_id: str
    observations: tuple[BenchmarkObservation, ...]

    @property
    def total(self) -> int:
        return len(self.observations)

    @property
    def errors(self) -> int:
        return sum(observation.error is not None for observation in self.observations)

    @property
    def route_accuracy(self) -> float:
        return self._ratio(
            (observation.route_correct for observation in self.observations)
        )

    @property
    def payload_accuracy(self) -> float:
        return self._ratio(
            (observation.payload_correct for observation in self.observations)
        )

    @property
    def unknown_safety(self) -> float:
        unknown_observations = (
            observation
            for observation in self.observations
            if observation.expected_unknown
        )
        return self._ratio(
            observation.unknown_safe for observation in unknown_observations
        )

    @property
    def latency_ms(self) -> float:
        latencies = [observation.latency_ms for observation in self.observations]
        return float(median(latencies)) if latencies else 0.0

    @property
    def p95_latency_ms(self) -> float:
        latencies = sorted(observation.latency_ms for observation in self.observations)
        if not latencies:
            return 0.0
        index = min(len(latencies) - 1, max(0, math.ceil(len(latencies) * 0.95) - 1))
        return float(latencies[index])

    def to_dict(self) -> dict[str, object]:
        return {
            "adapter_id": self.adapter_id,
            "metrics": {
                "total": self.total,
                "errors": self.errors,
                "route_accuracy": self.route_accuracy,
                "payload_accuracy": self.payload_accuracy,
                "unknown_safety": self.unknown_safety,
                "latency_ms": self.latency_ms,
                "p95_latency_ms": self.p95_latency_ms,
            },
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
        }

    @staticmethod
    def _ratio(values: Any) -> float:
        materialized = list(values)
        return sum(materialized) / len(materialized) if materialized else 0.0


def load_cases(path: Path) -> tuple[str, tuple[BenchmarkCase, ...]]:
    """Load and validate benchmark expectations from versioned JSON data."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"benchmark cases not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"benchmark cases contain invalid JSON: {exc.msg}") from exc

    try:
        root = payload["hybrid_perception_benchmark"]
        if not isinstance(root, dict):
            raise TypeError("hybrid_perception_benchmark must be an object")
        version = _non_empty_string(root["version"], "version")
        raw_cases = root["cases"]
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError("cases must be a non-empty array")
        cases: list[BenchmarkCase] = []
        seen_ids: set[str] = set()
        for index, raw_case in enumerate(raw_cases):
            if not isinstance(raw_case, dict):
                raise TypeError(f"cases[{index}] must be an object")
            case_id = _non_empty_string(raw_case["id"], f"cases[{index}].id")
            if case_id in seen_ids:
                raise ValueError(f"duplicate benchmark case id: {case_id}")
            seen_ids.add(case_id)
            cases.append(
                BenchmarkCase(
                    case_id=case_id,
                    text=raw_case["text"],
                    expected_intent=raw_case["expected_intent"],
                    expect_payload=raw_case["expect_payload"],
                    expect_unknown=raw_case["expect_unknown"],
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid benchmark cases at {path}: {exc}") from exc

    return version, tuple(cases)


def evaluate_adapter(
    adapter_id: str,
    adapter: PerceptionAdapter,
    cases: tuple[BenchmarkCase, ...] | list[BenchmarkCase],
    *,
    memory: MemoryStore | None = None,
    context: DialogueContext | None = None,
) -> BenchmarkReport:
    """Evaluate an adapter without allowing benchmark inputs to mutate its contract."""

    adapter_id = _non_empty_string(adapter_id, "adapter_id")
    observations: list[BenchmarkObservation] = []
    for case in cases:
        owned_memory = memory is None
        case_memory = memory
        if case_memory is None:
            from little.memory.store import MemoryStore as RuntimeMemoryStore

            case_memory = RuntimeMemoryStore(":memory:", seed_ontology=False)
        started = time.perf_counter()
        error: str | None = None
        actual_intent = "error"
        payload_present = False
        try:
            frame = adapter.perceive(case.text, memory=case_memory, context=context)
            if not isinstance(frame, CandidateFrame):
                raise TypeError("perception adapter must return CandidateFrame")
            actual_intent = frame.intent
            payload_present = _has_payload(frame)
        except Exception as exc:  # benchmark records adapter failures per case
            error = f"{type(exc).__name__}: {exc}"
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if owned_memory:
                case_memory.close()

        route_correct = error is None and actual_intent == case.expected_intent
        payload_correct = error is None and payload_present == case.expect_payload
        unknown_safe = not case.expect_unknown or (
            error is None
            and actual_intent == case.expected_intent
            and not payload_present
        )
        observations.append(
            BenchmarkObservation(
                case_id=case.case_id,
                expected_intent=case.expected_intent,
                actual_intent=actual_intent,
                expected_payload=case.expect_payload,
                payload_present=payload_present,
                expected_unknown=case.expect_unknown,
                unknown_safe=unknown_safe,
                route_correct=route_correct,
                payload_correct=payload_correct,
                latency_ms=elapsed_ms,
                error=error,
            )
        )

    return BenchmarkReport(adapter_id=adapter_id, observations=tuple(observations))


def build_adapter(
    mode: str,
    *,
    model_id_or_path: str | None = None,
    device: str | None = None,
    subfolder: str | None = None,
    runtime: str | None = None,
    model_version: str | None = None,
    package_version: str | None = None,
    loader: Loader | None = None,
    runtime_policy: LayaRuntimePolicy | None = None,
) -> PerceptionAdapter:
    """Build only the explicitly requested benchmark adapter.

    Hybrid mode requires an explicit model path or identifier. It never falls
    back to deterministic perception when the optional Laya runtime is absent.
    """

    mode = _non_empty_string(mode, "mode")
    if mode == "deterministic":
        return _deterministic_adapter()
    if mode != "hybrid-laya":
        raise ValueError(f"unsupported benchmark mode: {mode}")
    selected_policy = runtime_policy or LayaRuntimePolicy.default()
    selected_model = model_id_or_path or selected_policy.model_load.model_id_or_path
    if selected_model is None or not selected_model.strip():
        raise ValueError("model_id_or_path is required for hybrid-laya mode")

    backend = LayaSDKBackend.from_installed(
        model_id_or_path=selected_model,
        device=device,
        subfolder=subfolder,
        runtime=runtime,
        model_version=model_version,
        package_version=package_version,
        loader=loader,
        policy=selected_policy,
    )
    return HybridPerceptionAdapter(backend, _deterministic_adapter())


def _deterministic_adapter() -> PerceptionAdapter:
    from little.language.perception import DeterministicPerceptionAdapter

    return DeterministicPerceptionAdapter()


def _has_payload(frame: CandidateFrame) -> bool:
    return any(
        bool(getattr(frame, field_name, None))
        for field_name in (
            "claims",
            "entities",
            "actions",
            "parsed_query",
            "parsed_queries",
            "question_parts",
        )
    )
