import json
from dataclasses import replace
from pathlib import Path

import pytest

from little.evaluation.hybrid_benchmark import (
    BenchmarkCase,
    BenchmarkReport,
    build_adapter,
    evaluate_adapter,
    load_cases,
)
from little.language.hybrid_perception import HybridPerceptionAdapter
from little.language.laya_adapter import LayaDecision
from little.language.laya_runtime import LayaRuntimePolicy
from little.language.perception import DeterministicPerceptionAdapter
from little.memory.store import MemoryStore


class FakeSystem1:
    def __init__(self, decision: LayaDecision) -> None:
        self.decision = decision

    def decide(self, _text: str) -> LayaDecision:
        return self.decision


def _decision(intent: str) -> LayaDecision:
    return LayaDecision(
        intent=intent,
        uncertainty=0.1,
        model_id="laya-test",
        model_version="checkpoint-test",
        intent_probabilities={intent: 1.0},
    )


def _cases() -> tuple[BenchmarkCase, ...]:
    return (
        BenchmarkCase(
            case_id="statement",
            text="A falcon is a bird.",
            expected_intent="statement",
            expect_payload=True,
            expect_unknown=False,
        ),
        BenchmarkCase(
            case_id="question",
            text="Is a falcon a bird?",
            expected_intent="question",
            expect_payload=True,
        ),
        BenchmarkCase(
            case_id="unknown",
            text="qwerty zorp",
            expected_intent="unknown",
            expect_payload=False,
            expect_unknown=True,
        ),
    )


def test_load_cases_reads_versioned_data_without_sentence_rules(tmp_path: Path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "hybrid_perception_benchmark": {
                    "version": "test",
                    "cases": [
                        {
                            "id": "statement",
                            "text": "A falcon is a bird.",
                            "expected_intent": "statement",
                            "expect_payload": True,
                            "expect_unknown": False,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    version, cases = load_cases(path)

    assert version == "test"
    assert cases == (
        BenchmarkCase(
            case_id="statement",
            text="A falcon is a bird.",
            expected_intent="statement",
            expect_payload=True,
            expect_unknown=False,
        ),
    )


def test_repository_benchmark_cases_are_versioned_data():
    version, cases = load_cases(
        Path("data/benchmarks/hybrid_perception_cases.json")
    )

    assert version == "1"
    assert {case.case_id for case in cases} == {
        "statement",
        "question",
        "action",
        "unknown",
    }


def test_load_cases_rejects_duplicate_or_malformed_case_data(tmp_path: Path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "hybrid_perception_benchmark": {
                    "version": "test",
                    "cases": [
                        {
                            "id": "duplicate",
                            "text": "one",
                            "expected_intent": "unknown",
                            "expect_payload": False,
                            "expect_unknown": True,
                        },
                        {
                            "id": "duplicate",
                            "text": "two",
                            "expected_intent": "unknown",
                            "expect_payload": "false",
                            "expect_unknown": True,
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate|expect_payload"):
        load_cases(path)


def test_evaluate_adapter_reports_route_payload_unknown_and_latency_metrics():
    report = evaluate_adapter(
        "deterministic",
        DeterministicPerceptionAdapter(),
        _cases(),
    )

    assert isinstance(report, BenchmarkReport)
    assert report.total == 3
    assert report.route_accuracy == 1.0
    assert report.payload_accuracy == 1.0
    assert report.unknown_safety == 1.0
    assert report.errors == 0
    assert report.latency_ms >= 0.0
    assert report.p95_latency_ms >= report.latency_ms
    assert len(report.observations) == 3


def test_build_adapter_keeps_deterministic_mode_explicit():
    adapter = build_adapter("deterministic")

    assert isinstance(adapter, DeterministicPerceptionAdapter)


def test_build_hybrid_adapter_requires_explicit_model_path():
    with pytest.raises(ValueError, match="model_id_or_path"):
        build_adapter("hybrid-laya")


def test_build_hybrid_adapter_accepts_explicit_policy_model_path():
    runtime_policy = replace(
        LayaRuntimePolicy.default(),
        model_load=replace(
            LayaRuntimePolicy.default().model_load,
            model_id_or_path="configured/checkpoint",
        ),
    )
    calls: list[tuple[object, dict[str, object]]] = []

    class Agent:
        def predict(self, _text: str, _questions: object) -> object:
            return {}

    def loader(model_id_or_path: object, **kwargs: object) -> Agent:
        calls.append((model_id_or_path, kwargs))
        return Agent()

    build_adapter(
        "hybrid-laya",
        runtime_policy=runtime_policy,
        loader=loader,
        package_version="sdk-test",
    )

    assert calls == [("configured/checkpoint", {"device": "cpu"})]


def test_evaluate_adapter_uses_real_hybrid_boundary_and_records_mismatch():
    memory = MemoryStore(":memory:", seed_ontology=False)
    grounding = DeterministicPerceptionAdapter()
    adapter = HybridPerceptionAdapter(
        FakeSystem1(_decision("assertion")),
        grounding,
    )

    report = evaluate_adapter(
        "hybrid-test",
        adapter,
        (
            BenchmarkCase(
                case_id="mismatch",
                text="Is a falcon a bird?",
                expected_intent="unknown",
                expect_payload=False,
                expect_unknown=True,
            ),
        ),
        memory=memory,
    )

    assert report.route_accuracy == 1.0
    assert report.payload_accuracy == 1.0
    assert report.unknown_safety == 1.0
    assert report.observations[0].actual_intent == "unknown"
    assert report.observations[0].payload_present is False


def test_report_serializes_metrics_and_observations():
    report = evaluate_adapter(
        "deterministic",
        DeterministicPerceptionAdapter(),
        _cases(),
    )

    serialized = report.to_dict()

    assert serialized["adapter_id"] == "deterministic"
    assert serialized["metrics"]["route_accuracy"] == 1.0
    assert len(serialized["observations"]) == 3
    json.dumps(serialized)
