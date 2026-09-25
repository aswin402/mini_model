import importlib
from pathlib import Path

import pytest

from little.language.laya_adapter import LayaDecision, LayaPerceptionAdapter
from little.language.laya_runtime import LayaRuntimePolicy, LayaSDKBackend
from little.memory.store import MemoryStore


class FakeAgent:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[str, object]] = []

    def predict(self, state: str, questions: object) -> object:
        self.calls.append((state, questions))
        return self.result


def _result(
    *,
    choice: str = "assertion",
    probabilities: object = None,
    confidence: object = 0.8,
    routing: object = None,
    model: object = None,
) -> dict[str, object]:
    return {
        "answers": {
            "intent": {
                "choice": choice,
                "probabilities": probabilities
                if probabilities is not None
                else {"assertion": 8, "query": 2},
                "confidence": confidence,
            }
        },
        "routing": routing if routing is not None else {"model_id": "laya-local"},
        "model": model if model is not None else {"version": "model-7"},
    }


def test_runtime_policy_loads_question_fields_and_model_settings():
    policy = LayaRuntimePolicy.load(Path("data/schemas"))

    assert policy.question_id == "intent"
    assert policy.question_definition["type"] == "choice"
    assert policy.result_answers_field == "answers"
    assert policy.answer_choice_field == "choice"
    assert policy.model_load.model_id_or_path is None
    assert policy.model_load.device == "cpu"
    assert policy.model_load.runtime == "agent"


def test_decide_calls_configured_prediction_and_normalizes_answer():
    agent = FakeAgent(_result())
    backend = LayaSDKBackend(agent, package_version="sdk-1")

    decision = backend.decide("A falcon is a bird.")

    assert isinstance(decision, LayaDecision)
    assert decision.intent == "assertion"
    assert decision.intent_probabilities == {"assertion": 0.8, "query": 0.2}
    assert decision.uncertainty == pytest.approx(0.2)
    assert decision.model_id == "laya-local"
    assert decision.model_version == "model-7"
    assert agent.calls == [
        (
            "A falcon is a bird.",
            {
                "intent": dict(LayaRuntimePolicy.default().question_definition),
            },
        )
    ]


def test_decide_uses_routing_model_metadata_when_available():
    result = _result(
        routing={"model_id": "routed-model", "model_version": "routed-3"},
        model={"version": "ignored-model-version"},
    )

    decision = LayaSDKBackend(FakeAgent(result), package_version="sdk-1").decide(
        "text"
    )

    assert decision.model_id == "routed-model"
    assert decision.model_version == "routed-3"


def test_decide_accepts_official_top_level_model_string():
    result = _result(routing={}, model="laya-rl-agent")
    decision = LayaSDKBackend(
        FakeAgent(result), package_version="sdk-1", model_version="checkpoint-1"
    ).decide("text")

    assert decision.model_id == "laya-rl-agent"
    assert decision.model_version == "checkpoint-1"


def test_decide_rejects_malformed_answer_shapes():
    malformed_results = [
        {},
        {"answers": {}},
        {"answers": {"intent": {"choice": "assertion"}}},
        _result(probabilities={"query": 1.0}),
        _result(probabilities={"assertion": 0, "query": 0}),
        _result(confidence=float("nan")),
        _result(routing={"model_id": ""}),
        _result(model={"version": ""}),
    ]

    for result in malformed_results:
        with pytest.raises((TypeError, ValueError), match=".+"):
            LayaSDKBackend(FakeAgent(result), package_version="sdk-1").decide(
                "text"
            )


def test_intent_only_bridge_remains_fail_closed_in_perception_adapter():
    backend = LayaSDKBackend(FakeAgent(_result()))
    frame = LayaPerceptionAdapter(backend).perceive(
        "A falcon is a bird.",
        memory=MemoryStore(":memory:", seed_ontology=False),
    )

    assert frame.intent == "unknown"
    assert frame.claims == []
    assert frame.actions == []
    assert frame.model_id == "laya-local"


def test_from_installed_uses_injected_loader_and_policy_settings():
    result = _result()
    calls: list[tuple[object, dict[str, object]]] = []
    agent = FakeAgent(result)

    def loader(model_id_or_path: object, **kwargs: object) -> FakeAgent:
        calls.append((model_id_or_path, kwargs))
        return agent

    backend = LayaSDKBackend.from_installed(
        model_id_or_path="local/checkpoint",
        subfolder="weights",
        loader=loader,
        package_version="sdk-2",
    )

    assert backend.agent is agent
    assert calls == [("local/checkpoint", {"device": "cpu", "subfolder": "weights"})]


def test_from_installed_requires_package_version_before_loading():
    loaded = False

    def loader(_model_id_or_path: object, **_kwargs: object) -> FakeAgent:
        nonlocal loaded
        loaded = True
        return FakeAgent(_result())

    with pytest.raises(RuntimeError, match="package version"):
        LayaSDKBackend.from_installed(
            model_id_or_path="local/checkpoint",
            loader=loader,
        )

    assert loaded is False


def test_missing_laya_package_raises_actionable_runtime_error(monkeypatch):
    def missing_import(name: str) -> object:
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(importlib, "import_module", missing_import)

    with pytest.raises(RuntimeError, match="Install the optional Laya SDK"):
        LayaSDKBackend.from_installed(
            model_id_or_path="local/checkpoint",
        )


def test_direct_construction_does_not_import_or_load_laya(monkeypatch):
    imported: list[str] = []

    def unexpected_import(name: str) -> object:
        imported.append(name)
        raise AssertionError("direct construction must not import Laya")

    monkeypatch.setattr(importlib, "import_module", unexpected_import)
    agent = FakeAgent(_result())

    backend = LayaSDKBackend(agent)

    assert backend.agent is agent
    assert imported == []


def test_from_installed_requires_explicit_model_id_or_policy_setting():
    with pytest.raises(RuntimeError, match="model_id_or_path"):
        LayaSDKBackend.from_installed(
            loader=lambda *_args, **_kwargs: FakeAgent(_result()),
            package_version="sdk-1",
        )
