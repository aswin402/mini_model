from dataclasses import replace
from pathlib import Path

from little.core.models import BeliefStatus
from little.language.parser import LearningEngine, SimpleParser
from little.language.parser_confidence_policy import ParserConfidencePolicy
from little.memory.store import MemoryStore


def test_parser_confidence_policy_loads_named_calibration_levels():
    policy = ParserConfidencePolicy.load(Path("data/schemas"))

    assert policy.version == "1"
    assert policy.certain == 1.0
    assert policy.derived == 0.95
    assert policy.unknown == 0.0
    assert policy.low == 0.1


def test_parser_result_uses_injected_certain_confidence(monkeypatch):
    policy = ParserConfidencePolicy(
        version="test",
        certain=0.77,
        derived=0.66,
        unknown=0.11,
        low=0.22,
    )
    monkeypatch.setattr(
        SimpleParser,
        "POLICY",
        replace(SimpleParser.POLICY, confidence=policy),
    )

    result = LearningEngine(MemoryStore(":memory:")).ask("who are you")

    assert result.status is BeliefStatus.SUPPORTED
    assert result.confidence == 0.77
