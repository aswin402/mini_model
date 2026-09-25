from pathlib import Path

import pytest

from little.inference.reasoning_policy import ReasoningPolicy


def test_reasoning_policy_loads_search_and_confidence_values():
    policy = ReasoningPolicy.load(Path("data/schemas"))

    assert policy.version == "1"
    assert policy.max_depth == 8
    assert policy.fast_confidence == pytest.approx(0.95)
    assert policy.verified_confidence == pytest.approx(0.90)
    assert policy.unknown_confidence == pytest.approx(0.0)
    assert policy.inheritance_confidence_factor == pytest.approx(0.95)
    assert policy.inference_unknown_confidence == pytest.approx(0.1)


def test_reasoning_policy_rejects_invalid_depth(tmp_path: Path):
    path = tmp_path / "reasoning_policy.json"
    path.write_text(
        '{"reasoning_policy": {"version": "1", "max_depth": 0, '
        '"fast_confidence": 0.95, "verified_confidence": 0.9, '
        '"unknown_confidence": 0.0, "inheritance_confidence_factor": 0.95, '
        '"inference_unknown_confidence": 0.1}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_depth"):
        ReasoningPolicy.load(tmp_path)
