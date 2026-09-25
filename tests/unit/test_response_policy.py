import json
from pathlib import Path

from little.core.models import BeliefStatus, InferenceResult
from little.core.response_policy import ResponsePolicy


def test_response_policy_controls_math_verbalization(tmp_path: Path):
    (tmp_path / "response_policy.json").write_text(
        json.dumps(
            {
                "format": "little.response_policy.v1",
                "response_policy": {
                    "math_templates": {
                        "is_prime_true": "PRIME {n} => {answer}",
                    },
                    "generic_templates": {
                        "numeric_result": "VALUE {answer}",
                        "value_result": "VALUE {answer}",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    policy = ResponsePolicy.load(tmp_path)
    result = InferenceResult(
        query="is 7 prime",
        status=BeliefStatus.SUPPORTED,
        answer=True,
        confidence=1.0,
        evidence=["Evaluated skill IS_PRIME(n=7) = True"],
    )

    assert result.verbalize(policy=policy) == "PRIME 7 => True"


def test_response_policy_routes_arbitrary_skill_without_code_branch(tmp_path: Path):
    (tmp_path / "response_policy.json").write_text(
        json.dumps(
            {
                "format": "little.response_policy.v1",
                "response_policy": {
                    "math_templates": {
                        "custom_score": "CUSTOM {value} => {answer}",
                    },
                    "generic_templates": {
                        "numeric_result": "VALUE {answer}",
                        "value_result": "VALUE {answer}",
                    },
                    "math_routes": {
                        "CUSTOM_SCORE": {
                            "template": "custom_score",
                            "arguments": {"value": "value"},
                            "answer": "raw",
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    policy = ResponsePolicy.load(tmp_path)
    result = InferenceResult(
        query="score 9",
        status=BeliefStatus.SUPPORTED,
        answer=42,
        confidence=1.0,
        evidence=["Evaluated skill CUSTOM_SCORE(value=9) = 42"],
    )

    assert result.verbalize(policy=policy) == "CUSTOM 9 => 42"
