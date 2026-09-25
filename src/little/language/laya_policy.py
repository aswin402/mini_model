"""Versioned numeric decision policy for the deterministic Laya fallback."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from little.core.runtime_paths import RuntimePaths
from typing import Any


def _number(path: Path, field: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} field {field!r} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise TypeError(f"{path} field {field!r} must be a finite number")
    return value


def _required(mapping: dict[str, Any], path: Path, field: str) -> Any:
    if field not in mapping:
        raise ValueError(f"{path} missing field {field!r}")
    return mapping[field]


def _probability(path: Path, field: str, value: Any) -> float:
    value = _number(path, field, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{path} field {field!r} must be between 0 and 1")
    return value


def _positive(path: Path, field: str, value: Any) -> float:
    value = _number(path, field, value)
    if value <= 0.0:
        raise ValueError(f"{path} field {field!r} must be greater than 0")
    return value


def _nonnegative(path: Path, field: str, value: Any) -> float:
    value = _number(path, field, value)
    if value < 0.0:
        raise ValueError(f"{path} field {field!r} must be non-negative")
    return value


def _nonnegative_integer(path: Path, field: str, value: Any) -> int:
    value = _number(path, field, value)
    if not value.is_integer() or value < 0.0:
        raise ValueError(f"{path} field {field!r} must be a non-negative integer")
    return int(value)


def _validate_probability(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Laya policy field {field!r} must be a probability")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"Laya policy field {field!r} must be between 0 and 1")


def _validate_nonnegative(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Laya policy field {field!r} must be a number")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"Laya policy field {field!r} must be non-negative")


@dataclass(frozen=True)
class LayaChoicePolicy:
    """Numeric scoring and temperature calibration for categorical choices."""

    baseline_prior: float
    minimum_token_length: int
    state_token_match_weight: float
    question_token_match_weight: float
    association_overlap_weight: float
    supported_probability_threshold: float
    temperature_buckets: dict[str, float]
    default_temperature_bucket: str

    def __post_init__(self) -> None:
        _validate_nonnegative(self.baseline_prior, "choice.baseline_prior")
        _validate_nonnegative(
            self.state_token_match_weight, "choice.state_token_match_weight"
        )
        _validate_nonnegative(
            self.question_token_match_weight, "choice.question_token_match_weight"
        )
        _validate_nonnegative(
            self.association_overlap_weight, "choice.association_overlap_weight"
        )
        _validate_probability(
            self.supported_probability_threshold,
            "choice.supported_probability_threshold",
        )
        if not self.temperature_buckets:
            raise ValueError("Laya choice temperature buckets cannot be empty")
        if (
            isinstance(self.minimum_token_length, bool)
            or not isinstance(self.minimum_token_length, int)
            or self.minimum_token_length < 0
        ):
            raise ValueError("Laya choice minimum token length must be non-negative")
        for bucket, value in self.temperature_buckets.items():
            if not isinstance(bucket, str) or not bucket.strip():
                raise ValueError(
                    "Laya choice temperature bucket names must be non-empty strings"
                )
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(
                    "Laya choice temperatures must be finite and greater than 0"
                )
        if self.default_temperature_bucket not in self.temperature_buckets:
            raise ValueError(
                "Laya default choice temperature bucket must be configured"
            )


@dataclass(frozen=True)
class LayaPropositionPolicy:
    """Numeric calibration and status thresholds for binary propositions."""

    neutral_probability: float
    minimum_token_length: int
    minimum_probability: float
    maximum_probability: float
    prior_true: float
    match_weight: float
    confidence_scale: float
    supported_probability_threshold: float
    contradicted_probability_threshold: float

    def __post_init__(self) -> None:
        for field, value in (
            ("neutral_probability", self.neutral_probability),
            ("minimum_probability", self.minimum_probability),
            ("maximum_probability", self.maximum_probability),
            ("prior_true", self.prior_true),
            ("supported_probability_threshold", self.supported_probability_threshold),
            (
                "contradicted_probability_threshold",
                self.contradicted_probability_threshold,
            ),
        ):
            _validate_probability(value, f"proposition.{field}")
        _validate_nonnegative(self.match_weight, "proposition.match_weight")
        _validate_nonnegative(
            self.confidence_scale, "proposition.confidence_scale"
        )
        if (
            isinstance(self.minimum_token_length, bool)
            or not isinstance(self.minimum_token_length, int)
            or self.minimum_token_length < 0
        ):
            raise ValueError(
                "Laya proposition minimum token length must be non-negative"
            )
        if self.minimum_probability > self.maximum_probability:
            raise ValueError("Laya proposition probability bounds are ordered incorrectly")
        if self.contradicted_probability_threshold > self.neutral_probability:
            raise ValueError(
                "Laya proposition contradicted threshold must not exceed neutral probability"
            )
        if self.supported_probability_threshold < self.neutral_probability:
            raise ValueError(
                "Laya proposition supported threshold must reach neutral probability"
            )


@dataclass(frozen=True)
class LayaDecisionPolicy:
    """Validated numeric policy for the deterministic, non-model fallback."""

    entropy_unknown_threshold: float
    entropy_probability_floor: float
    choice: LayaChoicePolicy
    proposition: LayaPropositionPolicy

    def __post_init__(self) -> None:
        _validate_probability(
            self.entropy_unknown_threshold,
            "entropy_unknown_threshold",
        )
        if (
            isinstance(self.entropy_probability_floor, bool)
            or not isinstance(self.entropy_probability_floor, (int, float))
            or not 0.0 < float(self.entropy_probability_floor) < 1.0
        ):
            raise ValueError(
                "Laya policy field 'entropy_probability_floor' must be between 0 and 1 exclusively"
            )
        if not isinstance(self.choice, LayaChoicePolicy):
            raise ValueError("Laya decision policy choice must be LayaChoicePolicy")
        if not isinstance(self.proposition, LayaPropositionPolicy):
            raise ValueError(
                "Laya decision policy proposition must be LayaPropositionPolicy"
            )

    @classmethod
    def load(cls, directory: Path) -> LayaDecisionPolicy:
        path = Path(directory) / "laya_policy.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Laya decision policy not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} contains invalid JSON: {exc.msg}") from exc

        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        data = _required(payload, path, "laya_policy")
        if not isinstance(data, dict):
            raise TypeError(f"{path} field 'laya_policy' must be an object")

        entropy_threshold = _probability(
            path,
            "laya_policy.entropy_unknown_threshold",
            _required(data, path, "entropy_unknown_threshold"),
        )
        entropy_floor = _number(
            path,
            "laya_policy.entropy_probability_floor",
            _required(data, path, "entropy_probability_floor"),
        )
        if not 0.0 < entropy_floor < 1.0:
            raise ValueError(
                f"{path} field 'laya_policy.entropy_probability_floor' must be between 0 and 1 exclusively"
            )

        raw_choice = _required(data, path, "choice")
        if not isinstance(raw_choice, dict):
            raise TypeError(f"{path} field 'choice' must be an object")
        raw_temperatures = _required(raw_choice, path, "temperature_buckets")
        if not isinstance(raw_temperatures, dict) or not raw_temperatures:
            raise TypeError(
                f"{path} field 'choice.temperature_buckets' must be a non-empty object"
            )
        temperatures: dict[str, float] = {}
        for bucket, value in raw_temperatures.items():
            if not isinstance(bucket, str) or not bucket.strip():
                raise TypeError(
                    f"{path} choice temperature bucket names must be non-empty strings"
                )
            temperatures[bucket] = _positive(
                path, f"choice.temperature_buckets.{bucket}", value
            )
        default_bucket = _required(raw_choice, path, "default_temperature_bucket")
        if not isinstance(default_bucket, str):
            raise TypeError(
                f"{path} field 'choice.default_temperature_bucket' must be a string"
            )

        choice = LayaChoicePolicy(
            baseline_prior=_number(
                path,
                "choice.baseline_prior",
                _required(raw_choice, path, "baseline_prior"),
            ),
            minimum_token_length=_nonnegative_integer(
                path,
                "choice.minimum_token_length",
                _required(raw_choice, path, "minimum_token_length"),
            ),
            state_token_match_weight=_nonnegative(
                path,
                "choice.state_token_match_weight",
                _required(raw_choice, path, "state_token_match_weight"),
            ),
            question_token_match_weight=_nonnegative(
                path,
                "choice.question_token_match_weight",
                _required(raw_choice, path, "question_token_match_weight"),
            ),
            association_overlap_weight=_nonnegative(
                path,
                "choice.association_overlap_weight",
                _required(raw_choice, path, "association_overlap_weight"),
            ),
            supported_probability_threshold=_probability(
                path,
                "choice.supported_probability_threshold",
                _required(raw_choice, path, "supported_probability_threshold"),
            ),
            temperature_buckets=temperatures,
            default_temperature_bucket=default_bucket,
        )

        raw_proposition = _required(data, path, "proposition")
        if not isinstance(raw_proposition, dict):
            raise TypeError(f"{path} field 'proposition' must be an object")
        proposition = LayaPropositionPolicy(
            neutral_probability=_probability(
                path,
                "proposition.neutral_probability",
                _required(raw_proposition, path, "neutral_probability"),
            ),
            minimum_token_length=_nonnegative_integer(
                path,
                "proposition.minimum_token_length",
                _required(raw_proposition, path, "minimum_token_length"),
            ),
            minimum_probability=_probability(
                path,
                "proposition.minimum_probability",
                _required(raw_proposition, path, "minimum_probability"),
            ),
            maximum_probability=_probability(
                path,
                "proposition.maximum_probability",
                _required(raw_proposition, path, "maximum_probability"),
            ),
            prior_true=_probability(
                path,
                "proposition.prior_true",
                _required(raw_proposition, path, "prior_true"),
            ),
            match_weight=_nonnegative(
                path,
                "proposition.match_weight",
                _required(raw_proposition, path, "match_weight"),
            ),
            confidence_scale=_nonnegative(
                path,
                "proposition.confidence_scale",
                _required(raw_proposition, path, "confidence_scale"),
            ),
            supported_probability_threshold=_probability(
                path,
                "proposition.supported_probability_threshold",
                _required(
                    raw_proposition, path, "supported_probability_threshold"
                ),
            ),
            contradicted_probability_threshold=_probability(
                path,
                "proposition.contradicted_probability_threshold",
                _required(
                    raw_proposition,
                    path,
                    "contradicted_probability_threshold",
                ),
            ),
        )
        return cls(
            entropy_unknown_threshold=entropy_threshold,
            entropy_probability_floor=entropy_floor,
            choice=choice,
            proposition=proposition,
        )

    @classmethod
    def default(cls) -> LayaDecisionPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
