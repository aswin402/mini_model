"""Optional lazy bridge to an installed Laya typed-decision runtime."""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from little.core.runtime_paths import RuntimePaths
from types import MappingProxyType
from typing import Any

from little.language.laya_adapter import LayaDecision


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _text(value, field_name)


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return value


def _probability(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return value


@dataclass(frozen=True)
class LayaModelLoadPolicy:
    model_id_or_path: str | None
    device: str
    subfolder: str | None
    runtime: str
    model_version: str | None


@dataclass(frozen=True)
class LayaRuntimePolicy:
    """Data-owned request and response mapping for the optional Laya SDK."""

    version: str
    question_id: str
    question_definition: Mapping[str, Any]
    prediction_method: str
    result_answers_field: str
    result_model_field: str
    result_routing_field: str
    routing_model_id_field: str
    routing_model_version_field: str
    model_id_field: str
    model_version_field: str
    answer_choice_field: str
    answer_probabilities_field: str
    answer_confidence_field: str
    model_load: LayaModelLoadPolicy

    @classmethod
    def load(cls, directory: Path) -> LayaRuntimePolicy:
        path = Path(directory) / "laya_runtime_policy.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            data = _mapping(payload, str(path))["laya_runtime_policy"]
            data = _mapping(data, "laya_runtime_policy")
            version = _text(data["version"], "version")
            question_id = _text(data["question_id"], "question_id")
            question_definition = _mapping(
                data["question_definition"], "question_definition"
            )
            prediction_method = _text(
                data["prediction_method"], "prediction_method"
            )

            fields = _mapping(data["result_fields"], "result_fields")
            answer_fields = _mapping(data["answer_fields"], "answer_fields")
            result_answers_field = _text(
                fields["answers"], "result_fields.answers"
            )
            result_model_field = _text(fields["model"], "result_fields.model")
            result_routing_field = _text(
                fields["routing"], "result_fields.routing"
            )
            routing_model_id_field = _text(
                fields["routing_model_id"], "result_fields.routing_model_id"
            )
            routing_model_version_field = _text(
                fields["routing_model_version"],
                "result_fields.routing_model_version",
            )
            model_id_field = _text(fields["model_id"], "result_fields.model_id")
            model_version_field = _text(
                fields["model_version"], "result_fields.model_version"
            )
            answer_choice_field = _text(
                answer_fields["choice"], "answer_fields.choice"
            )
            answer_probabilities_field = _text(
                answer_fields["probabilities"], "answer_fields.probabilities"
            )
            answer_confidence_field = _text(
                answer_fields["confidence"], "answer_fields.confidence"
            )

            load = _mapping(data["model_load"], "model_load")
            model_load = LayaModelLoadPolicy(
                model_id_or_path=_optional_text(
                    load.get("model_id_or_path"), "model_load.model_id_or_path"
                ),
                device=_text(load["device"], "model_load.device"),
                subfolder=_optional_text(
                    load.get("subfolder"), "model_load.subfolder"
                ),
                runtime=_text(load["runtime"], "model_load.runtime"),
                model_version=_optional_text(
                    load.get("model_version"), "model_load.model_version"
                ),
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid Laya runtime policy at {path}: {exc}") from exc

        return cls(
            version=version,
            question_id=question_id,
            question_definition=MappingProxyType(dict(question_definition)),
            prediction_method=prediction_method,
            result_answers_field=result_answers_field,
            result_model_field=result_model_field,
            result_routing_field=result_routing_field,
            routing_model_id_field=routing_model_id_field,
            routing_model_version_field=routing_model_version_field,
            model_id_field=model_id_field,
            model_version_field=model_version_field,
            answer_choice_field=answer_choice_field,
            answer_probabilities_field=answer_probabilities_field,
            answer_confidence_field=answer_confidence_field,
            model_load=model_load,
        )

    @classmethod
    def default(cls) -> LayaRuntimePolicy:
        return cls.load(RuntimePaths.default().schema_directory)


Loader = Callable[..., Any]


class LayaSDKBackend:
    """Adapt an explicitly loaded Laya agent to ``LayaDecisionBackend``."""

    def __init__(
        self,
        agent: Any,
        *,
        policy: LayaRuntimePolicy | None = None,
        package_version: str | None = None,
        model_version: str | None = None,
    ) -> None:
        if agent is None or not callable(
            getattr(agent, (policy or LayaRuntimePolicy.default()).prediction_method, None)
        ):
            raise TypeError("LayaSDKBackend requires an agent with a prediction method")
        self.agent = agent
        self.policy = policy or LayaRuntimePolicy.default()
        self.package_version = _optional_text(package_version, "package_version")
        self.model_version = _optional_text(model_version, "model_version")

    @classmethod
    def from_installed(
        cls,
        *,
        model_id_or_path: str | None = None,
        device: str | None = None,
        subfolder: str | None = None,
        policy: LayaRuntimePolicy | None = None,
        loader: Loader | None = None,
        package_version: str | None = None,
        model_version: str | None = None,
        runtime: str | None = None,
    ) -> LayaSDKBackend:
        selected_policy = policy or LayaRuntimePolicy.default()
        selected_runtime = runtime or selected_policy.model_load.runtime
        if selected_runtime not in {"agent", "router"}:
            raise RuntimeError(f"Unsupported Laya runtime: {selected_runtime}")
        selected_model = model_id_or_path or selected_policy.model_load.model_id_or_path
        if selected_runtime == "agent" and not selected_model:
            raise RuntimeError(
                "Laya model_id_or_path is required; configure it or pass it explicitly"
            )

        if loader is None:
            try:
                package = importlib.import_module("laya")
            except ModuleNotFoundError as exc:
                if exc.name == "laya":
                    raise RuntimeError(
                        "Install the optional Laya SDK before using from_installed"
                    ) from exc
                raise RuntimeError(
                    "The Laya SDK is installed but a runtime dependency is missing"
                ) from exc
            package_version = package_version or getattr(package, "__version__", None)
            if selected_runtime == "router":
                router = getattr(package, "Router", None)
                if not callable(router):
                    raise RuntimeError("Installed Laya SDK does not expose Router()")
                loader = router
            else:
                loader = getattr(package, "load", None)
                if not callable(loader):
                    raise RuntimeError("Installed Laya SDK does not expose load()")

        package_version = _optional_text(package_version, "package version")
        if package_version is None:
            raise RuntimeError(
                "Laya package version is required before loading the runtime"
            )

        load_kwargs: dict[str, object] = {
            "device": device or selected_policy.model_load.device,
        }
        if selected_runtime == "router":
            agent = loader(**load_kwargs)
        else:
            selected_subfolder = subfolder or selected_policy.model_load.subfolder
            if selected_subfolder is not None:
                load_kwargs["subfolder"] = selected_subfolder
            agent = loader(selected_model, **load_kwargs)
        return cls(
            agent,
            policy=selected_policy,
            package_version=package_version,
            model_version=(
                model_version or selected_policy.model_load.model_version
            ),
        )

    def decide(self, text: str) -> LayaDecision:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Laya input must be a non-empty string")
        predictor = getattr(self.agent, self.policy.prediction_method)
        result = predictor(
            text,
            {self.policy.question_id: dict(self.policy.question_definition)},
        )
        result_map = _mapping(result, "Laya prediction result")
        answers = _mapping(
            result_map.get(self.policy.result_answers_field),
            "Laya prediction answers",
        )
        answer = _mapping(
            answers.get(self.policy.question_id), "Laya intent answer"
        )
        intent = _text(
            answer.get(self.policy.answer_choice_field), "Laya selected intent"
        )
        raw_probabilities = _mapping(
            answer.get(self.policy.answer_probabilities_field),
            "Laya intent probabilities",
        )
        probabilities = self._normalize_probabilities(raw_probabilities)
        if intent not in probabilities:
            raise ValueError("Laya probabilities must include the selected intent")
        confidence = _probability(
            answer.get(self.policy.answer_confidence_field), "Laya confidence"
        )
        model_id, model_version = self._model_metadata(result_map)
        return LayaDecision(
            intent=intent,
            uncertainty=1.0 - confidence,
            model_id=model_id,
            model_version=model_version,
            intent_probabilities=probabilities,
            intent_alternatives=tuple(
                label for label in probabilities if label != intent
            ),
        )

    @staticmethod
    def _normalize_probabilities(raw: Mapping[str, Any]) -> dict[str, float]:
        if not raw:
            raise ValueError("Laya intent probabilities must be non-empty")
        values: dict[str, float] = {}
        for label, value in raw.items():
            label = _text(label, "Laya probability label")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("Laya probabilities must be numeric")
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("Laya probabilities must be finite and non-negative")
            values[label] = value
        total = sum(values.values())
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("Laya probabilities must have a positive finite sum")
        return {label: value / total for label, value in values.items()}

    def _model_metadata(self, result: Mapping[str, Any]) -> tuple[str, str]:
        routing_value = result.get(self.policy.result_routing_field)
        routing = (
            _mapping(routing_value, "Laya routing metadata")
            if routing_value is not None
            else {}
        )
        model_value = result.get(self.policy.result_model_field)
        if model_value is not None and not isinstance(model_value, Mapping):
            model_id = _text(model_value, "model id")
        else:
            model_id = None
        model = (
            _mapping(model_value, "Laya model metadata")
            if isinstance(model_value, Mapping)
            else {}
        )

        routing_model_id = self._metadata_value(
            routing, self.policy.routing_model_id_field, "routing model id"
        )
        if routing_model_id is not None:
            if model_id is None:
                model_id = routing_model_id
        if model_id is None:
            model_id = self._metadata_value(model, self.policy.model_id_field, "model id")
        if model_id is None:
            raise ValueError("Laya model id metadata is missing")

        model_version = self._metadata_value(
            routing,
            self.policy.routing_model_version_field,
            "routing model version",
        )
        if model_version is None:
            model_version = self._metadata_value(
                model, self.policy.model_version_field, "model version"
            )
        if model_version is None:
            model_version = self.model_version
        if model_version is None:
            raise ValueError("Laya model version metadata is missing")
        return model_id, model_version

    @staticmethod
    def _metadata_value(
        metadata: Mapping[str, Any], key: str, field_name: str
    ) -> str | None:
        if key not in metadata:
            return None
        return _text(metadata[key], field_name)
