"""Versioned pluralization rules shared by the language normalization paths."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


def _required_text(raw: Any, field: str, path: Path, index: int) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise TypeError(f"{path} rule {index} field {field!r} must be a non-empty string")
    return raw.strip()


def _string_tuple(raw: Any, field: str, path: Path, index: int) -> tuple[str, ...]:
    if not isinstance(raw, list) or not all(
        isinstance(value, str) and value.strip() for value in raw
    ):
        raise TypeError(f"{path} rule {index} field {field!r} must be a list of strings")
    return tuple(value.strip() for value in raw)


def _optional_text(raw: Any, field: str, path: Path, index: int) -> str | None:
    if raw is None:
        return None
    return _required_text(raw, field, path, index)


@dataclass(frozen=True)
class PluralRule:
    name: str
    suffixes: tuple[str, ...]
    minimum_length: int
    strip_length: int
    replacement: str
    required_word_source: str | None = None
    required_suffix_source: str | None = None
    excluded_suffix_source: str | None = None


@dataclass(frozen=True)
class PluralizationPolicy:
    rules: tuple[PluralRule, ...]

    @classmethod
    def load(cls, directory: Path) -> PluralizationPolicy:
        path = Path(directory) / "parser_plural_policy.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "little.parser_plural_policy.v1":
            raise ValueError(f"{path} has unsupported plural policy format")
        raw_rules = payload.get("rules")
        if not isinstance(raw_rules, list):
            raise TypeError(f"{path} field 'rules' must be a list")

        rules: list[PluralRule] = []
        for index, raw in enumerate(raw_rules):
            if not isinstance(raw, dict):
                raise TypeError(f"{path} rule {index} must be an object")
            minimum_length = raw.get("minimum_length")
            strip_length = raw.get("strip_length")
            if not isinstance(minimum_length, int) or isinstance(minimum_length, bool):
                raise TypeError(
                    f"{path} rule {index} field 'minimum_length' must be an integer"
                )
            if not isinstance(strip_length, int) or isinstance(strip_length, bool):
                raise TypeError(
                    f"{path} rule {index} field 'strip_length' must be an integer"
                )
            if minimum_length < 0 or strip_length < 0:
                raise ValueError(
                    f"{path} rule {index} lengths must be greater than or equal to zero"
                )
            replacement = raw.get("replacement")
            if not isinstance(replacement, str):
                raise TypeError(
                    f"{path} rule {index} field 'replacement' must be a string"
                )
            rules.append(
                PluralRule(
                    name=_required_text(raw.get("name"), "name", path, index),
                    suffixes=_string_tuple(raw.get("suffixes"), "suffixes", path, index),
                    minimum_length=minimum_length,
                    strip_length=strip_length,
                    replacement=replacement,
                    required_word_source=_optional_text(
                        raw.get("required_word_source"),
                        "required_word_source",
                        path,
                        index,
                    ),
                    required_suffix_source=_optional_text(
                        raw.get("required_suffix_source"),
                        "required_suffix_source",
                        path,
                        index,
                    ),
                    excluded_suffix_source=_optional_text(
                        raw.get("excluded_suffix_source"),
                        "excluded_suffix_source",
                        path,
                        index,
                    ),
                )
            )
        return cls(rules=tuple(rules))

    @classmethod
    def default(cls) -> PluralizationPolicy:
        return cls.load(RuntimePaths.default().schema_directory)

    def normalize(self, word: str, context: object) -> str:
        for rule in self.rules:
            if len(word) < rule.minimum_length:
                continue
            if not any(word.endswith(suffix) for suffix in rule.suffixes):
                continue
            if rule.required_word_source is not None:
                words = getattr(context, rule.required_word_source, ())
                if word not in words:
                    continue
            if rule.required_suffix_source is not None:
                suffixes = getattr(context, rule.required_suffix_source, ())
                if not any(word.endswith(suffix) for suffix in suffixes):
                    continue
            if rule.excluded_suffix_source is not None:
                suffixes = getattr(context, rule.excluded_suffix_source, ())
                if word.endswith(tuple(suffixes)):
                    continue
            return word[: -rule.strip_length] + rule.replacement
        return word
