"""Versioned lexical policy used by the fallback language parsers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from little.core.runtime_paths import RuntimePaths

from little.language.parser_confidence_policy import ParserConfidencePolicy
from little.language.question_policy import QuestionPatternPolicy
from little.core.semantic_predicate_policy import SemanticPredicatePolicy


@dataclass(frozen=True)
class ParserPolicy:
    leading_articles: tuple[str, ...]
    non_concept_words: frozenset[str]
    question_words: frozenset[str]
    action_verbs: dict[str, str]
    irregular_plurals: dict[str, str]
    invariable_words: frozenset[str]
    colors: frozenset[str]
    known_properties: frozenset[str]
    normalization_replacements: tuple[tuple[str, str], ...]
    isolated_greetings: frozenset[str]
    discourse_markers: tuple[str, ...]
    protected_discourse_prefixes: tuple[str, ...]
    polite_prefix_pattern: str
    slice_verbs: tuple[str, ...]
    greeting_prefixes: tuple[str, ...]
    pivot_prepositions: frozenset[str]
    verb_suffixes: tuple[str, ...]
    pivot_verbs: frozenset[str]
    excluded_pivot_verbs: frozenset[str]
    invalid_definition_markers: frozenset[str]
    invalid_category_markers: frozenset[str]
    numeric_skills: frozenset[str]
    open_query_predicates: frozenset[str]
    noun_auxiliaries: tuple[str, ...]
    plural_f_to_fe_words: frozenset[str]
    plural_es_suffixes: tuple[str, ...]
    plural_s_exceptions: tuple[str, ...]
    clause_delimiter_pattern: str
    confidence: ParserConfidencePolicy = field(
        default_factory=ParserConfidencePolicy.default
    )
    semantic: SemanticPredicatePolicy = field(
        default_factory=SemanticPredicatePolicy.default
    )
    question_patterns: QuestionPatternPolicy = field(
        default_factory=QuestionPatternPolicy.default
    )

    @classmethod
    def load(cls, directory: Path) -> ParserPolicy:
        path = Path(directory) / "parser_policy.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        data = payload.get("parser_policy")
        if not isinstance(data, dict):
            raise TypeError(f"{path} must contain 'parser_policy' object")

        def words(key: str) -> tuple[str, ...]:
            values = data.get(key)
            if not isinstance(values, list) or not all(
                isinstance(value, str) for value in values
            ):
                raise TypeError(f"{path} field {key!r} must be a list of strings")
            return tuple(value.strip().lower() for value in values if value.strip())

        def word_set(key: str) -> frozenset[str]:
            return frozenset(words(key))

        def upper_word_set(key: str) -> frozenset[str]:
            return frozenset(value.upper() for value in words(key))

        action_verbs = data.get("action_verbs")
        irregular_plurals = data.get("irregular_plurals")
        replacements = data.get("normalization_replacements")
        if not isinstance(action_verbs, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in action_verbs.items()
        ):
            raise TypeError(f"{path} field 'action_verbs' must map strings to strings")
        if not isinstance(irregular_plurals, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in irregular_plurals.items()
        ):
            raise TypeError(
                f"{path} field 'irregular_plurals' must map strings to strings"
            )
        if not isinstance(replacements, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in replacements.items()
        ):
            raise TypeError(
                f"{path} field 'normalization_replacements' must map strings to strings"
            )

        pattern = data.get("polite_prefix_pattern")
        if not isinstance(pattern, str):
            raise TypeError(f"{path} field 'polite_prefix_pattern' must be a string")
        clause_delimiter_pattern = data.get("clause_delimiter_pattern")
        if not isinstance(clause_delimiter_pattern, str):
            raise TypeError(
                f"{path} field 'clause_delimiter_pattern' must be a string"
            )

        semantic_path = Path(directory) / "semantic_predicate_policy.json"
        semantic = (
            SemanticPredicatePolicy.load(directory)
            if semantic_path.exists()
            else SemanticPredicatePolicy.default()
        )
        question_policy_path = Path(directory) / "parser_question_policy.json"
        question_patterns = (
            QuestionPatternPolicy.load(directory)
            if question_policy_path.exists()
            else QuestionPatternPolicy.default()
        )

        return cls(
            leading_articles=words("leading_articles"),
            non_concept_words=word_set("non_concept_words"),
            question_words=word_set("question_words"),
            action_verbs={
                key.strip().lower(): value.strip().lower()
                for key, value in action_verbs.items()
            },
            irregular_plurals={
                key.strip().lower(): value.strip().lower()
                for key, value in irregular_plurals.items()
            },
            invariable_words=word_set("invariable_words"),
            colors=word_set("colors"),
            known_properties=word_set("known_properties"),
            normalization_replacements=tuple(
                (key, value) for key, value in replacements.items()
            ),
            isolated_greetings=word_set("isolated_greetings"),
            discourse_markers=words("discourse_markers"),
            protected_discourse_prefixes=words("protected_discourse_prefixes"),
            polite_prefix_pattern=pattern,
            slice_verbs=words("slice_verbs"),
            greeting_prefixes=words("greeting_prefixes"),
            pivot_prepositions=word_set("pivot_prepositions"),
            verb_suffixes=words("verb_suffixes"),
            pivot_verbs=word_set("pivot_verbs"),
            excluded_pivot_verbs=word_set("excluded_pivot_verbs"),
            invalid_definition_markers=word_set("invalid_definition_markers"),
            invalid_category_markers=word_set("invalid_category_markers"),
            numeric_skills=upper_word_set("numeric_skills"),
            open_query_predicates=word_set("open_query_predicates"),
            noun_auxiliaries=words("noun_auxiliaries"),
            plural_f_to_fe_words=word_set("plural_f_to_fe_words"),
            plural_es_suffixes=words("plural_es_suffixes"),
            plural_s_exceptions=words("plural_s_exceptions"),
            clause_delimiter_pattern=clause_delimiter_pattern,
            confidence=ParserConfidencePolicy.default(),
            semantic=semantic,
            question_patterns=question_patterns,
        )

    @classmethod
    def default(cls) -> ParserPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
