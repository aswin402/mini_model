"""Candidate-only deterministic perception boundary."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from little.core.runtime_paths import RuntimePaths
from typing import TYPE_CHECKING, Protocol

from little.core.contracts import CandidateAction, CandidateClaim, CandidateFrame

if TYPE_CHECKING:
    from little.language.dialogue import DialogueContext
    from little.language.parser import SimpleParser
    from little.memory.store import MemoryStore


class PerceptionAdapter(Protocol):
    def perceive(
        self,
        text: str,
        *,
        memory: MemoryStore,
        context: DialogueContext | None = None,
    ) -> CandidateFrame: ...


@dataclass(frozen=True)
class PerceptionPolicy:
    version: str
    deterministic_claim_probability: float
    deterministic_frame_uncertainty: float
    question_frame_uncertainty: float
    unknown_frame_uncertainty: float
    action_probability: float

    @classmethod
    def load(cls, directory: Path) -> PerceptionPolicy:
        path = Path(directory) / "perception_policy.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            data = payload["perception_policy"]
            version = data["version"]
            if not isinstance(version, str) or not version:
                raise ValueError("version must be a non-empty string")
            keys = (
                "deterministic_claim_probability",
                "deterministic_frame_uncertainty",
                "question_frame_uncertainty",
                "unknown_frame_uncertainty",
                "action_probability",
            )
            values: dict[str, float] = {}
            for key in keys:
                value = data[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{key} must be a number in [0.0, 1.0]")
                try:
                    finite = math.isfinite(value)
                except OverflowError:
                    finite = False
                if not finite:
                    raise ValueError(f"{key} must be finite")
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{key} must be a number in [0.0, 1.0]")
                values[key] = float(value)
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid perception policy at {path}: {exc}") from exc
        except ValueError as exc:
            raise ValueError(f"invalid perception policy at {path}: {exc}") from exc
        return cls(version=version, **values)

    @classmethod
    def default(cls) -> PerceptionPolicy:
        return cls.load(RuntimePaths.default().schema_directory)


class DeterministicPerceptionAdapter:
    def __init__(
        self,
        policy: PerceptionPolicy | None = None,
        parser: type[SimpleParser] | None = None,
    ) -> None:
        self.policy = policy or PerceptionPolicy.default()
        self.parser = parser

    def perceive(
        self,
        text: str,
        *,
        memory: MemoryStore,
        context: DialogueContext | None = None,
    ) -> CandidateFrame:
        if not text.strip():
            raise ValueError("perception input must be non-empty")

        from little.language.parser import SimpleParser

        parser = self.parser or SimpleParser

        def construction_catalog(construction_type: str | None = None):
            configured = list(parser.CONSTRUCTIONS)
            stored = memory.list_constructions(construction_type=construction_type)
            if construction_type is None:
                return configured + stored
            return [
                construction
                for construction in configured
                if construction.construction_type == construction_type
            ] + stored

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?;\n])\s+", text)
            if sentence.strip()
        ]
        if len(sentences) > 1:
            question_sentences = [
                sentence
                for sentence in sentences
                if sentence.endswith("?") or parser.parse_question(sentence)
            ]
            if question_sentences:
                question_text = " ".join(question_sentences)
                resolved_question = (
                    context.resolve_anaphora_in_text(question_text)
                    if context
                    else question_text
                )
                known_concepts = {
                    concept.name for concept in memory.list_concepts()
                }
                question_parts, parsed_queries = parser.parse_compound_question(
                    resolved_question,
                    memory=memory,
                    constructions=construction_catalog(),
                    known_concepts=known_concepts,
                    last_subject=context.get_salient_subject() if context else None,
                )
                if question_parts:
                    return CandidateFrame.from_text(
                        text,
                        claims=[],
                        intent="question",
                        parsed_query=parsed_queries[0] if parsed_queries else None,
                        parsed_queries=parsed_queries,
                        question_parts=question_parts,
                        uncertainty=self.policy.question_frame_uncertainty,
                        model_id="deterministic-parser",
                        model_version=self.policy.version,
                    )
            claims: list[CandidateClaim] = []
            previous_subject = context.get_salient_subject() if context else None
            for index, sentence in enumerate(sentences):
                if sentence.endswith("?") or parser.parse_question(sentence):
                    continue
                if index == 0 and context is not None:
                    sentence = context.resolve_anaphora_in_text(sentence)
                resolved_sentence = parser.resolve_anaphora(
                    sentence, previous_subject
                )
                triples = parser.CONSTRUCTION_ENGINE.parse_from_catalog(
                    resolved_sentence, construction_catalog("statement")
                )
                if not triples:
                    triples = parser.parse_statement(resolved_sentence)
                for triple in triples:
                    claims.append(
                        CandidateClaim(
                            triple.subject, triple.predicate, triple.object_,
                            probability=self.policy.deterministic_claim_probability,
                            positive=not triple.is_negative,
                            is_property=triple.is_property,
                        )
                    )
                if triples:
                    previous_subject = triples[0].subject
            return CandidateFrame.from_text(
                text,
                claims=claims,
                intent="statement" if claims else "unknown",
                uncertainty=(
                    self.policy.deterministic_frame_uncertainty
                    if claims else self.policy.unknown_frame_uncertainty
                ),
                model_id="deterministic-parser",
                model_version=self.policy.version,
            )
        resolved_text = context.resolve_anaphora_in_text(text) if context else text
        known_concepts = {concept.name for concept in memory.list_concepts()}
        question_parts: list[str] = []
        parsed_queries = []
        if "?" in resolved_text:
            question_parts, parsed_queries = parser.parse_compound_question(
                resolved_text,
                memory=memory,
                constructions=construction_catalog(),
                known_concepts=known_concepts,
                last_subject=context.get_salient_subject() if context else None,
            )
        if question_parts:
            return CandidateFrame.from_text(
                text,
                claims=[],
                intent="question",
                parsed_query=parsed_queries[0] if parsed_queries else None,
                parsed_queries=parsed_queries,
                question_parts=question_parts,
                uncertainty=self.policy.question_frame_uncertainty,
                model_id="deterministic-parser",
                model_version=self.policy.version,
            )
        question = parser.CONSTRUCTION_ENGINE.parse_question_from_catalog_with_procedural(
            resolved_text, construction_catalog()
        )
        if question is None:
            question = parser.parse_question(
                resolved_text,
                known_concepts=known_concepts,
            )
        if question is not None or resolved_text.strip().endswith("?"):
            return CandidateFrame.from_text(
                text,
                claims=[],
                intent="question",
                parsed_query=question,
                uncertainty=self.policy.question_frame_uncertainty,
                model_id="deterministic-parser",
                model_version=self.policy.version,
            )

        action = parser.CONSTRUCTION_ENGINE.parse_action_from_catalog(
            resolved_text, construction_catalog("action")
        )
        if action is None:
            action = parser.parse_action(resolved_text)
        if action is not None:
            name, arguments = action
            return CandidateFrame.from_text(
                text,
                claims=[],
                actions=[
                    CandidateAction(
                        name=name,
                        arguments={str(key): str(value) for key, value in arguments.items()},
                        probability=self.policy.action_probability,
                    )
                ],
                intent="action",
                uncertainty=self.policy.deterministic_frame_uncertainty,
                model_id="deterministic-parser",
                model_version=self.policy.version,
            )

        triples = parser.CONSTRUCTION_ENGINE.parse_from_catalog(
            resolved_text, construction_catalog("statement")
        )
        if not triples:
            triples = parser.parse_statement(resolved_text)
        if not triples:
            return CandidateFrame.from_text(
                text,
                claims=[],
                intent="unknown",
                uncertainty=self.policy.unknown_frame_uncertainty,
                model_id="deterministic-parser",
                model_version=self.policy.version,
            )

        claims = [
            CandidateClaim(
                triple.subject,
                triple.predicate,
                triple.object_,
                probability=self.policy.deterministic_claim_probability,
                positive=not triple.is_negative,
                is_property=triple.is_property,
            )
            for triple in triples
        ]
        return CandidateFrame.from_text(
            text,
            claims=claims,
            intent="statement",
            uncertainty=self.policy.deterministic_frame_uncertainty,
            model_id="deterministic-parser",
            model_version=self.policy.version,
        )
