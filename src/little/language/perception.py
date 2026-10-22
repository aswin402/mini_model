"""Candidate-only deterministic perception boundary."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from little.core.contracts import CandidateAction, CandidateClaim, CandidateFrame
from little.core.runtime_paths import RuntimePaths


def _split_sentences_after_punct(text: str) -> list[str]:
    """Split text into sentences after [.!?;\\n] followed by whitespace without regex."""
    out: list[str] = []
    n = len(text)
    last = 0
    i = 0
    while i < n:
        if text[i] in ".!?;\n" and i + 1 < n and text[i + 1].isspace():
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            seg = text[last : i + 1].strip()
            if seg:
                out.append(seg)
            last = j
            i = j
            continue
        i += 1
    tail = text[last:].strip()
    if tail:
        out.append(tail)
    return out


def _is_comparative_query(resolved_text: str) -> bool:
    """Detect comparative/superlative query syntax without regex."""
    stripped = resolved_text.strip()
    words = stripped.rstrip("?.!").lower().split()
    if len(words) >= 3 and words[0] in ("who", "which") and words[1] == "is":
        if words[2] == "the" and len(words) >= 4 and words[3].isalpha():
            return True
        if words[2].isalpha():
            return True
    if stripped.endswith("?") and "is" in words and "than" in words:
        is_idx = words.index("is")
        than_idx = words.index("than")
        if is_idx + 1 < than_idx < len(words) - 1:
            return True
    return False

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
                    raise ValueError(f"{key} must be a number in [0.0, 1.0]")  # noqa: TRY004
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
        frame_extractor: Any | None = None,
    ) -> None:
        self.policy = policy or PerceptionPolicy.default()
        self.parser = parser
        if frame_extractor is None:
            from little.neural.frame_extractor import NeuralFrameExtractor
            self.frame_extractor = NeuralFrameExtractor()
        else:
            self.frame_extractor = frame_extractor

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

        from little.procedural.math_story import MathStorySolver

        if MathStorySolver().is_math_story(text):
            return CandidateFrame.from_text(
                text,
                claims=[],
                intent="question",
                parsed_query=None,
                parsed_queries=[],
                question_parts=[text],
                uncertainty=self.policy.question_frame_uncertainty,
                model_id="math-story-solver",
                model_version=self.policy.version,
            )

        sentences = _split_sentences_after_punct(text)
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
                elif text.strip().endswith("?"):
                    return CandidateFrame.from_text(
                        text,
                        claims=[],
                        intent="question",
                        parsed_query=None,
                        parsed_queries=[],
                        question_parts=[text],
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
                    sub_claims = self.frame_extractor.extract_candidate_claims(resolved_sentence)
                    if sub_claims:
                        claims.extend(sub_claims)
                        previous_subject = sub_claims[0].subject
                    else:
                        triples = parser.parse_statement(resolved_sentence)
                        if triples:
                            for triple in triples:
                                claims.append(
                                    CandidateClaim(
                                        triple.subject, triple.predicate, triple.object_,
                                        probability=self.policy.deterministic_claim_probability,
                                        positive=not triple.is_negative,
                                        is_property=triple.is_property,
                                    )
                                )
                            previous_subject = triples[0].subject
                else:
                    for triple in triples:
                        claims.append(
                            CandidateClaim(
                                triple.subject, triple.predicate, triple.object_,
                                probability=self.policy.deterministic_claim_probability,
                                positive=not triple.is_negative,
                                is_property=triple.is_property,
                            )
                        )
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

        from little.dynamics.counterfactual_policy import CounterfactualPolicy
        from little.procedural.math_story import MathStorySolver

        cf_prefixes = CounterfactualPolicy.default().counterfactual_prefixes
        is_counterfactual = (
            any(resolved_text.lower().strip().startswith(p) for p in cf_prefixes)
            or " vs " in resolved_text.lower()
        )
        is_math = MathStorySolver().is_math_story(resolved_text)
        teaching_prefixes = (
            "did you know that",
            "do you know that",
            "i learned that",
            "note that",
            "remember that",
            "we know that",
        )
        if any(resolved_text.lower().strip().startswith(p) for p in teaching_prefixes):
            extracted = self.frame_extractor.extract_candidate_claims(resolved_text)
            if extracted:
                return CandidateFrame.from_text(
                    text,
                    claims=extracted,
                    intent="statement",
                    uncertainty=self.policy.deterministic_frame_uncertainty,
                    model_id="neural-frame-extractor",
                    model_version=self.policy.version,
                )

        is_comparative_query = _is_comparative_query(resolved_text)

        if (
            question is not None
            or resolved_text.strip().endswith("?")
            or is_counterfactual
            or is_math
            or is_comparative_query
        ):
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

        # Check specialized syntactic patterns that SimpleParser cannot parse (passive voice, parenthetical relative clauses)
        pass_frames = self.frame_extractor._extract_passive(resolved_text, text)
        if pass_frames:
            claims = self.frame_extractor.to_candidate_claims(pass_frames)
            return CandidateFrame.from_text(
                text,
                claims=claims,
                intent="statement",
                uncertainty=self.policy.deterministic_frame_uncertainty,
                model_id="neural-frame-extractor",
                model_version=self.policy.version,
            )

        rel_frames = self.frame_extractor._extract_relative_clause(resolved_text, text)
        if rel_frames:
            claims = self.frame_extractor.to_candidate_claims(rel_frames)
            return CandidateFrame.from_text(
                text,
                claims=claims,
                intent="statement",
                uncertainty=self.policy.deterministic_frame_uncertainty,
                model_id="neural-frame-extractor",
                model_version=self.policy.version,
            )

        triples = parser.CONSTRUCTION_ENGINE.parse_from_catalog(
            resolved_text, construction_catalog("statement")
        )
        if not triples:
            triples = parser.parse_statement(resolved_text)
        if not triples:
            extracted_claims = self.frame_extractor.extract_candidate_claims(resolved_text)
            if extracted_claims:
                return CandidateFrame.from_text(
                    text,
                    claims=extracted_claims,
                    intent="statement",
                    uncertainty=self.policy.deterministic_frame_uncertainty,
                    model_id="neural-frame-extractor",
                    model_version=self.policy.version,
                )
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
