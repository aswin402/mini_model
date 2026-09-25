"""Natural Language Statement and Question Parser for LITTLE.

Parses simple declarative statements into Semantic Triples without requiring
a monolithic LLM, and dispatches learning and inference operations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from little.core.contracts import (
    CandidateClaim,
    CandidateFrame,
    CommitStatus,
    ParsedQuery,
    SourceType,
    VerificationResult,
    VerificationStatus,
)
from little.core.grammar_registry import GrammarRegistry
from little.core.runtime_policy import RuntimePolicy
from little.core.models import (
    BeliefStatus,
    Construction,
    InferenceResult,
    LearningResult,
    UpdateType,
)
from little.dynamics.cfc import ContinuousDynamicsEngine
from little.inference.engine import InferenceEngine
from little.inference.invariant_gates import DeepSeekInvariantVerifier
from little.language.construction import ConstructionEngine
from little.language.dialogue import DialogueContext
from little.language.math_policy import MathCapture
from little.language.parser_policy import ParserPolicy
from little.language.perception import DeterministicPerceptionAdapter, PerceptionAdapter
from little.memory.store import MemoryStore
from little.procedural.runner import SkillRunner
from little.procedural.math_cas import UnitConversionGraph
from little.procedural.skills import register_builtin_skills


@dataclass
class ParsedTriple:
    subject: str
    predicate: str
    object_: str
    is_property: bool = False
    is_negative: bool = False


class _ParserPolicyCompatibilityMeta(type):
    """Expose a legacy default without storing it on the parser base class."""

    def __getattr__(cls, name: str) -> Any:
        if name == "POLICY":
            return ParserPolicy.default()
        if name == "CONSTRUCTIONS":
            return GrammarRegistry.default()
        if name == "UNIT_CONVERSIONS":
            return UnitConversionGraph()
        if name == "CONSTRUCTION_ENGINE":
            return ConstructionEngine
        if name == "QUESTION_PATTERNS":
            return ParserPolicy.default().question_patterns
        if name == "MATH_PATTERNS":
            return ParserPolicy.default().math_patterns
        if name == "TEMPORAL_PATTERNS":
            return ParserPolicy.default().temporal_patterns
        if name == "SEMANTIC_PATTERNS":
            return ParserPolicy.default().semantic_patterns
        raise AttributeError(name)


class SimpleParser(metaclass=_ParserPolicyCompatibilityMeta):
    """Robust pattern and rule-based parser for English statements and questions."""

    @classmethod
    def _policy(cls) -> ParserPolicy:
        """Resolve a bound or legacy policy without storing a base global default."""
        configured = getattr(cls, "POLICY", None)
        return configured if configured is not None else ParserPolicy.default()

    @classmethod
    def _question_patterns(cls):
        """Resolve explicitly bound or policy-owned special question patterns."""
        configured = cls.__dict__.get("QUESTION_PATTERNS")
        return (
            configured
            if configured is not None
            else cls._policy().question_patterns
        )

    @classmethod
    def _math_patterns(cls):
        """Resolve explicitly bound or policy-owned arithmetic patterns."""
        configured = cls.__dict__.get("MATH_PATTERNS")
        return configured if configured is not None else cls._policy().math_patterns

    @classmethod
    def _temporal_patterns(cls):
        """Resolve duration-question patterns from the active policy."""
        configured = cls.__dict__.get("TEMPORAL_PATTERNS")
        return (
            configured
            if configured is not None
            else cls._policy().temporal_patterns
        )

    @classmethod
    def _semantic_patterns(cls):
        """Resolve direct semantic-question patterns from the active policy."""
        configured = cls.__dict__.get("SEMANTIC_PATTERNS")
        return (
            configured
            if configured is not None
            else cls._policy().semantic_patterns
        )

    @classmethod
    def normalize_text(cls, text: str) -> str:
        s = text.strip()
        for source, replacement in cls._policy().normalization_replacements:
            token_pattern = rf"(?<!\w){re.escape(source)}(?!\w)"
            s = re.sub(token_pattern, replacement, s, flags=re.IGNORECASE)

        # Isolated greetings must NOT be stripped away to empty
        if s.lower() in cls._policy().isolated_greetings:
            return s

        # Conversational discourse markers e.g. "so what is 10+10", "hey what can u do", "well tell me..."
        protected = cls._policy().protected_discourse_prefixes
        if not any(
            re.match(rf"^{re.escape(prefix)}\b", s, re.IGNORECASE)
            for prefix in protected
        ):
            markers = "|".join(re.escape(marker) for marker in cls._policy().discourse_markers)
            s = re.sub(
                rf"^(?:(?:{markers})\b[\s,]*)+",
                "",
                s,
                flags=re.IGNORECASE,
            ).strip()

        # Conversational polite prefixes e.g. "Can you calculate ...", "Could you please tell me ...", "Do you know ..."
        s = re.sub(
            cls._policy().polite_prefix_pattern,
            "",
            s,
            flags=re.IGNORECASE,
        ).strip()
        return s

    @classmethod
    def is_valid_concept(cls, c: str) -> bool:
        """Check if candidate text constitutes a valid domain concept rather than a question or pronoun."""
        if not c or c.strip() in ("", "?"):
            return False
        clean = c.strip().lower()
        if clean in cls._policy().non_concept_words or clean == "?":
            return False
        words = set(clean.split())
        return not any(
            w in cls._policy().non_concept_words or w in cls._policy().question_words
            for w in words
        )

    @classmethod
    def resolve_anaphora(
        cls,
        text: str,
        last_subject: str | None,
        last_object: str | None = None,
    ) -> str:
        """Resolve pronouns (e.g. 'it', 'its', 'they', 'he', 'she') using dialogue context."""
        if not last_subject:
            return text

        s = text.strip()

        # If sentence is a conditional with an internal antecedent (e.g. "If an animal is a canine, then it is..."),
        # do not override the bound pronoun with the previous dialogue subject!
        if re.match(r"^if\s+", s, re.IGNORECASE):
            return s

        # 1. Possessive pronouns: "what is its color", "what color is it", "tell me about its parts"
        s = re.sub(
            r"\b(?:its|his|her|their)\s+color\b",
            f"color of {last_subject}",
            s,
            flags=re.IGNORECASE,
        )
        s = re.sub(
            r"\b(?:its|his|her|their)\b",
            last_subject,
            s,
            flags=re.IGNORECASE,
        )

        # 2. Subject personal pronouns: "is it an animal", "does it have wings", "it is grey", "can it fly"
        # Only replace personal pronouns: it, he, she, they.
        # DO NOT replace "that" as a whole word because "that" is commonly a relative pronoun
        # ("mammals that live in the ocean") or complementizer ("know that...").
        s = re.sub(
            r"\b(?:it|he|she|they)\b",
            last_subject,
            s,
            flags=re.IGNORECASE,
        )

        # 3. Demonstrative pronouns at sentence start or question focus: "this is...", "what is this", "is that..."
        s = re.sub(
            r"^(?:this|that)\s+",
            f"{last_subject} ",
            s,
            flags=re.IGNORECASE,
        )
        s = re.sub(
            r"\b(is|was|about)\s+(?:this|that)\b",
            rf"\1 {last_subject}",
            s,
            flags=re.IGNORECASE,
        )
        return s

    @classmethod
    def clean_noun(cls, text: str) -> str:
        raw_lower = text.strip().lower()
        if raw_lower in ("a", "an", "the", "this", "that", "these", "those"):
            return ""
        articles = "|".join(
            re.escape(article) for article in cls._policy().leading_articles
        )
        s = re.sub(rf"^(?:{articles})\s+", "", raw_lower).strip()
        if not s:
            return ""
        if s in cls._policy().irregular_plurals:
            return cls._policy().irregular_plurals[s]
        if s in cls._policy().invariable_words:
            return s
        # Multi-word nouns (e.g. "living things" -> "living thing", "apple slices" -> "apple slice")
        tokens = s.split()
        if len(tokens) > 1:
            last = cls.clean_noun(tokens[-1])
            return " ".join(tokens[:-1] + [last])

        if s.endswith("ies") and len(s) > 4:
            s = s[:-3] + "y"
        elif s.endswith("ves") and len(s) > 4:
            if s in cls._policy().plural_f_to_fe_words:
                s = s[:-3] + "fe"
            else:
                s = s[:-3] + "f"
        elif s.endswith("es") and len(s) > 4:
            if s.endswith(cls._policy().plural_es_suffixes):
                s = s[:-2]
            else:
                s = s[:-1]
        elif (
            s.endswith("s")
            and not s.endswith(cls._policy().plural_s_exceptions)
            and len(s) > 3
        ):
            s = s[:-1]
        return s

    @classmethod
    def split_conjoined_items(cls, text: str) -> list[str]:
        """Split conjoined list: 'wheels, an engine, and doors' -> ['wheel', 'engine', 'door']."""
        raw = text.strip()
        parts = [
            p.strip() for p in re.split(r",\s*(?:and\s+)?|\s+and\s+", raw) if p.strip()
        ]
        results = []
        for p in parts:
            c = cls.clean_noun(p)
            if c and cls.is_valid_concept(c) and c not in results:
                results.append(c)
        return (
            results
            if results
            else (
                [cls.clean_noun(raw)]
                if cls.is_valid_concept(cls.clean_noun(raw))
                else []
            )
        )

    @classmethod
    def parse_statement(cls, text: str) -> list[ParsedTriple]:
        clean = cls.normalize_text(text).rstrip(".").strip()
        triples: list[ParsedTriple] = []

        # 0a. Coordinate compound sentences: "A dog is an animal and it has fur", "A falcon is a bird and it hunts rodents"
        coord_pattern = r",?\s+and\s+(?=(?:it|they|he|she|this|that|these|those|(?:a|an|the)\s+[a-z]+)\s+[a-z]+)\b"
        m_coord = re.search(coord_pattern, clean, flags=re.IGNORECASE)
        if m_coord:
            coord_parts = [
                p.strip()
                for p in re.split(coord_pattern, clean, flags=re.IGNORECASE)
                if p.strip()
            ]
            if len(coord_parts) > 1:
                curr_subject = None
                for cp in coord_parts:
                    cp_resolved = cls.resolve_anaphora(cp, curr_subject)
                    cp_triples = cls.parse_statement(cp_resolved)
                    if cp_triples:
                        curr_subject = cp_triples[0].subject
                        for ct in cp_triples:
                            if ct not in triples:
                                triples.append(ct)
                if triples:
                    return triples

        # 0b. Subject-level Restrictive Relative Clauses:
        # "The animal that has gills and swims in water is a fish"
        # "An animal that eats meat is a carnivore"
        m_subj_rel = re.match(
            r"^(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s-]+?)\s+(?:that|which|who)\s+(.+?)\s+(?:is|are)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s-]+)$",
            clean,
            re.IGNORECASE,
        )
        if m_subj_rel:
            category = cls.clean_noun(m_subj_rel.group(1))
            rel_body = m_subj_rel.group(2).strip()
            concept = cls.clean_noun(m_subj_rel.group(3))

            if cls.is_valid_concept(concept) and cls.is_valid_concept(category):
                triples.append(
                    ParsedTriple(
                        subject=concept,
                        predicate=cls._policy().semantic.taxonomy,
                        object_=category,
                    )
                )
                sub_clauses = [
                    c.strip()
                    for c in re.split(r",\s*(?:and\s+)?|\s+and\s+", rel_body)
                    if c.strip()
                ]
                for sc in sub_clauses:
                    sub_stmt = f"{concept} {sc}"
                    sub_triples = cls.parse_statement(sub_stmt)
                    for st in sub_triples:
                        if st not in triples:
                            triples.append(st)
                return triples

        # 0c. Object-level Relative Clauses: "Whales are mammals that live in the ocean and have fins"
        m_rel = re.match(
            r"^(.*?\s+(?:is|are)\s+[^,]+?)\s+(?:that|which|who)\s+(.*)$",
            clean,
            re.IGNORECASE,
        )
        if m_rel:
            main_part = m_rel.group(1).strip()
            sub_part = m_rel.group(2).strip()
            main_triples = cls.parse_statement(main_part)
            if main_triples:
                subj = main_triples[0].subject
                triples.extend(main_triples)
                sub_clauses = [
                    c.strip()
                    for c in re.split(r",\s*(?:and\s+)?|\s+and\s+", sub_part)
                    if c.strip()
                ]
                for sc in sub_clauses:
                    sub_stmt = f"{subj} {sc}"
                    sub_triples = cls.parse_statement(sub_stmt)
                    for st in sub_triples:
                        if st not in triples:
                            triples.append(st)
                return triples

        # Prefer the versioned construction catalog for ordinary statements.
        # Compatibility regexes below remain only for grammar not yet captured
        # by the data pack; adding a new relation should not require Python code.
        catalog_triples = cls.CONSTRUCTION_ENGINE.parse_from_catalog(
            clean, cls.CONSTRUCTIONS
        )
        if catalog_triples:
            return [
                ParsedTriple(
                    subject=triple.subject,
                    predicate=(
                        cls._policy().semantic.taxonomy
                        if triple.predicate == cls.CONSTRUCTION_ENGINE._policy().semantic.taxonomy
                        else triple.predicate
                    ),
                    object_=triple.object_,
                    is_property=triple.is_property,
                    is_negative=triple.is_negative,
                )
                for triple in catalog_triples
            ]

        # 8c. Intransitive capability verbs: "Dolphins swim", "An eagle flies", "It swims"
        m_act = re.match(
            r"^(.*?)\s+(" + "|".join(sorted(cls._policy().action_verbs)) + r")$",
            clean,
            re.IGNORECASE,
        )
        if m_act:
            s = cls.clean_noun(m_act.group(1))
            verb = cls._policy().action_verbs[m_act.group(2).lower()]
            if cls.is_valid_concept(s):
                triples.append(
                    ParsedTriple(
                        subject=s,
                        predicate=cls._policy().semantic.capability,
                        object_=verb,
                    )
                )
                return triples

        # Fallback: split on a novel verb when no construction has been learned yet.
        m_verb = re.match(
            r"^(?:(?:a|an|the)\s+)?(.*?)\s+([a-z_]+)\s+(.*?)$",
            clean,
            re.IGNORECASE,
        )
        if m_verb:
            s = cls.clean_noun(m_verb.group(1))
            p = m_verb.group(2).lower()
            o = cls.clean_noun(m_verb.group(3))
            if cls.is_valid_concept(s) and cls.is_valid_concept(o):
                triples.append(ParsedTriple(subject=s, predicate=p, object_=o))

        return triples

    @classmethod
    def parse_action(cls, text: str) -> tuple[str, dict[str, Any]] | None:
        """Parse procedural actions like 'slice apple into 4 pieces'."""
        clean = text.strip().rstrip(".").strip()
        slice_verbs = "|".join(
            re.escape(verb) for verb in cls._policy().slice_verbs
        )
        m_slice = re.match(
            rf"^(?:{slice_verbs})\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s-]+?)\s+into\s+(\d+)\s+pieces?$",
            clean,
            re.IGNORECASE,
        )
        if m_slice:
            obj = cls.clean_noun(m_slice.group(1))
            count = int(m_slice.group(2))
            return ("SLICE", {"object": obj, "count": count})
        return None

    @staticmethod
    def _parse_math_number(value: str) -> int | float:
        return float(value) if "." in value else int(value)

    @classmethod
    def _parse_math_value(cls, value: str, value_type: str) -> Any:
        if value_type == "text":
            return value.strip()
        if value_type == "numbers":
            raw_numbers = re.split(
                r"\s*(?:,|\band\b)\s*|\s+",
                value.strip(),
                flags=re.IGNORECASE,
            )
            return [
                cls._parse_math_number(number)
                for number in raw_numbers
                if number
            ]
        return cls._parse_math_number(value)

    @staticmethod
    def _normalize_math_value(value: Any, normalizers: tuple[str, ...]) -> Any:
        for normalizer in normalizers:
            if normalizer == "lower" and isinstance(value, str):
                value = value.lower()
            elif normalizer == "float":
                value = float(value)
            elif normalizer == "caret_to_power" and isinstance(value, str):
                value = value.replace("^", "**")
        return value

    @classmethod
    def _parse_configured_math(
        cls, text: str
    ) -> tuple[str, str, dict[str, Any]] | None:
        """Apply configured arithmetic patterns and convert their captures."""
        for pattern in cls._math_patterns().patterns:
            match = re.match(pattern.pattern, text, re.IGNORECASE)
            if match is None:
                continue
            try:
                captures = pattern.captures or tuple(
                    MathCapture(
                        group=group,
                        name=name,
                        value_type=pattern.value_type,
                    )
                    for group, name in zip(
                        pattern.argument_groups, pattern.argument_names
                    )
                )
                arguments: dict[str, Any] = dict(pattern.fixed_arguments)
                for capture in captures:
                    value = cls._parse_math_value(
                        match.group(capture.group), capture.value_type
                    )
                    value = cls._normalize_math_value(value, capture.normalizers)
                    if (
                        capture.sign_group is not None
                        and match.group(capture.sign_group).strip() == "-"
                    ):
                        value = -value
                    if capture.include:
                        arguments[capture.name] = value
            except (IndexError, ValueError):
                continue

            skill = pattern.skill
            if pattern.operator_group is not None:
                operator = match.group(pattern.operator_group).lower()
                skill = pattern.skill_by_operator.get(operator)
            if skill is None:
                continue

            if pattern.reverse_arguments:
                first_name, second_name = pattern.argument_names[:2]
                arguments[first_name], arguments[second_name] = (
                    arguments[second_name],
                    arguments[first_name],
                )
            return (skill, "__math__", arguments)
        return None

    @classmethod
    def parse_question(
        cls, text: str, known_concepts: set[str] | None = None
    ) -> tuple[str, str, Any] | None:
        """Extract (subject, predicate, target) from natural English questions."""
        q = cls.normalize_text(text).rstrip("?").strip()

        # Strip leading conversational greetings or discourse markers before parsing the question
        # e.g. "hii who are you" -> "who are you", "hello can an eagle fly" -> "can an eagle fly"
        greeting_prefixes = "|".join(
            re.escape(prefix) for prefix in cls._policy().greeting_prefixes
        )
        m_greeting_prefix = re.match(
            rf"^(?:{greeting_prefixes})[,\s!]+(.+)$",
            q,
            re.IGNORECASE,
        )
        if m_greeting_prefix:
            sub_q = m_greeting_prefix.group(1).strip()
            sub_parsed = cls.parse_question(sub_q, known_concepts=known_concepts)
            if sub_parsed:
                return sub_parsed

        # Indirect / embedded polar questions: "if Paris is in Europe", "whether Paris is in Europe"
        m_if = re.match(r"^(?:if|whether)\s+(.+)$", q, re.IGNORECASE)
        if m_if and not re.search(r"\b(?:then|it\s+is|they\s+are)\b", q, re.IGNORECASE):
            inner = m_if.group(1).strip()
            stmt_triples = cls.parse_statement(inner)
            if stmt_triples:
                t = stmt_triples[0]
                return (t.subject, t.predicate, t.object_)
            m_inv = re.match(
                r"^(.+?)\s+(is|are|can|has|have|does|do)\s+(.+)$",
                inner,
                re.IGNORECASE,
            )
            if m_inv:
                inv_q = f"{m_inv.group(2)} {m_inv.group(1)} {m_inv.group(3)}"
                sub_parsed = cls.parse_question(inv_q, known_concepts=known_concepts)
                if sub_parsed:
                    return sub_parsed

        for pattern in cls._question_patterns().patterns:
            if re.match(pattern.pattern, q, re.IGNORECASE):
                return (pattern.subject, pattern.predicate, pattern.target)

        configured_math = cls._parse_configured_math(q)
        if configured_math is not None:
            return configured_math

        procedural = cls.CONSTRUCTION_ENGINE.parse_procedural_from_catalog(
            q, cls.CONSTRUCTIONS
        )
        if procedural:
            skill_name, args = procedural
            return (skill_name, "__math__", args)

        # 1. Arithmetic and text calculations are defined by the versioned math catalog.

        # 2. Temporal continuous-time routes are defined by the versioned catalog.
        for pattern in cls._temporal_patterns().patterns:
            match = re.match(pattern.pattern, q, re.IGNORECASE)
            if match:
                subject = cls.clean_noun(match.group(pattern.subject_group))
                duration = (
                    float(match.group(pattern.value_group)),
                    match.group(pattern.unit_group).lower(),
                )
                return (subject, pattern.predicate, duration)

        # 3. Direct semantic question routes are defined by the versioned catalog.
        for pattern in cls._semantic_patterns().patterns:
            match = re.match(pattern.pattern, q, re.IGNORECASE)
            if match is None:
                continue

            def captured(group: int) -> str:
                return match.group(group) or ""

            subject = (
                pattern.subject_literal
                if pattern.subject_literal is not None
                else next(
                    (
                        cls.clean_noun(captured(group))
                        for group in pattern.subject_groups
                        if captured(group).strip()
                    ),
                    "",
                )
            )
            target = (
                pattern.target_literal
                if pattern.target_literal is not None
                else next(
                    (
                        cls.clean_noun(captured(group))
                        for group in pattern.target_groups
                        if captured(group).strip()
                    ),
                    "",
                )
            )
            if pattern.validate_subject and not cls.is_valid_concept(subject):
                continue
            if pattern.validate_target and not cls.is_valid_concept(target):
                continue

            predicate = pattern.predicate
            if predicate is None and pattern.predicate_role is not None:
                predicate = getattr(cls._policy().semantic, pattern.predicate_role)
            if predicate is not None:
                return (subject, predicate, target)

        # Comparative and negative why routes are defined by the semantic catalog.

        # 3. Why is X a Y? "Why is an eagle an animal?"
        m_why_is = re.match(r"^why\s+(?:is|are)\s+(.+)$", q, re.IGNORECASE)
        if m_why_is:
            body = m_why_is.group(1).strip()

            if known_concepts:
                tokens = body.split()
                best_split = None
                for i in range(len(tokens) - 1, 0, -1):
                    cand_s = cls.clean_noun(" ".join(tokens[:i]))
                    cand_o = cls.clean_noun(" ".join(tokens[i:]))
                    if cand_s in known_concepts and cand_o in known_concepts:
                        best_split = (cand_s, cand_o)
                        break
                if best_split:
                    return (best_split[0], "__why_is_a__", best_split[1])

            tokens = body.split()
            if len(tokens) >= 2:
                s = cls.clean_noun(" ".join(tokens[:-1]))
                target = cls.clean_noun(tokens[-1])
                if cls.is_valid_concept(s) and cls.is_valid_concept(target):
                    return (s, "__why_is_a__", target)

        # 8. Capability: "Can an eagle fly?", "Can fish swim?", "Can whales communicate using songs?"
        m_can = re.match(r"^can\s+(.+?)\s+([a-zA-Z0-9_\s\-]+)$", q, re.IGNORECASE)
        if m_can:
            raw_s = m_can.group(1).strip()
            raw_act = m_can.group(2).strip()
            if known_concepts and (" " in raw_s or " " in raw_act):
                all_tokens = (raw_s + " " + raw_act).split()
                best_split = None
                for i in range(1, len(all_tokens)):
                    cs = cls.clean_noun(" ".join(all_tokens[:i]))
                    ca = cls.clean_noun(" ".join(all_tokens[i:]))
                    if cs in known_concepts and ca in known_concepts:
                        best_split = (cs, ca)
                        break
                    elif cs in known_concepts and cls.is_valid_concept(ca):
                        best_split = (cs, ca)
                if best_split:
                    return (
                        best_split[0],
                        cls._policy().semantic.capability,
                        best_split[1],
                    )
            s = cls.clean_noun(raw_s)
            act = cls.clean_noun(raw_act)
            if cls.is_valid_concept(s) and act:
                return (s, cls._policy().semantic.capability, act)
            if not cls.is_valid_concept(s):
                fallback_tokens = raw_act.split()
                if len(fallback_tokens) > 1:
                    s = cls.clean_noun(" ".join(fallback_tokens[:-1]))
                    act = cls.clean_noun(fallback_tokens[-1])
                    if cls.is_valid_concept(s) and act:
                        return (s, cls._policy().semantic.capability, act)

        # 8b. Capability with does/do: "Does a dolphin swim?", "Do birds fly?", "Does it swim?"
        m_does_act = re.match(
            r"^(?:does|do)\s+(.+?)\s+("
            + "|".join(sorted(cls._policy().action_verbs))
            + r")$",
            q,
            re.IGNORECASE,
        )
        if m_does_act:
            s = cls.clean_noun(m_does_act.group(1))
            act = cls._policy().action_verbs[m_does_act.group(2).lower()]
            if cls.is_valid_concept(s):
                return (s, cls._policy().semantic.capability, act)

        # 11. Property boolean query: "Is the apple red?", "Is it red?"
        colors = sorted(cls._policy().colors)
        m_is_color = re.match(
            r"^(?:is|are)\s+(.*?)\s+(" + "|".join(colors) + r")$", q, re.IGNORECASE
        )
        if m_is_color:
            s = cls.clean_noun(m_is_color.group(1))
            val = m_is_color.group(2).lower()
            return (s, cls._policy().semantic.color, val)

        # 13b. General transitive action questions: "Does a falcon hunt rodents?", "Does X verb Y?"
        m_does_trans = re.match(
            r"^(?:does|do)\s+(.+)$", q, re.IGNORECASE
        )
        if m_does_trans:
            body = m_does_trans.group(1).strip()
            tokens = body.split()
            if len(tokens) >= 2:
                best_split = None
                for i in range(1, len(tokens)):
                    cand_s = cls.clean_noun(" ".join(tokens[:i]))
                    cand_v = tokens[i].lower()
                    cand_o = cls.clean_noun(" ".join(tokens[i + 1 :]))
                    if (
                        cand_s
                        and cand_o
                        and cls.is_valid_concept(cand_s)
                        and cls.is_valid_concept(cand_o)
                    ):
                        if known_concepts and cand_s in known_concepts:
                            best_split = (cand_s, cand_v, cand_o)
                            break
                        if not best_split:
                            best_split = (cand_s, cand_v, cand_o)
                if best_split:
                    return best_split

        # 14. Property boolean query: "Is glass transparent?", "Is metal hard?", "Is ice cold?"
        known_props = sorted(cls._policy().known_properties)
        m_is_prop = re.match(
            r"^(?:is|are)\s+(.+?)\s+(" + "|".join(known_props) + r")$",
            q,
            re.IGNORECASE,
        )
        if m_is_prop:
            s = cls.clean_noun(m_is_prop.group(1))
            prop = m_is_prop.group(2).lower()
            if cls.is_valid_concept(s):
                return (s, cls._policy().semantic.property, prop)

        catalog_question = cls.CONSTRUCTION_ENGINE.parse_question_from_catalog(
            q, cls.CONSTRUCTIONS
        )
        if catalog_question:
            return catalog_question

        # 6. Concept definition query: "What is an apple?", "Who is Alice?", "Tell me about a dog"
        m_def = re.match(
            r"^(?:what|who)\s+(?:is|are)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s-]+)$",
            q,
            re.IGNORECASE,
        )
        if not m_def:
            m_def = re.match(
                r"^tell\s+me\s+about\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s-]+)$",
                q,
                re.IGNORECASE,
            )
        if m_def:
            target_noun = cls.clean_noun(m_def.group(1))
            if target_noun and target_noun not in cls._policy().non_concept_words:
                return (target_noun, "__definition__", None)

        # 5. Taxonomic queries: "Is a dog an animal?" / "Is an rtx 4090 hardware?" / "Are dogs animals?"
        m_tax = re.match(r"^(?:is|are)\s+(.+)$", q, re.IGNORECASE)
        if m_tax:
            body = m_tax.group(1).strip()

            # If an explicit article boundary exists in the middle: "is [a dog] [an animal]"
            # Consume optional leading article first so it doesn't match the middle article
            m_two_arts = re.match(
                r"^(?:(?:a|an|the)\s+)?(.+?)\s+(?:a|an|the)\s+(.+)$",
                body,
                re.IGNORECASE,
            )
            if m_two_arts:
                s = cls.clean_noun(m_two_arts.group(1))
                o = cls.clean_noun(m_two_arts.group(2))
                return (s, cls._policy().semantic.taxonomy, o)

            # If known_concepts is provided, find the optimal boundary
            if known_concepts:
                tokens = body.split()
                best_split = None
                for i in range(len(tokens) - 1, 0, -1):
                    cand_s = cls.clean_noun(" ".join(tokens[:i]))
                    cand_o = cls.clean_noun(" ".join(tokens[i:]))
                    if cand_s in known_concepts and cand_o in known_concepts:
                        best_split = (cand_s, cand_o)
                        break
                    elif cand_s in known_concepts or cand_o in known_concepts:
                        if not best_split:
                            best_split = (cand_s, cand_o)
                if best_split:
                    return (
                        best_split[0],
                        cls._policy().semantic.taxonomy,
                        best_split[1],
                    )

            # Fallback: predicate nominal (category) is the last token or tokens
            tokens = body.split()
            if len(tokens) >= 2:
                s = cls.clean_noun(" ".join(tokens[:-1]))
                o = cls.clean_noun(tokens[-1])
                return (s, cls._policy().semantic.taxonomy, o)

        return None

    @classmethod
    def split_compound_question(cls, text: str) -> list[str]:
        """Split a conjoined question into ordered question clauses."""
        clean = text.strip()
        if "?" in clean[:-1]:
            return [part.strip() for part in re.split(r"\?\s*", clean) if part.strip()]
        return [
            part.strip()
            for part in re.split(
                r",?\s+and\s+(?=(?:is|are|was|were|do|does|did|can|could|will|would|has|have|what|who|where|how|why)\b)",
                clean,
                flags=re.IGNORECASE,
            )
            if part.strip()
        ]

    @classmethod
    def parse_compound_question(
        cls,
        text: str,
        *,
        memory: MemoryStore | None = None,
        constructions: list[Construction] | None = None,
        known_concepts: set[str] | None = None,
        last_subject: str | None = None,
        last_object: str | None = None,
    ) -> tuple[list[str], list[ParsedQuery | None]]:
        """Parse each compound question clause once, preserving clause alignment."""
        parts = cls.split_compound_question(text)
        if len(parts) <= 1:
            return [], []

        known = known_concepts or set()
        resolved_parts: list[str] = []
        parsed_queries: list[ParsedQuery | None] = []
        subject = last_subject
        for part in parts:
            resolved = cls.resolve_anaphora(part, subject, last_object)
            parsed: ParsedQuery | None = None
            catalog = (
                constructions
                if constructions is not None
                else memory.list_constructions()
                if memory is not None
                else None
            )
            if catalog is not None:
                parsed = cls.CONSTRUCTION_ENGINE.parse_question_from_catalog_with_procedural(
                    resolved, catalog
                )
            if parsed is None:
                parsed = cls.parse_question(resolved, known_concepts=known)
            resolved_parts.append(part)
            parsed_queries.append(parsed)
            if parsed is not None:
                subject = parsed[0]
        return resolved_parts, parsed_queries


def configured_parser(
    policy: ParserPolicy | None = None,
    *,
    constructions: list[Construction] | None = None,
    unit_conversions: UnitConversionGraph | None = None,
) -> type[SimpleParser]:
    """Build an isolated parser class bound to one runtime policy bundle.

    ``SimpleParser`` keeps its classmethod API for compatibility with existing
    callers.  Runtime components use this factory so a custom policy is held
    on a private subclass instead of mutating the process-wide defaults.
    """
    active_policy = SimpleParser._policy() if policy is None else policy
    construction_engine = type(
        "ConfiguredConstructionEngine",
        (ConstructionEngine,),
        {"_POLICY": active_policy},
    )
    return type(
        "ConfiguredSimpleParser",
        (SimpleParser,),
        {
            "POLICY": active_policy,
            "CONSTRUCTIONS": list(
                SimpleParser.CONSTRUCTIONS if constructions is None else constructions
            ),
            "UNIT_CONVERSIONS": unit_conversions or UnitConversionGraph(),
            "CONSTRUCTION_ENGINE": construction_engine,
            "QUESTION_PATTERNS": active_policy.question_patterns,
            "MATH_PATTERNS": active_policy.math_patterns,
            "TEMPORAL_PATTERNS": active_policy.temporal_patterns,
            "SEMANTIC_PATTERNS": active_policy.semantic_patterns,
        },
    )


class LearningEngine:
    """Orchestrates learning interactions and question answering over persistent memory."""

    def __init__(
        self,
        memory: MemoryStore,
        perception: PerceptionAdapter | None = None,
        runtime_policy: RuntimePolicy | None = None,
    ) -> None:
        self.memory = memory
        explicit_runtime_policy = runtime_policy is not None
        self.runtime_policy = runtime_policy or RuntimePolicy.default()
        self.unit_conversions = UnitConversionGraph(self.runtime_policy.units)
        self.parser = configured_parser(
            self.runtime_policy.parser if explicit_runtime_policy else None,
            constructions=(
                self.runtime_policy.constructions
                if explicit_runtime_policy
                else None
            ),
            unit_conversions=self.unit_conversions,
        )
        self.semantic = (
            self.runtime_policy.semantic
            if explicit_runtime_policy
            else self.parser.POLICY.semantic
        )
        self.perception = perception or DeterministicPerceptionAdapter(
            policy=self.runtime_policy.perception,
            parser=self.parser,
        )
        self.transformation_policy = self.runtime_policy.transformation
        self.memory.ledger.transformation_policy = self.transformation_policy
        self.memory.ledger.dynamics_registry = self.runtime_policy.dynamics
        self.registry = self.memory.ledger.registry
        self.inference = InferenceEngine(
            memory,
            registry=self.registry,
            semantic_policy=self.semantic,
            policy=self.runtime_policy.reasoning,
        )
        self.verifier = DeepSeekInvariantVerifier(memory, registry=self.registry)
        self.last_subject: str | None = None
        self.last_object: str | None = None
        self.dialogue = DialogueContext()
        register_builtin_skills(
            self.memory,
            policy=self.runtime_policy.procedural,
            unit_policy=self.runtime_policy.units,
        )

    def learn(self, text: str) -> LearningResult:
        """Perceive once and delegate candidate decisions to the commit boundary."""
        frame = self.perception.perceive(
            text, memory=self.memory, context=self.dialogue
        )
        return self.commit_frame(frame, source_type=SourceType.USER)

    def commit_frame(
        self,
        frame: CandidateFrame,
        *,
        source_type: SourceType = SourceType.USER,
    ) -> LearningResult:
        """Verify and commit candidates without parsing the source text again."""
        text = frame.source_text
        if frame.actions:
            result = self.memory.ledger.commit_action(
                frame, frame.actions[0], source_type=source_type
            )
            if result.update_type is not UpdateType.NO_OP:
                self.last_subject = frame.actions[0].arguments.get("object")
            return result

        if not frame.claims:
            exp = self.memory.add_experience(input_text=text, extracted_triples=[])
            message = (
                f"Input was recognized as an inquiry or calculation, not a declarative statement: '{text}'"
                if frame.intent == "question"
                else f"Could not extract structured relations from: '{text}'"
            )
            return LearningResult(text, UpdateType.NO_OP, exp.id, message=message)

        extracted: list[dict[str, Any]] = []
        concepts_created: list[str] = []
        relations_created: list[str] = []
        update_types: list[UpdateType] = []
        accepted_claims: list[CandidateClaim] = []
        registry = self.memory.ledger.registry
        self.registry = registry
        self.verifier.registry = registry

        for claim in frame.claims:
            extracted.append(
                {
                    "subject": claim.subject,
                    "predicate": claim.predicate,
                    "object": claim.object,
                    "is_property": claim.is_property,
                    "is_negative": not claim.positive,
                }
            )
            evidence = self.memory.ledger.propose_relation(
                frame,
                claim,
                source_type=source_type,
                source_reference=f"frame:{frame.frame_id}",
                positive=claim.positive,
            )
            subject_before = self.memory.get_concept(claim.subject)
            relation_schema = registry.relation(claim.predicate)
            property_schema = (
                relation_schema
                if relation_schema and relation_schema.attribute_key is not None
                else None
            )
            if property_schema is not None:
                if not claim.positive:
                    verification = VerificationResult(
                        status=VerificationStatus.UNKNOWN,
                        reasons=["Negative property assertions need a separate policy"],
                    )
                else:
                        verification = self.verifier.verify_relation(
                            claim.subject, claim.predicate, claim.object
                        ).to_verification_result()
                decision = self.memory.ledger.commit_property(
                    evidence,
                    verification,
                    key=(
                        property_schema.attribute_key
                        if property_schema.attribute_key
                        else claim.predicate
                    ),
                    value=claim.object,
                    update_policy=(
                        property_schema.attribute_update_policy
                        if property_schema else "append"
                    ),
                    value_format=(
                        property_schema.attribute_value_format
                        if property_schema else "preserve"
                    ),
                )
                if decision.status is CommitStatus.ACCEPTED:
                    if subject_before is None:
                        concepts_created.append(claim.subject)
                    relations_created.append(
                        f"({claim.subject} {claim.predicate}={claim.object})"
                    )
                    update_types.append(UpdateType.PROPERTY_UPDATE)
                    accepted_claims.append(claim)
                continue

            if relation_schema is None:
                verification = VerificationResult(
                    status=VerificationStatus.UNKNOWN,
                    reasons=[f"Unregistered predicate: {claim.predicate}"],
                )
            else:
                verification = self.verifier.verify_relation(
                    claim.subject, claim.predicate, claim.object
                ).to_verification_result()
            object_before = self.memory.get_concept(claim.object)
            existing = (
                self.memory.get_relations(
                    subject_id=subject_before.id,
                    predicate=claim.predicate,
                    object_id=object_before.id,
                )
                if subject_before and object_before
                else []
            )
            decision = self.memory.ledger.commit_relation(
                evidence, verification, positive=claim.positive
            )
            if decision.status is CommitStatus.ACCEPTED:
                if subject_before is None:
                    concepts_created.append(claim.subject)
                if object_before is None:
                    concepts_created.append(claim.object)
                relations_created.append(
                    f"({claim.subject} {claim.predicate} {claim.object})"
                )
                update_types.append(
                    UpdateType.EVIDENCE_ADDITION if existing else UpdateType.NEW_RELATION
                )
                accepted_claims.append(claim)

        exp = self.memory.add_experience(input_text=text, extracted_triples=extracted)
        if accepted_claims:
            first = accepted_claims[0]
            self.last_subject = first.subject
            self.last_object = first.object
            self.dialogue.record_turn(
                speaker="user", text=text, entities=[first.subject, first.object]
            )
            self.dialogue.register_entity(name=first.subject, role="subject")
            self.dialogue.register_entity(name=first.object, role="object")

        if UpdateType.PROPERTY_UPDATE in update_types:
            primary = UpdateType.PROPERTY_UPDATE
        elif UpdateType.NEW_RELATION in update_types:
            primary = UpdateType.NEW_RELATION
        elif UpdateType.EVIDENCE_ADDITION in update_types:
            primary = UpdateType.EVIDENCE_ADDITION
        else:
            primary = UpdateType.NO_OP

        user_name = next(
            (claim for claim in accepted_claims
             if claim.subject == "user" and claim.predicate in ("name", "has_name")),
            None,
        )
        if user_name is not None:
            message = (
                f"Nice to meet you, {user_name.object.capitalize()}! "
                "I have recorded your name in my persistent memory."
            )
        elif primary is UpdateType.NO_OP:
            message = f"Could not verify structured relations from: '{text}'"
        else:
            message = f"Successfully learned: {', '.join(relations_created or concepts_created)}"
        return LearningResult(
            input_text=text, update_type=primary, experience_id=exp.id,
            concepts_created=list(dict.fromkeys(concepts_created)),
            relations_created=relations_created, message=message,
        )

    def ask(
        self,
        question: str,
        *,
        parsed_query: ParsedQuery | None = None,
        parsed_queries: list[ParsedQuery | None] | None = None,
        question_parts: list[str] | None = None,
    ) -> InferenceResult:
        """Answer questions by querying the knowledge graph via InferenceEngine."""
        payload_supplied = (
            parsed_query is not None
            or parsed_queries is not None
            or question_parts is not None
        )
        # 0a. Check for multi-step math word problems
        from little.procedural.math_story import MathStorySolver
        story_solver = MathStorySolver(unit_policy=self.runtime_policy.units)
        if story_solver.is_math_story(question):
            sol = story_solver.solve_story(question)
            if sol and sol.status == "SOLVED":
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.SUPPORTED,
                    answer=sol.result,
                confidence=self.parser.POLICY.confidence.certain,
                    evidence=[f"<math_trace>\n{sol.verbalize()}\n</math_trace>"],
                    trace=sol.steps,
                )

        # 0. Check for compound / conjoined questions: e.g. "Can an eagle fly and does it have wings?"
        q_clean = question.strip()
        sub_questions: list[str] = []
        supplied_queries: list[ParsedQuery | None] = []
        if payload_supplied and parsed_queries and len(parsed_queries) > 1:
            sub_questions = list(question_parts or [])
            supplied_queries = list(parsed_queries)
        elif not payload_supplied and "?" in q_clean[:-1]:
            # Multiple questions with question marks: "Can an eagle fly? Does it have wings?"
            sub_questions = [
                s.strip() for s in re.split(r"\?\s*", q_clean) if s.strip()
            ]
        elif not payload_supplied and re.search(
            r",?\s+and\s+(?=(?:is|are|was|were|do|does|did|can|could|will|would|has|have|what|who|where|how|why)\b)",
            q_clean,
            flags=re.IGNORECASE,
        ):
            sub_questions = [
                s.strip()
                for s in re.split(
                    r",?\s+and\s+(?=(?:is|are|was|were|do|does|did|can|could|will|would|has|have|what|who|where|how|why)\b)",
                    q_clean,
                    flags=re.IGNORECASE,
                )
                if s.strip()
            ]

        if len(sub_questions) > 1:
            sub_results: list[InferenceResult] = []
            for index, sq in enumerate(sub_questions):
                sq_clean = sq.strip()
                parsed_part = (
                    supplied_queries[index]
                    if index < len(supplied_queries)
                    else None
                )
                if payload_supplied:
                    is_q = parsed_part is not None or sq_clean.endswith("?")
                else:
                    is_q = (
                        sq_clean.endswith("?")
                        or self.parser.parse_question(sq_clean) is not None
                        or self.parser.parse_question(f"{sq_clean}?") is not None
                        or sq_clean.lower().startswith(
                            (
                                "what",
                                "who",
                                "where",
                                "how",
                                "why",
                                "which",
                                "is",
                                "are",
                                "do",
                                "does",
                                "did",
                                "can",
                                "could",
                                "will",
                                "would",
                                "has",
                                "have",
                                "calculate",
                            )
                        )
                    )
                if not is_q and self.parser.parse_statement(sq_clean):
                    learn_res = self.learn(sq_clean)
                    sub_results.append(
                        InferenceResult(
                            query=sq_clean,
                            status=BeliefStatus.SUPPORTED,
                            answer=learn_res.message,
                confidence=self.parser.POLICY.confidence.certain,
                            evidence=[
                                f"Learned: {', '.join(learn_res.relations_created or [learn_res.message])}"
                            ],
                            trace=[
                                "Learned declarative statement in compound conversational turn"
                            ],
                        )
                    )
                else:
                    sq_resolved = (
                        sq_clean
                        if payload_supplied
                        else self.parser.resolve_anaphora(
                            sq_clean, self.last_subject, self.last_object
                        )
                    )
                    sq_formatted = (
                        sq_resolved if sq_resolved.endswith("?") else f"{sq_resolved}?"
                    )
                    if payload_supplied:
                        sub_res = self.ask(
                            sq_formatted,
                            parsed_query=parsed_part,
                            parsed_queries=[],
                            question_parts=[],
                        )
                    else:
                        sub_res = self.ask(sq_formatted)
                    sub_results.append(sub_res)

            all_bool = all(isinstance(sr.answer, bool) for sr in sub_results)
            if all_bool:
                all_true = all(sr.answer is True for sr in sub_results)
                any_false = any(sr.answer is False for sr in sub_results)

                if all_true:
                    comb_status = BeliefStatus.SUPPORTED
                    comb_answer: Any = True
                    conn = ", and "
                elif any_false:
                    comb_status = BeliefStatus.REFUTED
                    comb_answer = False
                    conn = ", but "
                else:
                    comb_status = BeliefStatus.UNKNOWN
                    comb_answer = None
                    conn = ", but "
            else:
                all_supported = all(
                    sr.status == BeliefStatus.SUPPORTED for sr in sub_results
                )
                comb_status = (
                    BeliefStatus.SUPPORTED if all_supported else BeliefStatus.UNKNOWN
                )
                comb_answer = [sr.answer for sr in sub_results]
                conn = ", and "

            verbalized_parts = [sr.verbalize().rstrip(".") for sr in sub_results]
            for idx in range(1, len(verbalized_parts)):
                vp = verbalized_parts[idx]
                if vp and vp[0].isupper() and not vp.startswith("I "):
                    verbalized_parts[idx] = vp[0].lower() + vp[1:]
            verbalized_combined = conn.join(verbalized_parts) + "."

            comb_evidence = [f"Compound question: {verbalized_combined}"]
            for i, sr in enumerate(sub_results, 1):
                for e in sr.evidence:
                    comb_evidence.append(f"[Part {i}] {e}")

            comb_trace = [
                f"Evaluating compound question with {len(sub_results)} sub-queries:"
            ]
            for i, sr in enumerate(sub_results, 1):
                comb_trace.append(f"--- Sub-query {i}: '{sub_questions[i - 1]}' ---")
                comb_trace.extend(sr.trace)

            min_conf = min(sr.confidence for sr in sub_results) if sub_results else 0.0

            return InferenceResult(
                query=question,
                status=comb_status,
                answer=comb_answer,
                confidence=min_conf,
                evidence=comb_evidence,
                trace=comb_trace,
            )

        # Resolve anaphora using dialogue context
        resolved_q = self.dialogue.resolve_anaphora_in_text(question)
        if resolved_q == question:
            resolved_q = self.parser.resolve_anaphora(
                question, self.last_subject, self.last_object
            )
        known = {c.name.lower() for c in self.memory.list_concepts()}

        if payload_supplied:
            parsed = parsed_query or next(
                (candidate for candidate in (parsed_queries or []) if candidate),
                None,
            )
        else:
            # 1. Check dynamic Construction Grammar in SQLite FIRST (zero hardcoded regexes)
            catalog = list(self.parser.CONSTRUCTIONS) + self.memory.list_constructions()
            parsed = self.parser.CONSTRUCTION_ENGINE.parse_question_from_catalog_with_procedural(
                resolved_q, catalog
            )
            if not parsed:
                parsed = self.parser.CONSTRUCTION_ENGINE.parse_question_from_catalog_with_procedural(
                    question, catalog
                )
            # 2. Fallback to SimpleParser for specialized queries (e.g. temporal ODEs) if needed
            if not parsed:
                parsed = self.parser.parse_question(resolved_q, known_concepts=known)
            if not parsed:
                parsed = self.parser.parse_question(question, known_concepts=known)

        if not parsed:
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=self.parser.POLICY.confidence.unknown,
                evidence=[],
                trace=[f"Question pattern not recognized: '{question}'"],
            )

        subj, pred, target = parsed

        # Update dialogue focus if valid concept
        if (
            self.parser.is_valid_concept(subj)
            and not pred.startswith("__")
            and pred != "little"
        ) or (pred == "__definition__" and self.parser.is_valid_concept(subj)):
            self.last_subject = subj
            self.dialogue.register_entity(name=subj, role="subject")
        if isinstance(target, str) and self.parser.is_valid_concept(target):
            self.last_object = target
            self.dialogue.register_entity(name=target, role="object")

        # User Identity query handler: "what is my name", "who am i", "do you know my name"
        if pred == "__user_name__":
            user_c = self.memory.get_concept("user")
            name = user_c.attributes.get("name") if user_c else None
            if not name and user_c:
                rels = self.memory.get_relations(subject_id=user_c.id)
                for r in rels:
                    if r.predicate in self.semantic.identity:
                        target_c = self.memory.get_concept(r.object_id)
                        if target_c:
                            name = target_c.name.capitalize()
                            break
            if name:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.SUPPORTED,
                    answer=f"Your name is {name}.",
                confidence=self.parser.POLICY.confidence.certain,
                    evidence=[f"User profile in persistent memory: name={name}"],
                    trace=["Retrieved user identity attribute from concept 'user'"],
                )
            else:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer="I do not know your name yet. What should I call you?",
                confidence=self.parser.POLICY.confidence.unknown,
                    evidence=["User profile has no name recorded"],
                    trace=["Concept 'user' has no stored 'name' attribute"],
                )

        # Bot Name query handler: "what is your name", "tell me your name"
        if pred == "__identity_name__":
            return InferenceResult(
                query=question,
                status=BeliefStatus.SUPPORTED,
                answer="My name is LITTLE (Lightweight In-memory Transitive & Temporal Learning Engine).",
                confidence=self.parser.POLICY.confidence.certain,
                evidence=["Self-identity specification"],
                trace=["Dialogue self-knowledge retrieval"],
            )

        # Chitchat query handlers:
        if pred == "__chitchat_greeting__":
            return InferenceResult(
                query=question,
                status=BeliefStatus.SUPPORTED,
                answer="Hello! I am LITTLE (Lightweight In-memory Transitive & Temporal Learning Engine). How can I help you reason, compute, or learn today?",
                confidence=self.parser.POLICY.confidence.certain,
                evidence=["Conversational greeting acknowledged"],
                trace=["Dialogue interaction"],
            )

        if pred == "__chitchat_thanks__":
            return InferenceResult(
                query=question,
                status=BeliefStatus.SUPPORTED,
                answer="You're very welcome! I am always here to reason, learn new concepts, and compute with zero hallucination.",
                confidence=self.parser.POLICY.confidence.certain,
                evidence=["Conversational pleasantry acknowledged"],
                trace=["Dialogue interaction"],
            )

        if pred == "__chitchat_how_are_you__":
            return InferenceResult(
                query=question,
                status=BeliefStatus.SUPPORTED,
                answer="I am operating at 100% nominal efficiency. Continuous memory is active, ODE states are stable, and my inference engine is ready to reason.",
                confidence=self.parser.POLICY.confidence.certain,
                evidence=["Architecture operational status normal"],
                trace=["LITTLE Cognitive Architecture v0.1.0"],
            )

        if pred == "__chitchat_praise__":
            return InferenceResult(
                query=question,
                status=BeliefStatus.SUPPORTED,
                answer="Thank you! I strive for deterministic accuracy, continuous learning, and multi-hop logical precision.",
                confidence=self.parser.POLICY.confidence.certain,
                evidence=["Positive reinforcement received"],
                trace=["Dialogue interaction"],
            )

        if pred == "__chitchat_fact__":
            relations = self.memory.get_relations()
            candidates = [
                r
                for r in relations
                if r.predicate
                in {
                    self.semantic.taxonomy,
                    self.semantic.location,
                    self.semantic.whole,
                    self.semantic.capability,
                    self.semantic.composition,
                    self.semantic.habitat,
                    self.semantic.part,
                }
                and r.weight_positive > r.weight_negative
            ]
            import random

            selected_rel = (
                random.choice(candidates)
                if candidates
                else (relations[0] if relations else None)
            )
            if selected_rel:
                s_c = self.memory.get_concept(selected_rel.subject_id)
                o_c = self.memory.get_concept(selected_rel.object_id)
                s_n = s_c.name if s_c else selected_rel.subject_id
                o_n = o_c.name if o_c else selected_rel.object_id
                p = selected_rel.predicate
                s_art = "an" if s_n and s_n[0] in "aeiou" else "a"
                o_art = "an" if o_n and o_n[0] in "aeiou" else "a"
                if p == self.semantic.taxonomy:
                    fact_str = f"Here is a verified fact from my memory: {s_art.capitalize()} {s_n} is {o_art} {o_n}."
                elif p == self.semantic.location:
                    fact_str = f"Here is a verified fact from my memory: {s_n.capitalize()} is located in {o_n.capitalize()}."
                elif p == self.semantic.whole:
                    fact_str = f"Here is a verified fact from my memory: {s_n.capitalize()} has {o_art} {o_n}."
                elif p == self.semantic.capability:
                    fact_str = f"Here is a verified fact from my memory: {s_art.capitalize()} {s_n} can {o_n}."
                elif p == self.semantic.composition:
                    fact_str = f"Here is a verified fact from my memory: {s_n.capitalize()} is made of {o_n}."
                elif p == self.semantic.habitat:
                    fact_str = f"Here is a verified fact from my memory: {s_art.capitalize()} {s_n} lives in {o_n}."
                elif p == self.semantic.part:
                    fact_str = f"Here is a verified fact from my memory: {s_art.capitalize()} {s_n} is part of {o_art} {o_n}."
                else:
                    fact_str = (
                        f"Here is a verified fact from my memory: ({s_n} {p} {o_n})."
                    )
                fact_str += " Unlike statistical LLMs, I maintain exact deductive proofs and zero hallucination."
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.SUPPORTED,
                    answer=fact_str,
                confidence=self.parser.POLICY.confidence.certain,
                    evidence=[f"{s_n} {p} {o_n}"],
                    trace=["Retrieved verified relation from semantic memory"],
                )

        # Identity and capability query:
        if pred == "__identity__":
            bio = (
                "I am LITTLE (Lightweight In-memory Transitive & Temporal Learning Engine), "
                "a continuous-learning cognitive architecture. Unlike statistical LLMs that guess tokens, "
                "I operate on an explicit semantic knowledge graph with 0% catastrophic forgetting.\n\n"
                "Here are the core things I can do:\n"
                "1. 🧠 Multi-Hop Deductive Reasoning: I learn taxonomies (e.g. 'A tiger is a feline', 'A feline is an animal') and prove multi-step transitive truths.\n"
                "2. 🛡️ Invariant Checking & Refutation: I detect contradictions and refute falsehoods using disjoint constraints (e.g. 'mammal is disjoint with reptile').\n"
                "3. ⏳ Continuous Physical Dynamics: I solve differential equations (CfC ODEs) to simulate physical decay over time (e.g. apple browning and oxidation over elapsed hours).\n"
                "4. ⚡ Exact Procedural Mathematics: I execute Python algorithms with 100% precision in sub-milliseconds (e.g. 10+10, factorials, primes, fibonacci, palindromes).\n"
                "5. 🔍 Active Epistemic Curiosity: When I encounter an unknown concept, I ask questions to expand my memory rather than hallucinating."
            )
            return InferenceResult(
                query=question,
                status=BeliefStatus.SUPPORTED,
                answer=bio,
                confidence=self.parser.POLICY.confidence.certain,
                evidence=["Self-identity and capabilities specification"],
                trace=["LITTLE Cognitive Architecture v0.1.0"],
            )

        # Concept definition query:
        if pred == "__definition__":
            # Special case: check if subj matches the user's recorded name
            user_c = self.memory.get_concept("user")
            user_name = user_c.attributes.get("name") if user_c else None
            if user_name and subj.lower() == str(user_name).strip().lower():
                capitalized = str(user_name).capitalize()
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.SUPPORTED,
                    answer=f"{capitalized} is you! You introduced yourself as {capitalized}.",
                confidence=self.parser.POLICY.confidence.certain,
                    evidence=[f"User profile identity: name={user_name}"],
                    trace=["Matched query concept with recorded user name"],
                )

            if subj.lower() in ("little", "mivi", "mivi_model"):
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.SUPPORTED,
                    answer="I am LITTLE (Lightweight In-memory Transitive & Temporal Learning Engine), a continuous-learning cognitive architecture.",
                confidence=self.parser.POLICY.confidence.certain,
                    evidence=["Self-identity specification"],
                    trace=["Self-concept definition retrieval"],
                )

            concept = self.memory.get_concept(subj)
            if not concept:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer=None,
                confidence=self.parser.POLICY.confidence.unknown,
                    evidence=[],
                    trace=[f"Concept '{subj}' is unknown in memory."],
                )

            desc_parts: list[str] = []
            out_rels = self.memory.get_relations(subject_id=concept.id)
            is_a_rels = [
                self.memory.get_concept(r.object_id).name
                for r in out_rels
                if r.predicate == self.semantic.taxonomy
                and self.memory.get_concept(r.object_id)
            ]
            part_of_rels = [
                self.memory.get_concept(r.object_id).name
                for r in out_rels
                if r.predicate == self.semantic.part
                and self.memory.get_concept(r.object_id)
            ]
            has_rels = [
                self.memory.get_concept(r.object_id).name
                for r in out_rels
                if r.predicate == self.semantic.whole
                and self.memory.get_concept(r.object_id)
            ]
            disjoint_rels = [
                self.memory.get_concept(r.object_id).name
                for r in out_rels
                if r.predicate == "disjoint_with"
                and self.memory.get_concept(r.object_id)
            ]

            in_rels = self.memory.get_relations(object_id=concept.id)
            parts = [
                self.memory.get_concept(r.subject_id).name
                for r in in_rels
                if r.predicate == self.semantic.part
                and self.memory.get_concept(r.subject_id)
            ]

            if is_a_rels:
                desc_parts.append(f"is a {', '.join(is_a_rels)}")
            if part_of_rels:
                desc_parts.append(f"is part of {', '.join(part_of_rels)}")
            if has_rels:
                desc_parts.append(f"has {', '.join(has_rels)}")
            if parts:
                desc_parts.append(f"consists of parts: {', '.join(parts)}")
            if disjoint_rels:
                desc_parts.append(f"is not a {', '.join(disjoint_rels)}")

            attrs = self.inference.get_inherited_attributes(subj)
            if attrs:
                attr_strs = []
                for k, v in attrs.items():
                    val = ", ".join(v) if isinstance(v, list) else str(v)
                    attr_strs.append(f"{k}={val}")
                desc_parts.append(f"attributes: ({', '.join(attr_strs)})")

            if not desc_parts:
                explanation = f"'{concept.name}' is registered in memory (confidence: {concept.confidence:.1f})."
            else:
                explanation = f"{concept.name.capitalize()} {'; '.join(desc_parts)}."

            return InferenceResult(
                query=question,
                status=BeliefStatus.SUPPORTED,
                answer=explanation,
                confidence=concept.confidence,
                evidence=[f"Retrieved concept definition for '{concept.name}'"],
                trace=[f"Knowledge graph query for concept ID: {concept.id}"],
            )

        if target is None:
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=self.parser.POLICY.confidence.unknown,
                evidence=[],
                trace=[
                    f"Question handler '{pred}' did not provide a graph target."
                ],
            )

        # Procedural Mathematics query:
        if pred == "__math__":
            skill = self.memory.get_skill(subj)
            if not skill:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer=None,
                confidence=self.parser.POLICY.confidence.unknown,
                    evidence=[],
                    trace=[f"Skill '{subj}' is not registered in procedural memory."],
                )
            exec_res = SkillRunner.execute(skill, **target)
            if exec_res.success:
                arg_repr = ", ".join(f"{k}={v}" for k, v in target.items())
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.SUPPORTED,
                    answer=exec_res.result,
                confidence=self.parser.POLICY.confidence.certain,
                    evidence=[
                        f"Evaluated skill {subj}({arg_repr}) = {exec_res.result}"
                    ],
                    trace=exec_res.trace or [],
                )
            return InferenceResult(
                query=question,
                status=BeliefStatus.REFUTED,
                answer=None,
                confidence=self.parser.POLICY.confidence.unknown,
                evidence=[],
                trace=[f"Execution failed: {exec_res.error}"],
            )

        # Continuous Dynamics queries:
        if pred in ("__temporal_color__", "__temporal_condition__"):
            val, unit = target
            conversion = self.unit_conversions.convert(float(val), unit, "s")
            if conversion.status != "SOLVED":
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer=None,
                    confidence=self.parser.POLICY.confidence.unknown,
                    evidence=[],
                    trace=[
                        f"Unable to convert temporal unit '{unit}' to seconds."
                    ],
                )
            dt = float(conversion.result)

            concept = self.memory.get_concept(subj)
            if not concept:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer=None,
                confidence=self.parser.POLICY.confidence.unknown,
                    evidence=[],
                    trace=[f"Concept '{subj}' is unknown."],
                )

            is_exposed = concept.attributes.get("exposed_flesh", False)
            state = ContinuousDynamicsEngine.create_initial_state(
                exposed_to_air=is_exposed
            )
            evolved = ContinuousDynamicsEngine.evolve_state(state, delta_seconds=dt)

            if pred == "__temporal_color__":
                base_color = concept.attributes.get("interior_color", "white")
                perceived_color = ContinuousDynamicsEngine.get_perceived_color(
                    evolved, base_color
                )
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.SUPPORTED,
                    answer=perceived_color,
                confidence=self.parser.POLICY.confidence.derived,
                    evidence=[
                        f"ODE state at dt={dt:.0f}s: oxidation={evolved.oxidation:.2f}, tau={evolved.tau_seconds:.0f}s -> {perceived_color}"
                    ],
                    trace=[
                        f"Continuous CfC evolution: initial oxidation={state.oxidation}, exposed={is_exposed}",
                        f"After {val} {unit} ({dt:.0f} seconds): oxidation={evolved.oxidation:.4f}",
                        f"Perceived color mapped to: '{perceived_color}'",
                    ],
                )
            elif pred == "__temporal_condition__":
                condition = ContinuousDynamicsEngine.get_perceived_condition(evolved)
                is_fresh = condition == "fresh"
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.SUPPORTED if is_fresh else BeliefStatus.REFUTED,
                    answer=is_fresh,
                confidence=self.parser.POLICY.confidence.derived,
                    evidence=[
                        f"ODE state at dt={dt:.0f}s: freshness={evolved.freshness:.2f} -> {condition}"
                    ],
                    trace=[
                        f"Continuous CfC evolution: freshness={evolved.freshness:.4f}",
                        f"Condition descriptor: '{condition}'",
                    ],
                )

        # Property boolean query: "Is the apple red?", "Is it red?"
        if pred == self.semantic.color and target != "?":
            concept = self.memory.get_concept(subj)
            if not concept:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer=None,
                confidence=self.parser.POLICY.confidence.unknown,
                    evidence=[],
                    trace=[f"Concept '{subj}' is unknown."],
                )
            attrs = self.inference.get_inherited_attributes(subj)
            actual_color = attrs.get(self.semantic.color)
            if actual_color:
                actual_list = (
                    [c.lower() for c in actual_color]
                    if isinstance(actual_color, list)
                    else [str(actual_color).lower()]
                )
                target_str = str(target).lower()
                if target_str in actual_list:
                    return InferenceResult(
                        query=question,
                        status=BeliefStatus.SUPPORTED,
                        answer=True,
                confidence=self.parser.POLICY.confidence.derived,
                        evidence=[f"{subj} color is {target_str}"],
                        trace=[f"Matched property 'color'={target_str}"],
                    )
                else:
                    return InferenceResult(
                        query=question,
                        status=BeliefStatus.REFUTED,
                        answer=False,
                confidence=self.parser.POLICY.confidence.derived,
                        evidence=[
                            f"{subj} color is {', '.join(actual_list)}, not {target_str}"
                        ],
                        trace=[
                            f"Property 'color'={actual_list} contradicts target '{target_str}'"
                        ],
                    )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=self.parser.POLICY.confidence.low,
                evidence=[],
                trace=[f"No color recorded for '{subj}'."],
            )

        # Special query: "What color is X?"
        if pred == self.semantic.color and target == "?":
            concept = self.memory.get_concept(subj)
            if not concept:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer=None,
                confidence=self.parser.POLICY.confidence.unknown,
                    evidence=[],
                    trace=[f"Concept '{subj}' is unknown."],
                )
            attrs = self.inference.get_inherited_attributes(subj)
            color = attrs.get(self.semantic.color)
            if color:
                val_str = ", ".join(color) if isinstance(color, list) else str(color)
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.SUPPORTED,
                    answer=f"The color of {subj} is {val_str}.",
                confidence=self.parser.POLICY.confidence.derived,
                    evidence=[f"{subj} color is {val_str}"],
                    trace=[f"Retrieved property 'color' for '{subj}': {val_str}"],
                )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=self.parser.POLICY.confidence.low,
                evidence=[],
                trace=[f"No color property recorded for '{subj}'."],
            )

        # WH-Location query: "Where is X?", "Where is X located?"
        if pred == self.semantic.location and target == "?":
            concept = self.memory.get_concept(subj)
            if not concept:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer=None,
                confidence=self.parser.POLICY.confidence.unknown,
                    evidence=[],
                    trace=[f"Concept '{subj}' is unknown in memory."],
                )
            curr_id = concept.id
            loc_names: list[str] = []
            visited: set[str] = {curr_id}
            while True:
                rels = self.memory.get_relations(
                    subject_id=curr_id, predicate=self.semantic.location
                )
                pos_rels = [
                    r
                    for r in rels
                    if r.weight_positive > r.weight_negative
                    and r.object_id not in visited
                ]
                if not pos_rels:
                    break
                next_id = pos_rels[0].object_id
                visited.add(next_id)
                c = self.memory.get_concept(next_id)
                if c:
                    loc_names.append(c.name.capitalize())
                curr_id = next_id
            if loc_names:
                ans_str = f"{subj.capitalize()} is located in {', '.join(loc_names)}."
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.SUPPORTED,
                    answer=ans_str,
                confidence=self.parser.POLICY.confidence.certain,
                    evidence=[f"{subj} located_in {' -> '.join(loc_names)}"],
                    trace=["Resolved location hierarchy in semantic memory"],
                )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=self.parser.POLICY.confidence.low,
                evidence=[],
                trace=[f"No location recorded for '{subj}'."],
            )

        # WH-Habitat query: "Where does X live?"
        if pred == self.semantic.habitat and target == "?":
            concept = self.memory.get_concept(subj)
            if not concept:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer=None,
                confidence=self.parser.POLICY.confidence.unknown,
                    evidence=[],
                    trace=[f"Concept '{subj}' is unknown."],
                )
            cand_ids = [concept.id] + self.inference.get_ancestor_ids(
                concept.id, predicate=self.semantic.taxonomy
            )
            for cid in cand_ids:
                rels = self.memory.get_relations(
                    subject_id=cid, predicate=self.semantic.habitat
                )
                pos_rels = [r for r in rels if r.weight_positive > r.weight_negative]
                if pos_rels:
                    obj_c = self.memory.get_concept(pos_rels[0].object_id)
                    if obj_c:
                        anc_c = self.memory.get_concept(cid)
                        anc_str = (
                            f" (inherited from {anc_c.name})"
                            if cid != concept.id and anc_c
                            else ""
                        )
                        s_art = "An" if subj[0].lower() in "aeiou" else "A"
                        ans_str = f"{s_art} {subj} lives in {obj_c.name}."
                        return InferenceResult(
                            query=question,
                            status=BeliefStatus.SUPPORTED,
                            answer=ans_str,
                confidence=self.parser.POLICY.confidence.derived,
                            evidence=[f"{concept.name} lives_in {obj_c.name}{anc_str}"],
                            trace=[
                                f"Habitat resolved via knowledge graph: {pos_rels[0].id}"
                            ],
                        )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=self.parser.POLICY.confidence.low,
                evidence=[],
                trace=[f"No habitat recorded for '{subj}'."],
            )

        # WH-Diet query: "What does X eat?"
        if pred == self.semantic.diet and target == "?":
            concept = self.memory.get_concept(subj)
            if not concept:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer=None,
                confidence=self.parser.POLICY.confidence.unknown,
                    evidence=[],
                    trace=[f"Concept '{subj}' is unknown."],
                )
            cand_ids = [concept.id] + self.inference.get_ancestor_ids(
                concept.id, predicate=self.semantic.taxonomy
            )
            for cid in cand_ids:
                rels = self.memory.get_relations(
                    subject_id=cid, predicate=self.semantic.diet
                )
                pos_rels = [r for r in rels if r.weight_positive > r.weight_negative]
                if pos_rels:
                    obj_c = self.memory.get_concept(pos_rels[0].object_id)
                    if obj_c:
                        anc_c = self.memory.get_concept(cid)
                        anc_str = (
                            f", because it is a {anc_c.name}"
                            if cid != concept.id and anc_c and anc_c.name != subj
                            else ""
                        )
                        s_art = "An" if subj[0].lower() in "aeiou" else "A"
                        ans_str = f"{s_art} {subj} eats {obj_c.name}{anc_str}."
                        return InferenceResult(
                            query=question,
                            status=BeliefStatus.SUPPORTED,
                            answer=ans_str,
                confidence=self.parser.POLICY.confidence.derived,
                            evidence=[f"{concept.name} eats {obj_c.name}"],
                            trace=[
                                f"Diet resolved via knowledge graph: {pos_rels[0].id}"
                            ],
                        )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=self.parser.POLICY.confidence.low,
                evidence=[],
                trace=[f"No diet recorded for '{subj}'."],
            )

        # WH-Capability query: "What can X do?"
        if pred == self.semantic.capability and target == "?":
            concept = self.memory.get_concept(subj)
            if not concept:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer=None,
                confidence=self.parser.POLICY.confidence.unknown,
                    evidence=[],
                    trace=[f"Concept '{subj}' is unknown."],
                )
            cand_ids = [concept.id] + self.inference.get_ancestor_ids(
                concept.id, predicate=self.semantic.taxonomy
            )
            actions: list[str] = []
            for cid in cand_ids:
                rels = self.memory.get_relations(
                    subject_id=cid, predicate=self.semantic.capability
                )
                for r in rels:
                    if r.weight_positive > r.weight_negative:
                        obj_c = self.memory.get_concept(r.object_id)
                        if obj_c and obj_c.name not in actions:
                            actions.append(obj_c.name)
            if actions:
                s_art = "An" if subj[0].lower() in "aeiou" else "A"
                ans_str = f"{s_art} {subj} can {', '.join(actions)}."
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.SUPPORTED,
                    answer=ans_str,
                confidence=self.parser.POLICY.confidence.derived,
                    evidence=[f"{subj} can {', '.join(actions)}"],
                    trace=["Resolved capabilities from concept and ancestors"],
                )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=self.parser.POLICY.confidence.low,
                evidence=[],
                trace=[f"No capabilities recorded for '{subj}'."],
            )

        # WH-Material query: "What is X made of?"
        if pred == self.semantic.composition and target == "?":
            concept = self.memory.get_concept(subj)
            if not concept:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer=None,
                confidence=self.parser.POLICY.confidence.unknown,
                    evidence=[],
                    trace=[f"Concept '{subj}' is unknown."],
                )
            rels = self.memory.get_relations(
                subject_id=concept.id, predicate=self.semantic.composition
            )
            pos_rels = [r for r in rels if r.weight_positive > r.weight_negative]
            if pos_rels:
                obj_c = self.memory.get_concept(pos_rels[0].object_id)
                if obj_c:
                    ans_str = f"{subj.capitalize()} is made of {obj_c.name}."
                    return InferenceResult(
                        query=question,
                        status=BeliefStatus.SUPPORTED,
                        answer=ans_str,
                confidence=self.parser.POLICY.confidence.derived,
                        evidence=[f"{subj} made_of {obj_c.name}"],
                        trace=["Resolved composition from memory"],
                    )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=self.parser.POLICY.confidence.low,
                evidence=[],
                trace=[f"No composition recorded for '{subj}'."],
            )

        # WH-Features/Parts query: "What does X have?", "What parts does X have?"
        if pred == self.semantic.whole and target == "?":
            concept = self.memory.get_concept(subj)
            if not concept:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer=None,
                confidence=self.parser.POLICY.confidence.unknown,
                    evidence=[],
                    trace=[f"Concept '{subj}' is unknown."],
                )
            cand_ids = [concept.id] + self.inference.get_ancestor_ids(
                concept.id, predicate=self.semantic.taxonomy
            )
            has_items: list[str] = []
            for cid in cand_ids:
                for r in self.memory.get_relations(
                    subject_id=cid, predicate=self.semantic.whole
                ):
                    if r.weight_positive > r.weight_negative:
                        obj_c = self.memory.get_concept(r.object_id)
                        if obj_c and obj_c.name not in has_items:
                            has_items.append(obj_c.name)
                for r in self.memory.get_relations(
                    object_id=cid, predicate=self.semantic.part
                ):
                    if r.weight_positive > r.weight_negative:
                        sc = self.memory.get_concept(r.subject_id)
                        if sc and sc.name not in has_items:
                            has_items.append(sc.name)
            if has_items:
                items_fmt = []
                for it in has_items:
                    art = "an" if it[0].lower() in "aeiou" else "a"
                    items_fmt.append(f"{art} {it}")
                ans_str = f"{subj.capitalize()} has {', '.join(items_fmt)}."
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.SUPPORTED,
                    answer=ans_str,
                confidence=self.parser.POLICY.confidence.derived,
                    evidence=[f"{subj} has {', '.join(has_items)}"],
                    trace=[
                        "Resolved features and parts via semantic memory and duality"
                    ],
                )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=self.parser.POLICY.confidence.low,
                evidence=[],
                trace=[f"No parts or features recorded for '{subj}'."],
            )

        # WH-Purpose query: "What is X used for?"
        if pred == self.semantic.purpose and target == "?":
            concept = self.memory.get_concept(subj)
            if not concept:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer=None,
                confidence=self.parser.POLICY.confidence.unknown,
                    evidence=[],
                    trace=[f"Concept '{subj}' is unknown."],
                )
            cand_ids = [concept.id] + self.inference.get_ancestor_ids(
                concept.id, predicate=self.semantic.taxonomy
            )
            purposes: list[str] = []
            for cid in cand_ids:
                rels = self.memory.get_relations(
                    subject_id=cid, predicate=self.semantic.purpose
                )
                for r in rels:
                    if r.weight_positive > r.weight_negative:
                        obj_c = self.memory.get_concept(r.object_id)
                        if obj_c and obj_c.name not in purposes:
                            purposes.append(obj_c.name)
            if purposes:
                s_art = "An" if subj[0].lower() in "aeiou" else "A"
                ans_str = f"{s_art} {subj} is used for {', '.join(purposes)}."
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.SUPPORTED,
                    answer=ans_str,
                confidence=self.parser.POLICY.confidence.derived,
                    evidence=[f"{subj} used_for {', '.join(purposes)}"],
                    trace=[
                        "Resolved usage and purpose via semantic memory and inheritance"
                    ],
                )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=self.parser.POLICY.confidence.low,
                evidence=[],
                trace=[f"No purpose or usage recorded for '{subj}'."],
            )

        # WH-Causality query: "What does X cause?"
        if pred == self.semantic.causality and target == "?":
            concept = self.memory.get_concept(subj)
            if not concept:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer=None,
                confidence=self.parser.POLICY.confidence.unknown,
                    evidence=[],
                    trace=[f"Concept '{subj}' is unknown."],
                )
            cand_ids = [concept.id] + self.inference.get_ancestor_ids(
                concept.id, predicate=self.semantic.taxonomy
            )
            effects: list[str] = []
            for cid in cand_ids:
                rels = self.memory.get_relations(
                    subject_id=cid, predicate=self.semantic.causality
                )
                for r in rels:
                    if r.weight_positive > r.weight_negative:
                        obj_c = self.memory.get_concept(r.object_id)
                        if obj_c and obj_c.name not in effects:
                            effects.append(obj_c.name)
            if effects:
                ans_str = f"{subj.capitalize()} causes {', '.join(effects)}."
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.SUPPORTED,
                    answer=ans_str,
                confidence=self.parser.POLICY.confidence.derived,
                    evidence=[f"{subj} causes {', '.join(effects)}"],
                    trace=["Resolved causality from semantic memory"],
                )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=self.parser.POLICY.confidence.low,
                evidence=[],
                trace=[f"No effects recorded for '{subj}'."],
            )

        # WH-Reverse Causality: "What causes X?"
        if (
            subj == "?"
            and pred == self.semantic.causality
            and isinstance(target, str)
        ):
            concept = self.memory.get_concept(target)
            if not concept:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer=None,
                confidence=self.parser.POLICY.confidence.unknown,
                    evidence=[],
                    trace=[f"Concept '{target}' is unknown."],
                )
            rels = self.memory.get_relations(
                object_id=concept.id, predicate=self.semantic.causality
            )
            causes: list[str] = []
            for r in rels:
                if r.weight_positive > r.weight_negative:
                    sc = self.memory.get_concept(r.subject_id)
                    if sc and sc.name not in causes:
                        causes.append(sc.name)
            if causes:
                ans_str = (
                    f"{', '.join(c.capitalize() for c in causes)} causes {target}."
                )
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.SUPPORTED,
                    answer=ans_str,
                confidence=self.parser.POLICY.confidence.derived,
                    evidence=[f"{', '.join(causes)} causes {target}"],
                    trace=["Resolved causes from semantic memory"],
                )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=self.parser.POLICY.confidence.low,
                evidence=[],
                trace=[f"No causes recorded for '{target}'."],
            )

        # WH-Properties query: "What properties does X have?"
        if pred == self.semantic.property and target == "?":
            concept = self.memory.get_concept(subj)
            if not concept:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer=None,
                confidence=self.parser.POLICY.confidence.unknown,
                    evidence=[],
                    trace=[f"Concept '{subj}' is unknown."],
                )
            cand_ids = [concept.id] + self.inference.get_ancestor_ids(
                concept.id, predicate=self.semantic.taxonomy
            )
            props: list[str] = []
            for cid in cand_ids:
                rels = self.memory.get_relations(
                    subject_id=cid, predicate=self.semantic.property
                )
                for r in rels:
                    if r.weight_positive > r.weight_negative:
                        obj_c = self.memory.get_concept(r.object_id)
                        if obj_c and obj_c.name not in props:
                            props.append(obj_c.name)
            if props:
                ans_str = f"{subj.capitalize()} is {', '.join(props)}."
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.SUPPORTED,
                    answer=ans_str,
                confidence=self.parser.POLICY.confidence.derived,
                    evidence=[f"{subj} has_property {', '.join(props)}"],
                    trace=["Resolved properties from semantic memory"],
                )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=self.parser.POLICY.confidence.low,
                evidence=[],
                trace=[f"No properties recorded for '{subj}'."],
            )

        # Why-Questions:
        # 1. Why is X a Y?
        if pred == "__why_is_a__":
            res = self.inference.infer(
                subject=subj,
                predicate=self.semantic.taxonomy,
                target=target,
            )
            if res.status == BeliefStatus.SUPPORTED:
                trans_step = next(
                    (t for t in res.trace if "Transitive path discovered" in t), None
                )
                if trans_step:
                    path_str = trans_step.split(": ")[1].split(" (")[0]
                    nodes = [n.strip() for n in path_str.split("->")]
                    chain_items = []
                    for n in nodes[1:]:
                        art = "an" if n and n[0].lower() in "aeiou" else "a"
                        chain_items.append(f"{art} {n}")
                    chain_desc = ", which is ".join(chain_items)
                    s_art = "an" if nodes[0] and nodes[0][0].lower() in "aeiou" else "a"
                    proof_ans = f"Because {s_art} {nodes[0]} is {chain_desc}."
                elif res.evidence:
                    proof_ans = f"Because {res.evidence[0]}."
                else:
                    proof_ans = f"Because {subj} is explicitly verified as a {target} in memory."
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.SUPPORTED,
                    answer=proof_ans,
                    confidence=res.confidence,
                    evidence=res.evidence,
                    trace=res.trace,
                )
            elif res.status == BeliefStatus.REFUTED:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.REFUTED,
                    answer=f"Actually, {subj} is not a {target}. {res.verbalize()}",
                    confidence=res.confidence,
                    evidence=res.evidence,
                    trace=res.trace,
                )
            else:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer=f"I do not know whether {subj} is a {target} yet, so I cannot deduce a proof.",
                confidence=self.parser.POLICY.confidence.unknown,
                    evidence=[],
                    trace=res.trace,
                )

        # 2. Why is X located in Y?
        if pred == "__why_located_in__":
            res = self.inference.infer(
                subject=subj, predicate=self.semantic.location, target=target
            )
            if res.status == BeliefStatus.SUPPORTED:
                trans_step = next(
                    (t for t in res.trace if "Transitive path discovered" in t), None
                )
                if trans_step:
                    path_str = trans_step.split(": ")[1].split(" (")[0]
                    nodes = [n.strip() for n in path_str.split("->")]
                    chain_desc = ", which is located in ".join(
                        n.capitalize() for n in nodes[1:]
                    )
                    proof_ans = (
                        f"Because {nodes[0].capitalize()} is located in {chain_desc}."
                    )
                elif res.evidence:
                    proof_ans = f"Because {res.evidence[0]}."
                else:
                    proof_ans = f"Because {subj.capitalize()} is located in {target.capitalize()}."
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.SUPPORTED,
                    answer=proof_ans,
                    confidence=res.confidence,
                    evidence=res.evidence,
                    trace=res.trace,
                )
            return res

        # 3. Why is X not a Y?
        if pred == "__why_not__":
            res = self.inference.infer(
                subject=subj,
                predicate=self.semantic.taxonomy,
                target=target,
            )
            if res.status == BeliefStatus.REFUTED:
                disj_step = next(
                    (t for t in res.trace if "Disjoint constraint triggered" in t), None
                )
                if disj_step:
                    reason = disj_step.split(": ")[1]
                    proof_ans = f"Because {reason}."
                else:
                    proof_ans = (
                        f"Because {subj} and {target} are mutually exclusive in memory."
                    )
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.SUPPORTED,
                    answer=proof_ans,
                    confidence=res.confidence,
                    evidence=res.evidence,
                    trace=res.trace,
                )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=f"I do not have evidence that {subj} is incompatible with {target}.",
                confidence=self.parser.POLICY.confidence.unknown,
                evidence=[],
                trace=res.trace,
            )

        # Standard relational query
        return self.inference.infer(subject=subj, predicate=pred, target=target)
