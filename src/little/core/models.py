"""Core domain models and data structures for the LITTLE cognitive architecture.

Defines the fundamental building blocks: Concepts, Entities, Relations,
Experiences, Beliefs, and Inference/Learning results.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from little.core.response_policy import ResponsePolicy
from little.core.semantic_predicate_policy import SemanticPredicatePolicy


def _default_concept_confidence() -> float:
    """Load the configured confidence for concepts created without evidence."""
    from little.memory.memory_policy import MemoryPolicy

    return MemoryPolicy.default().concept_confidence


def _default_construction_confidence() -> float:
    """Load the configured confidence for uncalibrated constructions."""
    from little.core.construction_policy import ConstructionPolicy

    return ConstructionPolicy.default().default_confidence


class ConceptStatus(str, Enum):
    ACTIVE = "ACTIVE"
    CANDIDATE = "CANDIDATE"
    HYPOTHESIS = "HYPOTHESIS"
    ARCHIVED = "ARCHIVED"


class BeliefStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    UNKNOWN = "UNKNOWN"
    AMBIGUOUS = "AMBIGUOUS"
    CONTRADICTED = "CONTRADICTED"


class UpdateType(str, Enum):
    NEW_CONCEPT = "NEW_CONCEPT"
    NEW_ENTITY = "NEW_ENTITY"
    NEW_RELATION = "NEW_RELATION"
    PROPERTY_UPDATE = "PROPERTY_UPDATE"
    EVIDENCE_ADDITION = "EVIDENCE_ADDITION"
    DUPLICATE = "DUPLICATE"
    CONTRADICTION = "CONTRADICTION"
    REVISION = "REVISION"
    NO_OP = "NO_OP"


def current_iso_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Concept:
    """A semantic abstraction representing a class of objects, properties, or ideas."""

    id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    category: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence: float = field(default_factory=_default_concept_confidence)
    status: ConceptStatus = ConceptStatus.ACTIVE
    created_at: str = field(default_factory=current_iso_timestamp)

    @classmethod
    def create(
        cls,
        name: str,
        category: str | None = None,
        aliases: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
        confidence: float | None = None,
    ) -> Concept:
        clean_name = name.strip().lower()
        return cls(
            id=generate_id("concept"),
            name=clean_name,
            aliases=[a.strip().lower() for a in (aliases or [])],
            category=category.strip().lower() if category else None,
            attributes=attributes or {},
            confidence=(
                _default_concept_confidence()
                if confidence is None
                else confidence
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class Entity:
    """A specific, grounded instance of a concept (e.g. 'my_apple_01', 'Alice')."""

    id: str
    name: str
    concept_id: str
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=current_iso_timestamp)

    @classmethod
    def create(
        cls, name: str, concept_id: str, properties: dict[str, Any] | None = None
    ) -> Entity:
        return cls(
            id=generate_id("entity"),
            name=name.strip(),
            concept_id=concept_id,
            properties=properties or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Relation:
    """A typed edge connecting two concepts or entities in the semantic world model.

    Supports positive and negative evidence accumulation based on NARS
    (Non-Axiomatic Reasoning) principles.
    """

    id: str
    subject_id: str
    predicate: str  # e.g., 'is_a', 'has_part', 'color', 'can_be', 'disjoint_with'
    object_id: str
    weight_positive: int = 1
    weight_negative: int = 0
    confidence: float = 0.5
    source_experience_id: str | None = None
    created_at: str = field(default_factory=current_iso_timestamp)

    def __post_init__(self) -> None:
        self.recompute_confidence()

    def recompute_confidence(self) -> float:
        total = self.weight_positive + self.weight_negative
        if total == 0:
            self.confidence = 0.0
        else:
            # Evidence-grounded confidence: c = w / (w + 1)
            self.confidence = round(self.weight_positive / (total + 1.0), 4)
        return self.confidence

    def add_evidence(self, positive: bool = True) -> None:
        if positive:
            self.weight_positive += 1
        else:
            self.weight_negative += 1
        self.recompute_confidence()

    @classmethod
    def create(
        cls,
        subject_id: str,
        predicate: str,
        object_id: str,
        source_experience_id: str | None = None,
        positive: bool = True,
    ) -> Relation:
        rel = cls(
            id=generate_id("rel"),
            subject_id=subject_id,
            predicate=predicate.strip().lower(),
            object_id=object_id,
            weight_positive=1 if positive else 0,
            weight_negative=0 if positive else 1,
            source_experience_id=source_experience_id,
        )
        rel.recompute_confidence()
        return rel

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Experience:
    """Episodic record of a learning event or interaction."""

    id: str
    input_text: str
    extracted_triples: list[dict[str, Any]] = field(default_factory=list)
    source: str = "user"
    timestamp: str = field(default_factory=current_iso_timestamp)

    @classmethod
    def create(
        cls,
        input_text: str,
        extracted_triples: list[dict[str, Any]] | None = None,
        source: str = "user",
    ) -> Experience:
        return cls(
            id=generate_id("exp"),
            input_text=input_text.strip(),
            extracted_triples=extracted_triples or [],
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Belief:
    """An asserted or derived proposition accompanied by confidence and evidence trace."""

    proposition: str
    status: BeliefStatus
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class InferenceResult:
    """Result returned by the reasoning engine in response to a query."""

    query: str
    status: BeliefStatus
    answer: bool | str | None
    confidence: float
    evidence: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)

    @property
    def is_supported(self) -> bool:
        return self.status == BeliefStatus.SUPPORTED

    @property
    def is_unknown(self) -> bool:
        return self.status == BeliefStatus.UNKNOWN

    def verbalize(self, policy: ResponsePolicy | None = None) -> str:
        """Convert formal symbolic inference into fluent, natural English with reasoning context."""
        semantic = SemanticPredicatePolicy.default()
        response = policy or ResponsePolicy.default()
        # 0. Compound question verbalization
        if self.evidence and any(
            e.startswith("Compound question:") for e in self.evidence
        ):
            cq_e = next(e for e in self.evidence if e.startswith("Compound question:"))
            return cq_e.replace("Compound question: ", "").strip()

        # 1. Identity, Definition, or extensive descriptive answers
        if isinstance(self.answer, str) and len(self.answer) > 25 and not self.evidence:
            return self.answer

        # Concept definitions (e.g. "What is an apple?")
        if self.evidence and any(
            "Retrieved concept definition" in e for e in self.evidence
        ):
            return str(self.answer)

        # 2. Procedural mathematics and algorithmic execution
        if self.evidence and any("Evaluated skill" in e for e in self.evidence):
            eval_e = next(e for e in self.evidence if "Evaluated skill" in e)
            m_skill = re.search(
                r"Evaluated skill ([A-Z_]+)\((.*)\) = ", eval_e
            ) or re.search(r"Evaluated skill ([A-Z_]+)\((.*?)\)", eval_e)
            if m_skill:
                s_name = m_skill.group(1)
                args_str = m_skill.group(2)
                args_map = {}
                for part in args_str.split(", "):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        args_map[k.strip()] = v.strip()

                if s_name == "IS_PRIME":
                    n = args_map.get("n", "")
                    if self.answer is True:
                        return response.math("is_prime_true", n=n, answer=self.answer)
                    return response.math("is_prime_false", n=n, answer=self.answer)
                elif s_name == "PALINDROME":
                    txt = args_map.get("text", "")
                    if self.answer is True:
                        return response.math(
                            "palindrome_true", text=txt, answer=self.answer
                        )
                    return response.math(
                        "palindrome_false", text=txt, answer=self.answer
                    )
                elif s_name == "NTH_PRIME":
                    n = args_map.get("n", "")
                    formatted = (
                        f"{self.answer:,}"
                        if isinstance(self.answer, int)
                        else f"{self.answer}"
                    )
                    return response.math("nth_prime", n=n, answer=formatted)
                elif s_name == "FACTORIAL":
                    n = args_map.get("n", "")
                    formatted = (
                        f"{self.answer:,}"
                        if isinstance(self.answer, int)
                        else f"{self.answer}"
                    )
                    return response.math("factorial", n=n, answer=formatted)
                elif s_name == "FIBONACCI":
                    n = args_map.get("n", "")
                    formatted = (
                        f"{self.answer:,}"
                        if isinstance(self.answer, int)
                        else f"{self.answer}"
                    )
                    return response.math("fibonacci", n=n, answer=formatted)
                elif s_name == "REVERSE_STRING":
                    txt = args_map.get("text", "")
                    return response.math("reverse_string", text=txt, answer=self.answer)
                elif s_name == "SQRT":
                    n = args_map.get("n", "")
                    return response.math("sqrt", n=n, answer=self.answer)
                elif s_name == "GCD":
                    a = args_map.get("a", "")
                    b = args_map.get("b", "")
                    return response.math(
                        "gcd", a=a, b=b, answer=self.answer
                    )
                elif s_name == "LCM":
                    a = args_map.get("a", "")
                    b = args_map.get("b", "")
                    return response.math("lcm", a=a, b=b, answer=self.answer)
                elif s_name == "PERCENT":
                    p = args_map.get("percent", "")
                    t = args_map.get("total", "")
                    return response.math(
                        "percent", percent=p, total=t, answer=self.answer
                    )
                elif s_name == "AVERAGE":
                    return response.math("average", answer=self.answer)
                elif s_name == "SOLVE_LINEAR":
                    return response.math("solve_linear", answer=self.answer)
                elif s_name == "SOLVE_QUADRATIC":
                    if isinstance(self.answer, list):
                        if len(self.answer) == 1:
                            return response.math(
                                "solve_quadratic_single", answer=self.answer[0]
                            )
                        return response.math(
                            "solve_quadratic_multiple",
                            answer=", ".join(str(s) for s in self.answer),
                        )
                    return response.math("solve_quadratic_value", answer=self.answer)
                elif s_name == "CALCULUS_DERIVATIVE":
                    expr = args_map.get("expression", "")
                    return response.math(
                        "calculus_derivative", expression=expr, answer=self.answer
                    )
                elif s_name == "SIMPLIFY_EXPR":
                    expr = args_map.get("expression", "")
                    return response.math(
                        "simplify_expression", expression=expr, answer=self.answer
                    )
                elif s_name == "UNIT_CONVERT":
                    val = args_map.get("value", "")
                    u_from = args_map.get("from_unit", "")
                    u_to = args_map.get("to_unit", "")
                    return response.math(
                        "unit_convert",
                        value=val,
                        from_unit=u_from,
                        answer=self.answer,
                        to_unit=u_to,
                    )
                elif s_name == "EVAL_EXPR":
                    expr = args_map.get("expression", "")
                    return response.math(
                        "evaluate_expression", expression=expr, answer=self.answer
                    )

            if isinstance(self.answer, (int, float)):
                formatted = (
                    f"{self.answer:,}"
                    if isinstance(self.answer, int)
                    else f"{self.answer}"
                )
                return response.generic("numeric_result", answer=formatted)
            return response.generic("value_result", answer=self.answer)

        # 3. Continuous-Time Physical Dynamics (CfC / Liquid ODEs)
        if any("Continuous CfC evolution" in t for t in self.trace):
            if self.answer in ("brown", "light brown", "white"):
                return (
                    f"After the elapsed time, it turns {self.answer}. "
                    "Its exposed surface undergoes continuous enzymatic oxidation calculated via differential equations."
                )
            elif self.answer is False and any("freshness" in t for t in self.trace):
                return "No, it is no longer fresh. Continuous-time decay indicates it has oxidized and decomposed over time."
            elif self.answer is True and any("freshness" in t for t in self.trace):
                return "Yes, it is still fresh at this time."

        # 4. Property queries (e.g. color)
        if self.evidence and any(
            "color is" in e or "has color" in e for e in self.evidence
        ):
            col_e = next(
                e for e in self.evidence if "color is" in e or "has color" in e
            )
            if " color is " in col_e:
                s, c = col_e.split(" color is ", 1)
                return f"Based on my knowledge base, the {s} is {c}."
            elif " has color: " in col_e:
                s, c = col_e.split(" has color: ", 1)
                return f"Based on my knowledge base, the {s} is {c}."

        # 5. Transitive Deductive Reasoning & Inherited Axioms (Supported True)
        if self.is_supported and self.answer is True:
            trans_step = next(
                (t for t in self.trace if "Transitive path discovered" in t), None
            )
            if trans_step:
                try:
                    path_str = trans_step.split(": ")[1].split(" (")[0]
                    nodes = [n.strip() for n in path_str.split("->")]
                    pred_used = semantic.taxonomy
                    for t in self.trace:
                        if "Evaluating transitive path via '" in t:
                            pred_used = t.split("via '")[1].split("'")[0]
                            break

                    if pred_used == semantic.location:
                        chain_desc = ", which is located in ".join(
                            n.capitalize() for n in nodes[1:]
                        )
                        return (
                            f"Yes, {nodes[0].capitalize()} is located in {nodes[-1].capitalize()}. "
                            f"I deduced this across {len(nodes) - 1} logical steps: "
                            f"{nodes[0].capitalize()} is located in {chain_desc}."
                        )

                    if pred_used == "larger_than":
                        chain_desc = ", which is larger than ".join(
                            n.capitalize() for n in nodes[1:]
                        )
                        return (
                            f"Yes, {nodes[0].capitalize()} is larger than {nodes[-1].capitalize()}. "
                            f"Deductive chain: {nodes[0].capitalize()} is larger than {chain_desc}."
                        )

                    chain_items = []
                    for n in nodes[1:]:
                        art = "an" if n and n[0].lower() in "aeiou" else "a"
                        chain_items.append(f"{art} {n}")
                    chain_desc = ", which is ".join(chain_items)
                    s_art = "an" if nodes[0] and nodes[0][0].lower() in "aeiou" else "a"
                    e_art = (
                        "an" if nodes[-1] and nodes[-1][0].lower() in "aeiou" else "a"
                    )
                    return (
                        f"Yes, {s_art} {nodes[0]} is {e_art} {nodes[-1]}. "
                        f"I deduced this across {len(nodes) - 1} logical steps: "
                        f"{s_art} {nodes[0]} is {chain_desc}."
                    )
                except (IndexError, ValueError):
                    pass

            q_clean = self.query.strip("()").replace(",", " ") if self.query else ""
            q_parts = q_clean.split()
            q_subj = q_parts[0] if len(q_parts) >= 1 else ""
            q_pred = q_parts[1] if len(q_parts) >= 2 else ""
            q_obj = " ".join(q_parts[2:]) if len(q_parts) >= 3 else ""

            # Part-whole duality & inheritance verbalization
            if q_pred == semantic.whole:
                s_art = "an" if q_subj and q_subj[0].lower() in "aeiou" else "a"
                o_art = "an" if q_obj and q_obj[0].lower() in "aeiou" else "a"
                if len(self.evidence) >= 2:
                    p1 = self.evidence[0].split()
                    p2 = self.evidence[1].split()
                    anc = p1[2] if len(p1) >= 3 else ""
                    anc_art = "an" if anc and anc[0].lower() in "aeiou" else "a"
                    if len(p2) >= 3 and p2[1] == semantic.part:
                        return (
                            f"Yes, {s_art} {q_subj} has {o_art} {q_obj}. "
                            f"I deduced this because {s_art} {q_subj} is {anc_art} {anc}, "
                            f"and {o_art} {q_obj} is part of {anc_art} {anc}."
                        )
                    elif len(p2) >= 3 and p2[1] == semantic.whole:
                        return (
                            f"Yes, {s_art} {q_subj} has {o_art} {q_obj}. "
                            f"I deduced this because {s_art} {q_subj} is {anc_art} {anc}, "
                            f"which has {o_art} {q_obj}."
                        )
                elif len(self.evidence) == 1:
                    p1 = self.evidence[0].split()
                    if len(p1) >= 3 and p1[1] == semantic.part:
                        return f"Yes, {s_art} {q_subj} has {o_art} {q_obj}, because {o_art} {q_obj} is part of {s_art} {q_subj}."
                    elif len(p1) >= 3 and p1[1] == semantic.whole:
                        subj_str = (
                            q_subj.capitalize()
                            if q_subj.lower()
                            in ("earth", "mars", "jupiter", "saturn", "sun", "moon")
                            else f"{s_art} {q_subj}"
                        )
                        return f"Yes, {subj_str} has {o_art} {q_obj}."

            if q_pred == semantic.part:
                s_art = "an" if q_subj and q_subj[0].lower() in "aeiou" else "a"
                o_art = "an" if q_obj and q_obj[0].lower() in "aeiou" else "a"
                if len(self.evidence) >= 2:
                    isa_ev = next(
                        (e for e in self.evidence if f" {semantic.taxonomy} " in e),
                        None,
                    )
                    other_ev = next(
                        (
                            e
                            for e in self.evidence
                            if f" {semantic.taxonomy} " not in e
                        ),
                        None,
                    )
                    if isa_ev and other_ev:
                        isa_parts = isa_ev.split()
                        other_parts = other_ev.split()
                        anc = isa_parts[2] if len(isa_parts) >= 3 else ""
                        anc_art = "an" if anc and anc[0].lower() in "aeiou" else "a"
                        if len(other_parts) >= 3 and other_parts[1] == semantic.part:
                            return (
                                f"Yes, {s_art} {q_subj} is part of {o_art} {q_obj}. "
                                f"I deduced this because {s_art} {q_subj} is part of {anc_art} {anc}, "
                                f"and {o_art} {q_obj} is {anc_art} {anc}."
                            )
                        elif len(other_parts) >= 3 and other_parts[1] == semantic.whole:
                            return (
                                f"Yes, {s_art} {q_subj} is part of {o_art} {q_obj}. "
                                f"I deduced this because {anc_art.capitalize()} {anc} has {s_art} {q_subj}, "
                                f"and {o_art} {q_obj} is {anc_art} {anc}."
                            )
                elif len(self.evidence) == 1:
                    p1 = self.evidence[0].split()
                    if len(p1) >= 3 and p1[1] == semantic.whole:
                        return f"Yes, {s_art} {q_subj} is part of {o_art} {q_obj}, because {o_art} {q_obj} has {s_art} {q_subj}."
                    elif len(p1) >= 3 and p1[1] == semantic.part:
                        return f"Yes, {s_art} {q_subj} is part of {o_art} {q_obj}."

            # Inherited relation verbalization (e.g. eagle is_a bird -> bird can fly)
            inh_step = next(
                (
                    t
                    for t in self.trace
                    if "Inherited relation from ancestor" in t or "Inherited" in t
                ),
                None,
            )
            if inh_step and len(self.evidence) >= 2:
                try:
                    p1 = self.evidence[0].split()
                    p2 = self.evidence[1].split()
                    s_art = "an" if p1[0] and p1[0][0].lower() in "aeiou" else "a"
                    anc_art = "an" if p1[2] and p1[2][0].lower() in "aeiou" else "a"
                    rel_p = p2[1]
                    rel_o = " ".join(p2[2:])
                    if rel_p == semantic.capability:
                        return f"Yes, {s_art} {p1[0]} can {rel_o}. I deduced this because {s_art} {p1[0]} is {anc_art} {p1[2]}, which can {rel_o}."
                    elif rel_p == semantic.habitat:
                        return f"Yes, {s_art} {p1[0]} lives in {rel_o}. I deduced this because {s_art} {p1[0]} is {anc_art} {p1[2]}, which lives in {rel_o}."
                    elif rel_p == semantic.whole:
                        o_art = "an" if rel_o and rel_o[0].lower() in "aeiou" else "a"
                        return f"Yes, {s_art} {p1[0]} has {o_art} {rel_o}. I deduced this because {s_art} {p1[0]} is {anc_art} {p1[2]}, which has {o_art} {rel_o}."
                    elif rel_p == semantic.diet:
                        return f"Yes, {s_art} {p1[0]} eats {rel_o}. I deduced this because {s_art} {p1[0]} is {anc_art} {p1[2]}, which eats {rel_o}."
                    elif rel_p == semantic.composition:
                        return f"Yes, {s_art} {p1[0]} is made of {rel_o}. I deduced this because {s_art} {p1[0]} is {anc_art} {p1[2]}, which is made of {rel_o}."
                    elif rel_p == semantic.purpose:
                        return f"Yes, {s_art} {p1[0]} is used for {rel_o}. I deduced this because {s_art} {p1[0]} is {anc_art} {p1[2]}, which is used for {rel_o}."
                    elif rel_p == semantic.causality:
                        return f"Yes, {p1[0].capitalize()} causes {rel_o}. I deduced this because {s_art} {p1[0]} is {anc_art} {p1[2]}, which causes {rel_o}."
                    elif rel_p == semantic.property:
                        return f"Yes, {s_art} {p1[0]} is {rel_o}. I deduced this because {s_art} {p1[0]} is {anc_art} {p1[2]}, which is {rel_o}."
                except (IndexError, ValueError):
                    pass

            if self.evidence:
                parts = self.evidence[0].split()
                if len(parts) >= 3 and parts[1] == semantic.taxonomy:
                    s_art = "an" if parts[0] and parts[0][0].lower() in "aeiou" else "a"
                    o_target = " ".join(parts[2:])
                    o_art = "an" if o_target and o_target[0].lower() in "aeiou" else "a"
                    return f"Yes, {s_art} {parts[0]} is {o_art} {o_target}."
                elif len(parts) >= 3 and parts[1] == semantic.whole:
                    o_target = " ".join(parts[2:])
                    o_art = "an" if o_target and o_target[0].lower() in "aeiou" else "a"
                    return f"Yes, {parts[0].capitalize()} has {o_art} {o_target}."
                elif len(parts) >= 3 and parts[1] == semantic.location:
                    return f"Yes, {parts[0].capitalize()} is located in {' '.join(parts[2:]).capitalize()}."
                elif len(parts) >= 3 and parts[1] == semantic.composition:
                    return f"Yes, {parts[0].capitalize()} is made of {' '.join(parts[2:])}."
                elif len(parts) >= 3 and parts[1] == semantic.purpose:
                    s_art = "an" if parts[0] and parts[0][0].lower() in "aeiou" else "a"
                    return f"Yes, {s_art} {parts[0]} is used for {' '.join(parts[2:])}."
                elif len(parts) >= 3 and parts[1] == semantic.causality:
                    return f"Yes, {parts[0].capitalize()} causes {' '.join(parts[2:])}."
                elif len(parts) >= 3 and parts[1] == semantic.property:
                    s_art = "an" if parts[0] and parts[0][0].lower() in "aeiou" else "a"
                    return f"Yes, {s_art} {parts[0]} is {' '.join(parts[2:])}."
                elif len(parts) >= 3 and parts[1] == semantic.habitat:
                    return (
                        f"Yes, {parts[0].capitalize()} lives in {' '.join(parts[2:])}."
                    )
                elif len(parts) >= 3 and parts[1] == semantic.capability:
                    s_art = "an" if parts[0] and parts[0][0].lower() in "aeiou" else "a"
                    return f"Yes, {s_art} {parts[0]} can {' '.join(parts[2:])}."
                elif len(parts) >= 3 and parts[1] == semantic.part:
                    s_art = "an" if parts[0] and parts[0][0].lower() in "aeiou" else "a"
                    o_target = " ".join(parts[2:])
                    o_art = "an" if o_target and o_target[0].lower() in "aeiou" else "a"
                    return f"Yes, {s_art} {parts[0]} is part of {o_art} {o_target}."
                elif len(parts) >= 3 and parts[1] == "larger_than":
                    return f"Yes, {parts[0].capitalize()} is larger than {' '.join(parts[2:]).capitalize()}."
            return "Yes, that is correct based on my knowledge base."

        # 6. Disjoint Refutation / Invariant Proof (Refuted False)
        if self.status == BeliefStatus.REFUTED and self.answer is False:
            if any("Reverse order confirmed" in e for e in self.evidence):
                rev_info = next(
                    e for e in self.evidence if "Reverse order confirmed" in e
                )
                chain = rev_info.split(": ")[1]
                return f"No, that is incorrect. In fact, {chain}."

            if any("NOT (" in e for e in self.evidence):
                not_e = next(e for e in self.evidence if "NOT (" in e)
                content = not_e.replace("NOT (", "").rstrip(")")
                parts = content.split()
                if len(parts) >= 3 and parts[1] == semantic.capability:
                    s_art = "an" if parts[0][0].lower() in "aeiou" else "a"
                    return (
                        f"No, {s_art} {parts[0]} cannot {' '.join(parts[2:])}. "
                        "My knowledge base confirms this explicit negative fact."
                    )

            disj_step = next(
                (t for t in self.trace if "Disjoint constraint triggered" in t), None
            )
            if disj_step:
                try:
                    reason = disj_step.split(": ")[1]
                    return f"No, that is impossible. My knowledge base confirms that {reason}."
                except (IndexError, ValueError):
                    pass
            return "No, that statement contradicts known facts in my memory."

        # 7. Open-World Unknown
        if self.is_unknown:
            if isinstance(self.answer, str) and self.answer:
                return self.answer
            return (
                "I do not know whether that is true yet. I don't have enough observations in my knowledge base, "
                "and unlike statistical LLMs, I avoid guessing or hallucinating facts."
            )

        if self.answer is not None:
            return str(self.answer)

        return "I do not have enough information to answer that question."

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["verbalized"] = self.verbalize()
        return d


@dataclass
class LearningResult:
    """Result of processing a learning interaction."""

    input_text: str
    update_type: UpdateType
    experience_id: str
    concepts_created: list[str] = field(default_factory=list)
    relations_created: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["update_type"] = self.update_type.value
        return d


@dataclass
class Skill:
    """A deterministic procedure or computational capability stored in procedural memory."""

    id: str
    name: str
    description: str = ""
    parameters: list[str] = field(default_factory=list)
    code_body: str = ""
    created_at: str = field(default_factory=current_iso_timestamp)

    @classmethod
    def create(
        cls,
        name: str,
        parameters: list[str],
        code_body: str,
        description: str = "",
    ) -> Skill:
        clean_name = name.strip().upper()
        return cls(
            id=f"skill_{clean_name.lower()}",
            name=clean_name,
            parameters=[p.strip() for p in parameters],
            code_body=code_body.strip(),
            description=description.strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DynamicState:
    """Continuous physical state of an entity governed by continuous-time dynamical ODEs."""

    entity_id: str
    freshness: float = 1.0  # 1.0 = completely fresh, 0.0 = completely decayed
    oxidation: float = 0.0  # 0.0 = unexposed, 1.0 = fully oxidized/brown
    temperature: float = 20.0  # Celsius
    time_constant: float = 3600.0  # tau in seconds for decay/oxidation
    last_updated: str = field(default_factory=current_iso_timestamp)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Construction:
    """A Construction Grammar pairing of form (token pattern) and meaning (semantic frame).

    Grammar rules are stored in memory, not hardcoded into engine source code.
    """

    id: str
    name: str
    pattern_tokens: list[str]  # e.g. ["{X}", "is", "a", "{Y}"]
    slot_roles: dict[str, str]  # e.g. {"X": "subject", "Y": "object"}
    predicate_template: str  # e.g. "is_a", "disjoint_with", "part_of"
    construction_type: str = "statement"  # "statement", "question", "action"
    is_negative: bool = False
    is_property: bool = False
    confidence: float = field(default_factory=_default_construction_confidence)
    evidence_positive: int = 1
    evidence_negative: int = 0
    created_at: str = field(default_factory=current_iso_timestamp)

    @classmethod
    def create(
        cls,
        name: str,
        pattern_tokens: list[str],
        slot_roles: dict[str, str],
        predicate_template: str,
        construction_type: str = "statement",
        is_negative: bool = False,
        is_property: bool = False,
        confidence: float | None = None,
    ) -> Construction:
        return cls(
            id=generate_id("cxn"),
            name=name.strip().lower(),
            pattern_tokens=[t.strip().lower() for t in pattern_tokens],
            slot_roles={k.strip(): v.strip().lower() for k, v in slot_roles.items()},
            predicate_template=predicate_template.strip().lower(),
            construction_type=construction_type.strip().lower(),
            is_negative=is_negative,
            is_property=is_property,
            confidence=(
                _default_construction_confidence()
                if confidence is None
                else confidence
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
