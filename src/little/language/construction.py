"""Construction Grammar (CxG) and Language Acquisition Engine for LITTLE.

Implements memory-grounded parsing where grammatical constructions are stored
as relational records in SQLite rather than hardcoded in Python regex patterns.

Core components:
1. Tokenizer & ClauseSegmenter: boundary detection and contraction expansion.
2. ConstructionMatcher: matches token streams against memory-stored constructions.
3. OpenPivotLearner: extracts relational pivots (verbs + prepositions) from novel
   sentences and registers new constructions into SQLite dynamically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from little.core.construction_policy import ConstructionPolicy
from little.core.models import Construction
from little.language.parser_policy import ParserPolicy
from little.memory.store import MemoryStore


@dataclass
class MatchResult:
    """Result of matching an input token sequence against a Construction."""

    construction: Construction
    bindings: dict[str, str]
    confidence: float


@dataclass
class ParsedConstructionTriple:
    subject: str
    predicate: str
    object_: str
    is_property: bool = False
    is_negative: bool = False
    construction_id: str | None = None


class ConstructionEngine:
    """Executes grammar matching, clause segmentation, and dynamic grammar acquisition."""

    @classmethod
    def _policy(cls) -> ParserPolicy:
        """Resolve an optional bound policy without global mutable state."""
        configured = getattr(cls, "_POLICY", None)
        return configured if configured is not None else ParserPolicy.default()

    @classmethod
    def normalize_text(cls, text: str) -> str:
        """Expand English contractions and strip irregular whitespace."""
        s = text.strip()
        for source, replacement in cls._policy().normalization_replacements:
            token_pattern = rf"(?<!\w){re.escape(source)}(?!\w)"
            s = re.sub(token_pattern, replacement, s, flags=re.IGNORECASE)
        return s

    @classmethod
    def clean_noun(cls, text: str) -> str:
        """Strip articles, auxiliary adverbs, and normalize plural forms."""
        articles = "|".join(
            re.escape(article) for article in cls._policy().leading_articles
        )
        s = re.sub(
            rf"^(?:{articles})\s+", "", text.strip().lower(), flags=re.IGNORECASE
        ).strip()
        for aux in cls._policy().noun_auxiliaries:
            if s.endswith(f" {aux}"):
                s = s[: -len(aux) - 1].strip()
            if s.startswith(f"{aux} "):
                s = s[len(aux) + 1 :].strip()

        if not s:
            return ""
        if s in cls._policy().irregular_plurals:
            return cls._policy().irregular_plurals[s]
        if s in cls._policy().invariable_words:
            return s
        tokens = s.split()
        if len(tokens) > 1:
            last = cls.clean_noun(tokens[-1])
            return " ".join(tokens[:-1] + [last])

        # Plural inflection normalization
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
    def is_valid_concept(cls, c: str) -> bool:
        """Check if candidate text constitutes a valid domain concept rather than a question or pronoun."""
        if not c:
            return False
        clean = c.strip().lower()
        if clean in cls._policy().non_concept_words:
            return False
        words = set(clean.split())
        return not any(
            w in cls._policy().non_concept_words or w in cls._policy().question_words
            for w in words
        )

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """Convert normalized string into lowercase tokens, stripping sentence punctuation."""
        clean = cls.normalize_text(text).rstrip(".!?").strip()
        # Keep commas as list-boundary tokens so a slot can later normalize
        # "wheels, an engine, and headlights" into separate objects.
        tokens = re.findall(r"[a-zA-Z0-9_\-\+\*\/\^]+|,", clean.lower())
        return tokens

    @classmethod
    def segment_clauses(cls, text: str) -> list[str]:
        """Decompose compound or complex sentences into primary declarative clauses."""
        clean = cls.normalize_text(text).strip()
        clean = re.sub(r"^unlike\s+[^,]+,\s*", "", clean, flags=re.IGNORECASE)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
        clauses: list[str] = []
        for sent in sentences:
            sent_clean = sent.rstrip(".!?").strip()
            parts = re.split(
                cls._policy().clause_delimiter_pattern,
                sent_clean,
                flags=re.IGNORECASE,
            )
            for p in parts:
                p_str = p.strip()
                if p_str and len(p_str) > 2:
                    clauses.append(p_str)
        return clauses if clauses else [clean.rstrip(".!?").strip()]

    @classmethod
    def match_tokens(
        cls, tokens: list[str], construction: Construction
    ) -> dict[str, str] | None:
        """Attempt to bind slots in a construction pattern against a token sequence.

        Supports arbitrary interleaving of literal anchor tokens and variable slots,
        e.g.:
        - ["{X}", "is", "a", "{Y}"]
        - ["is", "{X}", "an", "{Y}"]
        - ["slice", "{X}", "into", "{COUNT}", "pieces"]
        - ["who", "are", "you"]
        """
        pattern = construction.pattern_tokens
        if not pattern or not tokens:
            return None

        # 1. Segment pattern into alternating literals and slots
        segments: list[tuple[str, Any]] = []
        current_lit: list[str] = []
        for elem in pattern:
            if elem.startswith("{") and elem.endswith("}"):
                if current_lit:
                    segments.append(("lit", current_lit))
                    current_lit = []
                segments.append(("slot", elem[1:-1].upper()))
            else:
                current_lit.append(elem.lower())
        if current_lit:
            segments.append(("lit", current_lit))

        # 2. Sequential segmented alignment
        t_idx = 0
        bindings: dict[str, str] = {}

        for s_i, (seg_type, seg_val) in enumerate(segments):
            if seg_type == "lit":
                lit_len = len(seg_val)
                if s_i == 0:
                    # Initial literal must match at token start
                    if tokens[:lit_len] != seg_val:
                        return None
                    t_idx = lit_len
                elif s_i == len(segments) - 1:
                    # Final literal must match at token end
                    if tokens[-lit_len:] != seg_val or t_idx > len(tokens) - lit_len:
                        return None
                    prev_slot = segments[s_i - 1][1]
                    slot_tokens = tokens[t_idx : len(tokens) - lit_len]
                    if not slot_tokens:
                        return None
                    bindings[prev_slot] = " ".join(slot_tokens)
                    t_idx = len(tokens)
                else:
                    # Intermediate literal: must leave at least 1 token for preceding slot
                    min_start = (
                        t_idx + 1
                        if (s_i > 0 and segments[s_i - 1][0] == "slot")
                        else t_idx
                    )
                    found = -1
                    for cand in range(min_start, len(tokens) - lit_len + 1):
                        if tokens[cand : cand + lit_len] == seg_val:
                            found = cand
                            break
                    if found == -1:
                        return None

                    prev_slot = segments[s_i - 1][1]
                    slot_tokens = tokens[t_idx:found]
                    if not slot_tokens:
                        return None
                    bindings[prev_slot] = " ".join(slot_tokens)
                    t_idx = found + lit_len
            elif seg_type == "slot":
                if s_i == len(segments) - 1:
                    # Trailing slot consumes all remaining tokens
                    slot_tokens = tokens[t_idx:]
                    if not slot_tokens:
                        return None
                    bindings[seg_val] = " ".join(slot_tokens)
                    t_idx = len(tokens)

        return bindings if t_idx == len(tokens) else None

    @classmethod
    def parse_from_catalog(
        cls,
        text: str,
        constructions: list[Construction],
        construction_type: str = "statement",
    ) -> list[ParsedConstructionTriple]:
        """Parse text using an explicitly supplied construction catalog.

        This is the pure, side-effect-free counterpart to
        :meth:`parse_with_constructions`.  It is used by the parser before its
        compatibility fallbacks, so a grammar pack can add a new relation
        without adding a Python regex or changing parser code.
        """
        candidates = [
            construction
            for construction in constructions
            if construction.construction_type == construction_type
        ]

        def anchor_count(construction: Construction) -> int:
            return sum(
                1
                for token in construction.pattern_tokens
                if not (token.startswith("{") and token.endswith("}"))
            )

        ordered = sorted(
            candidates,
            key=lambda construction: (
                anchor_count(construction),
                construction.confidence,
            ),
            reverse=True,
        )
        results: list[ParsedConstructionTriple] = []

        for clause in cls.segment_clauses(text):
            tokens = cls.tokenize(clause)
            if not tokens:
                continue

            for construction in ordered:
                bindings = cls.match_tokens(tokens, construction)
                if bindings is None:
                    continue

                triples = cls._triples_from_catalog_match(construction, bindings)
                if triples:
                    results.extend(triples)
                    break

        return results

    @classmethod
    def _triples_from_catalog_match(
        cls, construction: Construction, bindings: dict[str, str]
    ) -> list[ParsedConstructionTriple]:
        """Convert a matched construction into normalized semantic triples."""
        role_to_slot = {
            value.lower(): key.upper()
            for key, value in construction.slot_roles.items()
        }
        subject_slot = role_to_slot.get("subject", "X")
        object_slot = role_to_slot.get("object", "Y")
        subject_raw = bindings.get(subject_slot) or bindings.get("X", "")
        object_raw = bindings.get(object_slot) or bindings.get("Y", "")

        if construction.predicate_template == "__user_name__":
            name_raw = (
                bindings.get("NAME")
                or bindings.get("name")
                or bindings.get("X")
                or ""
            )
            name = cls.clean_noun(name_raw)
            if not name:
                return []
            return [
                ParsedConstructionTriple(
                    subject="user",
                    predicate="has_name",
                    object_=name,
                    is_property=True,
                    construction_id=construction.id,
                )
            ]

        subject = cls.clean_noun(subject_raw)
        if not subject or not cls.is_valid_concept(subject):
            return []

        predicate = construction.predicate_template
        is_property = construction.is_property
        conjoined = [
            cls.clean_noun(part)
            for part in re.split(r",\s*(?:and\s+)?|\s+and\s+", object_raw)
            if part.strip()
        ]
        if len(conjoined) > 1:
            if not all(cls.is_valid_concept(part) for part in conjoined):
                return []
            objects = conjoined
        else:
            object_ = cls.clean_noun(object_raw)
            if not object_ or not cls.is_valid_concept(object_):
                return []
            objects = [object_]

        if (
            predicate == cls._policy().semantic.taxonomy
            and len(objects) == 1
            and objects[0] in cls._policy().colors
        ):
            predicate = cls._policy().semantic.color
            is_property = True

        return [
            ParsedConstructionTriple(
                subject=subject,
                predicate=predicate,
                object_=part,
                is_property=is_property,
                is_negative=construction.is_negative,
                construction_id=construction.id,
            )
            for part in objects
        ]

    @classmethod
    def parse_with_constructions(
        cls,
        text: str,
        memory: MemoryStore,
        construction_type: str = "statement",
    ) -> list[ParsedConstructionTriple]:
        """Parse text by matching against persistent constructions stored in SQLite.

        Applies the Construction Grammar Principle of Specificity:
        Constructions with more anchor tokens (e.g. 'is not a' > 'is not' > 'is')
        are evaluated first to pre-empt overly generic patterns.
        """
        clauses = cls.segment_clauses(text)
        results: list[ParsedConstructionTriple] = []
        stored_constructions = memory.list_constructions(
            construction_type=construction_type
        )

        # Sort by specificity (number of literal anchor tokens descending)
        def anchor_count(cxn: Construction) -> int:
            return sum(
                1
                for t in cxn.pattern_tokens
                if not (t.startswith("{") and t.endswith("}"))
            )

        sorted_constructions = sorted(
            stored_constructions,
            key=lambda c: (anchor_count(c), c.confidence),
            reverse=True,
        )

        for clause in clauses:
            tokens = cls.tokenize(clause)
            if not tokens:
                continue

            matched = False
            for cxn in sorted_constructions:
                bindings = cls.match_tokens(tokens, cxn)
                if bindings is None:
                    continue

                triples = cls._triples_from_catalog_match(cxn, bindings)
                if not triples:
                    continue

                results.extend(triples)
                memory.add_construction_evidence(cxn.id, positive=True)
                matched = True
                break

            # If no known construction matched, invoke Open Pivot Learner
            if not matched and construction_type == "statement":
                learned_triple = cls.learn_open_pivot(tokens, memory)
                if learned_triple:
                    results.append(learned_triple)

        return results

    @classmethod
    def learn_open_pivot(
        cls, tokens: list[str], memory: MemoryStore
    ) -> ParsedConstructionTriple | None:
        """Open Pivot Grammar Learner:

        When a sentence has no matching construction in SQLite (e.g. 'Apples originated
        in Central Asia'), dynamically detect relational pivots (verbs + prepositions),
        hypothesize a new Construction, save it to SQLite, and extract the triple.
        """
        if len(tokens) < 3:
            return None

        # Search for verbal pivot candidate in tokens[1:-1]
        best_pivot: tuple[int, int] | None = None

        # Check for multi-word pivot (e.g. "originated in", "belongs to", "borders on")
        for i in range(1, len(tokens) - 1):
            if (
                i + 1 < len(tokens) - 1
                and tokens[i + 1] in cls._policy().pivot_prepositions
            ):
                best_pivot = (i, i + 2)
                break

        if not best_pivot:
            # Check single word verbal pivot (past tense -ed, 3rd person -s, or base)
            for i in range(1, len(tokens) - 1):
                word = tokens[i]
                if (
                    word.endswith(cls._policy().verb_suffixes)
                    or word in cls._policy().pivot_verbs
                ) and word not in cls._policy().excluded_pivot_verbs:
                    best_pivot = (i, i + 1)
                    break

        if not best_pivot:
            # Fallback: middle token as pivot
            mid = len(tokens) // 2
            best_pivot = (mid, mid + 1)

        p_start, p_end = best_pivot
        subj_tokens = tokens[:p_start]
        pivot_tokens = tokens[p_start:p_end]
        obj_tokens = tokens[p_end:]

        if not subj_tokens or not obj_tokens:
            return None

        s_clean = cls.clean_noun(" ".join(subj_tokens))
        o_clean = cls.clean_noun(" ".join(obj_tokens))
        predicate_name = "_".join(pivot_tokens)

        if (
            not s_clean
            or not o_clean
            or not cls.is_valid_concept(s_clean)
            or not cls.is_valid_concept(o_clean)
        ):
            return None

        # Dynamically register the new learned construction in SQLite memory!
        cxn_name = f"cxn_{predicate_name}"
        pattern = ["{X}"] + pivot_tokens + ["{Y}"]
        slot_roles = {"X": "subject", "Y": "object"}

        new_cxn = Construction.create(
            name=cxn_name,
            pattern_tokens=pattern,
            slot_roles=slot_roles,
            predicate_template=predicate_name,
            construction_type="statement",
            confidence=ConstructionPolicy.default().learned_confidence,
        )
        saved_id = memory.save_construction(new_cxn)

        return ParsedConstructionTriple(
            subject=s_clean,
            predicate=predicate_name,
            object_=o_clean,
            is_property=False,
            is_negative=False,
            construction_id=saved_id,
        )

    @classmethod
    def parse_question_with_constructions(
        cls, text: str, memory: MemoryStore
    ) -> tuple[str, str, Any] | None:
        """Parse natural language question using stored question constructions."""
        return cls.parse_question_from_catalog_with_procedural(
            text, memory.list_constructions()
        )

    @classmethod
    def parse_question_from_catalog_with_procedural(
        cls, text: str, constructions: list[Construction]
    ) -> tuple[str, str, Any] | None:
        """Parse a question or procedural request from one explicit catalog."""
        proc = cls.parse_procedural_from_catalog(text, constructions)
        if proc:
            skill_name, proc_args = proc
            return (skill_name, "__math__", proc_args)
        return cls.parse_question_from_catalog(text, constructions)

    @classmethod
    def parse_question_from_catalog(
        cls, text: str, constructions: list[Construction]
    ) -> tuple[str, str, Any] | None:
        """Parse a question from an explicit catalog without touching memory."""
        tokens = cls.tokenize(text)
        if not tokens:
            return None

        question_constructions = [
            construction
            for construction in constructions
            if construction.construction_type == "question"
        ]

        def anchor_count(cxn: Construction) -> int:
            return sum(
                1
                for t in cxn.pattern_tokens
                if not (t.startswith("{") and t.endswith("}"))
            )

        sorted_q = sorted(
            question_constructions,
            key=lambda c: (anchor_count(c), c.confidence),
            reverse=True,
        )

        for cxn in sorted_q:
            bindings = cls.match_tokens(tokens, cxn)
            if bindings is not None:
                pred = cxn.predicate_template
                role_to_slot = {v.lower(): k.upper() for k, v in cxn.slot_roles.items()}
                subj_slot = role_to_slot.get("subject", "X")
                obj_slot = role_to_slot.get("object", "Y")

                if pred in ("__identity__", "__identity_name__"):
                    return ("little", "__identity__", None)

                if pred == "__query_user_name__":
                    return ("user", "__user_name__", None)

                if pred == "__definition__":
                    s_raw = bindings.get(subj_slot) or bindings.get("X", "")
                    if any(
                        marker in s_raw.lower()
                        for marker in cls._policy().invalid_definition_markers
                    ):
                        continue
                    s_clean = cls.clean_noun(s_raw)
                    if s_clean and s_clean not in cls._policy().non_concept_words:
                        return (s_clean, "__definition__", None)

                s_raw = bindings.get(subj_slot) or bindings.get("X", "")
                o_raw = bindings.get(obj_slot) or bindings.get("Y", "")

                # Guardrail for generic categorical questions:
                # If s_raw or o_raw contains "than", "used for", or math symbols, it is NOT a simple is_a relation
                if (
                    any(
                        marker in s_raw.lower()
                        for marker in cls._policy().invalid_category_markers
                    )
                    or (
                        o_raw
                        and any(
                            marker in o_raw.lower()
                            for marker in cls._policy().invalid_category_markers
                        )
                    )
                ) and pred == cls._policy().semantic.taxonomy:
                    continue

                s_clean = cls.clean_noun(s_raw)
                o_clean = cls.clean_noun(o_raw)

                # Open WH query target placeholder handling (e.g. "what is X used for")
                if (
                    pred in cls._policy().open_query_predicates
                    and not o_clean
                    and s_clean
                ):
                    return (s_clean, pred, "?")

                if s_clean and o_clean:
                    return (s_clean, pred, o_clean)

        return None

    @classmethod
    def parse_procedural_with_constructions(
        cls, text: str, memory: MemoryStore
    ) -> tuple[str, dict[str, Any]] | None:
        """Parse procedural skill inquiries (e.g. math operations) using stored procedural constructions."""
        return cls.parse_procedural_from_catalog(
            text, memory.list_constructions(construction_type="procedural")
        )

    @classmethod
    def parse_procedural_from_catalog(
        cls, text: str, constructions: list[Construction]
    ) -> tuple[str, dict[str, Any]] | None:
        """Parse a procedural request using an explicit construction catalog."""
        tokens = cls.tokenize(text)
        if not tokens:
            return None

        procedural_constructions = [
            construction
            for construction in constructions
            if construction.construction_type == "procedural"
        ]

        def anchor_count(cxn: Construction) -> int:
            return sum(
                1
                for t in cxn.pattern_tokens
                if not (t.startswith("{") and t.endswith("}"))
            )

        sorted_proc = sorted(
            procedural_constructions,
            key=lambda c: (anchor_count(c), c.confidence),
            reverse=True,
        )

        for cxn in sorted_proc:
            bindings = cls.match_tokens(tokens, cxn)
            if bindings is not None:
                skill_name = cxn.predicate_template
                args: dict[str, Any] = {}
                valid = True
                for slot_key, param_name in cxn.slot_roles.items():
                    raw_val = (
                        bindings.get(slot_key.upper())
                        or bindings.get(slot_key.lower())
                        or bindings.get(param_name.upper())
                        or bindings.get(param_name.lower())
                    )
                    if raw_val is None:
                        valid = False
                        break
                    raw_val_str = str(raw_val).strip()
                    try:
                        if "." in raw_val_str:
                            args[param_name] = float(raw_val_str)
                        else:
                            args[param_name] = int(raw_val_str)
                    except ValueError:
                        args[param_name] = raw_val_str
                if valid:
                    # Enforce that arithmetic operations MUST receive numeric arguments
                    if skill_name.upper() in cls._policy().numeric_skills and not all(
                        isinstance(v, (int, float)) for v in args.values()
                    ):
                        continue
                    return (skill_name.upper(), args)

        return None

    @classmethod
    def parse_action_with_constructions(
        cls, text: str, memory: MemoryStore
    ) -> tuple[str, dict[str, Any]] | None:
        """Parse physical actions (e.g. slicing) using stored action constructions."""
        return cls.parse_action_from_catalog(
            text, memory.list_constructions(construction_type="action")
        )

    @classmethod
    def parse_action_from_catalog(
        cls, text: str, constructions: list[Construction]
    ) -> tuple[str, dict[str, Any]] | None:
        """Parse a physical action using an explicit construction catalog."""
        tokens = cls.tokenize(text)
        if not tokens:
            return None

        stored_act_cxns = [
            construction
            for construction in constructions
            if construction.construction_type == "action"
        ]

        for cxn in stored_act_cxns:
            bindings = cls.match_tokens(tokens, cxn)
            if bindings is not None:
                role_to_slot = {v.lower(): k.upper() for k, v in cxn.slot_roles.items()}
                obj_slot = role_to_slot.get("object", "X")
                count_slot = role_to_slot.get("count", "COUNT")

                obj_raw = bindings.get(obj_slot) or bindings.get("X", "")
                count_raw = bindings.get(count_slot) or bindings.get("COUNT", "1")
                obj_clean = cls.clean_noun(obj_raw)
                try:
                    count = int(count_raw)
                except ValueError:
                    count = 1

                if obj_clean:
                    return (
                        cxn.predicate_template,
                        {"object": obj_clean, "count": count},
                    )

        return None
