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

from little.core.models import Construction
from little.memory.store import MemoryStore

NON_CONCEPT_WORDS = {
    "who",
    "what",
    "where",
    "when",
    "why",
    "how",
    "you",
    "me",
    "i",
    "he",
    "she",
    "it",
    "we",
    "they",
    "them",
    "this",
    "that",
    "these",
    "those",
    "hi",
    "hello",
    "hey",
    "hii",
    "ok",
    "okay",
    "yes",
    "no",
}

SINGULAR_EXCEPTIONS = {
    "mars",
    "paris",
    "lens",
    "series",
    "species",
    "physics",
    "mathematics",
    "news",
    "status",
    "canvas",
    "atlantis",
    "asia",
}

# Clause boundary splitters for complex real-world text (e.g. Wikipedia)
CLAUSE_DELIMITERS = re.compile(
    r"(?:,\s*(?:where|which|who|whom|whose|although|because|while|since|but|and)\s+|\s*;\s*|\s*\b(?:until|whereas)\s+)",
    re.IGNORECASE,
)


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

    LEADING_ARTICLE = re.compile(
        r"^(?:a|an|the|this|that|these|those|every|all)\s+", re.IGNORECASE
    )

    @classmethod
    def normalize_text(cls, text: str) -> str:
        """Expand English contractions and strip irregular whitespace."""
        s = text.strip()
        s = re.sub(r"\bwhat['’]?s\b", "what is", s, flags=re.IGNORECASE)
        s = re.sub(r"\bwho['’]?s\b", "who is", s, flags=re.IGNORECASE)
        s = re.sub(r"\bwhere['’]?s\b", "where is", s, flags=re.IGNORECASE)
        s = re.sub(r"\bhow['’]?s\b", "how is", s, flags=re.IGNORECASE)
        s = re.sub(r"\bisn['’]?t\b", "is not", s, flags=re.IGNORECASE)
        s = re.sub(r"\baren['’]?t\b", "are not", s, flags=re.IGNORECASE)
        s = re.sub(r"\bcan['’]?t\b", "cannot", s, flags=re.IGNORECASE)
        s = re.sub(r"\bdon['’]?t\b", "do not", s, flags=re.IGNORECASE)
        s = re.sub(r"\bdoesn['’]?t\b", "does not", s, flags=re.IGNORECASE)
        return s

    @classmethod
    def clean_noun(cls, text: str) -> str:
        """Strip articles, auxiliary adverbs, and normalize plural forms."""
        s = cls.LEADING_ARTICLE.sub("", text.strip().lower()).strip()
        for aux in ["was formerly", "is formerly", "was originally", "does not", "did not", "cannot"]:
            if s.endswith(f" {aux}"):
                s = s[: -len(aux) - 1].strip()
            if s.startswith(f"{aux} "):
                s = s[len(aux) + 1 :].strip()

        if s in SINGULAR_EXCEPTIONS:
            return s
        # Plural inflection normalization
        if s.endswith("ies") and len(s) > 4:
            s = s[:-3] + "y"
        elif s.endswith("s") and not s.endswith("ss") and len(s) > 3:
            s = s[:-1]
        return s

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """Convert normalized string into lowercase tokens, stripping sentence punctuation."""
        clean = cls.normalize_text(text).rstrip(".!?").strip()
        # Extract alphanumeric words and basic mathematical symbols
        tokens = re.findall(r"[a-zA-Z0-9_\-\+\*\/\^]+", clean.lower())
        return tokens

    @classmethod
    def segment_clauses(cls, text: str) -> list[str]:
        """Decompose compound or complex sentences into primary declarative clauses."""
        clean = cls.normalize_text(text).rstrip(".!?").strip()
        clean = re.sub(r"^unlike\s+[^,]+,\s*", "", clean, flags=re.IGNORECASE)
        parts = CLAUSE_DELIMITERS.split(clean)
        clauses = [p.strip() for p in parts if p and len(p.strip()) > 3]
        return clauses if clauses else [clean]


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
                if bindings is not None:
                    # Invert slot roles to find the slot variable for subject and object
                    role_to_slot = {
                        v.lower(): k.upper() for k, v in cxn.slot_roles.items()
                    }
                    subj_slot = role_to_slot.get("subject", "X")
                    obj_slot = role_to_slot.get("object", "Y")

                    s_raw = bindings.get(subj_slot) or bindings.get("X", "")
                    o_raw = bindings.get(obj_slot) or bindings.get("Y", "")

                    s_clean = cls.clean_noun(s_raw)
                    o_clean = cls.clean_noun(o_raw)

                    if (
                        s_clean
                        and o_clean
                        and s_clean not in NON_CONCEPT_WORDS
                        and o_clean not in NON_CONCEPT_WORDS
                    ):
                        is_prop = cxn.is_property
                        pred = cxn.predicate_template
                        if pred == "is_a" and o_clean in {
                            "red",
                            "green",
                            "blue",
                            "yellow",
                            "black",
                            "white",
                            "brown",
                            "purple",
                            "orange",
                            "grey",
                            "gray",
                            "pink",
                        }:
                            pred = "color"
                            is_prop = True

                        results.append(
                            ParsedConstructionTriple(
                                subject=s_clean,
                                predicate=pred,
                                object_=o_clean,
                                is_property=is_prop,
                                is_negative=cxn.is_negative,
                                construction_id=cxn.id,
                            )
                        )
                        # Reinforce construction in memory
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

        # Common relational prepositions that attach to verbs
        prepositions = {
            "in",
            "to",
            "from",
            "with",
            "by",
            "of",
            "on",
            "at",
            "for",
            "as",
            "into",
        }

        # Search for verbal pivot candidate in tokens[1:-1]
        best_pivot: tuple[int, int] | None = None

        # Check for multi-word pivot (e.g. "originated in", "belongs to", "borders on")
        for i in range(1, len(tokens) - 1):
            if i + 1 < len(tokens) - 1 and tokens[i + 1] in prepositions:
                best_pivot = (i, i + 2)
                break

        if not best_pivot:
            # Check single word verbal pivot (past tense -ed, 3rd person -s, or base)
            for i in range(1, len(tokens) - 1):
                word = tokens[i]
                if (
                    word.endswith(("ed", "es", "s"))
                    or word in {"discovered", "produced", "created", "borders", "eats"}
                ) and word not in {"is", "are", "not", "has", "have", "can"}:
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
            or s_clean in NON_CONCEPT_WORDS
            or o_clean in NON_CONCEPT_WORDS
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
            confidence=0.8,
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
        tokens = cls.tokenize(text)
        if not tokens:
            return None

        # 1. Identity questions: "who are you", "what are you"
        if tokens in (
            ["who", "are", "you"],
            ["what", "are", "you"],
            ["who", "is", "little"],
        ):
            return ("little", "__identity__", None)

        stored_q_cxns = memory.list_constructions(construction_type="question")

        def anchor_count(cxn: Construction) -> int:
            return sum(
                1
                for t in cxn.pattern_tokens
                if not (t.startswith("{") and t.endswith("}"))
            )

        sorted_q = sorted(
            stored_q_cxns, key=lambda c: (anchor_count(c), c.confidence), reverse=True
        )

        for cxn in sorted_q:
            bindings = cls.match_tokens(tokens, cxn)
            if bindings is not None:
                pred = cxn.predicate_template
                role_to_slot = {v.lower(): k.upper() for k, v in cxn.slot_roles.items()}
                subj_slot = role_to_slot.get("subject", "X")
                obj_slot = role_to_slot.get("object", "Y")

                if pred == "__identity__":
                    return ("little", "__identity__", None)

                if pred == "__definition__":
                    s_raw = bindings.get(subj_slot) or bindings.get("X", "")
                    s_clean = cls.clean_noun(s_raw)
                    if s_clean and s_clean not in NON_CONCEPT_WORDS:
                        return (s_clean, "__definition__", None)

                s_raw = bindings.get(subj_slot) or bindings.get("X", "")
                o_raw = bindings.get(obj_slot) or bindings.get("Y", "")
                s_clean = cls.clean_noun(s_raw)
                o_clean = cls.clean_noun(o_raw)

                if s_clean and o_clean:
                    return (s_clean, pred, o_clean)

        return None

    @classmethod
    def parse_action_with_constructions(
        cls, text: str, memory: MemoryStore
    ) -> tuple[str, dict[str, Any]] | None:
        """Parse physical actions (e.g. slicing) using stored action constructions."""
        tokens = cls.tokenize(text)
        if not tokens:
            return None

        stored_act_cxns = memory.list_constructions(construction_type="action")

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
