"""Natural Language Statement and Question Parser for LITTLE.

Parses simple declarative statements into Semantic Triples without requiring
a monolithic LLM, and dispatches learning and inference operations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from little.core.models import (
    BeliefStatus,
    InferenceResult,
    LearningResult,
    UpdateType,
)
from little.dynamics.cfc import ContinuousDynamicsEngine
from little.dynamics.transformations import TransformationEngine
from little.inference.engine import InferenceEngine
from little.memory.store import MemoryStore
from little.procedural.runner import SkillRunner
from little.procedural.skills import register_builtin_skills

NON_CONCEPT_WORDS = {
    "who", "what", "where", "when", "why", "how",
    "you", "me", "i", "he", "she", "it", "we", "they", "them",
    "this", "that", "these", "those",
    "hi", "hello", "hey", "hii", "ok", "okay", "yes", "no",
}


@dataclass
class ParsedTriple:
    subject: str
    predicate: str
    object_: str
    is_property: bool = False
    is_negative: bool = False


class SimpleParser:
    """Robust pattern and rule-based parser for English statements and questions."""

    # Leading article pattern only at the start of a noun phrase
    LEADING_ARTICLE = re.compile(
        r"^(?:a|an|the|this|that|these|those|every|all)\s+", re.IGNORECASE
    )

    @classmethod
    def normalize_text(cls, text: str) -> str:
        s = text.strip()
        # English contractions normalization
        s = re.sub(r"\bwhat['’]?s\b", "what is", s, flags=re.IGNORECASE)
        s = re.sub(r"\bwho['’]?s\b", "who is", s, flags=re.IGNORECASE)
        s = re.sub(r"\bwhere['’]?s\b", "where is", s, flags=re.IGNORECASE)
        s = re.sub(r"\bhow['’]?s\b", "how is", s, flags=re.IGNORECASE)
        s = re.sub(r"\bisn['’]?t\b", "is not", s, flags=re.IGNORECASE)
        s = re.sub(r"\baren['’]?t\b", "are not", s, flags=re.IGNORECASE)
        s = re.sub(r"\bcan['’]?t\b", "cannot", s, flags=re.IGNORECASE)
        return s

    @classmethod
    def clean_noun(cls, text: str) -> str:
        s = cls.LEADING_ARTICLE.sub("", text.strip().lower()).strip()
        # Exceptions that end with 's' but are singular
        if s in {"mars", "paris", "lens", "series", "species", "physics", "mathematics", "news", "status", "canvas", "atlantis"}:
            return s
        # Handle plural to singular basic normalization
        if s.endswith("ies") and len(s) > 4:
            s = s[:-3] + "y"
        elif s.endswith("s") and not s.endswith("ss") and len(s) > 3:
            s = s[:-1]
        return s

    @classmethod
    def parse_statement(cls, text: str) -> list[ParsedTriple]:
        clean = cls.normalize_text(text).rstrip(".").strip()
        triples: list[ParsedTriple] = []

        # 1. Negative / Disjoint statements: "An animal is not a vehicle" / "Animals are not vehicles"
        m_neg = re.match(
            r"^(.*?)\s+(?:is not|are not|cannot be)\s+(.*?)$", clean, re.IGNORECASE
        )
        if m_neg:
            s = cls.clean_noun(m_neg.group(1))
            o = cls.clean_noun(m_neg.group(2))
            if s not in NON_CONCEPT_WORDS and o not in NON_CONCEPT_WORDS:
                triples.append(
                    ParsedTriple(
                        subject=s, predicate="disjoint_with", object_=o, is_negative=False
                    )
                )
                return triples

        # 2. "disjoint with" / "different from"
        m_disj = re.match(
            r"^(.*?)\s+(?:is disjoint with|is different from|are disjoint with)\s+(.*?)$",
            clean,
            re.IGNORECASE,
        )
        if m_disj:
            s = cls.clean_noun(m_disj.group(1))
            o = cls.clean_noun(m_disj.group(2))
            if s not in NON_CONCEPT_WORDS and o not in NON_CONCEPT_WORDS:
                triples.append(
                    ParsedTriple(
                        subject=s, predicate="disjoint_with", object_=o, is_negative=False
                    )
                )
                return triples

        # 3. Property statements with adjectives: "The apple is green", "The sky is blue"
        colors = {
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
        }
        m_color = re.match(
            r"^(.*?)\s+(?:is|are)\s+(" + "|".join(colors) + r")$", clean, re.IGNORECASE
        )
        if m_color:
            s = cls.clean_noun(m_color.group(1))
            val = m_color.group(2).lower()
            if s not in NON_CONCEPT_WORDS:
                triples.append(
                    ParsedTriple(
                        subject=s, predicate="color", object_=val, is_property=True
                    )
                )
                return triples

        # 4. "has a" / "owns a" / "have"
        m_has = re.match(r"^(.*?)\s+(?:has|owns|have)\s+(.*?)$", clean, re.IGNORECASE)
        if m_has:
            s = cls.clean_noun(m_has.group(1))
            o = cls.clean_noun(m_has.group(2))
            if s not in NON_CONCEPT_WORDS and o not in NON_CONCEPT_WORDS:
                triples.append(ParsedTriple(subject=s, predicate="has", object_=o))
                return triples

        # 5. Taxonomic "is a" / "are" classification: "A dog is an animal", "Dogs are animals"
        m_is_a = re.match(r"^(.*?)\s+(?:is|are)\s+(.*?)$", clean, re.IGNORECASE)
        if m_is_a:
            s = cls.clean_noun(m_is_a.group(1))
            o = cls.clean_noun(m_is_a.group(2))
            if s not in NON_CONCEPT_WORDS and o not in NON_CONCEPT_WORDS:
                triples.append(ParsedTriple(subject=s, predicate="is_a", object_=o))
                return triples

        # 6. Fallback: split on common verbs
        m_verb = re.match(r"^(.*?)\s+([a-z_]+)\s+(.*?)$", clean, re.IGNORECASE)
        if m_verb:
            s = cls.clean_noun(m_verb.group(1))
            p = m_verb.group(2).lower()
            o = cls.clean_noun(m_verb.group(3))
            if s not in NON_CONCEPT_WORDS and o not in NON_CONCEPT_WORDS:
                triples.append(ParsedTriple(subject=s, predicate=p, object_=o))

        return triples

    @classmethod
    def parse_action(cls, text: str) -> tuple[str, dict[str, Any]] | None:
        """Parse procedural actions like 'slice apple into 4 pieces'."""
        clean = text.strip().rstrip(".").strip()
        m_slice = re.match(
            r"^(?:slice|cut)\s+(?:an?|the)?\s*([a-zA-Z0-9_\s-]+?)\s+into\s+(\d+)\s+pieces?$",
            clean,
            re.IGNORECASE,
        )
        if m_slice:
            obj = cls.clean_noun(m_slice.group(1))
            count = int(m_slice.group(2))
            return ("SLICE", {"object": obj, "count": count})
        return None

    @classmethod
    def parse_question(
        cls, text: str, known_concepts: set[str] | None = None
    ) -> tuple[str, str, Any] | None:
        """Extract (subject, predicate, target) from natural English questions."""
        q = cls.normalize_text(text).rstrip("?").strip()

        # 0. Identity query: "who are you", "what are you", "what is little"
        if re.match(
            r"^(?:who|what)\s+(?:are|is)\s+(?:you|little|mivi|mivi_model)(?:\s+model|\s+ai)?$",
            q,
            re.IGNORECASE,
        ) or re.match(
            r"^(?:what\s+can\s+you\s+do|what\s+are\s+your\s+capabilities)$",
            q,
            re.IGNORECASE,
        ):
            return ("little", "__identity__", None)

        # 1. Arithmetic calculations: "4+4", "4 + 4", "What is 123 + 456?", "whats 4+4", "50 * 25"
        m_calc_sym = re.match(
            r"^(?:(?:what\s+is|calculate|solve|eval|evaluate)\s+)?(\d+(?:\.\d+)?)\s*([\+\-\*\/\^]|\*\*)\s*(\d+(?:\.\d+)?)$",
            q,
            re.IGNORECASE,
        )
        if m_calc_sym:
            num1 = (
                float(m_calc_sym.group(1))
                if "." in m_calc_sym.group(1)
                else int(m_calc_sym.group(1))
            )
            op_sym = m_calc_sym.group(2)
            num2 = (
                float(m_calc_sym.group(3))
                if "." in m_calc_sym.group(3)
                else int(m_calc_sym.group(3))
            )
            sym_map = {
                "+": "ADD",
                "-": "SUBTRACT",
                "*": "MULTIPLY",
                "/": "DIVIDE",
                "^": "POWER",
                "**": "POWER",
            }
            return (sym_map[op_sym], "__math__", {"a": num1, "b": num2})

        m_calc_word = re.match(
            r"^(?:(?:what\s+is|calculate|solve|eval|evaluate)\s+)?(\d+(?:\.\d+)?)\s+(plus|minus|times|multiplied by|divided by|to the power of)\s+(\d+(?:\.\d+)?)$",
            q,
            re.IGNORECASE,
        )
        if m_calc_word:
            num1 = (
                float(m_calc_word.group(1))
                if "." in m_calc_word.group(1)
                else int(m_calc_word.group(1))
            )
            op_word = m_calc_word.group(2).lower()
            num2 = (
                float(m_calc_word.group(3))
                if "." in m_calc_word.group(3)
                else int(m_calc_word.group(3))
            )
            word_map = {
                "plus": "ADD",
                "minus": "SUBTRACT",
                "times": "MULTIPLY",
                "multiplied by": "MULTIPLY",
                "divided by": "DIVIDE",
                "to the power of": "POWER",
            }
            return (word_map[op_word], "__math__", {"a": num1, "b": num2})

        m_fact = re.match(
            r"^(?:(?:what\s+is|calculate)\s+)?(?:the\s+)?factorial\s+(?:of\s+)?(\d+)$",
            q,
            re.IGNORECASE,
        )
        if not m_fact:
            m_fact = re.match(
                r"^(?:(?:what\s+is|calculate)\s+)?(\d+)!$",
                q,
                re.IGNORECASE,
            )
        if m_fact:
            n = int(m_fact.group(1))
            return ("FACTORIAL", "__math__", {"n": n})

        m_fib = re.match(
            r"^(?:(?:what\s+is|calculate)\s+)?(?:the\s+)?(?:fibonacci|fib)\s+(?:number\s+)?(?:of\s+|for\s+)?(\d+)$",
            q,
            re.IGNORECASE,
        )
        if m_fib:
            return ("FIBONACCI", "__math__", {"n": int(m_fib.group(1))})

        m_prime = re.match(r"^(?:is\s+)?(\d+)\s+prime$", q, re.IGNORECASE)
        if not m_prime:
            m_prime = re.match(r"^prime\s+(\d+)$", q, re.IGNORECASE)
        if m_prime:
            return ("IS_PRIME", "__math__", {"n": int(m_prime.group(1))})

        m_rev = re.match(
            r"^(?:(?:what\s+is|calculate)\s+)?(?:the\s+)?reverse\s+(?:of\s+)?['\"]?([^'\"]+)['\"]?$",
            q,
            re.IGNORECASE,
        )
        if m_rev:
            return ("REVERSE_STRING", "__math__", {"text": m_rev.group(1).strip()})

        m_pal = re.match(
            r"^(?:is\s+)?['\"]?([^'\"]+)['\"]?\s+(?:a\s+)?palindrome$",
            q,
            re.IGNORECASE,
        )
        if m_pal:
            return ("PALINDROME", "__math__", {"text": m_pal.group(1).strip()})

        # 2. Temporal Continuous-Time queries: "What color is the apple slice after 2 hours?"
        m_temp_color = re.match(
            r"^what color is\s+(?:an?|the)?\s*(.*?)\s+after\s+(\d+(?:\.\d+)?)\s+(seconds?|minutes?|hours?|days?)$",
            q,
            re.IGNORECASE,
        )
        if m_temp_color:
            s = cls.clean_noun(m_temp_color.group(1))
            val = float(m_temp_color.group(2))
            unit = m_temp_color.group(3).lower()
            return (s, "__temporal_color__", (val, unit))

        m_temp_fresh = re.match(
            r"^is\s+(?:an?|the)?\s*(.*?)\s+fresh\s+after\s+(\d+(?:\.\d+)?)\s+(seconds?|minutes?|hours?|days?)$",
            q,
            re.IGNORECASE,
        )
        if m_temp_fresh:
            s = cls.clean_noun(m_temp_fresh.group(1))
            val = float(m_temp_fresh.group(2))
            unit = m_temp_fresh.group(3).lower()
            return (s, "__temporal_condition__", (val, unit))

        # 3. "What color is the apple?" / "What is the color of the apple?"
        m_color = re.match(r"^what color is\s+(.*?)$", q, re.IGNORECASE)
        if m_color:
            s = cls.clean_noun(m_color.group(1))
            return (s, "color", "?")

        # 4. "Is an apple slice part of an apple?" / "Is a wheel part of a car?"
        m_part = re.match(
            r"^(?:is|are)\s+(.+?)\s+(?:part of|a part of)\s+(.+)$", q, re.IGNORECASE
        )
        if m_part:
            s = cls.clean_noun(m_part.group(1))
            o = cls.clean_noun(m_part.group(2))
            return (s, "part_of", o)

        # 5. "Does Alice have a dog?"
        m_does_have = re.match(
            r"^does\s+(.*?)\s+(?:have|own)\s+(.*?)$", q, re.IGNORECASE
        )
        if m_does_have:
            s = cls.clean_noun(m_does_have.group(1))
            o = cls.clean_noun(m_does_have.group(2))
            return (s, "has", o)

        # 6. Concept definition query: "What is an apple?", "Who is Alice?", "Tell me about a dog"
        m_def = re.match(
            r"^(?:what|who)\s+(?:is|are)\s+(?:an?|the)?\s*([a-zA-Z0-9_\s-]+)$",
            q,
            re.IGNORECASE,
        )
        if not m_def:
            m_def = re.match(
                r"^tell\s+me\s+about\s+(?:an?|the)?\s*([a-zA-Z0-9_\s-]+)$",
                q,
                re.IGNORECASE,
            )
        if m_def:
            target_noun = cls.clean_noun(m_def.group(1))
            if target_noun and target_noun not in NON_CONCEPT_WORDS:
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
                return (s, "is_a", o)

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
                    return (best_split[0], "is_a", best_split[1])

            # Fallback: predicate nominal (category) is the last token or tokens
            tokens = body.split()
            if len(tokens) >= 2:
                s = cls.clean_noun(" ".join(tokens[:-1]))
                o = cls.clean_noun(tokens[-1])
                return (s, "is_a", o)

        return None


class LearningEngine:
    """Orchestrates learning interactions and question answering over persistent memory."""

    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory
        self.inference = InferenceEngine(memory)
        register_builtin_skills(self.memory)

    def learn(self, text: str) -> LearningResult:
        """Process an input statement or action, extract concepts & relations, and persist them."""
        # 0. Check procedural actions (e.g. slicing, transformations)
        action = SimpleParser.parse_action(text)
        if action:
            act_name, act_args = action
            if act_name == "SLICE":
                res = TransformationEngine.slice_object(
                    self.memory,
                    object_name=act_args["object"],
                    num_pieces=act_args["count"],
                )
                return LearningResult(
                    input_text=text,
                    update_type=UpdateType.NEW_ENTITY,
                    experience_id=res.entities_created[0].id if res.entities_created else "",
                    concepts_created=[res.slice_concept],
                    relations_created=res.relations_created,
                    message=res.message,
                )

        triples = SimpleParser.parse_statement(text)

        if not triples:
            exp = self.memory.add_experience(input_text=text, extracted_triples=[])
            return LearningResult(
                input_text=text,
                update_type=UpdateType.NO_OP,
                experience_id=exp.id,
                message=f"Could not extract structured relations from: '{text}'",
            )

        extracted_dicts: list[dict[str, Any]] = []
        concepts_created: list[str] = []
        relations_created: list[str] = []
        update_types: list[UpdateType] = []

        for t in triples:
            extracted_dicts.append(
                {
                    "subject": t.subject,
                    "predicate": t.predicate,
                    "object": t.object_,
                    "is_property": t.is_property,
                    "is_negative": t.is_negative,
                }
            )

            # 1. Resolve or create subject concept
            subj_concept = self.memory.get_concept(t.subject)
            if not subj_concept:
                subj_concept = self.memory.create_concept(name=t.subject)
                concepts_created.append(subj_concept.name)
                update_types.append(UpdateType.NEW_CONCEPT)

            # 2. Handle property vs relational edge
            if t.is_property:
                # Store attribute directly on concept
                existing_attrs = subj_concept.attributes
                prop_key = t.predicate
                prop_val = t.object_

                # Multi-valued attribute handling (e.g. apple colors: red, green)
                if prop_key in existing_attrs:
                    current_val = existing_attrs[prop_key]
                    if isinstance(current_val, list):
                        if prop_val not in current_val:
                            current_val.append(prop_val)
                    elif current_val != prop_val:
                        existing_attrs[prop_key] = [current_val, prop_val]
                else:
                    existing_attrs[prop_key] = prop_val

                self.memory.update_concept_attributes(subj_concept.id, existing_attrs)
                update_types.append(UpdateType.PROPERTY_UPDATE)
            else:
                # Relational edge: resolve or create object concept
                obj_concept = self.memory.get_concept(t.object_)
                if not obj_concept:
                    obj_concept = self.memory.create_concept(name=t.object_)
                    concepts_created.append(obj_concept.name)
                    update_types.append(UpdateType.NEW_CONCEPT)

                # Check if relation already existed
                existing_rels = self.memory.get_relations(
                    subject_id=subj_concept.id,
                    predicate=t.predicate,
                    object_id=obj_concept.id,
                )

                self.memory.add_relation(
                    subject_id=subj_concept.id,
                    predicate=t.predicate,
                    object_id=obj_concept.id,
                    positive=not t.is_negative,
                )
                relations_created.append(
                    f"({subj_concept.name} {t.predicate} {obj_concept.name})"
                )

                if existing_rels:
                    update_types.append(UpdateType.EVIDENCE_ADDITION)
                else:
                    update_types.append(UpdateType.NEW_RELATION)

        # Log episodic experience
        exp = self.memory.add_experience(
            input_text=text, extracted_triples=extracted_dicts
        )

        # Prioritize domain-level action (property or relation) over raw concept instantiation
        if UpdateType.PROPERTY_UPDATE in update_types:
            primary_update = UpdateType.PROPERTY_UPDATE
        elif UpdateType.NEW_RELATION in update_types:
            primary_update = UpdateType.NEW_RELATION
        elif UpdateType.EVIDENCE_ADDITION in update_types:
            primary_update = UpdateType.EVIDENCE_ADDITION
        elif update_types:
            primary_update = update_types[0]
        else:
            primary_update = UpdateType.NO_OP
        return LearningResult(
            input_text=text,
            update_type=primary_update,
            experience_id=exp.id,
            concepts_created=concepts_created,
            relations_created=relations_created,
            message=f"Successfully learned: {', '.join(relations_created or concepts_created)}",
        )

    def ask(self, question: str) -> InferenceResult:
        """Answer questions by querying the knowledge graph via InferenceEngine."""
        known = {c.name.lower() for c in self.memory.list_concepts()}
        parsed = SimpleParser.parse_question(question, known_concepts=known)
        if not parsed:
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=0.0,
                evidence=[],
                trace=[f"Question pattern not recognized: '{question}'"],
            )

        subj, pred, target = parsed

        # Identity query:
        if pred == "__identity__":
            bio = (
                "I am LITTLE (Lightweight In-memory Transitive & Temporal Learning Engine), "
                "a continuous-learning cognitive architecture. Instead of predicting next tokens statistically like an LLM, "
                "I maintain an explicit semantic knowledge graph in persistent memory, execute exact multi-hop deductive logic "
                "with zero catastrophic forgetting, simulate continuous physical dynamics (e.g. apple browning over time), "
                "execute exact Python algorithms with 0% error, and actively ask questions when uncertain."
            )
            return InferenceResult(
                query=question,
                status=BeliefStatus.SUPPORTED,
                answer=bio,
                confidence=1.0,
                evidence=["Self-identity specification"],
                trace=["LITTLE Cognitive Architecture v0.1.0"],
            )

        # Concept definition query:
        if pred == "__definition__":
            concept = self.memory.get_concept(subj)
            if not concept:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer=None,
                    confidence=0.0,
                    evidence=[],
                    trace=[f"Concept '{subj}' is unknown in memory."],
                )

            desc_parts: list[str] = []
            out_rels = self.memory.get_relations(subject_id=concept.id)
            is_a_rels = [
                self.memory.get_concept(r.object_id).name
                for r in out_rels
                if r.predicate == "is_a" and self.memory.get_concept(r.object_id)
            ]
            part_of_rels = [
                self.memory.get_concept(r.object_id).name
                for r in out_rels
                if r.predicate == "part_of" and self.memory.get_concept(r.object_id)
            ]
            has_rels = [
                self.memory.get_concept(r.object_id).name
                for r in out_rels
                if r.predicate == "has" and self.memory.get_concept(r.object_id)
            ]
            disjoint_rels = [
                self.memory.get_concept(r.object_id).name
                for r in out_rels
                if r.predicate == "disjoint_with" and self.memory.get_concept(r.object_id)
            ]

            in_rels = self.memory.get_relations(object_id=concept.id)
            parts = [
                self.memory.get_concept(r.subject_id).name
                for r in in_rels
                if r.predicate == "part_of" and self.memory.get_concept(r.subject_id)
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

        # Procedural Mathematics query:
        if pred == "__math__":
            skill = self.memory.get_skill(subj)
            if not skill:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer=None,
                    confidence=0.0,
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
                    confidence=1.0,
                    evidence=[
                        f"Evaluated skill {subj}({arg_repr}) = {exec_res.result}"
                    ],
                    trace=exec_res.trace or [],
                )
            return InferenceResult(
                query=question,
                status=BeliefStatus.REFUTED,
                answer=None,
                confidence=0.0,
                evidence=[],
                trace=[f"Execution failed: {exec_res.error}"],
            )

        # Continuous Dynamics queries:
        if pred in ("__temporal_color__", "__temporal_condition__"):
            sec_mult = {
                "second": 1.0,
                "seconds": 1.0,
                "minute": 60.0,
                "minutes": 60.0,
                "hour": 3600.0,
                "hours": 3600.0,
                "day": 86400.0,
                "days": 86400.0,
            }
            val, unit = target
            dt = val * sec_mult.get(unit, 1.0)

            concept = self.memory.get_concept(subj)
            if not concept:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer=None,
                    confidence=0.0,
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
                    confidence=0.95,
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
                    confidence=0.95,
                    evidence=[
                        f"ODE state at dt={dt:.0f}s: freshness={evolved.freshness:.2f} -> {condition}"
                    ],
                    trace=[
                        f"Continuous CfC evolution: freshness={evolved.freshness:.4f}",
                        f"Condition descriptor: '{condition}'",
                    ],
                )

        # Special query: "What color is X?"
        if pred == "color" and target == "?":
            concept = self.memory.get_concept(subj)
            if not concept:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer=None,
                    confidence=0.0,
                    evidence=[],
                    trace=[f"Concept '{subj}' is unknown."],
                )
            attrs = self.inference.get_inherited_attributes(subj)
            color = attrs.get("color")
            if color:
                val_str = ", ".join(color) if isinstance(color, list) else str(color)
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.SUPPORTED,
                    answer=val_str,
                    confidence=0.95,
                    evidence=[f"{subj} color is {val_str}"],
                    trace=[f"Retrieved property 'color' for '{subj}': {val_str}"],
                )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=0.1,
                evidence=[],
                trace=[f"No color property recorded for '{subj}'."],
            )

        # Standard relational query
        return self.inference.infer(subject=subj, predicate=pred, target=target)
