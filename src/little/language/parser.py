"""Natural Language Statement and Question Parser for LITTLE.

Parses simple declarative statements into Semantic Triples without requiring
a monolithic LLM, and dispatches learning and inference operations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar

from little.core.models import (
    BeliefStatus,
    InferenceResult,
    LearningResult,
    UpdateType,
)
from little.dynamics.cfc import ContinuousDynamicsEngine
from little.dynamics.transformations import TransformationEngine
from little.inference.engine import InferenceEngine
from little.language.construction import ConstructionEngine
from little.language.dialogue import DialogueContext
from little.memory.store import MemoryStore
from little.procedural.runner import SkillRunner
from little.procedural.skills import register_builtin_skills

NON_CONCEPT_WORDS = {
    "?",
    "a",
    "an",
    "the",
    "who",
    "what",
    "where",
    "when",
    "why",
    "how",
    "which",
    "you",
    "u",
    "ur",
    "your",
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
    "and",
    "or",
    "but",
    "because",
    "hi",
    "hello",
    "hey",
    "hii",
    "ok",
    "okay",
    "yes",
    "no",
    "so",
    "well",
    "now",
    "then",
    "things you can do",
    "things u can do",
    "so what",
    "my",
    "mine",
    "our",
    "ours",
    "myself",
    "yourself",
    "my name",
    "your name",
}

ACTION_VERBS: dict[str, str] = {
    "fly": "fly",
    "flies": "fly",
    "swim": "swim",
    "swims": "swim",
    "run": "run",
    "runs": "run",
    "jump": "jump",
    "jumps": "jump",
    "walk": "walk",
    "walks": "walk",
    "crawl": "crawl",
    "crawls": "crawl",
    "sing": "sing",
    "sings": "sing",
    "bark": "bark",
    "barks": "bark",
    "meow": "meow",
    "meows": "meow",
    "roar": "roar",
    "roars": "roar",
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
        # Internet slang & shorthand normalization
        s = re.sub(r"\bu\b", "you", s, flags=re.IGNORECASE)
        s = re.sub(r"\bur\b", "your", s, flags=re.IGNORECASE)
        s = re.sub(r"\br\b", "are", s, flags=re.IGNORECASE)
        s = re.sub(r"\bwat\b", "what", s, flags=re.IGNORECASE)
        s = re.sub(r"\bplz\b", "please", s, flags=re.IGNORECASE)
        s = re.sub(r"\bidk\b", "i do not know", s, flags=re.IGNORECASE)

        # English contractions normalization
        s = re.sub(r"\bwhat['’]?s\b", "what is", s, flags=re.IGNORECASE)
        s = re.sub(r"\bwho['’]?s\b", "who is", s, flags=re.IGNORECASE)
        s = re.sub(r"\bwhere['’]?s\b", "where is", s, flags=re.IGNORECASE)
        s = re.sub(r"\bhow['’]?s\b", "how is", s, flags=re.IGNORECASE)
        s = re.sub(r"\bisn['’]?t\b", "is not", s, flags=re.IGNORECASE)
        s = re.sub(r"\baren['’]?t\b", "are not", s, flags=re.IGNORECASE)
        s = re.sub(r"\bcan['’]?t\b", "cannot", s, flags=re.IGNORECASE)

        # Isolated greetings must NOT be stripped away to empty
        if re.match(r"^(?:hey|heyy|hello|hi|hii|yo|howdy)$", s, re.IGNORECASE):
            return s

        # Conversational discourse markers e.g. "so what is 10+10", "hey what can u do", "well tell me..."
        if not re.match(r"^well\s+done\b", s, re.IGNORECASE):
            s = re.sub(
                r"^(?:(?:so|well|hey|now|ok|okay|then|and|also|just)\b[\s,]*)+",
                "",
                s,
                flags=re.IGNORECASE,
            ).strip()

        # Conversational polite prefixes e.g. "Can you calculate ...", "Could you please tell me ...", "Do you know ..."
        s = re.sub(
            r"^(?:(?:can|could|would)\s+you\s+(?:please\s+)?(?:tell\s+me\s+)?|(?:do|would)\s+you\s+know\s+|(?:please\s+)?tell\s+me\s+|please\s+)",
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
        if clean in NON_CONCEPT_WORDS or clean == "?":
            return False
        words = set(clean.split())
        question_words = {"who", "what", "where", "when", "why", "how", "which"}
        return not any(w in NON_CONCEPT_WORDS or w in question_words for w in words)

    IRREGULAR_PLURALS: ClassVar[dict[str, str]] = {
        "mice": "mouse",
        "children": "child",
        "geese": "goose",
        "teeth": "tooth",
        "feet": "foot",
        "people": "person",
        "men": "man",
        "women": "woman",
        "oxen": "ox",
        "leaves": "leaf",
        "knives": "knife",
        "wolves": "wolf",
        "halves": "half",
        "lives": "life",
        "loaves": "loaf",
        "thieves": "thief",
        "cacti": "cactus",
        "fungi": "fungus",
        "nuclei": "nucleus",
        "phenomena": "phenomenon",
        "criteria": "criterion",
        "analyses": "analysis",
        "crises": "crisis",
        "diagnoses": "diagnosis",
        "wings": "wing",
        "fins": "fin",
        "wheels": "wheel",
        "viruses": "virus",
        "walruses": "walrus",
        "campuses": "campus",
        "bonuses": "bonus",
        "circuses": "circus",
        "menus": "menu",
        "gurus": "guru",
        "emus": "emu",
    }

    INVARIABLE_WORDS: ClassVar[set[str]] = {
        "mars",
        "venus",
        "uranus",
        "phobos",
        "deimos",
        "celsius",
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
        "united states",
        "sheep",
        "deer",
        "fish",
        "aircraft",
        "salmon",
        "trout",
        "spacecraft",
        "swine",
        "bison",
        "water",
        "ice",
        "steam",
        "meat",
        "air",
        "grass",
        "glass",
        "brass",
        "compass",
        "mass",
        "bass",
        "gas",
        "virus",
        "fungus",
        "cactus",
        "nucleus",
        "radius",
        "walrus",
        "octopus",
        "platypus",
        "rhinoceros",
        "hippopotamus",
        "alps",
        "athens",
        "brussels",
        "scissors",
    }

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
        s = cls.LEADING_ARTICLE.sub("", raw_lower).strip()
        if not s:
            return ""
        if s in cls.IRREGULAR_PLURALS:
            return cls.IRREGULAR_PLURALS[s]
        if s in cls.INVARIABLE_WORDS:
            return s
        # Multi-word nouns (e.g. "living things" -> "living thing", "apple slices" -> "apple slice")
        tokens = s.split()
        if len(tokens) > 1:
            last = cls.clean_noun(tokens[-1])
            return " ".join(tokens[:-1] + [last])

        if s.endswith("ies") and len(s) > 4:
            s = s[:-3] + "y"
        elif s.endswith("ves") and len(s) > 4:
            if s in ("knives", "lives", "wives"):
                s = s[:-3] + "fe"
            else:
                s = s[:-3] + "f"
        elif s.endswith("es") and len(s) > 4:
            if s.endswith(("oes", "xes", "sses", "ches", "shes")):
                s = s[:-2]
            else:
                s = s[:-1]
        elif s.endswith("s") and not s.endswith(("ss", "us", "is")) and len(s) > 3:
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
                triples.append(ParsedTriple(subject=concept, predicate="is_a", object_=category))
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

        # 0b2. User self-introduction: "My name is Aswin", "Call me Aswin", "I am Aswin", "My name's Aswin"
        m_user_intro = re.match(
            r"^(?:(?:ok|okay|well|hello|hi|hey)[,\s]+)?(?:my\s+name\s+is|my\s+name['’]s|i\s+am|call\s+me)\s+([A-Za-z0-9_-]+)(?:[,\s]+and\s+.*)?$",
            clean,
            re.IGNORECASE,
        )
        if m_user_intro:
            uname = m_user_intro.group(1).strip()
            if uname.lower() not in NON_CONCEPT_WORDS:
                triples.append(
                    ParsedTriple(
                        subject="user",
                        predicate="name",
                        object_=uname,
                        is_property=True,
                    )
                )
                return triples

        # 0c. Universal Quantifiers: "All felines are carnivores", "Every tiger is a cat"
        m_univ = re.match(
            r"^(?:all|every|each)\s+(.*?)\s+(?:are|is)\s+(?:(?:a|an|the)\s+)?(.*?)$",
            clean,
            re.IGNORECASE,
        )
        if m_univ:
            s = cls.clean_noun(m_univ.group(1))
            o = cls.clean_noun(m_univ.group(2))
            if cls.is_valid_concept(s) and cls.is_valid_concept(o):
                triples.append(ParsedTriple(subject=s, predicate="is_a", object_=o))
                return triples

        # 0d. Conditional Rules: "If an animal is a feline then it is a carnivore"
        m_cond = re.match(
            r"^if\s+(?:(?:a|an|the)\s+)?(.*?)\s+(?:is|are)\s+(?:(?:a|an|the)\s+)?(.*?)[,\s]+then\s+(?:it|they)\s+(?:is|are)\s+(?:(?:a|an|the)\s+)?(.*?)$",
            clean,
            re.IGNORECASE,
        )
        if m_cond:
            antecedent_obj = cls.clean_noun(m_cond.group(2))
            consequent_obj = cls.clean_noun(m_cond.group(3))
            if cls.is_valid_concept(antecedent_obj) and cls.is_valid_concept(
                consequent_obj
            ):
                triples.append(
                    ParsedTriple(
                        subject=antecedent_obj, predicate="is_a", object_=consequent_obj
                    )
                )
                return triples

        # 1. Negative / Disjoint statements: "An animal is not a vehicle" / "Animals are not vehicles"
        m_neg = re.match(
            r"^(.*?)\s+(?:is not|are not|cannot be)\s+(.*?)$", clean, re.IGNORECASE
        )
        if m_neg:
            s = cls.clean_noun(m_neg.group(1))
            o = cls.clean_noun(m_neg.group(2))
            if cls.is_valid_concept(s) and cls.is_valid_concept(o):
                triples.append(
                    ParsedTriple(
                        subject=s,
                        predicate="disjoint_with",
                        object_=o,
                        is_negative=False,
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
            if cls.is_valid_concept(s) and cls.is_valid_concept(o):
                triples.append(
                    ParsedTriple(
                        subject=s,
                        predicate="disjoint_with",
                        object_=o,
                        is_negative=False,
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
            if cls.is_valid_concept(s):
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
            raw_o = m_has.group(2)
            conjoined = cls.split_conjoined_items(raw_o)
            if cls.is_valid_concept(s) and conjoined:
                for o in conjoined:
                    triples.append(ParsedTriple(subject=s, predicate="has", object_=o))
                return triples

        # 4b. Mereology / Part-Whole: "Leaves are part of plants", "A wheel is part of a car"
        m_part = re.match(
            r"^(.*?)\s+(?:is part of|are part of|is a part of|are parts of)\s+(.*?)$",
            clean,
            re.IGNORECASE,
        )
        if m_part:
            s = cls.clean_noun(m_part.group(1))
            o = cls.clean_noun(m_part.group(2))
            if cls.is_valid_concept(s) and cls.is_valid_concept(o):
                triples.append(ParsedTriple(subject=s, predicate="part_of", object_=o))
                return triples

        # 5. Geographic & Topological containment: "Paris is located in France"
        m_loc = re.match(
            r"^(.*?)\s+(?:is located in|are located in|is in|are in)\s+(.*?)$",
            clean,
            re.IGNORECASE,
        )
        if m_loc:
            s = cls.clean_noun(m_loc.group(1))
            o = cls.clean_noun(m_loc.group(2))
            if cls.is_valid_concept(s) and cls.is_valid_concept(o):
                triples.append(
                    ParsedTriple(subject=s, predicate="located_in", object_=o)
                )
                return triples

        # 6. Habitat: "Fish live in water", "A salmon lives in water"
        m_live = re.match(
            r"^(.*?)\s+(?:lives in|live in)\s+(.*?)$", clean, re.IGNORECASE
        )
        if m_live:
            s = cls.clean_noun(m_live.group(1))
            o = cls.clean_noun(m_live.group(2))
            if cls.is_valid_concept(s) and cls.is_valid_concept(o):
                triples.append(ParsedTriple(subject=s, predicate="lives_in", object_=o))
                return triples

        # 7. Material Composition: "Ice is made of water"
        m_made = re.match(
            r"^(.*?)\s+(?:is made of|are made of)\s+(.*?)$", clean, re.IGNORECASE
        )
        if m_made:
            s = cls.clean_noun(m_made.group(1))
            o = cls.clean_noun(m_made.group(2))
            if cls.is_valid_concept(s) and cls.is_valid_concept(o):
                triples.append(ParsedTriple(subject=s, predicate="made_of", object_=o))
                return triples

        # 8. Capability: "Birds can fly", "Fish can swim and jump"
        m_can = re.match(r"^(.*?)\s+can\s+(.*?)$", clean, re.IGNORECASE)
        if m_can:
            s = cls.clean_noun(m_can.group(1))
            raw_acts = m_can.group(2)
            conjoined = cls.split_conjoined_items(raw_acts)
            if cls.is_valid_concept(s) and conjoined:
                for act in conjoined:
                    triples.append(
                        ParsedTriple(subject=s, predicate="can", object_=act)
                    )
                return triples

        # 8b. Negative capability: "Penguins cannot fly", "Ostriches can't fly"
        m_cannot = re.match(
            r"^(.*?)\s+(?:cannot|can not|cant|can't)\s+(.*?)$", clean, re.IGNORECASE
        )
        if m_cannot:
            s = cls.clean_noun(m_cannot.group(1))
            raw_acts = m_cannot.group(2)
            conjoined = cls.split_conjoined_items(raw_acts)
            if cls.is_valid_concept(s) and conjoined:
                for act in conjoined:
                    triples.append(
                        ParsedTriple(
                            subject=s, predicate="can", object_=act, is_negative=True
                        )
                    )
                return triples

        # 8c. Intransitive capability verbs: "Dolphins swim", "An eagle flies", "It swims"
        m_act = re.match(
            r"^(.*?)\s+(" + "|".join(ACTION_VERBS.keys()) + r")$",
            clean,
            re.IGNORECASE,
        )
        if m_act:
            s = cls.clean_noun(m_act.group(1))
            verb = ACTION_VERBS[m_act.group(2).lower()]
            if cls.is_valid_concept(s):
                triples.append(ParsedTriple(subject=s, predicate="can", object_=verb))
                return triples

        # 9. Dietary: "Carnivores eat meat", "A tiger eats meat", "Bears eat meat and plant"
        m_eat = re.match(r"^(.*?)\s+(?:eats|eat)\s+(.*?)$", clean, re.IGNORECASE)
        if m_eat:
            s = cls.clean_noun(m_eat.group(1))
            raw_e = m_eat.group(2)
            conjoined = cls.split_conjoined_items(raw_e)
            if cls.is_valid_concept(s) and conjoined:
                for o in conjoined:
                    triples.append(ParsedTriple(subject=s, predicate="eats", object_=o))
                return triples

        # 9b. Purpose / Usage: "A hammer is used for hitting nails", "Knives are used to cut"
        m_used = re.match(
            r"^(.*?)\s+(?:is used for|are used for|is used to|are used to)\s+(.*?)$",
            clean,
            re.IGNORECASE,
        )
        if m_used:
            s = cls.clean_noun(m_used.group(1))
            raw_u = m_used.group(2)
            parts = [
                p.strip()
                for p in re.split(r",\s*(?:and\s+)?|\s+and\s+", raw_u)
                if p.strip()
            ]
            for p in parts:
                p_clean = cls.clean_noun(p)
                if cls.is_valid_concept(s) and p_clean:
                    triples.append(
                        ParsedTriple(subject=s, predicate="used_for", object_=p_clean)
                    )
            if triples:
                return triples

        # 9c. Causality: "Fire causes smoke and heat"
        m_cause = re.match(
            r"^(.*?)\s+(?:causes|cause)\s+(.*?)$",
            clean,
            re.IGNORECASE,
        )
        if m_cause:
            s = cls.clean_noun(m_cause.group(1))
            raw_c = m_cause.group(2)
            conjoined = cls.split_conjoined_items(raw_c)
            if cls.is_valid_concept(s) and conjoined:
                for eff in conjoined:
                    triples.append(
                        ParsedTriple(subject=s, predicate="causes", object_=eff)
                    )
                return triples

        # 10. Taxonomic "is a" / "are" classification: "A dog is an animal", "Dogs are animals"
        m_is_a = re.match(r"^(.*?)\s+(?:is|are)\s+(.*?)$", clean, re.IGNORECASE)
        if m_is_a:
            s = cls.clean_noun(m_is_a.group(1))
            raw_o = m_is_a.group(2)
            conjoined = cls.split_conjoined_items(raw_o)
            if cls.is_valid_concept(s) and conjoined:
                for o in conjoined:
                    triples.append(ParsedTriple(subject=s, predicate="is_a", object_=o))
                return triples

        # 11. Fallback: split on common verbs
        m_verb = re.match(r"^(.*?)\s+([a-z_]+)\s+(.*?)$", clean, re.IGNORECASE)
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
        m_slice = re.match(
            r"^(?:slice|cut)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s-]+?)\s+into\s+(\d+)\s+pieces?$",
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

        # Strip leading conversational greetings or discourse markers before parsing the question
        # e.g. "hii who are you" -> "who are you", "hello can an eagle fly" -> "can an eagle fly"
        m_greeting_prefix = re.match(
            r"^(?:hi|hii|hello|hey|heyy|howdy|yo|greetings|good\s+morning|good\s+afternoon|good\s+evening|ok|okay|so|well|please|tell\s+me)[,\s!]+(.+)$",
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

        # 0. Identity / Capability query: "who are you", "what can you do", "what are the things u can do"
        if re.match(
            r"^(?:who|what)\s+(?:are|is)\s+(?:you|little|mivi|mivi_model)(?:\s+model|\s+ai)?$",
            q,
            re.IGNORECASE,
        ) or re.match(
            r"^(?:(?:tell\s+me\s+)?what(?:\s+are)?\s+(?:all\s+)?(?:the\s+)?(?:things\s+)?(?:you|u)\s+can\s+do(?:\s+for\s+me)?|"
            r"what\s+can\s+(?:you|u)\s+do(?:\s+for\s+me)?|"
            r"what\s+do\s+(?:you|u)\s+do|"
            r"what\s+are\s+(?:your|ur)\s+(?:capabilities|features|skills|functions)|"
            r"what\s+is\s+(?:your|ur)\s+purpose|"
            r"how\s+can\s+(?:you|u)\s+help(?:\s+me)?)$",
            q,
            re.IGNORECASE,
        ):
            return ("little", "__identity__", None)

        # 0c. User Identity queries: "what is my name", "who am i", "do you know my name"
        if re.match(
            r"^(?:what\s+is\s+my\s+name|what['’]s\s+my\s+name|who\s+am\s+i|do\s+you\s+know\s+my\s+name|tell\s+me\s+my\s+name)$",
            q,
            re.IGNORECASE,
        ):
            return ("user", "__user_name__", None)

        # 0d. Bot Name queries: "what is your name", "what's your name", "tell me your name"
        if re.match(
            r"^(?:what\s+is\s+(?:your|ur|you)\s+name|what['’]s\s+(?:your|ur|you)\s+name|tell\s+me\s+(?:your|ur|you)\s+name)$",
            q,
            re.IGNORECASE,
        ):
            return ("little", "__identity_name__", None)

        # 0b. Conversational Chitchat & Polite Greetings
        if re.match(
            r"^(?:hello|hi|hii|hey|heyy|greetings|good\s+morning|good\s+afternoon|good\s+evening|howdy|yo)(?:\s+little|\s+there)?$",
            q,
            re.IGNORECASE,
        ):
            return ("little", "__chitchat_greeting__", None)

        if re.match(
            r"^(?:thank\s+you(?:\s+so\s+much|\s+very\s+much)?|thanks(?:\s+so\s+much|\s+a\s+lot|\s+a\s+bunch|\s+very\s+much)?|thx|ty|many\s+thanks|i\s+appreciate\s+it)$",
            q,
            re.IGNORECASE,
        ):
            return ("little", "__chitchat_thanks__", None)

        if re.match(
            r"^(?:how\s+are\s+you(?:\s+doing)?|how\s+r\s+u|how\s+do\s+you\s+do|how\s+is\s+it\s+going|how\s+are\s+things)$",
            q,
            re.IGNORECASE,
        ):
            return ("little", "__chitchat_how_are_you__", None)

        if re.match(
            r"^(?:good\s+job|great\s+job|well\s+done|nice\s+work|awesome|amazing|cool|nice|you\s+are\s+smart|you\s+are\s+great|you\s+did\s+great)$",
            q,
            re.IGNORECASE,
        ):
            return ("little", "__chitchat_praise__", None)

        if re.match(
            r"^(?:(?:tell\s+me\s+)?a\s+fact|(?:tell\s+me\s+)?something(?:\s+cool|\s+interesting|\s+you\s+know)?|give\s+me\s+a\s+fact|what\s+do\s+you\s+know)$",
            q,
            re.IGNORECASE,
        ):
            return ("little", "__chitchat_fact__", None)

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

        # Natural language arithmetic word forms:
        m_add = re.match(
            r"^(?:(?:what\s+is|calculate)\s+)?add\s+(\d+(?:\.\d+)?)\s+(?:and|to)\s+(\d+(?:\.\d+)?)$",
            q,
            re.IGNORECASE,
        )
        if m_add:
            n1 = float(m_add.group(1)) if "." in m_add.group(1) else int(m_add.group(1))
            n2 = float(m_add.group(2)) if "." in m_add.group(2) else int(m_add.group(2))
            return ("ADD", "__math__", {"a": n1, "b": n2})

        m_sum = re.match(
            r"^(?:(?:what\s+is|calculate)\s+)?(?:the\s+)?(?:sum|total)\s+of\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)$",
            q,
            re.IGNORECASE,
        )
        if m_sum:
            n1 = float(m_sum.group(1)) if "." in m_sum.group(1) else int(m_sum.group(1))
            n2 = float(m_sum.group(2)) if "." in m_sum.group(2) else int(m_sum.group(2))
            return ("ADD", "__math__", {"a": n1, "b": n2})

        m_mul = re.match(
            r"^(?:(?:what\s+is|calculate)\s+)?multiply\s+(\d+(?:\.\d+)?)\s+(?:by|and)\s+(\d+(?:\.\d+)?)$",
            q,
            re.IGNORECASE,
        )
        if m_mul:
            n1 = float(m_mul.group(1)) if "." in m_mul.group(1) else int(m_mul.group(1))
            n2 = float(m_mul.group(2)) if "." in m_mul.group(2) else int(m_mul.group(2))
            return ("MULTIPLY", "__math__", {"a": n1, "b": n2})

        m_prod = re.match(
            r"^(?:(?:what\s+is|calculate)\s+)?(?:the\s+)?product\s+of\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)$",
            q,
            re.IGNORECASE,
        )
        if m_prod:
            n1 = (
                float(m_prod.group(1))
                if "." in m_prod.group(1)
                else int(m_prod.group(1))
            )
            n2 = (
                float(m_prod.group(2))
                if "." in m_prod.group(2)
                else int(m_prod.group(2))
            )
            return ("MULTIPLY", "__math__", {"a": n1, "b": n2})

        m_div = re.match(
            r"^(?:(?:what\s+is|calculate)\s+)?divide\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)$",
            q,
            re.IGNORECASE,
        )
        if m_div:
            n1 = float(m_div.group(1)) if "." in m_div.group(1) else int(m_div.group(1))
            n2 = float(m_div.group(2)) if "." in m_div.group(2) else int(m_div.group(2))
            return ("DIVIDE", "__math__", {"a": n1, "b": n2})

        m_sub = re.match(
            r"^(?:(?:what\s+is|calculate)\s+)?subtract\s+(\d+(?:\.\d+)?)\s+from\s+(\d+(?:\.\d+)?)$",
            q,
            re.IGNORECASE,
        )
        if m_sub:
            n1 = float(m_sub.group(1)) if "." in m_sub.group(1) else int(m_sub.group(1))
            n2 = float(m_sub.group(2)) if "." in m_sub.group(2) else int(m_sub.group(2))
            return ("SUBTRACT", "__math__", {"a": n2, "b": n1})

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

        m_nth_prime = re.match(
            r"^(?:(?:what\s+is|calculate|find)\s+)?(?:the\s+)?(\d+)(?:st|nd|rd|th)\s+prime(?:\s+number)?$",
            q,
            re.IGNORECASE,
        )
        if m_nth_prime:
            return ("NTH_PRIME", "__math__", {"n": int(m_nth_prime.group(1))})

        m_rev = re.match(
            r"^(?:(?:what\s+is|calculate)\s+)?(?:the\s+)?reverse\s+(?:of\s+)?['\"]?([^'\"]+)['\"]?$",
            q,
            re.IGNORECASE,
        )
        if m_rev:
            return ("REVERSE_STRING", "__math__", {"text": m_rev.group(1).strip()})

        m_pal = re.match(
            r"^(?:is\s+)?['\"]?([^'\"]+?)['\"]?\s+(?:a\s+)?palindrome$",
            q,
            re.IGNORECASE,
        )
        if m_pal:
            return ("PALINDROME", "__math__", {"text": m_pal.group(1).strip()})

        m_sqrt = re.match(
            r"^(?:(?:what\s+is|calculate|find)\s+)?(?:the\s+)?(?:square\s+root|sqrt)\s+(?:of\s+)?(\d+(?:\.\d+)?)$",
            q,
            re.IGNORECASE,
        )
        if m_sqrt:
            num = (
                float(m_sqrt.group(1))
                if "." in m_sqrt.group(1)
                else int(m_sqrt.group(1))
            )
            return ("SQRT", "__math__", {"n": num})

        m_gcd = re.match(
            r"^(?:(?:what\s+is|calculate|find)\s+)?(?:the\s+)?(?:gcd|greatest\s+common\s+divisor)\s+(?:of\s+)?(\d+)\s+and\s+(\d+)$",
            q,
            re.IGNORECASE,
        )
        if m_gcd:
            return (
                "GCD",
                "__math__",
                {"a": int(m_gcd.group(1)), "b": int(m_gcd.group(2))},
            )

        m_lcm = re.match(
            r"^(?:(?:what\s+is|calculate|find)\s+)?(?:the\s+)?lcm\s+(?:of\s+)?(\d+)\s+and\s+(\d+)$",
            q,
            re.IGNORECASE,
        )
        if m_lcm:
            return (
                "LCM",
                "__math__",
                {"a": int(m_lcm.group(1)), "b": int(m_lcm.group(2))},
            )

        m_pct = re.match(
            r"^(?:(?:what\s+is|calculate|find)\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent)\s+(?:of\s+)(\d+(?:\.\d+)?)$",
            q,
            re.IGNORECASE,
        )
        if m_pct:
            p = float(m_pct.group(1)) if "." in m_pct.group(1) else int(m_pct.group(1))
            t = float(m_pct.group(2)) if "." in m_pct.group(2) else int(m_pct.group(2))
            return ("PERCENT", "__math__", {"percent": p, "total": t})

        m_sq = re.match(
            r"^(?:(?:what\s+is|calculate|find)\s+)?(?:the\s+)?square\s+of\s+(\d+(?:\.\d+)?)$",
            q,
            re.IGNORECASE,
        )
        if not m_sq:
            m_sq = re.match(r"^(\d+(?:\.\d+)?)\s+squared$", q, re.IGNORECASE)
        if m_sq:
            num = float(m_sq.group(1)) if "." in m_sq.group(1) else int(m_sq.group(1))
            return ("POWER", "__math__", {"a": num, "b": 2})

        m_cb = re.match(
            r"^(?:(?:what\s+is|calculate|find)\s+)?(?:the\s+)?cube\s+of\s+(\d+(?:\.\d+)?)$",
            q,
            re.IGNORECASE,
        )
        if not m_cb:
            m_cb = re.match(r"^(\d+(?:\.\d+)?)\s+cubed$", q, re.IGNORECASE)
        if m_cb:
            num = float(m_cb.group(1)) if "." in m_cb.group(1) else int(m_cb.group(1))
            return ("POWER", "__math__", {"a": num, "b": 3})

        m_avg = re.match(
            r"^(?:(?:what\s+is|calculate|find)\s+)?(?:the\s+)?(?:average|mean)\s+(?:of\s+)?((?:\d+(?:\.\d+)?(?:,\s*|\s+and\s+|\s+))+\d+(?:\.\d+)?)$",
            q,
            re.IGNORECASE,
        )
        if m_avg:
            raw_nums = m_avg.group(1).replace("and", ",")
            nums = [
                float(x.strip()) if "." in x.strip() else int(x.strip())
                for x in raw_nums.split(",")
                if x.strip()
            ]
            return ("AVERAGE", "__math__", {"numbers": nums})

        m_lin = re.match(
            r"^(?:solve\s+)?(-?\d+(?:\.\d+)?)\s*\*?\s*x\s*([+-])\s*(\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)$",
            q,
            re.IGNORECASE,
        )
        if m_lin:
            a_v = float(m_lin.group(1))
            sign = m_lin.group(2)
            b_raw = float(m_lin.group(3))
            b_v = -b_raw if sign == "-" else b_raw
            c_v = float(m_lin.group(4))
            return ("SOLVE_LINEAR", "__math__", {"a": a_v, "b": b_v, "c": c_v})

        # Quadratic & polynomial equations: "Solve x^2 - 5x + 6 = 0", "Solve 2x^2 + 5x - 3 = 0"
        m_quad = re.match(r"^solve\s+(.+)$", q, re.IGNORECASE)
        if m_quad:
            body = m_quad.group(1).strip()
            if any(term in body for term in ("x^2", "x**2", "x ^ 2", "x * * 2")):
                return (
                    "SOLVE_QUADRATIC",
                    "__math__",
                    {"equation": body},
                )

        # Calculus derivatives: "What is the derivative of x^3 + 2x?", "Derivative of x^2 + 3x"
        m_diff = re.match(
            r"^(?:(?:what\s+is\s+)?(?:the\s+)?derivative\s+of|diff(?:erentiate)?)\s+([x0-9\s+\-*/^.()]+)$",
            q,
            re.IGNORECASE,
        )
        if m_diff:
            return (
                "CALCULUS_DERIVATIVE",
                "__math__",
                {"expression": m_diff.group(1).strip()},
            )

        # Algebraic simplification: "Simplify 2x + 3x + 5"
        m_simp = re.match(
            r"^(?:(?:can\s+you\s+)?simplify)\s+([x0-9\s+\-*/^.()]+)$",
            q,
            re.IGNORECASE,
        )
        if m_simp:
            return (
                "SIMPLIFY_EXPR",
                "__math__",
                {"expression": m_simp.group(1).strip()},
            )

        # Unit conversion: "Convert 100 km to miles", "Convert 25 celsius to fahrenheit"
        m_conv = re.match(
            r"^(?:convert\s+)?(\d+(?:\.\d+)?)\s+([a-zA-Z]+)\s+(?:to|in|into)\s+([a-zA-Z]+)$",
            q,
            re.IGNORECASE,
        )
        if m_conv:
            val = (
                float(m_conv.group(1))
                if "." in m_conv.group(1)
                else int(m_conv.group(1))
            )
            return (
                "UNIT_CONVERT",
                "__math__",
                {
                    "value": val,
                    "from_unit": m_conv.group(2).lower(),
                    "to_unit": m_conv.group(3).lower(),
                },
            )

        m_expr = re.match(
            r"^(?:(?:what\s+is|calculate|compute|evaluate)\s+)?\s*(\(?\s*\d+(?:\.\d+)?\s*(?:[+\-*/^]|(?:\*\*))\s*[-+*/()0-9\s.^]+)$",
            q,
            re.IGNORECASE,
        )
        if m_expr:
            clean_expr = m_expr.group(1).strip().replace("^", "**")
            return ("EVAL_EXPR", "__math__", {"expression": clean_expr})

        # 2. Temporal Continuous-Time queries: "What color is the apple slice after 2 hours?"
        m_temp_color = re.match(
            r"^what color is\s+(?:(?:a|an|the)\s+)?(.*?)\s+after\s+(\d+(?:\.\d+)?)\s+(seconds?|minutes?|hours?|days?)$",
            q,
            re.IGNORECASE,
        )
        if m_temp_color:
            s = cls.clean_noun(m_temp_color.group(1))
            val = float(m_temp_color.group(2))
            unit = m_temp_color.group(3).lower()
            return (s, "__temporal_color__", (val, unit))

        m_temp_fresh = re.match(
            r"^is\s+(?:(?:a|an|the)\s+)?(.*?)\s+fresh\s+after\s+(\d+(?:\.\d+)?)\s+(seconds?|minutes?|hours?|days?)$",
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

        # Comparatives: "Is an elephant bigger than a mouse?", "Is the sun larger than the earth?"
        m_comp = re.match(
            r"^(?:is|are)\s+(.+?)\s+(?:bigger\s+than|larger\s+than|greater\s+than)\s+(.+)$",
            q,
            re.IGNORECASE,
        )
        if m_comp:
            s = cls.clean_noun(m_comp.group(1))
            o = cls.clean_noun(m_comp.group(2))
            if cls.is_valid_concept(s) and cls.is_valid_concept(o):
                return (s, "larger_than", o)

        # WH-Information Seeking Queries:
        # Location: "Where is Paris?", "Where is Paris located?"
        m_wh_loc = re.match(
            r"^where\s+(?:is|are)\s+(.+?)(?:\s+located)?$", q, re.IGNORECASE
        )
        if m_wh_loc:
            s = cls.clean_noun(m_wh_loc.group(1))
            if cls.is_valid_concept(s):
                return (s, "located_in", "?")

        # Habitat: "Where does a salmon live?", "Where do fish live?", "Where a salmon lives"
        m_wh_live = re.match(
            r"^where\s+(?:(?:does|do)\s+(.+?)\s+live|(.+?)\s+lives?)$", q, re.IGNORECASE
        )
        if not m_wh_live:
            m_wh_live = re.match(
                r"^where\s+can\s+(?:i|we|one)?\s*(?:find|see)\s+(.+)$",
                q,
                re.IGNORECASE,
            )
        if m_wh_live:
            raw_s = m_wh_live.group(1) or m_wh_live.group(2)
            s = cls.clean_noun(raw_s)
            if cls.is_valid_concept(s):
                return (s, "lives_in", "?")

        # Diet: "What does a lion eat?", "What do carnivores eat?"
        m_wh_eat = re.match(r"^what\s+(?:does|do)\s+(.+?)\s+eat$", q, re.IGNORECASE)
        if m_wh_eat:
            s = cls.clean_noun(m_wh_eat.group(1))
            if cls.is_valid_concept(s):
                return (s, "eats", "?")

        # Capability: "What can an eagle do?", "What can birds do?"
        m_wh_can = re.match(r"^what\s+can\s+(.+?)\s+do$", q, re.IGNORECASE)
        if m_wh_can:
            s = cls.clean_noun(m_wh_can.group(1))
            if cls.is_valid_concept(s):
                return (s, "can", "?")

        # Material: "What is ice made of?", "What is steam made of?"
        m_wh_mat = re.match(
            r"^what\s+(?:is|are)\s+(.+?)\s+made\s+of$", q, re.IGNORECASE
        )
        if m_wh_mat:
            s = cls.clean_noun(m_wh_mat.group(1))
            if cls.is_valid_concept(s):
                return (s, "made_of", "?")

        # Parts / Features: "What parts does a car have?", "What does a bird have?"
        m_wh_has = re.match(
            r"^what\s+(?:parts?\s+)?(?:does|do)\s+(.+?)\s+have$", q, re.IGNORECASE
        )
        if m_wh_has:
            s = cls.clean_noun(m_wh_has.group(1))
            if cls.is_valid_concept(s):
                return (s, "has", "?")

        # Purpose / Usage: "What is a hammer used for?", "What is a knife used for?"
        m_wh_used = re.match(
            r"^what\s+(?:is|are)\s+(.+?)\s+used\s+for$", q, re.IGNORECASE
        )
        if m_wh_used:
            s = cls.clean_noun(m_wh_used.group(1))
            if cls.is_valid_concept(s):
                return (s, "used_for", "?")

        # Cause / Effect: "What does fire cause?", "What does exercise cause?"
        m_wh_causes = re.match(
            r"^what\s+(?:does|do)\s+(.+?)\s+cause$", q, re.IGNORECASE
        )
        if m_wh_causes:
            s = cls.clean_noun(m_wh_causes.group(1))
            if cls.is_valid_concept(s):
                return (s, "causes", "?")

        # Reverse Cause: "What causes rain?", "What causes cancer?"
        m_wh_caused_by = re.match(r"^what\s+causes\s+(.+)$", q, re.IGNORECASE)
        if m_wh_caused_by:
            o = cls.clean_noun(m_wh_caused_by.group(1))
            if cls.is_valid_concept(o):
                return ("?", "causes", o)

        # Properties: "What properties does glass have?"
        m_wh_prop = re.match(
            r"^what\s+properties\s+(?:does|do)\s+(.+?)\s+have$", q, re.IGNORECASE
        )
        if m_wh_prop:
            s = cls.clean_noun(m_wh_prop.group(1))
            if cls.is_valid_concept(s):
                return (s, "has_property", "?")

        # Why-Questions:
        # 1. Why is X not a Y? "Why is water not a solid?", "Why can't an animal be a vehicle?"
        m_why_not = re.match(
            r"^why\s+(?:is|are)\s+(.+?)\s+not\s+(?:(?:a|an|the)\s+)?(.+)$",
            q,
            re.IGNORECASE,
        )
        if not m_why_not:
            m_why_not = re.match(
                r"^why\s+can['’]?t\s+(.+?)\s+be\s+(?:(?:a|an|the)\s+)?(.+)$",
                q,
                re.IGNORECASE,
            )
        if m_why_not:
            s = cls.clean_noun(m_why_not.group(1))
            target = cls.clean_noun(m_why_not.group(2))
            if cls.is_valid_concept(s) and cls.is_valid_concept(target):
                return (s, "__why_not__", target)

        # 2. Why is X in Y? "Why is Paris in Europe?", "Why is Paris located in France?"
        m_why_loc = re.match(
            r"^why\s+(?:is|are)\s+(.+?)\s+(?:located\s+in|in)\s+(.+)$",
            q,
            re.IGNORECASE,
        )
        if m_why_loc:
            s = cls.clean_noun(m_why_loc.group(1))
            target = cls.clean_noun(m_why_loc.group(2))
            if cls.is_valid_concept(s) and cls.is_valid_concept(target):
                return (s, "__why_located_in__", target)

        # 3. Why is X a Y? "Why is an eagle an animal?"
        m_why_is = re.match(r"^why\s+(?:is|are)\s+(.+)$", q, re.IGNORECASE)
        if m_why_is:
            body = m_why_is.group(1).strip()
            m_two_arts = re.match(
                r"^(?:(?:a|an|the)\s+)?(.+?)\s+(?:a|an|the)\s+(.+)$",
                body,
                re.IGNORECASE,
            )
            if m_two_arts:
                s = cls.clean_noun(m_two_arts.group(1))
                target = cls.clean_noun(m_two_arts.group(2))
                if cls.is_valid_concept(s) and cls.is_valid_concept(target):
                    return (s, "__why_is_a__", target)

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

        # 4. "Is an apple slice part of an apple?" / "Is a wheel part of a car?"
        m_part = re.match(
            r"^(?:is|are)\s+(.+?)\s+(?:part of|a part of)\s+(.+)$", q, re.IGNORECASE
        )
        if m_part:
            s = cls.clean_noun(m_part.group(1))
            o = cls.clean_noun(m_part.group(2))
            return (s, "part_of", o)

        # 5. Geographic: "Is Paris located in France?", "Is Paris in France?"
        m_loc = re.match(
            r"^(?:is|are)\s+(.+?)\s+(?:located in|in)\s+(.+)$", q, re.IGNORECASE
        )
        if m_loc:
            s = cls.clean_noun(m_loc.group(1))
            o = cls.clean_noun(m_loc.group(2))
            if cls.is_valid_concept(s) and cls.is_valid_concept(o):
                return (s, "located_in", o)

        # 6. Habitat: "Does a salmon live in water?", "Do fish live in water?"
        m_live = re.match(r"^(?:does|do)\s+(.+?)\s+live\s+in\s+(.+)$", q, re.IGNORECASE)
        if m_live:
            s = cls.clean_noun(m_live.group(1))
            o = cls.clean_noun(m_live.group(2))
            if cls.is_valid_concept(s) and cls.is_valid_concept(o):
                return (s, "lives_in", o)

        # 7. Material: "Is ice made of water?"
        m_made = re.match(r"^(?:is|are)\s+(.+?)\s+made\s+of\s+(.+)$", q, re.IGNORECASE)
        if m_made:
            s = cls.clean_noun(m_made.group(1))
            o = cls.clean_noun(m_made.group(2))
            if cls.is_valid_concept(s) and cls.is_valid_concept(o):
                return (s, "made_of", o)

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
                    return (best_split[0], "can", best_split[1])
            s = cls.clean_noun(raw_s)
            act = cls.clean_noun(raw_act)
            if cls.is_valid_concept(s) and act:
                return (s, "can", act)

        # 8b. Capability with does/do: "Does a dolphin swim?", "Do birds fly?", "Does it swim?"
        m_does_act = re.match(
            r"^(?:does|do)\s+(.+?)\s+(" + "|".join(ACTION_VERBS.keys()) + r")$",
            q,
            re.IGNORECASE,
        )
        if m_does_act:
            s = cls.clean_noun(m_does_act.group(1))
            act = ACTION_VERBS[m_does_act.group(2).lower()]
            if cls.is_valid_concept(s):
                return (s, "can", act)

        # 9. Dietary: "Does a tiger eat meat?", "Do carnivores eat meat?"
        m_eat = re.match(r"^(?:does|do)\s+(.+?)\s+eat\s+(.+)$", q, re.IGNORECASE)
        if m_eat:
            s = cls.clean_noun(m_eat.group(1))
            o = cls.clean_noun(m_eat.group(2))
            if cls.is_valid_concept(s) and cls.is_valid_concept(o):
                return (s, "eats", o)

        # 10. Possession: "Does Alice have a dog?", "Do birds have wings?"
        m_does_have = re.match(
            r"^(?:does|do)\s+(.*?)\s+(?:have|own)\s+(.*?)$", q, re.IGNORECASE
        )
        if m_does_have:
            s = cls.clean_noun(m_does_have.group(1))
            o = cls.clean_noun(m_does_have.group(2))
            return (s, "has", o)

        # 11. Property boolean query: "Is the apple red?", "Is it red?"
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
        m_is_color = re.match(
            r"^(?:is|are)\s+(.*?)\s+(" + "|".join(colors) + r")$", q, re.IGNORECASE
        )
        if m_is_color:
            s = cls.clean_noun(m_is_color.group(1))
            val = m_is_color.group(2).lower()
            return (s, "color", val)

        # 12. Purpose boolean query: "Is a hammer used for hitting nails?", "Is a knife used to cut?"
        m_is_used = re.match(
            r"^(?:is|are)\s+(.+?)\s+used\s+(?:for|to)\s+(.+)$", q, re.IGNORECASE
        )
        if m_is_used:
            s = cls.clean_noun(m_is_used.group(1))
            o = cls.clean_noun(m_is_used.group(2))
            if cls.is_valid_concept(s) and o:
                return (s, "used_for", o)

        # 13. Cause boolean query: "Does fire cause smoke?", "Does exercise cause fitness?"
        m_does_cause = re.match(
            r"^(?:does|do)\s+(.+?)\s+cause\s+(.+)$", q, re.IGNORECASE
        )
        if m_does_cause:
            s = cls.clean_noun(m_does_cause.group(1))
            o = cls.clean_noun(m_does_cause.group(2))
            if cls.is_valid_concept(s) and cls.is_valid_concept(o):
                return (s, "causes", o)

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
        known_props = {
            "transparent",
            "fragile",
            "conductive",
            "hard",
            "soft",
            "cold",
            "hot",
            "flexible",
            "absorbent",
            "warm-blooded",
            "cold-blooded",
        }
        m_is_prop = re.match(
            r"^(?:is|are)\s+(.+?)\s+(" + "|".join(known_props) + r")$",
            q,
            re.IGNORECASE,
        )
        if m_is_prop:
            s = cls.clean_noun(m_is_prop.group(1))
            prop = m_is_prop.group(2).lower()
            if cls.is_valid_concept(s):
                return (s, "has_property", prop)

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
        self.last_subject: str | None = None
        self.last_object: str | None = None
        self.dialogue = DialogueContext()
        register_builtin_skills(self.memory)

    def learn(self, text: str) -> LearningResult:
        """Process an input statement or action, extract concepts & relations, and persist them."""
        # Multi-sentence input handler: "A dog is an animal. It has fur. It can bark."
        raw_sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?;\n])\s+", text)
            if s.strip() and not s.strip().endswith("?")
        ]
        if len(raw_sentences) > 1:
            all_concepts: list[str] = []
            all_relations: list[str] = []
            all_updates: list[UpdateType] = []
            for sent in raw_sentences:
                sub_res = self.learn(sent)
                if sub_res.update_type != UpdateType.NO_OP:
                    all_concepts.extend(sub_res.concepts_created)
                    all_relations.extend(sub_res.relations_created)
                    all_updates.append(sub_res.update_type)
            if all_relations or all_concepts:
                primary = all_updates[0] if all_updates else UpdateType.NEW_RELATION
                return LearningResult(
                    input_text=text,
                    update_type=primary,
                    experience_id="",
                    concepts_created=list(dict.fromkeys(all_concepts)),
                    relations_created=list(dict.fromkeys(all_relations)),
                    message=f"Successfully learned: {', '.join(all_relations or all_concepts)}",
                )

        # Dialogue coreference resolution
        resolved_text = self.dialogue.resolve_anaphora_in_text(text)
        if resolved_text == text:
            resolved_text = SimpleParser.resolve_anaphora(
                text, self.last_subject, self.last_object
            )

        # Guard: If the text is recognized as a question or calculation, do NOT treat it as a statement to be learned!
        norm = SimpleParser.normalize_text(resolved_text)
        if (
            SimpleParser.parse_question(text) is not None
            or SimpleParser.parse_question(norm) is not None
            or SimpleParser.parse_question(resolved_text) is not None
            or text.strip().endswith("?")
            or resolved_text.strip().endswith("?")
        ):
            exp = self.memory.add_experience(input_text=text, extracted_triples=[])
            return LearningResult(
                input_text=text,
                update_type=UpdateType.NO_OP,
                experience_id=exp.id,
                message=f"Input was recognized as an inquiry or calculation, not a declarative statement: '{text}'",
            )

        # 0. Check procedural actions via Construction Grammar or fallback
        action = ConstructionEngine.parse_action_with_constructions(
            resolved_text, self.memory
        )
        if not action:
            action = SimpleParser.parse_action(resolved_text)

        if action:
            act_name, act_args = action
            if act_name.upper() == "SLICE":
                res = TransformationEngine.slice_object(
                    self.memory,
                    object_name=act_args["object"],
                    num_pieces=act_args["count"],
                )
                self.last_subject = act_args["object"]
                return LearningResult(
                    input_text=text,
                    update_type=UpdateType.NEW_ENTITY,
                    experience_id=res.entities_created[0].id
                    if res.entities_created
                    else "",
                    concepts_created=[res.slice_concept],
                    relations_created=res.relations_created,
                    message=res.message,
                )

        # 1. Parse statements: try dynamic Construction Grammar (with Open Pivot Learning) FIRST,
        # grounded in memory-stored constructions in SQLite rather than hardcoded Python regexes.
        # Fall back to SimpleParser for complex multi-clause/relative clause syntax.
        cxn_triples = ConstructionEngine.parse_with_constructions(
            resolved_text, self.memory
        )
        triples: list[ParsedTriple] = []
        if cxn_triples:
            for ct in cxn_triples:
                triples.append(
                    ParsedTriple(
                        subject=ct.subject,
                        predicate=ct.predicate,
                        object_=ct.object_,
                        is_property=ct.is_property,
                        is_negative=ct.is_negative,
                    )
                )
        else:
            triples = SimpleParser.parse_statement(resolved_text)

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
                relations_created.append(f"({subj_concept.name} {prop_key}={prop_val})")
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

        if triples:
            self.last_subject = triples[0].subject
            self.last_object = triples[0].object_
            ents = [triples[0].subject]
            if triples[0].object_:
                ents.append(triples[0].object_)
            self.dialogue.record_turn(speaker="user", text=text, entities=ents)
            self.dialogue.register_entity(name=triples[0].subject, role="subject")
            if triples[0].object_:
                self.dialogue.register_entity(name=triples[0].object_, role="object")

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
        if any(
            t.subject == "user" and t.predicate in ("name", "has_name")
            for t in triples
        ):
            u_t = next(
                t
                for t in triples
                if t.subject == "user" and t.predicate in ("name", "has_name")
            )
            user_c = self.memory.get_concept("user")
            if user_c:
                user_attrs = dict(user_c.attributes)
                user_attrs["name"] = u_t.object_.capitalize()
                self.memory.update_concept_attributes(user_c.id, user_attrs)
            return LearningResult(
                input_text=text,
                update_type=primary_update,
                experience_id=exp.id,
                concepts_created=concepts_created,
                relations_created=relations_created,
                message=f"Nice to meet you, {u_t.object_.capitalize()}! I have recorded your name in my persistent memory.",
            )

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
        # 0. Check for compound / conjoined questions: e.g. "Can an eagle fly and does it have wings?"
        q_clean = question.strip()
        sub_questions: list[str] = []
        if "?" in q_clean[:-1]:
            # Multiple questions with question marks: "Can an eagle fly? Does it have wings?"
            sub_questions = [
                s.strip() for s in re.split(r"\?\s*", q_clean) if s.strip()
            ]
        elif re.search(
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
            for sq in sub_questions:
                sq_clean = sq.strip()
                is_q = (
                    sq_clean.endswith("?")
                    or SimpleParser.parse_question(sq_clean) is not None
                    or SimpleParser.parse_question(f"{sq_clean}?") is not None
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
                if not is_q and SimpleParser.parse_statement(sq_clean):
                    learn_res = self.learn(sq_clean)
                    sub_results.append(
                        InferenceResult(
                            query=sq_clean,
                            status=BeliefStatus.SUPPORTED,
                            answer=learn_res.message,
                            confidence=1.0,
                            evidence=[
                                f"Learned: {', '.join(learn_res.relations_created or [learn_res.message])}"
                            ],
                            trace=[
                                "Learned declarative statement in compound conversational turn"
                            ],
                        )
                    )
                else:
                    sq_resolved = SimpleParser.resolve_anaphora(
                        sq_clean, self.last_subject, self.last_object
                    )
                    sq_formatted = (
                        sq_resolved if sq_resolved.endswith("?") else f"{sq_resolved}?"
                    )
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
            resolved_q = SimpleParser.resolve_anaphora(
                question, self.last_subject, self.last_object
            )
        known = {c.name.lower() for c in self.memory.list_concepts()}

        # 1. Check dynamic Construction Grammar in SQLite FIRST (zero hardcoded regexes)
        parsed = ConstructionEngine.parse_question_with_constructions(
            resolved_q, self.memory
        )
        if not parsed:
            parsed = ConstructionEngine.parse_question_with_constructions(
                question, self.memory
            )
        # 2. Fallback to SimpleParser for specialized queries (e.g. temporal ODEs) if needed
        if not parsed:
            parsed = SimpleParser.parse_question(resolved_q, known_concepts=known)
        if not parsed:
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

        # Update dialogue focus if valid concept
        if (
            SimpleParser.is_valid_concept(subj)
            and not pred.startswith("__")
            and pred != "little"
        ) or (pred == "__definition__" and SimpleParser.is_valid_concept(subj)):
            self.last_subject = subj
            self.dialogue.register_entity(name=subj, role="subject")
        if isinstance(target, str) and SimpleParser.is_valid_concept(target):
            self.last_object = target
            self.dialogue.register_entity(name=target, role="object")

        # User Identity query handler: "what is my name", "who am i", "do you know my name"
        if pred == "__user_name__":
            user_c = self.memory.get_concept("user")
            name = user_c.attributes.get("name") if user_c else None
            if not name and user_c:
                rels = self.memory.get_relations(subject_id=user_c.id)
                for r in rels:
                    if r.predicate in ("has_name", "name", "is"):
                        target_c = self.memory.get_concept(r.object_id)
                        if target_c:
                            name = target_c.name.capitalize()
                            break
            if name:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.SUPPORTED,
                    answer=f"Your name is {name}.",
                    confidence=1.0,
                    evidence=[f"User profile in persistent memory: name={name}"],
                    trace=["Retrieved user identity attribute from concept 'user'"],
                )
            else:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer="I do not know your name yet. What should I call you?",
                    confidence=0.0,
                    evidence=["User profile has no name recorded"],
                    trace=["Concept 'user' has no stored 'name' attribute"],
                )

        # Bot Name query handler: "what is your name", "tell me your name"
        if pred == "__identity_name__":
            return InferenceResult(
                query=question,
                status=BeliefStatus.SUPPORTED,
                answer="My name is LITTLE (Lightweight In-memory Transitive & Temporal Learning Engine).",
                confidence=1.0,
                evidence=["Self-identity specification"],
                trace=["Dialogue self-knowledge retrieval"],
            )

        # Chitchat query handlers:
        if pred == "__chitchat_greeting__":
            return InferenceResult(
                query=question,
                status=BeliefStatus.SUPPORTED,
                answer="Hello! I am LITTLE (Lightweight In-memory Transitive & Temporal Learning Engine). How can I help you reason, compute, or learn today?",
                confidence=1.0,
                evidence=["Conversational greeting acknowledged"],
                trace=["Dialogue interaction"],
            )

        if pred == "__chitchat_thanks__":
            return InferenceResult(
                query=question,
                status=BeliefStatus.SUPPORTED,
                answer="You're very welcome! I am always here to reason, learn new concepts, and compute with zero hallucination.",
                confidence=1.0,
                evidence=["Conversational pleasantry acknowledged"],
                trace=["Dialogue interaction"],
            )

        if pred == "__chitchat_how_are_you__":
            return InferenceResult(
                query=question,
                status=BeliefStatus.SUPPORTED,
                answer="I am operating at 100% nominal efficiency. Continuous memory is active, ODE states are stable, and my inference engine is ready to reason.",
                confidence=1.0,
                evidence=["Architecture operational status normal"],
                trace=["LITTLE Cognitive Architecture v0.1.0"],
            )

        if pred == "__chitchat_praise__":
            return InferenceResult(
                query=question,
                status=BeliefStatus.SUPPORTED,
                answer="Thank you! I strive for deterministic accuracy, continuous learning, and multi-hop logical precision.",
                confidence=1.0,
                evidence=["Positive reinforcement received"],
                trace=["Dialogue interaction"],
            )

        if pred == "__chitchat_fact__":
            relations = self.memory.get_relations()
            candidates = [
                r
                for r in relations
                if r.predicate
                in (
                    "is_a",
                    "located_in",
                    "has",
                    "can",
                    "made_of",
                    "lives_in",
                    "part_of",
                )
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
                if p == "is_a":
                    fact_str = f"Here is a verified fact from my memory: {s_art.capitalize()} {s_n} is {o_art} {o_n}."
                elif p == "located_in":
                    fact_str = f"Here is a verified fact from my memory: {s_n.capitalize()} is located in {o_n.capitalize()}."
                elif p == "has":
                    fact_str = f"Here is a verified fact from my memory: {s_n.capitalize()} has {o_art} {o_n}."
                elif p == "can":
                    fact_str = f"Here is a verified fact from my memory: {s_art.capitalize()} {s_n} can {o_n}."
                elif p == "made_of":
                    fact_str = f"Here is a verified fact from my memory: {s_n.capitalize()} is made of {o_n}."
                elif p == "lives_in":
                    fact_str = f"Here is a verified fact from my memory: {s_art.capitalize()} {s_n} lives in {o_n}."
                elif p == "part_of":
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
                    confidence=1.0,
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
                confidence=1.0,
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
                    confidence=1.0,
                    evidence=[f"User profile identity: name={user_name}"],
                    trace=["Matched query concept with recorded user name"],
                )

            if subj.lower() in ("little", "mivi", "mivi_model"):
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.SUPPORTED,
                    answer="I am LITTLE (Lightweight In-memory Transitive & Temporal Learning Engine), a continuous-learning cognitive architecture.",
                    confidence=1.0,
                    evidence=["Self-identity specification"],
                    trace=["Self-concept definition retrieval"],
                )

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
                if r.predicate == "disjoint_with"
                and self.memory.get_concept(r.object_id)
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

        # Property boolean query: "Is the apple red?", "Is it red?"
        if pred == "color" and target != "?":
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
            actual_color = attrs.get("color")
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
                        confidence=0.95,
                        evidence=[f"{subj} color is {target_str}"],
                        trace=[f"Matched property 'color'={target_str}"],
                    )
                else:
                    return InferenceResult(
                        query=question,
                        status=BeliefStatus.REFUTED,
                        answer=False,
                        confidence=0.95,
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
                confidence=0.1,
                evidence=[],
                trace=[f"No color recorded for '{subj}'."],
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
                    answer=f"The color of {subj} is {val_str}.",
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

        # WH-Location query: "Where is X?", "Where is X located?"
        if pred == "located_in" and target == "?":
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
            curr_id = concept.id
            loc_names: list[str] = []
            visited: set[str] = {curr_id}
            while True:
                rels = self.memory.get_relations(
                    subject_id=curr_id, predicate="located_in"
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
                    confidence=1.0,
                    evidence=[f"{subj} located_in {' -> '.join(loc_names)}"],
                    trace=["Resolved location hierarchy in semantic memory"],
                )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=0.1,
                evidence=[],
                trace=[f"No location recorded for '{subj}'."],
            )

        # WH-Habitat query: "Where does X live?"
        if pred == "lives_in" and target == "?":
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
            cand_ids = [concept.id] + self.inference.get_ancestor_ids(
                concept.id, predicate="is_a"
            )
            for cid in cand_ids:
                rels = self.memory.get_relations(subject_id=cid, predicate="lives_in")
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
                            confidence=0.95,
                            evidence=[f"{concept.name} lives_in {obj_c.name}{anc_str}"],
                            trace=[
                                f"Habitat resolved via knowledge graph: {pos_rels[0].id}"
                            ],
                        )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=0.1,
                evidence=[],
                trace=[f"No habitat recorded for '{subj}'."],
            )

        # WH-Diet query: "What does X eat?"
        if pred == "eats" and target == "?":
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
            cand_ids = [concept.id] + self.inference.get_ancestor_ids(
                concept.id, predicate="is_a"
            )
            for cid in cand_ids:
                rels = self.memory.get_relations(subject_id=cid, predicate="eats")
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
                            confidence=0.95,
                            evidence=[f"{concept.name} eats {obj_c.name}"],
                            trace=[
                                f"Diet resolved via knowledge graph: {pos_rels[0].id}"
                            ],
                        )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=0.1,
                evidence=[],
                trace=[f"No diet recorded for '{subj}'."],
            )

        # WH-Capability query: "What can X do?"
        if pred == "can" and target == "?":
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
            cand_ids = [concept.id] + self.inference.get_ancestor_ids(
                concept.id, predicate="is_a"
            )
            actions: list[str] = []
            for cid in cand_ids:
                rels = self.memory.get_relations(subject_id=cid, predicate="can")
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
                    confidence=0.95,
                    evidence=[f"{subj} can {', '.join(actions)}"],
                    trace=["Resolved capabilities from concept and ancestors"],
                )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=0.1,
                evidence=[],
                trace=[f"No capabilities recorded for '{subj}'."],
            )

        # WH-Material query: "What is X made of?"
        if pred == "made_of" and target == "?":
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
            rels = self.memory.get_relations(subject_id=concept.id, predicate="made_of")
            pos_rels = [r for r in rels if r.weight_positive > r.weight_negative]
            if pos_rels:
                obj_c = self.memory.get_concept(pos_rels[0].object_id)
                if obj_c:
                    ans_str = f"{subj.capitalize()} is made of {obj_c.name}."
                    return InferenceResult(
                        query=question,
                        status=BeliefStatus.SUPPORTED,
                        answer=ans_str,
                        confidence=0.95,
                        evidence=[f"{subj} made_of {obj_c.name}"],
                        trace=["Resolved composition from memory"],
                    )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=0.1,
                evidence=[],
                trace=[f"No composition recorded for '{subj}'."],
            )

        # WH-Features/Parts query: "What does X have?", "What parts does X have?"
        if pred == "has" and target == "?":
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
            cand_ids = [concept.id] + self.inference.get_ancestor_ids(
                concept.id, predicate="is_a"
            )
            has_items: list[str] = []
            for cid in cand_ids:
                for r in self.memory.get_relations(subject_id=cid, predicate="has"):
                    if r.weight_positive > r.weight_negative:
                        obj_c = self.memory.get_concept(r.object_id)
                        if obj_c and obj_c.name not in has_items:
                            has_items.append(obj_c.name)
                for r in self.memory.get_relations(object_id=cid, predicate="part_of"):
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
                    confidence=0.95,
                    evidence=[f"{subj} has {', '.join(has_items)}"],
                    trace=[
                        "Resolved features and parts via semantic memory and duality"
                    ],
                )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=0.1,
                evidence=[],
                trace=[f"No parts or features recorded for '{subj}'."],
            )

        # WH-Purpose query: "What is X used for?"
        if pred == "used_for" and target == "?":
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
            cand_ids = [concept.id] + self.inference.get_ancestor_ids(
                concept.id, predicate="is_a"
            )
            purposes: list[str] = []
            for cid in cand_ids:
                rels = self.memory.get_relations(subject_id=cid, predicate="used_for")
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
                    confidence=0.95,
                    evidence=[f"{subj} used_for {', '.join(purposes)}"],
                    trace=[
                        "Resolved usage and purpose via semantic memory and inheritance"
                    ],
                )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=0.1,
                evidence=[],
                trace=[f"No purpose or usage recorded for '{subj}'."],
            )

        # WH-Causality query: "What does X cause?"
        if pred == "causes" and target == "?":
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
            cand_ids = [concept.id] + self.inference.get_ancestor_ids(
                concept.id, predicate="is_a"
            )
            effects: list[str] = []
            for cid in cand_ids:
                rels = self.memory.get_relations(subject_id=cid, predicate="causes")
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
                    confidence=0.95,
                    evidence=[f"{subj} causes {', '.join(effects)}"],
                    trace=["Resolved causality from semantic memory"],
                )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=0.1,
                evidence=[],
                trace=[f"No effects recorded for '{subj}'."],
            )

        # WH-Reverse Causality: "What causes X?"
        if subj == "?" and pred == "causes" and isinstance(target, str):
            concept = self.memory.get_concept(target)
            if not concept:
                return InferenceResult(
                    query=question,
                    status=BeliefStatus.UNKNOWN,
                    answer=None,
                    confidence=0.0,
                    evidence=[],
                    trace=[f"Concept '{target}' is unknown."],
                )
            rels = self.memory.get_relations(object_id=concept.id, predicate="causes")
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
                    confidence=0.95,
                    evidence=[f"{', '.join(causes)} causes {target}"],
                    trace=["Resolved causes from semantic memory"],
                )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=0.1,
                evidence=[],
                trace=[f"No causes recorded for '{target}'."],
            )

        # WH-Properties query: "What properties does X have?"
        if pred == "has_property" and target == "?":
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
            cand_ids = [concept.id] + self.inference.get_ancestor_ids(
                concept.id, predicate="is_a"
            )
            props: list[str] = []
            for cid in cand_ids:
                rels = self.memory.get_relations(
                    subject_id=cid, predicate="has_property"
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
                    confidence=0.95,
                    evidence=[f"{subj} has_property {', '.join(props)}"],
                    trace=["Resolved properties from semantic memory"],
                )
            return InferenceResult(
                query=question,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=0.1,
                evidence=[],
                trace=[f"No properties recorded for '{subj}'."],
            )

        # Why-Questions:
        # 1. Why is X a Y?
        if pred == "__why_is_a__":
            res = self.inference.infer(subject=subj, predicate="is_a", target=target)
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
                    confidence=0.0,
                    evidence=[],
                    trace=res.trace,
                )

        # 2. Why is X located in Y?
        if pred == "__why_located_in__":
            res = self.inference.infer(
                subject=subj, predicate="located_in", target=target
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
            res = self.inference.infer(subject=subj, predicate="is_a", target=target)
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
                confidence=0.0,
                evidence=[],
                trace=res.trace,
            )

        # Standard relational query
        return self.inference.infer(subject=subj, predicate=pred, target=target)
