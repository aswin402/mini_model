"""Dialogue Context and Anaphora / Coreference Resolution Engine.

Maintains multi-turn conversational state, tracks entity salience across speaker turns,
and resolves pronouns ('it', 'they', 'that', 'the former', 'the latter') without statistical guessing.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class SalientEntity:
    """Represents an entity mentioned in conversational dialogue with syntactic/semantic tags."""

    name: str
    raw_text: str = ""
    category: Optional[str] = None
    is_animate: bool = False
    is_person: bool = False
    is_plural: bool = False
    role: str = "subject"  # "subject", "object", "topic"
    turn_index: int = 0
    timestamp: float = field(default_factory=time.time)
    salience_score: float = 1.0


@dataclass
class DialogueTurn:
    """Represents a single conversational turn between user and agent."""

    turn_index: int
    speaker: str  # "user" | "little"
    text: str
    entities: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class DialogueContext:
    """Manages conversational dialogue state and deterministic pronoun resolution."""

    def __init__(self) -> None:
        self.turns: List[DialogueTurn] = []
        self.entity_stack: List[SalientEntity] = []
        self.active_subject: Optional[str] = None
        self.active_object: Optional[str] = None
        self.current_turn_index: int = 0

    def record_turn(
        self,
        speaker: str,
        text: str,
        entities: Optional[List[str]] = None,
    ) -> DialogueTurn:
        """Log a dialogue turn and advance conversation index."""
        self.current_turn_index += 1
        turn = DialogueTurn(
            turn_index=self.current_turn_index,
            speaker=speaker,
            text=text,
            entities=entities or [],
        )
        self.turns.append(turn)
        return turn

    def register_entity(
        self,
        name: str,
        raw_text: str = "",
        category: Optional[str] = None,
        is_animate: bool = False,
        is_person: bool = False,
        is_plural: bool = False,
        role: str = "subject",
    ) -> SalientEntity:
        """Register a salient concept/entity into the conversation recency stack."""
        clean_name = name.strip().lower()

        # Decay prior salience scores slightly with each new entity introduction
        for e in self.entity_stack:
            e.salience_score *= 0.85

        entity = SalientEntity(
            name=clean_name,
            raw_text=raw_text or name,
            category=category,
            is_animate=is_animate,
            is_person=is_person,
            is_plural=is_plural,
            role=role,
            turn_index=self.current_turn_index,
            salience_score=1.0 if role == "subject" else 0.8,
        )

        self.entity_stack.append(entity)
        if role == "subject":
            self.active_subject = clean_name
        elif role == "object":
            self.active_object = clean_name

        # Associate with the active turn if present
        if self.turns:
            if clean_name not in self.turns[-1].entities:
                self.turns[-1].entities.append(clean_name)

        return entity

    def get_salient_subject(self) -> Optional[str]:
        """Return the most salient subject concept."""
        if self.active_subject:
            return self.active_subject
        for e in reversed(self.entity_stack):
            if e.role == "subject":
                return e.name
        return self.entity_stack[-1].name if self.entity_stack else None

    def resolve_pronoun(self, pronoun: str) -> Optional[str]:
        """Deterministically resolve a pronoun or referring expression to its salient antecedent."""
        p_clean = pronoun.strip().lower()

        # 1. Structural references: 'the former', 'the latter'
        if p_clean in ("the former", "former"):
            # Find the most recent turn with at least 2 distinct entities
            for turn in reversed(self.turns):
                if len(turn.entities) >= 2:
                    return turn.entities[0]
            # Fallback to entity stack
            if len(self.entity_stack) >= 2:
                # Find entities from the same turn
                last_turn = self.entity_stack[-1].turn_index
                turn_entities = [e.name for e in self.entity_stack if e.turn_index == last_turn]
                if len(turn_entities) >= 2:
                    return turn_entities[0]
                return self.entity_stack[-2].name
            return None

        if p_clean in ("the latter", "latter"):
            for turn in reversed(self.turns):
                if len(turn.entities) >= 2:
                    return turn.entities[-1]
            if len(self.entity_stack) >= 2:
                return self.entity_stack[-1].name
            return None

        # 2. Plural pronouns: 'they', 'them', 'their', 'theirs', 'these', 'those'
        if p_clean in ("they", "them", "their", "theirs", "these", "those"):
            for e in reversed(self.entity_stack):
                if e.is_plural:
                    return e.name
            # Fallback to collective set if multiple entities exist in last turn
            if self.turns and len(self.turns[-1].entities) >= 2:
                return self.turns[-1].entities[-1]
            return None

        # 3. Person-specific pronouns: 'he', 'him', 'his', 'she', 'her'
        if p_clean in ("he", "him", "his", "she", "her", "hers"):
            for e in reversed(self.entity_stack):
                if e.role == "subject" and (e.is_person or e.category in ("person", "human", "character")):
                    return e.name
            for e in reversed(self.entity_stack):
                if e.is_person or e.category in ("person", "human", "character"):
                    return e.name
            return None

        # 4. Singular non-human pronouns: 'it', 'its', 'that', 'this'
        if p_clean in ("it", "its", "that", "this"):
            # Centering Theory: Prioritize subject role (topic) of the most recent turns
            for e in reversed(self.entity_stack):
                if e.role == "subject" and not e.is_plural and not e.is_person:
                    return e.name
            if self.active_subject:
                return self.active_subject
            # Fallback to non-subject entity if no subject is found
            for e in reversed(self.entity_stack):
                if not e.is_plural and not e.is_person:
                    return e.name
            if self.active_object:
                return self.active_object

        return None

    def resolve_anaphora_in_text(self, text: str) -> str:
        """Replace referring pronouns in text with resolved antecedent nouns."""
        s = text.strip()
        if re.match(r"^if\s+", s, re.IGNORECASE):
            return text

        words = text.split()
        resolved_words: List[str] = []

        # Multi-word referring expressions first
        t_mod = text
        if "the former" in t_mod.lower():
            resolved = self.resolve_pronoun("the former")
            if resolved:
                t_mod = re.sub(r"\bthe former\b", resolved, t_mod, flags=re.IGNORECASE)

        if "the latter" in t_mod.lower():
            resolved = self.resolve_pronoun("the latter")
            if resolved:
                t_mod = re.sub(r"\bthe latter\b", resolved, t_mod, flags=re.IGNORECASE)

        # Single word pronoun replacement
        def replace_token(match: re.Match) -> str:
            token = match.group(0)
            t_lower = token.lower()
            start = match.start()
            end = match.end()

            # Guard: Do not replace 'that', 'these', 'those' when acting as relative pronouns or determiners
            if t_lower in ("that", "these", "those"):
                prefix = t_mod[:start].strip()
                if prefix:
                    last_word = prefix.split()[-1].lower()
                    if last_word not in (
                        "is", "are", "was", "were", "do", "does", "did",
                        "if", "and", "or", "but", "so", "than", "as", "like",
                    ):
                        # Preceded by a content noun -> relative clause (e.g. "mammals that live")
                        return token
                suffix = t_mod[end:].strip()
                if suffix:
                    first_word = suffix.split()[0].lower()
                    if first_word not in (
                        "is", "are", "was", "were", "has", "have", "can",
                        "will", "would", "do", "does", "did", "in", "on", "at", "?", ".",
                    ):
                        # Followed by a noun -> determiner (e.g. "that animal")
                        return token

            if t_lower in ("it", "they", "them", "that", "these", "those"):
                res = self.resolve_pronoun(t_lower)
                if res:
                    return res if token.islower() else res.capitalize()
            return token

        pattern = r"\b(it|they|them|that|these|those)\b"
        resolved_text = re.sub(pattern, replace_token, t_mod, flags=re.IGNORECASE)
        return resolved_text
