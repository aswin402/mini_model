"""Dialogue Context and Anaphora / Coreference Resolution Engine.

Maintains multi-turn conversational state, tracks entity salience across speaker turns,
and resolves pronouns ('it', 'they', 'that', 'the former', 'the latter') without statistical guessing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from little.language.dialogue_policy import DialogueReferencePolicy


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _matches_prefix_guard(text: str, pattern: str) -> bool:
    if not pattern:
        return False
    s = text.strip()
    if pattern == r"^if\s+":
        return len(s) > 2 and s[:2].lower() == "if" and s[2].isspace()
    clean_pat = pattern.lstrip("^")
    if clean_pat.endswith(r"\s+"):
        kw = clean_pat[:-3].strip()
        return (
            bool(kw)
            and len(s) > len(kw)
            and s[: len(kw)].lower() == kw.lower()
            and s[len(kw)].isspace()
        )
    return s.lower().startswith(clean_pat.lower())


def _find_word_phrase_ci(text: str, phrase: str, start_idx: int = 0) -> int:
    if not text or not phrase:
        return -1
    t_low = text.lower()
    p_low = phrase.lower()
    p_len = len(p_low)
    n = len(text)
    i = start_idx
    while i <= n - p_len:
        idx = t_low.find(p_low, i)
        if idx == -1:
            return -1
        end = idx + p_len
        left_ok = idx == 0 or not _is_word_char(text[idx - 1])
        right_ok = end == n or not _is_word_char(text[end])
        if left_ok and right_ok:
            return idx
        i = idx + 1
    return -1


def _replace_word_phrase_ci(text: str, phrase: str, replacement: str) -> str:
    if not text or not phrase:
        return text
    p_len = len(phrase)
    out: list[str] = []
    last = 0
    idx = _find_word_phrase_ci(text, phrase, 0)
    while idx != -1:
        out.append(text[last:idx])
        out.append(replacement)
        last = idx + p_len
        idx = _find_word_phrase_ci(text, phrase, last)
    out.append(text[last:])
    return "".join(out)


@dataclass
class SalientEntity:
    """Represents an entity mentioned in conversational dialogue with syntactic/semantic tags."""

    name: str
    raw_text: str = ""
    category: str | None = None
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
    entities: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class DialogueContext:
    """Manages conversational dialogue state and deterministic pronoun resolution."""

    def __init__(self, policy: DialogueReferencePolicy | None = None) -> None:
        self.policy = policy if policy is not None else DialogueReferencePolicy.default()
        self.turns: list[DialogueTurn] = []
        self.entity_stack: list[SalientEntity] = []
        self.active_subject: str | None = None
        self.active_object: str | None = None
        self.current_turn_index: int = 0

    def record_turn(
        self,
        speaker: str,
        text: str,
        entities: list[str] | None = None,
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
        category: str | None = None,
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
        if self.turns and clean_name not in self.turns[-1].entities:
            self.turns[-1].entities.append(clean_name)

        return entity

    def get_salient_subject(self) -> str | None:
        """Return the most salient subject concept."""
        if self.active_subject:
            return self.active_subject
        for e in reversed(self.entity_stack):
            if e.role == "subject":
                return e.name
        return self.entity_stack[-1].name if self.entity_stack else None

    def resolve_pronoun(self, pronoun: str) -> str | None:
        """Deterministically resolve a pronoun or referring expression to its salient antecedent."""
        p_clean = pronoun.strip().lower()

        # 1. Structural references: 'the former', 'the latter'
        if p_clean in self.policy.former_references:
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

        if p_clean in self.policy.latter_references:
            for turn in reversed(self.turns):
                if len(turn.entities) >= 2:
                    return turn.entities[-1]
            if len(self.entity_stack) >= 2:
                return self.entity_stack[-1].name
            return None

        # 2. Plural pronouns: 'they', 'them', 'their', 'theirs', 'these', 'those'
        if p_clean in self.policy.plural_pronouns:
            for e in reversed(self.entity_stack):
                if e.is_plural:
                    return e.name
            # Fallback to salient subject for generic plural or collective reference
            if self.active_subject:
                return self.active_subject
            for e in reversed(self.entity_stack):
                if e.role == "subject":
                    return e.name
            if self.turns and self.turns[-1].entities:
                return self.turns[-1].entities[0]
            if self.entity_stack:
                return self.entity_stack[-1].name
            return None

        # 3. Person-specific pronouns: 'he', 'him', 'his', 'she', 'her'
        if p_clean in self.policy.person_pronouns:
            for e in reversed(self.entity_stack):
                if e.role == "subject" and (
                    e.is_person or e.category in self.policy.person_categories
                ):
                    return e.name
            for e in reversed(self.entity_stack):
                if e.is_person or e.category in self.policy.person_categories:
                    return e.name
            return None

        # 4. Singular non-human pronouns: 'it', 'its', 'that', 'this'
        if p_clean in self.policy.singular_nonhuman_pronouns:
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
        if _matches_prefix_guard(s, self.policy.conditional_guard_pattern):
            return text

        # Multi-word referring expressions first
        t_mod = text
        for references in (
            self.policy.former_references,
            self.policy.latter_references,
        ):
            for reference in references:
                if _find_word_phrase_ci(t_mod, reference) != -1:
                    resolved = self.resolve_pronoun(reference)
                    if resolved:
                        t_mod = _replace_word_phrase_ci(t_mod, reference, resolved)
                    break

        if not self.policy.text_pronouns:
            return t_mod

        text_pronouns_lower = {p.lower() for p in self.policy.text_pronouns}
        relative_pronouns_lower = {p.lower() for p in self.policy.relative_pronouns}

        out: list[str] = []
        n = len(t_mod)
        i = 0
        while i < n:
            if _is_word_char(t_mod[i]):
                start = i
                while i < n and _is_word_char(t_mod[i]):
                    i += 1
                end = i
                token = t_mod[start:end]
                t_lower = token.lower()

                if t_lower in text_pronouns_lower:
                    skip = False
                    if t_lower in relative_pronouns_lower:
                        prefix = t_mod[:start].strip()
                        if prefix:
                            last_word = prefix.split()[-1].lower()
                            if last_word not in self.policy.relative_prefix_exclusions:
                                skip = True
                        if not skip:
                            suffix = t_mod[end:].strip()
                            if suffix:
                                first_word = suffix.split()[0].lower()
                                if first_word not in self.policy.relative_suffix_exclusions:
                                    skip = True
                    if not skip:
                        res = self.resolve_pronoun(t_lower)
                        if res:
                            out.append(res if token.islower() else res.capitalize())
                            continue
                out.append(token)
            else:
                out.append(t_mod[i])
                i += 1
        return "".join(out)
