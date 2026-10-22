"""Deterministic policy-based semantic frame extraction without regular expressions."""

from __future__ import annotations

from little.core.contracts import CandidateClaim
from little.neural.encoder import NeuralStateEncoder
from little.neural.models import ExtractedSemanticFrame
from little.neural.semantic_frame_policy import SemanticFramePolicy

# Adverbs commonly appearing between copula/verbs and predicates
ADVERB_WORDS: frozenset[str] = frozenset({
    "actually",
    "really",
    "truly",
    "simply",
    "basically",
    "generally",
    "usually",
    "typically",
    "certainly",
    "indeed",
    "literally",
})


def _strip_adverbs(text: str) -> str:
    """Remove filler adverbs at word boundaries without regular expressions."""
    segments = text.split(",")
    cleaned_segments: list[str] = []
    for seg in segments:
        kept = [w for w in seg.split() if w.lower().strip("?.!,:;") not in ADVERB_WORDS]
        cleaned_segments.append(" ".join(kept))
    return ", ".join(cleaned_segments).strip()


class NeuralFrameExtractor:
    """Extracts candidate semantic frames with policy phrase rules and bounded regex patterns."""

    def __init__(
        self,
        policy: SemanticFramePolicy | None = None,
        encoder: NeuralStateEncoder | None = None,
    ) -> None:
        self.policy = policy or SemanticFramePolicy.default()
        # Kept for constructor compatibility; extraction does not use embeddings.
        self.encoder = encoder

    def extract_frames(self, text: str) -> list[ExtractedSemanticFrame]:
        """Extract structured semantic frames from unstructured text."""
        raw_clean = text.strip()
        if not raw_clean:
            return []

        # 1. Strip conversational filler prefixes
        cleaned = self.policy.clean_filler(raw_clean)
        cleaned = cleaned.rstrip("?.! \t\n")
        cleaned = _strip_adverbs(cleaned)

        if not cleaned:
            return []

        frames: list[ExtractedSemanticFrame] = []

        # 2. Check relative clause parentheticals: "Cheetahs, which are felines, run fast"
        rel_frames = self._extract_relative_clause(cleaned, raw_clean)
        if rel_frames:
            return rel_frames

        # 3. Check passive constructions: "Euclidean geometry was discovered by Euclid"
        pass_frames = self._extract_passive(cleaned, raw_clean)
        if pass_frames:
            return pass_frames

        # 4. Check relational indicators configured in schema
        rel_ind_frames = self._extract_relational_indicators(cleaned, raw_clean)
        if rel_ind_frames:
            return rel_ind_frames

        # 5. Check direct transitive verbs: "Earth orbits Sun", "Enzymes catalyze reactions"
        verb_frames = self._extract_transitive_verb(cleaned, raw_clean)
        if verb_frames:
            return verb_frames

        return frames

    def to_candidate_claims(
        self, frames: list[ExtractedSemanticFrame]
    ) -> list[CandidateClaim]:
        """Convert deterministic extraction candidates into claims; verification is external."""
        claims: list[CandidateClaim] = []
        for f in frames:
            if f.confidence < self.policy.min_extraction_confidence:
                continue
            claims.append(
                CandidateClaim(
                    subject=f.subject,
                    predicate=f.predicate,
                    object=f.object_val,
                    probability=round(f.confidence, 4),
                    positive=not f.is_negative,
                    is_property=f.is_property,
                    attributes=f.attributes,
                )
            )
        return claims

    def extract_candidate_claims(self, text: str) -> list[CandidateClaim]:
        """Convenience method directly returning CandidateClaim list."""
        frames = self.extract_frames(text)
        return self.to_candidate_claims(frames)

    def _extract_relative_clause(
        self, text: str, raw_text: str
    ) -> list[ExtractedSemanticFrame] | None:
        """Extract frames from appositive/parenthetical relative clauses."""
        parts = [p.strip() for p in text.split(",")]
        if len(parts) < 3 or not parts[0] or not parts[1]:
            return None
        rel_clause = parts[1]
        rel_lower = rel_clause.lower()
        if rel_lower.startswith("which is "):
            rel_raw = rel_clause[9:].strip()
        elif rel_lower.startswith("which are "):
            rel_raw = rel_clause[10:].strip()
        else:
            return None
        main_part = ", ".join(parts[2:]).strip()
        if not main_part:
            return None

        subj = self._normalize_entity(parts[0])
        rel_obj = self._normalize_entity(rel_raw)

        frames: list[ExtractedSemanticFrame] = []
        if subj and rel_obj and subj != rel_obj:
            frames.append(
                ExtractedSemanticFrame(
                    frame_type="taxonomic_subsumption",
                    subject=subj,
                    predicate="is_a",
                    object_val=rel_obj,
                    confidence=0.90,
                    raw_text=raw_text,
                )
            )

        # Parse the main part: e.g. "run fast" or "has an ocean"
        if main_part:
            main_text = f"{subj} {main_part}"
            sub_frames = self.extract_frames(main_text)
            frames.extend(sub_frames)

        return frames if frames else None

    def _extract_passive(
        self, text: str, raw_text: str
    ) -> list[ExtractedSemanticFrame] | None:
        """Extract frames from passive agent constructions."""
        words = text.split()
        lower_words = [w.lower() for w in words]
        n = len(words)
        for act_pred, pass_phrase in self.policy.passive_patterns.items():
            p_words = pass_phrase.lower().split()
            plen = len(p_words)
            if n < plen + 3:
                continue
            for i in range(1, n - plen - 1):
                if (
                    lower_words[i] in ("was", "is", "were", "are")
                    and lower_words[i + 1 : i + 1 + plen] == p_words
                ):
                    obj = self._normalize_entity(" ".join(words[:i]))
                    subj = self._normalize_entity(" ".join(words[i + 1 + plen :]))
                    if subj and obj and subj != obj:
                        return [
                            ExtractedSemanticFrame(
                                frame_type="agentive_action",
                                subject=subj,
                                predicate=act_pred,
                                object_val=obj,
                                confidence=0.88,
                                raw_text=raw_text,
                            )
                        ]
        return None

    def _extract_relational_indicators(
        self, text: str, raw_text: str
    ) -> list[ExtractedSemanticFrame] | None:
        """Extract frames using schema-configured relational indicator phrases."""
        words = text.split()
        lower_words = [w.lower().strip(",:;.!?") for w in words]
        n = len(words)
        for rel_name, phrases in self.policy.relational_indicators.items():
            for phrase in phrases:
                p_words = phrase.lower().split()
                plen = len(p_words)
                if plen == 0 or n < plen + 2:
                    continue
                for i in range(1, n - plen):
                    if lower_words[i : i + plen] == p_words:
                        subj_raw = " ".join(words[:i]).strip()
                        obj_raw = " ".join(words[i + plen :]).strip()

                        subj = self._normalize_entity(subj_raw)
                        objs = self._split_conjunction(obj_raw)

                        frames: list[ExtractedSemanticFrame] = []
                        is_prop = rel_name == "has_property"
                        phrase_is_neg = self._is_negative(phrase)

                        for o in objs:
                            norm_o = self._normalize_entity(o)
                            if subj and norm_o and subj != norm_o:
                                frames.append(
                                    ExtractedSemanticFrame(
                                        frame_type=f"{rel_name}_frame",
                                        subject=subj,
                                        predicate=rel_name,
                                        object_val=norm_o,
                                        is_negative=phrase_is_neg,
                                        is_property=is_prop,
                                        confidence=0.85,
                                        raw_text=raw_text,
                                    )
                                )
                        if frames:
                            return frames
        return None

    def _extract_transitive_verb(
        self, text: str, raw_text: str
    ) -> list[ExtractedSemanticFrame] | None:
        """Extract frames with direct subject-verb-object structures."""
        if not all(c.isalnum() or c in "_ " for c in text):
            return None
        words = text.split()
        if len(words) < 3:
            return None

        skip_verbs = {
            "is",
            "are",
            "was",
            "were",
            "has",
            "have",
            "had",
            "can",
            "will",
            "did",
            "do",
            "not",
        }
        for i in range(1, len(words) - 1):
            subj_raw = " ".join(words[:i])
            verb = words[i].lower().strip()
            obj_raw = " ".join(words[i + 1 :])
            if not (2 <= len(subj_raw) <= 30 and 2 <= len(obj_raw) <= 40):
                continue
            if not (verb.isalpha() and 3 <= len(verb) <= 17):
                continue
            if verb in skip_verbs or verb in self.policy.stop_determiners:
                continue
            subj = self._normalize_entity(subj_raw)
            obj = self._normalize_entity(obj_raw)
            if subj and obj and subj != obj:
                return [
                    ExtractedSemanticFrame(
                        frame_type="transitive_action",
                        subject=subj,
                        predicate=verb,
                        object_val=obj,
                        confidence=0.75,
                        raw_text=raw_text,
                    )
                ]
        return None

    def _normalize_entity(self, phrase: str) -> str:
        """Clean determiners and whitespace while normalizing casing."""
        stripped = self.policy.strip_determiner(phrase.strip())
        stripped = stripped.strip(",:-. \t\n\r")
        # If all lowercase or mixed, strip plural trailing 's' if standard regular plural
        lower = stripped.lower()
        # Preserve known nonplural names while supporting legacy regular plurals.
        if (
            lower.endswith("s")
            and len(lower) > 3
            and not lower.endswith(("ss", "us", "is", "as"))
            and lower
            not in (
                "gas",
                "lens",
                "series",
                "species",
                "mars",
                "paris",
                "atlantis",
                "physics",
                "mathematics",
                "economics",
                "politics",
                "news",
            )
        ):
            lower = lower[:-1]
        return lower

    def _split_conjunction(self, obj_phrase: str) -> list[str]:
        """Split clean noun conjunctions (e.g. 'hydrogen and oxygen')."""
        clean = obj_phrase.strip()
        if " and " in clean and not any(w in clean for w in ("between", "both")):
            parts = clean.split(" and ")
            if len(parts) == 2 and all(len(p.strip().split()) <= 3 for p in parts):
                return [p.strip() for p in parts]
        return [clean]

    def _is_negative(self, text: str) -> bool:
        """Detect negation tokens in text without regular expressions."""
        words = [w.strip(",:;.!?") for w in text.lower().split() if w.strip(",:;.!?")]
        n = len(words)
        for neg in self.policy.negation_tokens:
            neg_words = neg.lower().split()
            k = len(neg_words)
            if k > 0 and any(words[idx : idx + k] == neg_words for idx in range(n - k + 1)):
                return True
        return False
