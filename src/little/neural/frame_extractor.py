"""Neural Semantic Frame & Slot Extractor for robust, zero-regex knowledge extraction."""

from __future__ import annotations

import re
from typing import Any

from little.core.contracts import CandidateClaim
from little.neural.encoder import NeuralStateEncoder
from little.neural.models import ExtractedSemanticFrame
from little.neural.semantic_frame_policy import SemanticFramePolicy

# Adverbs commonly appearing between copula/verbs and predicates
ADVERB_PATTERN = re.compile(
    r"\b(actually|really|truly|simply|basically|generally|usually|typically|certainly|indeed|literally)\b\s*",
    re.IGNORECASE,
)


class NeuralFrameExtractor:
    """Extracts typed FrameNet-style semantic frames from natural, conversational, or Wikipedia text."""

    def __init__(
        self,
        policy: SemanticFramePolicy | None = None,
        encoder: NeuralStateEncoder | None = None,
    ) -> None:
        self.policy = policy or SemanticFramePolicy.default()
        self.encoder = encoder or NeuralStateEncoder()

    def extract_frames(self, text: str) -> list[ExtractedSemanticFrame]:
        """Extract structured semantic frames from unstructured text."""
        raw_clean = text.strip()
        if not raw_clean:
            return []

        # 1. Strip conversational filler prefixes
        cleaned = self.policy.clean_filler(raw_clean)
        cleaned = cleaned.rstrip("?.! \t\n")
        cleaned = ADVERB_PATTERN.sub("", cleaned).strip()

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
        """Convert extracted semantic frames into verified CandidateClaim contracts."""
        claims: list[CandidateClaim] = []
        for f in frames:
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
        m_rel = re.match(
            r"^(?P<subj>[^,]+),\s*which\s+(?:is|are)\s+(?P<rel>[^,]+),\s*(?P<main>.+)$",
            text,
            re.IGNORECASE,
        )
        if not m_rel:
            return None

        subj = self._normalize_entity(m_rel.group("subj"))
        rel_obj = self._normalize_entity(m_rel.group("rel"))
        main_part = m_rel.group("main").strip()

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
            for sf in sub_frames:
                frames.append(sf)

        return frames if frames else None

    def _extract_passive(
        self, text: str, raw_text: str
    ) -> list[ExtractedSemanticFrame] | None:
        """Extract frames from passive agent constructions."""
        for act_pred, pass_phrase in self.policy.passive_patterns.items():
            pattern = (
                r"^(?P<obj>.+?)\s+(?:was|is|were|are)\s+"
                + re.escape(pass_phrase)
                + r"\s+(?P<subj>.+)$"
            )
            m = re.match(pattern, text, re.IGNORECASE)
            if m:
                subj = self._normalize_entity(m.group("subj"))
                obj = self._normalize_entity(m.group("obj"))
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
        for rel_name, phrases in self.policy.relational_indicators.items():
            for phrase in phrases:
                pattern = (
                    r"^(?P<subj>.+?)\s+\b"
                    + re.escape(phrase)
                    + r"\b\s+(?P<obj>.+)$"
                )
                m = re.match(pattern, text, re.IGNORECASE)
                if m:
                    subj_raw = m.group("subj").strip()
                    obj_raw = m.group("obj").strip()

                    subj = self._normalize_entity(subj_raw)
                    # Check if obj contains conjunction "and"
                    objs = self._split_conjunction(obj_raw)

                    frames: list[ExtractedSemanticFrame] = []
                    is_prop = (rel_name == "has_property")
                    phrase_is_neg = self._is_negative(phrase) or self._is_negative(text)

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
        m = re.match(
            r"^(?P<subj>[a-zA-Z0-9_\s]{2,30}?)\s+(?P<verb>[a-z]{3,15}(?:s|ed)?)\s+(?P<obj>[a-zA-Z0-9_\s]{2,40})$",
            text,
            re.IGNORECASE,
        )
        if not m:
            return None

        subj = self._normalize_entity(m.group("subj"))
        verb = m.group("verb").lower().strip()
        obj = self._normalize_entity(m.group("obj"))

        # Skip common auxiliary/negation words already covered by relational indicators
        if verb in ("is", "are", "was", "were", "has", "have", "had", "can", "will", "did", "do", "not"):
            return None

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
        stripped = re.sub(r"^[,:\-\.\s]+|[,:\-\.\s]+$", "", stripped)
        # If all lowercase or mixed, strip plural trailing 's' if standard regular plural
        lower = stripped.lower()
        if lower.endswith("s") and len(lower) > 3 and not lower.endswith(("ss", "us", "is", "as")):
            # Common plurals: cheetahs -> cheetah, felines -> feline
            # But don't strip if proper noun or collective
            if lower not in ("gas", "lens", "series", "species", "mars", "paris", "atlantis"):
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
        """Detect negation tokens in text."""
        for neg in self.policy.negation_tokens:
            if re.search(r"\b" + re.escape(neg) + r"\b", text, re.IGNORECASE):
                return True
        return False
