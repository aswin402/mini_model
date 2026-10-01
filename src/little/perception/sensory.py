"""Multimodal Sensory Grounding Engine (0° Episodic Instance Axis)."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from little.core.concept_knot import ContinuousPhysicalState
from little.core.models import Entity
from little.memory.store import MemoryStore


@dataclass
class SensoryObservation:
    """Structured sensory perception reading from one or more modalities."""

    modality: str
    features: dict[str, Any]
    source: str = "sensor"
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0


@dataclass
class GroundingResult:
    """Result of anchoring a sensory observation to a grounded Concept Knot."""

    observation: SensoryObservation
    matched_concept: str | None
    confidence: float
    entity_id: str | None
    episodic_instance_knot: dict[str, Any] = field(default_factory=dict)
    status: str = "UNKNOWN"  # "GROUNDED", "UNKNOWN", "MUTEX_REJECTED"
    reason: str = ""


# Typical biological / physical mass bounds (in grams) for invariant gating
_PHYSICAL_MASS_BOUNDS: dict[str, tuple[float, float]] = {
    "apple": (30.0, 1000.0),
    "orange": (40.0, 1200.0),
    "banana": (40.0, 800.0),
    "fruit": (1.0, 20000.0),
    "bird": (2.0, 150000.0),
    "falcon": (500.0, 2000.0),
    "hawk": (400.0, 2500.0),
    "eagle": (1500.0, 8000.0),
    "penguin": (1000.0, 50000.0),
    "dog": (500.0, 120000.0),
    "cat": (500.0, 15000.0),
    "fish": (1.0, 500000.0),
}


class SensoryGroundingEngine:
    """Grounds continuous multimodal sensory perceptions to discrete concept knots and episodic instances."""

    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory

    def parse_sensory_directive(self, text: str) -> SensoryObservation | None:
        """Parse natural language or structured directive into a SensoryObservation."""
        clean = text.strip()
        pattern = re.compile(
            r"^(?:sensory\s+observation|observe\s+sensor|perceive\s+sensor):\s*\[?(?P<body>[^\]]+)\]?",
            re.IGNORECASE,
        )
        match = pattern.search(clean)
        if not match:
            return None

        body = match.group("body").strip()
        features: dict[str, Any] = {}

        # Parse key-value pairs (comma or semicolon separated)
        pairs = re.split(r"[,;]", body)
        for pair in pairs:
            if ":" not in pair and "=" not in pair:
                continue
            delim = ":" if ":" in pair else "="
            k, v = pair.split(delim, 1)
            key = k.strip().lower()
            val_str = v.strip()

            # Normalize keys
            if key in ("mass", "weight"):
                key = "mass_grams"
            elif key in ("temp", "temperature"):
                key = "temperature_c"

            # Parse numeric values with units
            val_num_match = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*(?:g|grams?|kg|°?c|celsius)?$", val_str, re.IGNORECASE)
            if val_num_match:
                num = float(val_num_match.group(1))
                if "kg" in val_str.lower():
                    num *= 1000.0
                features[key] = num
            else:
                features[key] = val_str.lower().strip("\"'")

        return SensoryObservation(
            modality="multimodal",
            features=features,
            source="sensor",
            timestamp=time.time(),
        )

    def ground_observation(self, observation: SensoryObservation) -> GroundingResult:
        """Ground observation against knowledge store with invariant bounds checking."""
        features = observation.features

        # 1. Check explicit or candidate concept
        candidate_name = features.get("candidate")
        mass_grams = features.get("mass_grams")

        # Invariant Bound Check: Physical Mass
        if candidate_name and mass_grams is not None:
            bounds = _PHYSICAL_MASS_BOUNDS.get(candidate_name.lower())
            if bounds:
                min_m, max_m = bounds
                if not (min_m <= mass_grams <= max_m):
                    return GroundingResult(
                        observation=observation,
                        matched_concept=candidate_name,
                        confidence=0.0,
                        entity_id=None,
                        status="MUTEX_REJECTED",
                        reason=f"Physical mass bound violation: {candidate_name} cannot have mass {mass_grams:.1f}g (valid range: [{min_m}, {max_m}]).",
                    )

        # 2. Match against concepts in memory
        color = features.get("color")
        category = features.get("category")
        best_match: str | None = None
        best_score = 0.0

        concepts = self.memory.list_concepts()
        for c in concepts:
            c_name = c.name.lower()
            score = 0.0

            # Direct name/candidate match
            if candidate_name and candidate_name.lower() == c_name:
                score += 3.0

            # Category / Taxonomic match
            if category:
                if c.category and category in c.category.lower():
                    score += 2.0
                # Check is_a relations
                for r in self.memory.get_relations(subject_id=c.id):
                    if r.predicate in ("is_a", "subclass_of"):
                        obj = self.memory.get_concept(r.object_id)
                        if obj and category in obj.name.lower():
                            score += 2.0

            # Color / Visual feature match
            if color:
                if c.attributes.get("color", "").lower() == color:
                    score += 1.5
                for r in self.memory.get_relations(subject_id=c.id):
                    if r.predicate == "color":
                        obj = self.memory.get_concept(r.object_id)
                        if obj and color in obj.name.lower():
                            score += 1.5

            # Mass plausibility
            if mass_grams is not None:
                bounds = _PHYSICAL_MASS_BOUNDS.get(c_name)
                if bounds:
                    min_m, max_m = bounds
                    if min_m <= mass_grams <= max_m:
                        score += 1.0
                    else:
                        score -= 5.0  # Penalize severe bound violations

            if score > best_score:
                best_score = score
                best_match = c.name

        if not best_match or best_score < 1.5:
            # Fallback to general physical entity or unknown
            return GroundingResult(
                observation=observation,
                matched_concept=best_match,
                confidence=0.2,
                entity_id=None,
                status="UNKNOWN",
                reason="Sensory features insufficient or ambiguous for deterministic grounding.",
            )

        # 3. Ground entity instance at 0° Episodic axis
        matched_c = self.memory.get_concept(best_match)
        if not matched_c:
            return GroundingResult(
                observation=observation,
                matched_concept=best_match,
                confidence=0.0,
                entity_id=None,
                status="UNKNOWN",
            )

        # Final invariant bounds check on the selected best match
        if mass_grams is not None:
            bounds = _PHYSICAL_MASS_BOUNDS.get(best_match.lower())
            if bounds and not (bounds[0] <= mass_grams <= bounds[1]):
                return GroundingResult(
                    observation=observation,
                    matched_concept=best_match,
                    confidence=0.0,
                    entity_id=None,
                    status="MUTEX_REJECTED",
                    reason=f"Physical mass bound violation: {best_match} cannot have mass {mass_grams:.1f}g.",
                )

        entity_name = f"{matched_c.name}_instance_{int(time.time() * 1000)}"
        entity = self.memory.create_entity(
            name=entity_name,
            concept_id=matched_c.id,
            properties={
                "sensory_features": features,
                "modality": observation.modality,
                "confidence": min(1.0, round(best_score / 4.0, 2)),
                "physical_state": {
                    "freshness": 1.0,
                    "oxidation": 0.0,
                    "moisture": 0.86 if "fruit" in (category or "") or best_match == "apple" else 0.5,
                    "temperature": features.get("temperature_c", 20.0),
                    "mass_grams": mass_grams or 180.0,
                },
            },
        )

        # 4. Record grounded episodic instance in memory
        self.memory.add_experience(
            input_text=f"Observed grounded instance of {matched_c.name} via {observation.modality} sensor.",
            extracted_triples=[
                {"subject": entity.name, "predicate": "instance_of", "object": matched_c.name},
                {"subject": entity.name, "predicate": "observed_at", "object": "sensor_stream"},
            ],
            source="sensor",
        )

        episodic_knot = {
            "entity_id": entity.id,
            "entity_name": entity.name,
            "concept": matched_c.name,
            "axis": "0° Grounded Episodic Instance",
            "features": features,
        }

        return GroundingResult(
            observation=observation,
            matched_concept=matched_c.name,
            confidence=min(1.0, round(best_score / 4.0, 2)),
            entity_id=entity.id,
            episodic_instance_knot=episodic_knot,
            status="GROUNDED",
            reason=f"Successfully grounded observation to concept '{matched_c.name}' at 0° episodic axis.",
        )
