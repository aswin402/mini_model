"""LITTLE — Continual Concept Learning & Cognitive Architecture.

A modular AI architecture decoupling reasoning mechanisms from persistent,
inspectable episodic, semantic, and procedural memory.
"""

from little.core.models import (
    Belief,
    BeliefStatus,
    Concept,
    ConceptStatus,
    Entity,
    Experience,
    InferenceResult,
    LearningResult,
    Relation,
    UpdateType,
)
from little.inference.engine import InferenceEngine
from little.language.parser import LearningEngine, SimpleParser
from little.memory.store import MemoryStore

__version__ = "0.1.10"

__all__ = [
    "Belief",
    "BeliefStatus",
    "Concept",
    "ConceptStatus",
    "Entity",
    "Experience",
    "InferenceEngine",
    "InferenceResult",
    "LearningEngine",
    "LearningResult",
    "MemoryStore",
    "Relation",
    "SimpleParser",
    "UpdateType",
]
