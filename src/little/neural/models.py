"""Data models for continuous neural representations, routing decisions, and semantic frames."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Pathway(str, Enum):
    """The 6 specialized cognitive deliberation pathways in LITTLE."""

    CHAT = "chat"
    MATH = "math"
    TOOL = "tool"
    CODE = "code"
    POSET = "poset"
    PHYSICS = "physics"


@dataclass(frozen=True)
class NeuralState:
    """A continuous latent semantic workspace vector (S in R^512)."""

    embedding: tuple[float, ...]
    text: str
    tokens: tuple[str, ...]
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if len(self.embedding) != 512:
            raise ValueError(
                f"NeuralState embedding dimension must be 512, got {len(self.embedding)}"
            )


@dataclass(frozen=True)
class RoutingDecision:
    """The outcome of passing a NeuralState through the learned Neural Router."""

    pathway: Pathway
    confidence: float
    probabilities: dict[Pathway, float]
    entropy: float
    is_uncertain: bool
    latent_state: NeuralState


@dataclass(frozen=True)
class SemanticSlot:
    """A typed extraction slot (e.g. subject, predicate, object, modifier)."""

    slot_name: str
    text: str
    confidence: float
    start_pos: int = 0
    end_pos: int = 0


@dataclass(frozen=True)
class ExtractedSemanticFrame:
    """A FrameNet-style structured semantic frame extracted from unstructured natural language."""

    frame_type: str
    subject: str
    predicate: str
    object_val: str
    is_negative: bool = False
    is_property: bool = False
    confidence: float = 0.85
    attributes: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
