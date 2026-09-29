"""Data models for continuous neural representations and routing decisions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


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
