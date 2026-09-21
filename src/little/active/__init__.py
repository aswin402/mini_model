"""Active Learning and Curiosity Engine for LITTLE.

Implements uncertainty-directed questioning, active clarification loops,
and epistemic entropy reduction inspired by Non-Axiomatic Reasoning and Laya.
"""

from little.active.inquisitor import ActiveInquisitor, ClarificationPrompt

__all__ = ["ActiveInquisitor", "ClarificationPrompt"]
