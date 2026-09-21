"""Continuous-Time Dynamical Systems and Physical Entity Transformations for LITTLE.

Implements Closed-Form Continuous-Time (CfC) ODE state propagation inspired by
Liquid Foundation Models (LFM) and discrete topological transformations (e.g. slicing,
part decomposition).
"""

from little.dynamics.cfc import ContinuousDynamicsEngine, PhysicalState
from little.dynamics.transformations import TransformationEngine

__all__ = [
    "ContinuousDynamicsEngine",
    "PhysicalState",
    "TransformationEngine",
]
