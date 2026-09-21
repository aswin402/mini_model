"""Standard library of deterministic skills and procedural capabilities for LITTLE."""

from __future__ import annotations

from typing import TYPE_CHECKING

from little.core.models import Skill

if TYPE_CHECKING:
    from little.memory.store import MemoryStore


def get_builtin_skills() -> list[Skill]:
    """Return the fundamental procedural skills bootstrap library."""
    return [
        Skill.create(
            name="ADD",
            parameters=["a", "b"],
            code_body="a + b",
            description="Exact arithmetic addition of two numbers: a + b",
        ),
        Skill.create(
            name="SUBTRACT",
            parameters=["a", "b"],
            code_body="a - b",
            description="Exact arithmetic subtraction: a - b",
        ),
        Skill.create(
            name="MULTIPLY",
            parameters=["a", "b"],
            code_body="a * b",
            description="Exact arithmetic multiplication: a * b",
        ),
        Skill.create(
            name="DIVIDE",
            parameters=["a", "b"],
            code_body="a / b",
            description="Exact arithmetic division: a / b",
        ),
        Skill.create(
            name="POWER",
            parameters=["a", "b"],
            code_body="a ** b",
            description="Exact exponentiation: a ** b",
        ),
        Skill.create(
            name="FACTORIAL",
            parameters=["n"],
            code_body="math.factorial(int(n))",
            description="Factorial of non-negative integer n: n!",
        ),
        Skill.create(
            name="SLICE",
            parameters=["item", "count"],
            code_body="""
pieces = []
n = int(count)
for i in range(n):
    pieces.append({
        "name": f"{item}_slice_{i+1}",
        "part_of": item,
        "exposed_flesh": True,
        "skin_intact": False,
    })
return pieces
""",
            description="Physical transformation: cuts an entity or object into N discrete slices/parts.",
        ),
    ]


def register_builtin_skills(memory: MemoryStore) -> None:
    """Bootstrap procedural memory with core deterministic capabilities if not already present."""
    for skill in get_builtin_skills():
        memory.save_skill(skill)
