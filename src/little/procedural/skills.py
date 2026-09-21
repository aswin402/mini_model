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
        Skill.create(
            name="FIBONACCI",
            parameters=["n"],
            code_body="""
n_val = int(n)
if n_val <= 0:
    return 0
if n_val == 1:
    return 1
a, b = 0, 1
for _ in range(n_val):
    a, b = b, a + b
return a
""",
            description="Exact computation of the n-th Fibonacci number: Fib(n)",
        ),
        Skill.create(
            name="IS_PRIME",
            parameters=["n"],
            code_body="""
num = int(n)
if num < 2:
    return False
if num in (2, 3):
    return True
if num % 2 == 0 or num % 3 == 0:
    return False
i = 5
while i * i <= num:
    if num % i == 0 or num % (i + 2) == 0:
        return False
    i += 6
return True
""",
            description="Deterministic deterministic primality test for integer n",
        ),
        Skill.create(
            name="REVERSE_STRING",
            parameters=["text"],
            code_body="str(text)[::-1]",
            description="Reverse a string exactly: text[::-1]",
        ),
        Skill.create(
            name="PALINDROME",
            parameters=["text"],
            code_body="""
cleaned = "".join(c.lower() for c in str(text) if c.isalnum())
return cleaned == cleaned[::-1]
""",
            description="Verify whether a given string is a palindrome",
        ),
    ]


def register_builtin_skills(memory: MemoryStore) -> None:
    """Bootstrap procedural memory with core deterministic capabilities if not already present."""
    for skill in get_builtin_skills():
        memory.save_skill(skill)
