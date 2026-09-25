"""Data-backed deterministic skills and procedural capabilities for LITTLE."""

from __future__ import annotations

from typing import TYPE_CHECKING

from little.core.models import Skill
from little.procedural.skill_catalog import ProceduralSkillCatalog, SkillDefinition
from little.procedural.skill_policy import ProceduralSkillPolicy
from little.procedural.unit_conversion_policy import UnitConversionPolicy

if TYPE_CHECKING:
    from little.memory.store import MemoryStore


def _unit_conversion_code(policy: UnitConversionPolicy) -> str:
    """Compile the declarative conversion policy into sandbox-safe skill code."""
    adjacency: dict[str, list[tuple[str, float, float]]] = {}
    for edge in policy.edges:
        adjacency.setdefault(edge.source, []).append(
            (edge.target, edge.scale, edge.offset)
        )
        adjacency.setdefault(edge.target, []).append(
            (edge.source, 1.0 / edge.scale, -edge.offset / edge.scale)
        )

    return f"""
val = float(value)
aliases = {policy.aliases!r}
adjacency = {adjacency!r}
start_raw = str(from_unit).strip().lower()
goal_raw = str(to_unit).strip().lower()
start = aliases.get(start_raw, start_raw)
goal = aliases.get(goal_raw, goal_raw)
if start == goal:
    result = val
else:
    queue = [(start, val)]
    visited = {{start}}
    result = None
    index = 0
    while index < len(queue):
        node, current = queue[index]
        index += 1
        for neighbor, scale, offset in adjacency.get(node, []):
            if neighbor in visited:
                continue
            next_value = current * scale + offset
            if neighbor == goal:
                result = next_value
                queue = []
                break
            visited.add(neighbor)
            queue.append((neighbor, next_value))
        if result is not None:
            break
    if result is None:
        raise ValueError(f"Unsupported unit conversion from {{from_unit}} to {{to_unit}}")

if isinstance(result, float) and result.is_integer():
    return int(result)
return round(result, 4)
"""


def _slice_code(policy: ProceduralSkillPolicy) -> str:
    """Compile the configured physical slice output contract into skill code."""
    return f"""
pieces = []
n = int(count)
for i in range(n):
    pieces.append({{
        "name": f"{{item}}_{policy.slice_name_suffix}_{{i+1}}",
        "{policy.slice_part_key}": item,
        "{policy.slice_exposed_key}": {policy.slice_exposed_value},
        "{policy.slice_skin_key}": {policy.slice_skin_value},
    }})
return pieces
"""


def _skill_code(
    definition: SkillDefinition,
    policy: ProceduralSkillPolicy,
    unit_policy: UnitConversionPolicy,
) -> str:
    if definition.code is not None:
        return definition.code
    if definition.implementation == "slice":
        return _slice_code(policy)
    if definition.implementation == "unit_conversion":
        return _unit_conversion_code(unit_policy)
    raise ValueError(
        f"Unsupported procedural skill implementation {definition.implementation!r}"
    )


def get_builtin_skills(
    policy: ProceduralSkillPolicy | None = None,
    unit_policy: UnitConversionPolicy | None = None,
    catalog: ProceduralSkillCatalog | None = None,
) -> list[Skill]:
    """Return the enabled procedural skills from the configured catalog."""
    active_policy = policy or ProceduralSkillPolicy.default()
    active_unit_policy = unit_policy or UnitConversionPolicy.default()
    active_catalog = catalog or ProceduralSkillCatalog.default()

    skills = [
        Skill.create(
            name=definition.name,
            parameters=list(definition.parameters),
            code_body=_skill_code(definition, active_policy, active_unit_policy),
            description=definition.description,
        )
        for definition in active_catalog.definitions
        if definition.name in active_policy.enabled_skills
    ]
    return skills


def register_builtin_skills(
    memory: MemoryStore,
    policy: ProceduralSkillPolicy | None = None,
    unit_policy: UnitConversionPolicy | None = None,
    catalog: ProceduralSkillCatalog | None = None,
) -> None:
    """Bootstrap procedural memory with skills from the configured catalog."""
    for skill in get_builtin_skills(
        policy,
        unit_policy=unit_policy,
        catalog=catalog,
    ):
        memory.save_skill(skill)
