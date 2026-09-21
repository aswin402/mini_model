"""Unit tests for LITTLE procedural memory and skill execution."""

from little.core.models import Skill
from little.memory.store import MemoryStore
from little.procedural.runner import SkillRunner
from little.procedural.skills import register_builtin_skills


def test_custom_skill_execution():
    skill = Skill.create(
        name="HYPOTENUSE",
        parameters=["a", "b"],
        code_body="math.sqrt(a**2 + b**2)",
        description="Compute right triangle hypotenuse",
    )
    result = SkillRunner.execute(skill, a=3, b=4)
    assert result.success is True
    assert result.result == 5.0
    assert "Preparing execution" in result.trace[0]


def test_missing_parameter_handling():
    skill = Skill.create(
        name="ADD",
        parameters=["a", "b"],
        code_body="a + b",
    )
    result = SkillRunner.execute(skill, a=10)
    assert result.success is False
    assert "Missing required parameter 'b'" in result.error


def test_zero_division_handling():
    skill = Skill.create(
        name="DIVIDE",
        parameters=["a", "b"],
        code_body="a / b",
    )
    result = SkillRunner.execute(skill, a=10, b=0)
    assert result.success is False
    assert "Division by zero" in result.error


def test_builtin_skills_registration_and_persistence():
    store = MemoryStore(":memory:")
    register_builtin_skills(store)

    add_skill = store.get_skill("ADD")
    assert add_skill is not None
    assert add_skill.parameters == ["a", "b"]

    # Test unseen large number addition (guaranteed 100% exact symbolic precision)
    res = SkillRunner.execute(add_skill, a=9876543210123, b=1234567890987)
    assert res.success is True
    assert res.result == 9876543210123 + 1234567890987

    # Test slice skill
    slice_skill = store.get_skill("SLICE")
    assert slice_skill is not None
    slice_res = SkillRunner.execute(slice_skill, item="apple", count=4)
    assert slice_res.success is True
    assert len(slice_res.result) == 4
    assert slice_res.result[0]["name"] == "apple_slice_1"
    assert slice_res.result[0]["part_of"] == "apple"
    assert slice_res.result[0]["exposed_flesh"] is True
