"""Unit tests for LITTLE procedural memory and skill execution."""

import json
from pathlib import Path

from little.core.models import Skill
from little.memory.store import MemoryStore
from little.procedural.runner import SkillRunner
from little.procedural.skill_policy import ProceduralSkillPolicy
from little.procedural.skills import get_builtin_skills, register_builtin_skills


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


def test_slice_skill_output_contract_is_data_driven(tmp_path: Path):
    payload = json.loads(
        Path("data/schemas/procedural_skill_policy.json").read_text(
            encoding="utf-8"
        )
    )
    output = payload["procedural_skill_policy"]["slice_output"]
    output.update(
        {
            "name_suffix": "portion",
            "part_key": "component_of",
            "exposed_key": "inner_exposed",
            "skin_key": "boundary_intact",
            "exposed_value": False,
            "skin_value": True,
        }
    )
    policy_path = tmp_path / "procedural_skill_policy.json"
    policy_path.write_text(json.dumps(payload), encoding="utf-8")
    policy = ProceduralSkillPolicy.load(tmp_path)
    skill = next(
        skill for skill in get_builtin_skills(policy) if skill.name == "SLICE"
    )

    result = SkillRunner.execute(skill, item="apple", count=1)

    assert result.success is True
    assert result.result == [
        {
            "name": "apple_portion_1",
            "component_of": "apple",
            "inner_exposed": False,
            "boundary_intact": True,
        }
    ]


def test_builtin_skill_catalog_can_be_restricted_by_policy(tmp_path: Path):
    payload = json.loads(
        Path("data/schemas/procedural_skill_policy.json").read_text(
            encoding="utf-8"
        )
    )
    payload["procedural_skill_policy"]["enabled_skills"] = ["ADD"]
    policy_path = tmp_path / "procedural_skill_policy.json"
    policy_path.write_text(json.dumps(payload), encoding="utf-8")
    policy = ProceduralSkillPolicy.load(tmp_path)

    skills = get_builtin_skills(policy)

    assert [skill.name for skill in skills] == ["ADD"]


def test_skill_runner_uses_injected_execution_policy(tmp_path: Path):
    payload = json.loads(
        Path("data/schemas/procedural_skill_policy.json").read_text(
            encoding="utf-8"
        )
    )
    payload["procedural_skill_policy"]["execution"] = {
        "builtins": ["abs"],
        "modules": [],
        "helpers": [],
    }
    policy_path = tmp_path / "procedural_skill_policy.json"
    policy_path.write_text(json.dumps(payload), encoding="utf-8")
    policy = ProceduralSkillPolicy.load(tmp_path)
    skill = Skill.create(
        name="DISABLED_MATH",
        parameters=["value"],
        code_body="math.sqrt(value)",
    )

    result = SkillRunner.execute(skill, policy=policy, value=9)

    assert result.success is False
    assert "NameError" in result.error


def test_skill_execution_uses_a_fresh_builtin_namespace_per_run():
    mutating_skill = Skill.create(
        name="MUTATE_BUILTINS",
        parameters=[],
        code_body='__builtins__["leaked_name"] = "should not persist"\nreturn None',
    )
    probe_skill = Skill.create(
        name="PROBE_BUILTINS",
        parameters=[],
        code_body="leaked_name",
    )

    mutation = SkillRunner.execute(mutating_skill)
    probe = SkillRunner.execute(probe_skill)

    assert mutation.success is True
    assert probe.success is False
    assert "NameError" in probe.error
