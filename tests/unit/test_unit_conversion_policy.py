import json
from pathlib import Path

from little.procedural.math_cas import UnitConversionGraph
from little.procedural.runner import SkillRunner
from little.procedural.skills import get_builtin_skills
from little.procedural.unit_conversion_policy import UnitConversionPolicy


def test_unit_conversion_policy_loads_aliases_and_edges():
    policy = UnitConversionPolicy.load(Path("data/schemas"))

    assert policy.aliases["kilometers"] == "km"
    assert any(
        edge.source == "km" and edge.target == "m" and edge.scale == 1000.0
        for edge in policy.edges
    )


def test_unit_conversion_graph_uses_injected_data(tmp_path: Path):
    payload = json.loads(
        Path("data/schemas/unit_conversion_policy.json").read_text(
            encoding="utf-8"
        )
    )
    payload["unit_conversion_policy"]["aliases"]["furlong"] = "furlong"
    payload["unit_conversion_policy"]["edges"].append(
        {"source": "furlong", "target": "km", "scale": 0.201168}
    )
    policy_path = tmp_path / "unit_conversion_policy.json"
    policy_path.write_text(json.dumps(payload), encoding="utf-8")
    policy = UnitConversionPolicy.load(tmp_path)

    result = UnitConversionGraph(policy).convert(2, "furlong", "m")

    assert result.status == "SOLVED"
    assert result.result == 402.336


def test_unit_conversion_skill_uses_injected_data(tmp_path: Path):
    payload = json.loads(
        Path("data/schemas/unit_conversion_policy.json").read_text(
            encoding="utf-8"
        )
    )
    payload["unit_conversion_policy"]["aliases"]["furlong"] = "furlong"
    payload["unit_conversion_policy"]["edges"].append(
        {"source": "furlong", "target": "km", "scale": 0.201168}
    )
    policy_path = tmp_path / "unit_conversion_policy.json"
    policy_path.write_text(json.dumps(payload), encoding="utf-8")
    policy = UnitConversionPolicy.load(tmp_path)
    skill = next(skill for skill in get_builtin_skills(unit_policy=policy) if skill.name == "UNIT_CONVERT")

    result = SkillRunner.execute(skill, value=2, from_unit="furlong", to_unit="m")

    assert result.success is True
    assert result.result == 402.336
