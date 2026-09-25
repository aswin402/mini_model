from pathlib import Path

import pytest

from little.active.self_study_policy import SelfStudyPolicy


def test_self_study_policy_loads_data_owned_axes_and_calibration():
    policy = SelfStudyPolicy.load(Path("data/schemas"))

    assert policy.version == "1"
    assert policy.axis_names == (
        "taxonomy",
        "procedural",
        "mereology",
        "invariants",
        "dynamics",
        "episodic",
    )
    assert policy.hierarchy_axis == "taxonomy"
    assert policy.default_axis == "procedural"
    assert policy.no_analogy_axis == "unknown"
    assert policy.hypothesis_probability == pytest.approx(0.85)
    assert policy.hypothesis_uncertainty == pytest.approx(0.15)
    assert policy.max_steps == 5


def test_self_study_policy_rejects_duplicate_axes(tmp_path: Path):
    path = tmp_path / "self_study_policy.json"
    path.write_text(
        '{"self_study_policy": {"version": "1", "axes": ['
        '{"name": "taxonomy", "coverage": "relation"}, '
        '{"name": "taxonomy", "coverage": "relation"}], '
        '"hierarchy_axis": "taxonomy", "default_axis": "taxonomy", '
        '"no_analogy_axis": "unknown", "hypothesis_probability": 0.85, '
        '"hypothesis_uncertainty": 0.15, "model_id": "study", '
        '"model_version": "1", "max_steps": 1}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        SelfStudyPolicy.load(tmp_path)
