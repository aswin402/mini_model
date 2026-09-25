from pathlib import Path

from little.active.active_policy import ActiveLearningPolicy


def test_active_learning_policy_loads_guard_and_entropy_configuration():
    policy = ActiveLearningPolicy.load(Path("data/schemas"))

    assert policy.version == "1"
    assert "little" in policy.meta_words
    assert policy.default_predicate == "is_a"
    assert policy.binary_prior == 0.5
    assert policy.round_digits == 4
    assert "subject_unknown" in policy.prompt_templates
