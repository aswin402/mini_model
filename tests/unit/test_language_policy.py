from pathlib import Path

from little.knowledge.policy import LanguagePolicy
from little.language.laya_gatekeeper import LayaSystem1Gatekeeper


def test_language_policy_loads_routing_and_grounding_data():
    policy = LanguagePolicy.load(Path("data/schemas"))

    assert "is " in policy.question_prefixes
    assert "calculate" in policy.math_markers
    assert "apple" in policy.concept_associations
    assert "hello" in policy.greetings
    assert "quit" in policy.exit_commands
    assert "skip" in policy.cancel_words


def test_laya_grounding_uses_injected_policy_associations():
    policy = LanguagePolicy(
        math_markers=(),
        curiosity_markers=(),
        action_prefixes=(),
        question_prefixes=("is ",),
        concept_associations={"quartz": frozenset({"silica", "crystal"})},
    )
    gatekeeper = LayaSystem1Gatekeeper(policy=policy)

    result = gatekeeper.evaluate_choice(
        state="The sample contains silica crystal.",
        question="Which concept does this match?",
        options=["Quartz", "Bicycle"],
    )

    assert result.winner == "Quartz"
