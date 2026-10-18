import copy

import pytest

from little.memory.store import MemoryStore
from little.neural.decision_head import (
    EpistemicBeliefState,
    LayaDecisionHead,
    compute_brier_score,
    compute_ranked_probability_score,
)
from little.neural.frame_extractor import NeuralFrameExtractor
from little.neural.models import Pathway
from little.neural.router import NeuralRouter
from little.procedural.code_synthesizer import ASTCodeSynthesizer


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A car contains an engine", ("car", "has_part", "engine")),
        ("A car is composed of wheels", ("car", "has_part", "wheel")),
        ("Physics is a science", ("physics", "is_a", "science")),
        ("No Man Sky is a game", ("no man sky", "is_a", "game")),
    ],
)
def test_frame_direction_names_and_local_negation(text, expected):
    frame = NeuralFrameExtractor().extract_frames(text)[0]
    assert (frame.subject, frame.predicate, frame.object_val) == expected
    assert not frame.is_negative


def test_synthesizer_rejects_added_constraint():
    synth = ASTCodeSynthesizer()
    assert (
        synth.synthesize_from_request(
            "write a factorial function modulo 7", verify=False
        )
        is None
    )
    assert (
        synth.synthesize_from_request("write a factorial function", verify=False).name
        == "factorial"
    )


def test_noul_matches_whole_predicate_and_explicit_negation():
    memory = MemoryStore(":memory:")
    car = memory.create_concept("car")
    engine = memory.create_concept("engine")
    memory.add_relation(car.id, "has", engine.id)
    head = LayaDecisionHead(memory=memory)
    assert (
        head.evaluate_noul("A car has an engine").status
        == EpistemicBeliefState.SUPPORTED
    )
    assert (
        head.evaluate_noul("A car is an engine").status == EpistemicBeliefState.UNKNOWN
    )
    assert (
        head.evaluate_noul("A car does not have an engine").status
        == EpistemicBeliefState.CONTRADICTED
    )
    assert LayaDecisionHead().memory is None
    assert (
        LayaDecisionHead().evaluate_noul("A car has an engine").status
        == EpistemicBeliefState.UNKNOWN
    )


@pytest.mark.parametrize(
    "prob,target",
    [
        ({"a": 0.5, "b": 0.5}, "missing"),
        ([0.5, 0.5], -1),
        ([0.5, 0.5], 2),
        ([0.5, 0.5], True),
        ([0.2, 0.2], 0),
        ([float("nan"), 1], 0),
        ([], 0),
        ([-0.1, 1.1], 0),
    ],
)
def test_invalid_brier_inputs_fail(prob, target):
    with pytest.raises((TypeError, ValueError)):
        compute_brier_score(prob, target)


def test_invalid_rps_target_fails():
    with pytest.raises(ValueError):
        compute_ranked_probability_score([0.5, 0.5], 3)


def test_router_feedback_checkpoint_roundtrip():
    router = NeuralRouter()
    state = router.encoder.encode("unusual novel prompt")
    initial = router.route(state).probabilities[Pathway.PHYSICS]
    router.learn(state, Pathway.PHYSICS, learning_rate=0.4)
    assert router.route(state).probabilities[Pathway.PHYSICS] > initial
    checkpoint = router.checkpoint()
    restored = NeuralRouter.from_checkpoint(checkpoint)
    assert (
        restored.route("unusual novel prompt").probabilities
        == router.route("unusual novel prompt").probabilities
    )
    broken = copy.deepcopy(checkpoint)
    broken["weights"][0][0] = float("nan")
    with pytest.raises(ValueError):
        NeuralRouter.from_checkpoint(broken)


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_dim", 0),
        ("input_dim", 128),
        ("pathways", []),
        ("pathways", ["chat", "chat"]),
        ("uncertainty_margin", 1.1),
        ("entropy_threshold", -1),
    ],
)
def test_router_rejects_invalid_policies(tmp_path, field, value):
    import json

    from little.neural.router_policy import NeuralRouterPolicy

    payload = {
        "format": "little.neural_router_policy.v1",
        "input_dim": 512,
        "pathways": ["chat", "math"],
        field: value,
    }
    (tmp_path / "neural_router_policy.json").write_text(json.dumps(payload))
    with pytest.raises((ValueError, TypeError)):
        NeuralRouterPolicy.load(tmp_path)


@pytest.mark.parametrize(
    "field,value", [("temperature", 0), ("temperature", float("nan"))]
)
def test_router_rejects_invalid_temperature(field, value):
    with pytest.raises(ValueError):
        NeuralRouter(**{field: value})


def test_frame_policy_threshold_is_applied_and_validated():
    from dataclasses import replace

    from little.neural.semantic_frame_policy import SemanticFramePolicy

    policy = SemanticFramePolicy.default()
    assert (
        NeuralFrameExtractor(
            policy=replace(policy, min_extraction_confidence=0.95)
        ).extract_candidate_claims("A car contains an engine")
        == []
    )
    with pytest.raises(ValueError):
        replace(policy, min_extraction_confidence=-1)


def test_router_checkpoint_preserves_injected_encoder_policy():
    from dataclasses import replace

    from little.neural.encoder import NeuralStateEncoder
    from little.neural.router_policy import NeuralRouterPolicy

    policy = NeuralRouterPolicy.default()
    encoder = NeuralStateEncoder(
        replace(policy, semantic_clusters={Pathway.PHYSICS: ("novel",)})
    )
    router = NeuralRouter(policy=policy, encoder=encoder)
    restored = NeuralRouter.from_checkpoint(router.checkpoint())
    assert restored.route("novel").probabilities == router.route("novel").probabilities


@pytest.mark.parametrize(
    "prompt",
    [
        "write a factorial function returning odd values only",
        "write a fibonacci function in JavaScript",
        "write optimized factorial for a GPU",
    ],
)
def test_synthesis_refuses_unimplemented_modifiers(prompt):
    assert ASTCodeSynthesizer().synthesize_from_request(prompt, verify=False) is None


def test_synthesis_rejects_multiple_primitive_requests():
    assert (
        ASTCodeSynthesizer().synthesize_from_request(
            "write factorial fibonacci code", verify=False
        )
        is None
    )


def test_decision_learning_uses_temperature_scaled_cross_entropy_gradient():
    head = LayaDecisionHead(temperature=2)
    state = head.encoder.encode("new unusual prompt")
    before = head.classify_intent(state).probabilities
    biases = head._b[:]
    head.learn(state, "physics", learning_rate=0.2)
    for index, label in enumerate(head._intent_labels):
        expected = 0.2 * ((1 if label == "physics" else 0) - before[label]) / 2
        assert head._b[index] - biases[index] == pytest.approx(expected)
