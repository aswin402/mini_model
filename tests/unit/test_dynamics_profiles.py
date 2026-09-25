import json
from pathlib import Path

from little.dynamics.cfc import ContinuousDynamicsEngine
from little.dynamics.cfc_ode import CfCContinuousODE
from little.dynamics.profiles import DynamicsProfileRegistry
from little.dynamics.transformations import TransformationEngine
from little.dynamics.transformation_policy import TransformationPolicy
from little.memory.store import MemoryStore


def test_dynamics_profiles_are_loaded_from_data():
    registry = DynamicsProfileRegistry.load(Path("data/schemas"))

    perishable = registry.profile("perishable")
    inert = registry.profile("inert")

    assert perishable.tau_exposed_seconds == 1800.0
    assert "bread" in perishable.categories
    assert perishable.perishable is True
    assert inert.tau_exposed_seconds > perishable.tau_exposed_seconds
    assert registry.perception.color_labels == ("base", "light brown", "brown")
    assert registry.continuous.q10 == 2.0


def test_transformation_engine_has_no_process_global_policy_state():
    assert "POLICY" not in TransformationEngine.__dict__


def test_slice_uses_profile_for_material_not_object_name():
    memory = MemoryStore(":memory:")
    bread = memory.create_concept(
        "bread",
        category="bakery",
        attributes={"dynamics_profile": "perishable", "interior_color": "cream"},
    )

    result = TransformationEngine.slice_object(memory, bread.name, num_pieces=2)

    assert result.initial_state.tau_seconds == 1800.0
    assert result.slice_concept == "bread slice"
    assert result.initial_state.exposed_to_air is True


def test_transformation_uses_injected_relation_and_attribute_policy(
    tmp_path: Path,
):
    payload = json.loads(
        Path("data/schemas/transformation_policy.json").read_text(
            encoding="utf-8"
        )
    )
    config = payload["transformation_policy"]
    config["predicates"]["taxonomy"] = "classified_as"
    config["predicates"]["part"] = "component_of"
    config["predicates"]["interior_color_relations"] = ["inner_tone"]
    config["attributes"]["interior_color"] = "inner_tone"
    config["attributes"]["color"] = "surface_tone"
    config["defaults"]["category"] = "partition"
    config["defaults"]["slice_suffix"] = "portion"
    config["defaults"]["experience_source"] = "custom_transformation"
    policy_path = tmp_path / "transformation_policy.json"
    policy_path.write_text(json.dumps(payload), encoding="utf-8")
    policy = TransformationPolicy.load(tmp_path)

    memory = MemoryStore(":memory:", seed_ontology=False)
    apple = memory.create_concept("apple", attributes={"inner_tone": "white"})

    result = TransformationEngine.slice_object(
        memory,
        apple.name,
        num_pieces=2,
        policy=policy,
    )

    assert result.slice_concept == "apple portion"
    assert "component_of" in result.relations_created[0]
    slice_concept = memory.get_concept("apple portion")
    assert slice_concept is not None
    assert slice_concept.category == "partition"
    assert slice_concept.attributes["inner_tone"] == "white"
    assert memory.list_experiences(limit=1)[0].source == "custom_transformation"


def test_initial_state_defaults_come_from_profile_data():
    whole = ContinuousDynamicsEngine.create_initial_state(exposed_to_air=False)
    exposed = ContinuousDynamicsEngine.create_initial_state(exposed_to_air=True)

    assert whole.tau_seconds == 86400.0 * 14.0
    assert exposed.tau_seconds == 1800.0


def test_cfc_ode_parameters_are_loaded_from_profile_data(tmp_path: Path):
    payload = json.loads(
        Path("data/schemas/dynamics_profiles.json").read_text(encoding="utf-8")
    )
    payload["ode"]["exposed_decay_rate_per_hour"] = 0.2
    custom_path = tmp_path / "dynamics_profiles.json"
    custom_path.write_text(json.dumps(payload), encoding="utf-8")

    registry = DynamicsProfileRegistry.load(tmp_path)
    ode = CfCContinuousODE(profile_registry=registry)

    assert ode.exposed_decay_rate == 0.2
    assert ode.reference_temperature_c == 20.0


def test_continuous_defaults_and_thermal_model_use_injected_profile_data(
    tmp_path: Path,
):
    payload = json.loads(
        Path("data/schemas/dynamics_profiles.json").read_text(encoding="utf-8")
    )
    payload["continuous"].update(
        {
            "q10": 4.0,
            "min_rate_factor": 0.25,
            "max_rate_factor": 8.0,
            "default_temperature_c": 12.0,
            "initial_freshness": 0.8,
            "initial_oxidation": 0.1,
            "protected_color_oxidation_threshold": 0.2,
        }
    )
    custom_path = tmp_path / "dynamics_profiles.json"
    custom_path.write_text(json.dumps(payload), encoding="utf-8")
    registry = DynamicsProfileRegistry.load(tmp_path)

    state = ContinuousDynamicsEngine.create_initial_state(profile_registry=registry)
    effective_tau = ContinuousDynamicsEngine.compute_effective_tau(
        100.0, 30.0, profile_registry=registry
    )

    assert state.temperature == 12.0
    assert state.freshness == 0.8
    assert state.oxidation == 0.1
    assert effective_tau == 25.0
