"""Unit tests for Conversational, Identity, and Definition Query handling in LITTLE."""

from little.active.inquisitor import ActiveInquisitor
from little.core.models import BeliefStatus, UpdateType
from little.language.parser import LearningEngine, SimpleParser
from little.memory.store import MemoryStore


def test_conversational_pronoun_protection():
    """Verify that conversational sentences like 'who are you' do not create garbage concepts."""
    triples = SimpleParser.parse_statement("who are you")
    assert len(triples) == 0, (
        f"Expected no triples for conversational input, got: {triples}"
    )

    triples_hello = SimpleParser.parse_statement("hello")
    assert len(triples_hello) == 0

    triples_hii = SimpleParser.parse_statement("hii")
    assert len(triples_hii) == 0


def test_identity_query():
    """Verify that asking 'who are you' or 'what are you' returns LITTLE's identity."""
    store = MemoryStore(":memory:")
    engine = LearningEngine(store)

    for q in ["who are you", "who are you?", "what are you", "what can you do"]:
        res = engine.ask(q)
        assert res.status == BeliefStatus.SUPPORTED, f"Failed on: {q}"
        assert "LITTLE" in str(res.answer)
        assert res.confidence == 1.0


def test_concept_definition_query():
    """Verify that 'what is a dog/animal' returns rich definitions from memory."""
    store = MemoryStore(":memory:")
    engine = LearningEngine(store)

    engine.learn("A dog is an animal.")
    engine.learn("An animal is not a vehicle.")

    res_dog = engine.ask("What is a dog?")
    assert res_dog.status == BeliefStatus.SUPPORTED
    assert "animal" in str(res_dog.answer).lower()

    res_animal = engine.ask("What is an animal?")
    assert res_animal.status == BeliefStatus.SUPPORTED
    assert "vehicle" in str(res_animal.answer).lower()


def test_definition_unknown_curiosity_flow():
    """Verify that asking about an unknown concept prompts curiosity and learns its category."""
    store = MemoryStore(":memory:")
    engine = LearningEngine(store)
    inquisitor = ActiveInquisitor(store, engine)

    # Ask about unknown mango
    res = engine.ask("What is a mango?")
    assert res.status == BeliefStatus.UNKNOWN

    # Inquisitor inspects
    parsed = SimpleParser.parse_question("What is a mango?")
    assert parsed is not None
    s, p, o = parsed
    prompt = inquisitor.inspect_uncertainty(res, s, p, o)
    assert prompt is not None
    assert "mango" in prompt.question_for_user

    # User answers: "fruit"
    learn_res = inquisitor.resolve_response(prompt, "fruit")
    assert (
        "mango" in learn_res.concepts_created or "fruit" in learn_res.concepts_created
    )
    assert any("mango is_a fruit" in r for r in learn_res.relations_created)

    # Re-ask
    res_after = engine.ask("What is a mango?")
    assert res_after.status == BeliefStatus.SUPPORTED
    assert "fruit" in str(res_after.answer).lower()


def test_contractions_and_flexible_math():
    """Verify that contractions like 'whats 4+4' and raw math expressions work seamlessly."""
    store = MemoryStore(":memory:")
    engine = LearningEngine(store)

    assert engine.ask("whats 4+4").answer == 8
    assert engine.ask("what's 4+4").answer == 8
    assert engine.ask("4+4").answer == 8
    assert engine.ask("4 + 4").answer == 8
    assert engine.ask("5 * 5").answer == 25
    assert engine.ask("10 / 2").answer == 5.0
    assert engine.ask("2^8").answer == 256
    assert engine.ask("5!").answer == 120
    assert engine.ask("fibonacci 10").answer == 55
    assert engine.ask("is 7 prime").answer is True


def test_verbalize_and_conversational_thinking():
    """Verify that LITTLE speaks in fluent, articulate English and explains its reasoning."""
    store = MemoryStore(":memory:")
    engine = LearningEngine(store)

    # 1. Multi-hop Transitive Deduction Reasoning Verbalization
    engine.learn("A dog is a mammal.")
    engine.learn("A mammal is a vertebrate.")
    engine.learn("A vertebrate is an animal.")
    engine.learn("An animal is a living thing.")
    res_trans = engine.ask("Is a dog a living thing?")
    assert res_trans.status == BeliefStatus.SUPPORTED
    v_trans = res_trans.verbalize()
    assert "Yes, a dog is a living thing" in v_trans
    assert "I deduced this across" in v_trans
    assert "logical steps" in v_trans
    assert "vertebrate" in v_trans
    assert "animal" in v_trans

    # 2. Disjoint Refutation / Invariant Verbalization
    engine.learn("A mammal is not a reptile.")
    res_refuted = engine.ask("Is a dog a reptile?")
    assert res_refuted.status == BeliefStatus.REFUTED
    v_refuted = res_refuted.verbalize()
    assert "No, that is impossible" in v_refuted
    assert "disjoint" in v_refuted

    # 3. Procedural Mathematics & Algorithms Verbalization
    res_math = engine.ask("What is 123 + 456?")
    assert "The result is 579" in res_math.verbalize()

    res_prime = engine.ask("Is 104729 prime?")
    assert "Yes, 104729 is a prime number." == res_prime.verbalize()

    res_nth_prime = engine.ask("What is the 10th prime?")
    assert "The 10th prime number is 29." == res_nth_prime.verbalize()

    res_fact = engine.ask("What is the factorial of 5?")
    assert "The factorial of 5 (5!) is 120." == res_fact.verbalize()

    res_fib = engine.ask("What is the fibonacci of 10?")
    assert "The 10th Fibonacci number is 55." == res_fib.verbalize()

    res_pal = engine.ask("Is racecar a palindrome?")
    assert "Yes, 'racecar' is a palindrome" in res_pal.verbalize()

    # 4. Polite Conversational Prefix Stripping
    res_polite1 = engine.ask("Can you calculate 25 * 4?")
    assert res_polite1.answer == 100
    res_polite2 = engine.ask("Could you please tell me the 5th prime?")
    assert res_polite2.answer == 11

    # 5. Continuous Time Physical Decay (Liquid / CfC ODEs)
    engine.learn("Slice an apple into 4 pieces.")
    res_decay = engine.ask("What color is the apple slice after 2 hours?")
    v_decay = res_decay.verbalize()
    assert "brown" in v_decay.lower()
    assert "differential equations" in v_decay.lower() or "oxidation" in v_decay.lower()

    # 6. Epistemic Unknown (Zero Hallucination)
    res_unknown = engine.ask("Is an apple a spaceship?")
    assert res_unknown.status == BeliefStatus.UNKNOWN
    v_unk = res_unknown.verbalize()
    assert "I do not know" in v_unk
    assert "avoid guessing or hallucinating" in v_unk


def test_slang_discourse_markers_and_natural_math():
    """Verify that internet slang, conversational discourse markers, and natural math are parsed accurately."""
    store = MemoryStore(":memory:")
    engine = LearningEngine(store)

    # 1. Slang & capabilities
    for cap_q in [
        "what are the things u can do",
        "what are the things you can do",
        "what can u do",
        "what can you do for me",
        "tell me what u can do",
        "what are ur capabilities",
        "how can u help me",
    ]:
        res = engine.ask(cap_q)
        assert res.status == BeliefStatus.SUPPORTED, f"Failed on {cap_q}"
        assert "LITTLE" in str(res.answer)
        assert "Deductive Reasoning" in str(res.answer)

    # 2. Discourse marker + arithmetic
    assert engine.ask("so what is 10+10").answer == 20
    assert engine.ask("well what is 20 + 30").answer == 50
    assert engine.ask("hey calculate 5 * 12").answer == 60
    assert engine.ask("now solve 100 / 4").answer == 25.0

    # 3. Natural language verbal math
    assert engine.ask("add 15 and 25").answer == 40
    assert engine.ask("the sum of 100 and 250").answer == 350
    assert engine.ask("multiply 7 by 8").answer == 56
    assert engine.ask("product of 9 and 9").answer == 81
    assert engine.ask("divide 144 by 12").answer == 12.0
    assert engine.ask("subtract 20 from 100").answer == 80

    # 4. Critical Guard: questions must NOT be learned as declarative facts
    for q_like in [
        "so what is 10+10",
        "what is an apple?",
        "who are you",
        "what can u do",
    ]:
        learn_res = engine.learn(q_like)
        assert learn_res.update_type.value == "NO_OP", (
            f"Failed for {q_like}: got {learn_res.update_type}"
        )


def test_inquisitor_skip_and_question_guard():
    """Verify that inquisitor does not create garbage relations when user asks a question or skips."""
    store = MemoryStore(":memory:")
    engine = LearningEngine(store)
    inquisitor = ActiveInquisitor(store, engine)

    res = engine.ask("What is a pulsar?")
    prompt = inquisitor.inspect_uncertainty(res, "pulsar", "__definition__", None)
    assert prompt is not None

    # User says skip / idk
    res_skip = inquisitor.resolve_response(prompt, "skip")
    assert res_skip.update_type.value == "NO_OP"

    res_idk = inquisitor.resolve_response(prompt, "idk")
    assert res_idk.update_type.value == "NO_OP"

    # User types a question instead of answering
    res_q = inquisitor.resolve_response(prompt, "so what is 10+10")
    assert res_q.update_type.value == "NO_OP"

    # Memory must NOT have 'pulsar is_a so what is 10+10'
    assert store.get_concept("so what is 10+10") is None
    assert store.get_concept("things u can do") is None


def test_multiturn_anaphora_resolution():
    """Verify that pronouns ('it', 'its', 'this', 'that') resolve across multi-turn dialogue."""
    store = MemoryStore(":memory:")
    engine = LearningEngine(store)

    # Turn 1: Teach elephant
    engine.learn("An elephant is a mammal.")
    engine.learn("A mammal is an animal.")
    engine.learn("An elephant has a trunk.")

    # Turn 2: Query elephant definition
    res_def = engine.ask("What is an elephant?")
    assert res_def.status == BeliefStatus.SUPPORTED
    assert "elephant" in res_def.verbalize().lower()

    # Turn 3: "Is it an animal?" -> resolves to "Is elephant an animal?"
    res_it_animal = engine.ask("Is it an animal?")
    assert res_it_animal.status == BeliefStatus.SUPPORTED
    assert res_it_animal.answer is True
    assert "elephant" in res_it_animal.verbalize().lower()

    # Turn 4: "Does it have a trunk?" -> resolves to "Does elephant have a trunk?"
    res_it_trunk = engine.ask("Does it have a trunk?")
    assert res_it_trunk.status == BeliefStatus.SUPPORTED
    assert res_it_trunk.answer is True

    # Turn 5: "It is grey." -> learns property color=grey on elephant
    res_learn_color = engine.learn("It is grey.")
    assert res_learn_color.update_type.value == "PROPERTY_UPDATE"

    # Turn 6: "What color is it?" -> resolves to "What color is elephant?"
    res_what_color = engine.ask("What color is it?")
    assert res_what_color.status == BeliefStatus.SUPPORTED
    assert "grey" in str(res_what_color.answer).lower()

    # Turn 7: "Is it red?" -> refutes red
    res_is_red = engine.ask("Is it red?")
    assert res_is_red.status == BeliefStatus.REFUTED
    assert res_is_red.answer is False


def test_irregular_plurals_and_lemmatization():
    """Verify that irregular plurals and noun inflections are cleanly lemmatized."""
    store = MemoryStore(":memory:")
    engine = LearningEngine(store)

    # 1. Mice -> mouse
    engine.learn("Mice are mammals.")
    res_mouse = engine.ask("Is a mouse a mammal?")
    assert res_mouse.status == BeliefStatus.SUPPORTED
    assert res_mouse.answer is True

    # 2. Geese -> goose
    engine.learn("Geese are birds.")
    res_goose = engine.ask("Is a goose a bird?")
    assert res_goose.status == BeliefStatus.SUPPORTED
    assert res_goose.answer is True

    # 3. Wolves -> wolf, teeth -> tooth
    engine.learn("Wolves have teeth.")
    res_wolf = engine.ask("Does a wolf have teeth?")
    assert res_wolf.status == BeliefStatus.SUPPORTED

    # 4. Children -> child, people -> person
    engine.learn("Children are humans.")
    res_child = engine.ask("Is a child a human?")
    assert res_child.status == BeliefStatus.SUPPORTED

    # 5. Leaves -> leaf
    engine.learn("Leaves are part of plants.")
    res_leaf = engine.ask("Is a leaf part of a plant?")
    assert res_leaf.status == BeliefStatus.SUPPORTED


def test_conversational_chitchat_and_fact_realizer():
    """Verify that pleasantries and chitchat receive articulate, polite responses."""
    store = MemoryStore(":memory:")
    store.seed_ontology()
    engine = LearningEngine(store)

    # Thanks
    for thanks_q in ["thank you", "thanks so much", "thx", "thank you very much"]:
        res_thx = engine.ask(thanks_q)
        assert res_thx.status == BeliefStatus.SUPPORTED
        assert "welcome" in res_thx.verbalize().lower()

    # How are you
    for how_q in ["how are you", "how are you doing", "how r u", "how is it going"]:
        res_how = engine.ask(how_q)
        assert res_how.status == BeliefStatus.SUPPORTED
        assert (
            "100%" in res_how.verbalize()
            or "operational" in res_how.verbalize().lower()
        )

    # Praise
    for praise_q in ["good job", "great job", "awesome", "well done", "you are smart"]:
        res_praise = engine.ask(praise_q)
        assert res_praise.status == BeliefStatus.SUPPORTED
        assert (
            "deterministic" in res_praise.verbalize().lower()
            or "thank you" in res_praise.verbalize().lower()
        )

    # Fact request
    res_fact = engine.ask("tell me a fact")
    assert res_fact.status == BeliefStatus.SUPPORTED
    assert "verified fact" in res_fact.verbalize().lower()
    assert (
        "zero hallucination" in res_fact.verbalize().lower()
        or "deductive" in res_fact.verbalize().lower()
    )


def test_extended_mathematics_sqrt_and_gcd():
    """Verify exact procedural computation of SQRT and GCD."""
    store = MemoryStore(":memory:")
    engine = LearningEngine(store)

    # Square root
    assert engine.ask("what is the square root of 144").answer == 12
    assert engine.ask("sqrt 64").answer == 8
    assert engine.ask("calculate the square root of 25").answer == 5
    assert engine.ask("square root of 2").answer == 1.414214
    v_sqrt = engine.ask("what is the square root of 144").verbalize()
    assert "square root of 144 is 12" in v_sqrt

    # GCD
    assert engine.ask("what is the gcd of 24 and 36").answer == 12
    assert engine.ask("gcd of 100 and 25").answer == 25
    v_gcd = engine.ask("calculate the gcd of 24 and 36").verbalize()
    assert "greatest common divisor of 24 and 36 is 12" in v_gcd


def test_foundational_commonsense_ontology_and_multi_hop_reasoning():
    """Verify that LITTLE solves diverse commonsense queries with multi-hop proofs and zero hallucination."""
    store = MemoryStore(":memory:")
    store.seed_ontology()
    engine = LearningEngine(store)

    # 1. Astronomy
    assert engine.ask("Is the sun a star?").answer is True
    res_sun_planet = engine.ask("Is the sun a planet?")
    assert res_sun_planet.answer is False
    assert res_sun_planet.status == BeliefStatus.REFUTED
    assert "disjoint" in res_sun_planet.verbalize().lower()

    assert engine.ask("Is mars a planet?").answer is True
    assert engine.ask("Is jupiter a planet?").answer is True
    assert engine.ask("Does earth have a moon?").answer is True

    # 2. Biology & Multi-hop Transitivity
    # Eagle is bird -> animal -> living thing (3 hops)
    res_eagle = engine.ask("Is an eagle a living thing?")
    assert res_eagle.answer is True
    assert "3 logical steps" in res_eagle.verbalize()

    # Inherited capabilities
    res_fly = engine.ask("Can an eagle fly?")
    assert res_fly.answer is True
    assert "because an eagle is a bird, which can fly" in res_fly.verbalize().lower()

    res_swim = engine.ask("Can a salmon swim?")
    assert res_swim.answer is True

    res_fish_water = engine.ask("Does a salmon live in water?")
    assert res_fish_water.answer is True

    res_tiger_meat = engine.ask("Does a tiger eat meat?")
    assert res_tiger_meat.answer is True

    # 3. Geography & Transitive Containment
    res_paris = engine.ask("Is Paris located in France?")
    assert res_paris.answer is True

    res_paris_europe = engine.ask("Is Paris located in Europe?")
    assert res_paris_europe.answer is True
    assert (
        "Paris is located in France, which is located in Europe"
        in res_paris_europe.verbalize()
    )

    res_tokyo_asia = engine.ask("Is Tokyo located in Asia?")
    assert res_tokyo_asia.answer is True

    # 4. Physical States of Matter & Chemistry
    res_water_liq = engine.ask("Is water a liquid?")
    assert res_water_liq.answer is True

    res_water_sol = engine.ask("Is water a solid?")
    assert res_water_sol.answer is False
    assert res_water_sol.status == BeliefStatus.REFUTED

    res_ice_water = engine.ask("Is ice made of water?")
    assert res_ice_water.answer is True

    res_steam_gas = engine.ask("Is steam a gas?")
    assert res_steam_gas.answer is True

    # 5. Artifacts vs Living Things
    res_car_animal = engine.ask("Is a car an animal?")
    assert res_car_animal.answer is False
    assert res_car_animal.status == BeliefStatus.REFUTED


def test_wh_questions_and_information_seeking():
    """Verify information-seeking WH questions for location, habitat, diet, capability, and composition."""
    memory = MemoryStore(":memory:", seed_ontology=True)
    engine = LearningEngine(memory)

    # 1. Location
    res_loc = engine.ask("Where is Paris located?")
    assert res_loc.status == BeliefStatus.SUPPORTED
    assert "France, Europe" in res_loc.verbalize()

    # 2. Habitat (inherited from fish)
    res_hab = engine.ask("Where does a salmon live?")
    assert res_hab.status == BeliefStatus.SUPPORTED
    assert "lives in water" in res_hab.verbalize()

    # 3. Diet (inherited from carnivore)
    res_diet = engine.ask("What does a lion eat?")
    assert res_diet.status == BeliefStatus.SUPPORTED
    assert "eats meat" in res_diet.verbalize()

    # 4. Capability (inherited from bird)
    res_cap = engine.ask("What can an eagle do?")
    assert res_cap.status == BeliefStatus.SUPPORTED
    assert "can fly" in res_cap.verbalize()

    # 5. Material / Composition
    res_mat = engine.ask("What is ice made of?")
    assert res_mat.status == BeliefStatus.SUPPORTED
    assert "made of water" in res_mat.verbalize()

    # 6. Parts / Features (duality)
    res_parts = engine.ask("What does a car have?")
    assert res_parts.status == BeliefStatus.SUPPORTED
    assert "wheel" in res_parts.verbalize().lower()


def test_why_questions_and_deductive_proofs():
    """Verify deductive proofs and causal explanations generated for Why questions."""
    memory = MemoryStore(":memory:", seed_ontology=True)
    engine = LearningEngine(memory)

    # Taxonomic proof
    res_why_eagle = engine.ask("Why is an eagle an animal?")
    assert res_why_eagle.status == BeliefStatus.SUPPORTED
    assert (
        res_why_eagle.verbalize() == "Because an eagle is a bird, which is an animal."
    )

    # Geographic containment proof
    res_why_paris = engine.ask("Why is Paris in Europe?")
    assert res_why_paris.status == BeliefStatus.SUPPORTED
    assert (
        "Paris is located in France, which is located in Europe"
        in res_why_paris.verbalize()
    )

    # Disjoint incompatibility proof
    res_why_water = engine.ask("Why is water not a solid?")
    assert res_why_water.status == BeliefStatus.SUPPORTED
    assert "liquid is disjoint with solid" in res_why_water.verbalize()


def test_comparatives_and_strict_order():
    """Verify transitive comparatives and asymmetric refutation."""
    memory = MemoryStore(":memory:", seed_ontology=True)
    engine = LearningEngine(memory)

    # Forward comparative
    res_forward = engine.ask("Is an elephant bigger than a mouse?")
    assert res_forward.status == BeliefStatus.SUPPORTED
    assert res_forward.answer is True

    # Transitive chain comparative: sun > earth > moon
    res_sun_moon = engine.ask("Is the sun larger than the moon?")
    assert res_sun_moon.status == BeliefStatus.SUPPORTED
    assert res_sun_moon.answer is True

    # Asymmetric reverse refutation
    res_reverse = engine.ask("Is the moon larger than the sun?")
    assert res_reverse.status == BeliefStatus.REFUTED
    assert res_reverse.answer is False
    assert "sun > earth > moon" in res_reverse.verbalize()


def test_non_monotonic_flightless_override():
    """Verify non-monotonic reasoning: penguins inherit bird category but override flying capability."""
    memory = MemoryStore(":memory:", seed_ontology=True)
    engine = LearningEngine(memory)

    # General bird can fly
    res_bird = engine.ask("Can an eagle fly?")
    assert res_bird.answer is True

    # Penguin is a bird
    res_penguin_bird = engine.ask("Is a penguin a bird?")
    assert res_penguin_bird.answer is True

    # Flightless exception
    res_penguin_fly = engine.ask("Can a penguin fly?")
    assert res_penguin_fly.answer is False
    assert res_penguin_fly.status == BeliefStatus.REFUTED
    assert "cannot fly" in res_penguin_fly.verbalize()


def test_advanced_mathematical_skills():
    """Verify newly registered deterministic mathematical skills via natural language questions."""
    memory = MemoryStore(":memory:", seed_ontology=True)
    engine = LearningEngine(memory)

    # Percentage
    res_pct = engine.ask("What is 15 percent of 200?")
    assert res_pct.answer == 30
    assert "15% of 200 is 30" in res_pct.verbalize()

    # LCM
    res_lcm = engine.ask("What is the lcm of 12 and 18?")
    assert res_lcm.answer == 36
    assert "least common multiple" in res_lcm.verbalize()

    # Square & Cube
    res_sq = engine.ask("What is the square of 9?")
    assert res_sq.answer == 81

    res_cb = engine.ask("What is the cube of 4?")
    assert res_cb.answer == 64

    # Average / Mean
    res_avg = engine.ask("What is the average of 10, 20, 30?")
    assert res_avg.answer == 20
    assert "average (mean) is 20" in res_avg.verbalize()

    # Solve linear equation
    res_lin = engine.ask("Solve 2x + 4 = 12")
    assert res_lin.answer == 4
    assert "solution is x = 4" in res_lin.verbalize()

    # Arithmetic expression
    res_expr = engine.ask("Calculate (25 * 4) - 10")
    assert res_expr.answer == 90
    assert "result of (25 * 4) - 10 is 90" in res_expr.verbalize()


def test_multi_turn_anaphora_dialogue():
    """Verify multi-turn conversation with dynamic pronoun and focus resolution."""
    memory = MemoryStore(":memory:", seed_ontology=True)
    engine = LearningEngine(memory)

    # Turn 1: Learn new concept
    learn1 = engine.learn("A cheetah is a mammal.")
    assert "cheetah" in learn1.message

    # Turn 2: Learn property with anaphora "It" -> cheetah
    learn2 = engine.learn("It is fast.")
    assert "fast" in learn2.message

    # Turn 3: Ask with anaphora "it" -> cheetah
    res_animal = engine.ask("Is it an animal?")
    assert res_animal.answer is True
    assert "cheetah is a mammal, which is an animal" in res_animal.verbalize().lower()

    # Turn 4: Verify property inherited
    res_fast = engine.ask("Is it fast?")
    assert res_fast.answer is True

    # Turn 5: Concept definition
    res_def = engine.ask("What is it?")
    assert res_def.status == BeliefStatus.SUPPORTED
    assert "cheetah" in res_def.verbalize().lower()


def test_part_whole_duality_and_inheritance_verbalization():
    """Verify fluent verbalization of inherited part-whole relationships and duality."""
    memory = MemoryStore(":memory:", seed_ontology=True)
    engine = LearningEngine(memory)

    # 1. Inherited has from part_of on ancestor (dog is_a animal, heart part_of animal)
    res_dog_heart = engine.ask("Does a dog have a heart?")
    assert res_dog_heart.status == BeliefStatus.SUPPORTED
    assert res_dog_heart.answer is True
    v1 = res_dog_heart.verbalize()
    assert "Yes, a dog has a heart" in v1
    assert "dog is an animal" in v1
    assert "heart is part of an animal" in v1 or "has a heart" in v1

    # 2. Inherited part_of via ancestor
    res_heart_dog = engine.ask("Is a heart part of a dog?")
    assert res_heart_dog.status == BeliefStatus.SUPPORTED
    assert res_heart_dog.answer is True
    v2 = res_heart_dog.verbalize()
    assert "Yes, a heart is part of a dog" in v2
    assert "heart is part of an animal" in v2
    assert "dog is an animal" in v2

    # 3. Direct has
    res_car_wheel = engine.ask("Does a car have a wheel?")
    assert res_car_wheel.status == BeliefStatus.SUPPORTED
    assert res_car_wheel.answer is True
    assert "Yes, a car has a wheel." == res_car_wheel.verbalize()

    # 4. Planetary proper noun has
    res_earth_moon = engine.ask("Does earth have a moon?")
    assert res_earth_moon.status == BeliefStatus.SUPPORTED
    assert res_earth_moon.answer is True
    assert "Yes, Earth has a moon." == res_earth_moon.verbalize()


def test_greetings_and_intransitive_capability_flow():
    """Verify polite greetings and intransitive capability verb learning & query flow."""
    memory = MemoryStore(":memory:", seed_ontology=True)
    engine = LearningEngine(memory)

    # 1. Greetings
    for g in ["hello", "hi", "hey", "good morning", "greetings"]:
        res_g = engine.ask(g)
        assert res_g.status == BeliefStatus.SUPPORTED
        assert "Hello! I am LITTLE" in res_g.verbalize()

    # 2. Intransitive capability learning and anaphora
    learn1 = engine.learn("A dolphin is a mammal.")
    assert "dolphin is_a mammal" in learn1.message

    learn2 = engine.learn("It swims.")
    assert "dolphin can swim" in learn2.message

    # 3. Query capability with does/do
    res_swim = engine.ask("Does it swim?")
    assert res_swim.status == BeliefStatus.SUPPORTED
    assert res_swim.answer is True
    assert "Yes, a dolphin can swim." == res_swim.verbalize()

    res_dolphin_swim = engine.ask("Does a dolphin swim?")
    assert res_dolphin_swim.status == BeliefStatus.SUPPORTED
    assert res_dolphin_swim.answer is True


def test_sympy_algebra_calculus_and_units():
    """Verify exact execution of SymPy-powered algebra, quadratic equations, calculus, and units."""
    memory = MemoryStore(":memory:", seed_ontology=True)
    engine = LearningEngine(memory)

    # 1. Quadratic equations
    res_quad = engine.ask("Solve x^2 - 5x + 6 = 0")
    assert res_quad.status == BeliefStatus.SUPPORTED
    assert res_quad.answer == [2, 3]
    assert "solutions are x = 2, 3" in res_quad.verbalize()

    # 2. Calculus derivative
    res_diff = engine.ask("What is the derivative of x^3 + 2x?")
    assert res_diff.status == BeliefStatus.SUPPORTED
    assert "3*x^2 + 2" in str(res_diff.answer)

    # 3. Algebraic simplification
    res_simp = engine.ask("Simplify 2x + 3x + 5")
    assert res_simp.status == BeliefStatus.SUPPORTED
    assert "5*x + 5" in str(res_simp.answer)

    # 4. Unit conversions
    res_temp = engine.ask("Convert 25 celsius to fahrenheit")
    assert res_temp.answer == 77
    assert "25 celsius is equal to 77 fahrenheit" in res_temp.verbalize()

    res_dist = engine.ask("Convert 100 km to miles")
    assert round(float(res_dist.answer), 2) == 62.14

    res_mass = engine.ask("Convert 10 kg to pounds")
    assert round(float(res_mass.answer), 2) == 22.05

    res_time = engine.ask("Convert 2 hours to minutes")
    assert res_time.answer == 120


def test_expanded_world_knowledge_deductions():
    """Verify multi-hop deductions across astronomy, global geography, particle physics, and technology."""
    memory = MemoryStore(":memory:", seed_ontology=True)
    engine = LearningEngine(memory)

    # 1. Global geography transitivity
    res_dc = engine.ask("Is Washington located in North America?")
    assert res_dc.answer is True
    assert "United states, which is located in North america" in res_dc.verbalize()

    res_bj = engine.ask("Is Beijing located in Asia?")
    assert res_bj.answer is True

    res_cairo = engine.ask("Is Cairo located in Africa?")
    assert res_cairo.answer is True

    # 2. Astronomy & Moons
    assert engine.ask("Does Saturn have Titan?").answer is True
    assert engine.ask("Does Mars have Phobos?").answer is True
    res_sun_mars = engine.ask("Is the sun larger than mars?")
    assert res_sun_mars.answer is True

    # 3. Particle Physics & Chemistry
    assert engine.ask("Is an electron part of an atom?").answer is True
    assert engine.ask("Is a proton part of an atom?").answer is True

    # 4. Anatomy & Technology
    assert engine.ask("Does a human have a brain?").answer is True
    assert engine.ask("Does a computer have a cpu?").answer is True
    res_sw_hw = engine.ask("Is software hardware?")
    assert res_sw_hw.answer is False
    assert res_sw_hw.status == BeliefStatus.REFUTED


def test_complex_and_compound_english_parsing():
    """Verify learning from multi-sentence paragraphs, relative clauses, conjoined lists, and conditionals."""
    memory = MemoryStore(":memory:", seed_ontology=True)
    engine = LearningEngine(memory)

    # 1. Multi-sentence paragraph input with anaphora resolution
    res_para = engine.learn(
        "A golden retriever is a dog. It has thick fur. It can swim."
    )
    assert len(res_para.relations_created) == 3
    assert engine.ask("Is a golden retriever an animal?").answer is True
    assert engine.ask("Can a golden retriever swim?").answer is True
    assert engine.ask("Does a golden retriever have fur?").answer is True

    # 2. Relative clause learning
    res_rel = engine.learn("Dolphins are mammals that live in the ocean and have fins.")
    assert len(res_rel.relations_created) >= 2
    assert engine.ask("Is a dolphin an animal?").answer is True
    assert engine.ask("Does a dolphin live in the ocean?").answer is True
    assert engine.ask("Does a dolphin have fins?").answer is True

    # 3. Conjoined objects in direct clauses
    engine.learn("A truck has wheels, an engine, and headlights.")
    assert engine.ask("Does a truck have wheels?").answer is True
    assert engine.ask("Does a truck have headlights?").answer is True

    # 4. Universal quantifiers and conditional rules
    engine.learn("A tiger is a feline.")
    engine.learn("All felines are carnivores.")
    assert engine.ask("Is a tiger a carnivore?").answer is True

    engine.learn("A wolf is a canine.")
    engine.learn("If an animal is a canine, then it is a vertebrate.")
    assert engine.ask("Is a wolf a vertebrate?").answer is True

    # 5. Negative capability statements
    engine.learn("Ostriches cannot fly.")
    res_ost = engine.ask("Can an ostrich fly?")
    assert res_ost.answer is False
    assert res_ost.status == BeliefStatus.REFUTED


def test_compound_and_conjoined_questions():
    """Verify answering compound conjoined queries in a single conversational turn."""
    memory = MemoryStore(":memory:", seed_ontology=True)
    engine = LearningEngine(memory)

    # 1. Conjoined relational query with coreference: "Can an eagle fly and does it have wings?"
    res_eagle = engine.ask("Can an eagle fly and does it have wings?")
    assert res_eagle.answer is True
    assert res_eagle.status == BeliefStatus.SUPPORTED
    assert "fly" in res_eagle.verbalize().lower()
    assert "wing" in res_eagle.verbalize().lower()
    assert len(res_eagle.evidence) >= 2

    # 2. Conjoined math query: "What is 10 + 10 and what is 5 * 5?"
    res_math = engine.ask("What is 10 + 10 and what is 5 * 5?")
    assert res_math.status == BeliefStatus.SUPPORTED
    assert "20" in res_math.verbalize()
    assert "25" in res_math.verbalize()

    # 3. Mixed truth value: True and False
    engine.learn("Ostriches cannot fly.")
    res_mixed = engine.ask("Can an eagle fly and can an ostrich fly?")
    assert res_mixed.answer is False
    assert res_mixed.status == BeliefStatus.REFUTED


def test_user_identity_and_inquisitor_guardrails():
    """Verify greeting prefix stripping, user identity learning, and inquisitor meta-word guards."""
    from little.active.inquisitor import ActiveInquisitor

    memory = MemoryStore(":memory:", seed_ontology=True)
    engine = LearningEngine(memory)
    inquisitor = ActiveInquisitor(memory, engine)

    # 1. Greeting prefix stripped on identity question
    res_ident = engine.ask("hii who are you")
    assert res_ident.status == BeliefStatus.SUPPORTED
    assert "LITTLE" in res_ident.verbalize()

    # 2. Unknown user name returns clear conversational question
    res_name = engine.ask("what is my name")
    assert res_name.status == BeliefStatus.UNKNOWN
    assert "I do not know your name yet" in res_name.verbalize()

    # 3. Active inquisitor must NEVER ask questions about 'my name' or pronouns
    parsed = SimpleParser.parse_question("what is my name")
    assert parsed is not None
    s, p, o = parsed
    prompt = inquisitor.inspect_uncertainty(res_name, s, p, o)
    assert prompt is None

    # 4. Learning user name via statement
    learn_res = engine.learn("ok my name is aswin")
    assert learn_res.update_type != UpdateType.NO_OP
    assert "Aswin" in learn_res.message

    # 5. Query user name after learning
    res_name_after = engine.ask("what is my name")
    assert res_name_after.status == BeliefStatus.SUPPORTED
    assert "aswin" in res_name_after.verbalize().lower()

    # 6. Compound statement + question turn: "ok my name is aswin and what is your name"
    res_compound = engine.ask("ok my name is aswin and what is your name")
    assert res_compound.status == BeliefStatus.SUPPORTED
    assert "Aswin" in res_compound.verbalize()
    assert "LITTLE" in res_compound.verbalize()
