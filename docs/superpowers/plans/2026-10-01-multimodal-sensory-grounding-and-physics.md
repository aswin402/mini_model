# Multimodal / Sensory Grounding & Continuous Dynamic Physics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement multimodal sensory grounding (0° episodic axis) and continuous-time Closed-Form Continuous (CfC) physics simulation with Qualitative Process Theory (QPT) mass conservation and conversational inquiry.

**Architecture:**
- `src/little/perception/sensory.py`: Sensory observation data structures (`SensoryObservation`, `SensoryModality`), continuous feature extraction, concept matching, zero-hallucination bounds checking, and anchoring to the 0° Episodic axis of Concept Knots.
- `src/little/dynamics/cfc_ode.py` & `transformations.py`: Extended physical action jumps (`slice`, `peel`, `heat`, `chill`, `dehydrate`) with boundary topology transitions and strict mass conservation ($\sum m_i = M$).
- `src/little/dynamics/continuous_simulation.py`: Simulation engine tracking continuous-time physical trajectories $s(t) = \text{CfC}(s_0, \Delta t, T, \text{exposed})$ across hours/days.
- `src/little/core/cognitive_command.py` & `dialogue_surface.py`: Conversational directives and verbalizers for physical predictions, action jumps, and sensory perception.

**Tech Stack:** Pure CPU Python (no PyTorch/CUDA), math/numpy, SQLite memory store, JSON schemas.

## Global Constraints
- Pure CPU execution (no GPU/CUDA/PyTorch).
- Working RAM < 70 MB, process RSS < 500 MB.
- 100% backward compatibility: all 579 existing tests must pass.
- Open-World Assumption (OWA) compliance: zero hallucination, strict invariant checks.
- Data-driven schemas: actions and state variables configured in `data/schemas/`.

---

### Task 1: Action & State Variable Schema Extensions
**Files:**
- Modify: `data/schemas/action_schemas.json`
- Modify: `data/schemas/state_variables.json`
- Test: `tests/unit/test_schema_registry.py` (or existing registry tests)

**Interfaces:**
- Produces: New actions `peel`, `heat`, `chill`, `dehydrate` with argument types, defaults, and invariant specifications.

- [ ] **Step 1: Write the failing test for new action schemas**
- [ ] **Step 2: Update `data/schemas/action_schemas.json` and `data/schemas/state_variables.json`**
- [ ] **Step 3: Run schema registry tests and verify passing**
- [ ] **Step 4: Commit schema updates**

---

### Task 2: Extended Hybrid Automaton Action Jumps & QPT Mass Conservation
**Files:**
- Modify: `src/little/dynamics/cfc_ode.py`
- Modify: `src/little/dynamics/transformations.py`
- Test: `tests/unit/test_cfc_extended_physics.py`

**Interfaces:**
- Consumes: `ConceptKnot`, `ContinuousPhysicalState`, `SchemaRegistry`
- Produces: `apply_action_jump(knot, action, **kwargs)` supporting `slice`, `peel`, `heat`, `chill`, `dehydrate` with mass conservation.

- [ ] **Step 1: Write the failing tests in `tests/unit/test_cfc_extended_physics.py`**
- [ ] **Step 2: Implement action jumps in `src/little/dynamics/cfc_ode.py`**
- [ ] **Step 3: Implement transformation methods in `src/little/dynamics/transformations.py`**
- [ ] **Step 4: Run unit tests and verify passing**
- [ ] **Step 5: Commit extended physics action jumps**

---

### Task 3: Continuous-Time Physical Simulation Engine
**Files:**
- Create: `src/little/dynamics/continuous_simulation.py`
- Test: `tests/unit/test_continuous_simulation.py`

**Interfaces:**
- Consumes: `CfCContinuousODE`, `ContinuousPhysicalState`, `ConceptKnot`
- Produces: `PhysicalSimulationEngine.simulate(knot_or_name, delta_t_hours, ambient_temperature_c, skin_intact)` returning `SimulationResult` with state trajectory, decay milestones, and browning/moisture events.

- [ ] **Step 1: Write the failing tests in `tests/unit/test_continuous_simulation.py`**
- [ ] **Step 2: Implement `PhysicalSimulationEngine` in `src/little/dynamics/continuous_simulation.py`**
- [ ] **Step 3: Run simulation unit tests and verify passing**
- [ ] **Step 4: Commit continuous simulation engine**

---

### Task 4: Multimodal Sensory Grounding Engine (0° Episodic Axis)
**Files:**
- Create: `src/little/perception/sensory.py`
- Test: `tests/unit/test_sensory_grounding.py`

**Interfaces:**
- Consumes: `MemoryStore`, `ConceptKnot`, `ContinuousPhysicalState`
- Produces: `SensoryGroundingEngine.ground_observation(observation)` -> `GroundingResult`, `parse_sensory_directive(text)` -> `SensoryObservation`.

- [ ] **Step 1: Write failing tests in `tests/unit/test_sensory_grounding.py`**
- [ ] **Step 2: Implement `SensoryObservation` and `SensoryGroundingEngine` in `src/little/perception/sensory.py`**
- [ ] **Step 3: Run unit tests and verify passing**
- [ ] **Step 4: Commit sensory grounding engine**

---

### Task 5: Conversational Directives, Dialogue Verbalization & Integration Tests
**Files:**
- Modify: `src/little/language/dialogue_surface.py`
- Modify: `src/little/core/cognitive_command.py`
- Modify: `src/little/core/kernel.py`
- Create: `tests/integration/test_multimodal_physics_chat_integration.py`

**Interfaces:**
- Consumes: `PhysicalSimulationEngine`, `SensoryGroundingEngine`, `TransformationEngine`
- Produces: Dialogue commands `"simulate <concept> for <N> <unit>"`, `"what happens if I <action> <concept>"`, `"sensory observation: [...]"`.

- [ ] **Step 1: Write integration tests in `tests/integration/test_multimodal_physics_chat_integration.py`**
- [ ] **Step 2: Add verbalizers in `src/little/language/dialogue_surface.py`**
- [ ] **Step 3: Add cognitive commands in `src/little/core/cognitive_command.py` and attach to `CognitiveKernel`**
- [ ] **Step 4: Run integration tests and full test suite (all 579+ tests)**
- [ ] **Step 5: Commit completed multimodal physics chat integration**
