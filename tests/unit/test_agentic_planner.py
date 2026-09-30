"""Unit tests for the Multi-Step Agentic Planning Loop and ReAct Execution Engine."""

from pathlib import Path

from little.actions import (
    ActionPolicy,
    ActionRouter,
    AgenticPlanExecutionResult,
    AgenticPlanner,
    AgenticPlanningPolicy,
    Plan,
    PlanStep,
    PlanStepStatus,
    ToolDefinitionsPolicy,
    ToolExecutor,
    ToolGroundingEngine,
)
from little.memory.store import MemoryStore


def test_agentic_planning_policy_defaults():
    """Verify AgenticPlanningPolicy loads with valid bounds and invariants."""
    policy = AgenticPlanningPolicy.default()
    assert policy.max_steps == 8
    assert policy.max_retries_per_step == 2
    assert policy.enable_backtracking is True
    assert "calculator" in policy.chainable_tools
    assert "file_writer" in policy.chainable_tools
    assert "I_dag_acyclicity" in policy.invariant_gates


def test_is_compound_goal():
    """Verify detection of compound multi-step goals vs single tool commands."""
    planner = AgenticPlanner()

    # Single tool actions should NOT be compound
    assert not planner.is_compound_goal("calculate 25 * 4")
    assert not planner.is_compound_goal("what time is it?")
    assert not planner.is_compound_goal("search for Europa")
    assert not planner.is_compound_goal("read file test.txt")
    assert not planner.is_compound_goal("A falcon is a bird.")

    # Multi-step chains SHOULD be compound
    assert planner.is_compound_goal("calculate 25 * 4 and then write the result to calc.txt")
    assert planner.is_compound_goal("calculate 15 * 4, then write the answer to math.txt, and then read file math.txt")
    assert planner.is_compound_goal("search for Europa and then save to europa.txt")
    assert planner.is_compound_goal("step 1: calculate 10 + 20; step 2: write to sum.txt")


def test_decompose_goal_dag():
    """Verify that compound goals decompose into an ordered DAG with correct dependencies."""
    planner = AgenticPlanner()
    plan = planner.decompose_goal("calculate 25 * 4, then write the result to calc.txt, and then read file calc.txt")

    assert len(plan.steps) == 3
    assert plan.is_acyclic()

    # Step 1: calculator
    assert plan.steps[0].step_index == 1
    assert plan.steps[0].tool_name == "calculator"
    assert "25 * 4" in plan.steps[0].arguments_template["expression"]
    assert plan.steps[0].dependencies == ()

    # Step 2: file_writer
    assert plan.steps[1].step_index == 2
    assert plan.steps[1].tool_name == "file_writer"
    assert plan.steps[1].arguments_template["file_path"] == "calc.txt"
    assert 1 in plan.steps[1].dependencies

    # Step 3: file_reader
    assert plan.steps[2].step_index == 3
    assert plan.steps[2].tool_name == "file_reader"
    assert plan.steps[2].arguments_template["file_path"] == "calc.txt"
    assert 2 in plan.steps[2].dependencies


def test_dag_acyclicity_invariant():
    """Verify I_DAG gate detects cycles or illegal forward references."""
    # Valid acyclic plan
    step1 = PlanStep(step_index=1, description="calc", tool_name="calculator", arguments_template={"expression": "1+1"})
    step2 = PlanStep(step_index=2, description="write", tool_name="file_writer", arguments_template={"file_path": "a.txt", "content": "$prev"}, dependencies=(1,))
    valid_plan = Plan(plan_id="valid", goal="test", steps=(step1, step2))
    assert valid_plan.is_acyclic()

    # Invalid cyclic plan (step 1 depends on step 2)
    bad_step1 = PlanStep(step_index=1, description="calc", tool_name="calculator", arguments_template={"expression": "1+1"}, dependencies=(2,))
    bad_plan = Plan(plan_id="invalid", goal="test", steps=(bad_step1, step2))
    assert not bad_plan.is_acyclic()

    planner = AgenticPlanner()
    res = planner.execute_plan(bad_plan)
    assert not res.success
    assert "I_DAG violation" in (res.error or "")


def test_mutex_invariant_gate():
    """Verify I_mutex gate rejects plans attempting to read a file before it is written in the same plan."""
    planner = AgenticPlanner()
    step_read = PlanStep(step_index=1, description="read", tool_name="file_reader", arguments_template={"file_path": "out.txt"})
    step_write = PlanStep(step_index=2, description="write", tool_name="file_writer", arguments_template={"file_path": "out.txt", "content": "data"})
    bad_plan = Plan(plan_id="mutex_fail", goal="test", steps=(step_read, step_write))

    res = planner.execute_plan(bad_plan)
    assert not res.success
    assert "resource mutex violation" in (res.error or "")


def test_plan_execution_with_variable_binding(tmp_path: Path):
    """Verify end-to-end execution of a 3-step action chain with variable passing."""
    memory = MemoryStore(":memory:")
    action_policy = ActionPolicy.default(workspace_root=tmp_path)
    planner = AgenticPlanner(
        action_policy=action_policy,
        tool_executor=ToolExecutor(workspace_root=tmp_path),
        memory=memory,
    )

    goal = "calculate 15 * 8, then write the result to math_out.txt, and then read file math_out.txt"
    res = planner.plan_and_execute(goal)

    assert res.success
    assert len(res.steps) == 3
    assert res.final_output == "120"
    assert len(res.trace_hash) == 16
    assert "I_dag_acyclicity" in res.invariants_checked
    assert "I_ground_provenance" in res.invariants_checked

    # Verify that file was created on disk in workspace
    created_file = tmp_path / "math_out.txt"
    assert created_file.exists()
    assert created_file.read_text(encoding="utf-8") == "120"

    # Verify memory grounding
    exp_count = memory._conn.execute(
        "SELECT COUNT(*) AS c FROM experiences WHERE source = 'tool_execution';"
    ).fetchone()["c"]
    assert exp_count == 3


def test_backtracking_and_self_correction(tmp_path: Path):
    """Verify that planner automatically backtracks and repairs recoverable step errors."""
    action_policy = ActionPolicy.default(workspace_root=tmp_path)
    planner = AgenticPlanner(
        action_policy=action_policy,
        tool_executor=ToolExecutor(workspace_root=tmp_path),
    )

    # Missing import 'math' in Python snippet
    goal = "run python: print(math.sqrt(169)) and then write the result to root.txt"
    res = planner.plan_and_execute(goal)

    assert res.success
    assert res.backtracks_count >= 1
    assert "13.0" in str(res.steps[0].result.output)

    written = (tmp_path / "root.txt").read_text(encoding="utf-8")
    assert "13.0" in written


def test_step_failure_and_skip(tmp_path: Path):
    """Verify that when a step fails unrecoverably, dependent downstream steps are cleanly skipped."""
    action_policy = ActionPolicy.default(workspace_root=tmp_path)
    planner = AgenticPlanner(
        action_policy=action_policy,
        tool_executor=ToolExecutor(workspace_root=tmp_path),
    )

    # Step 1 attempts to read nonexistent file -> fails
    step1 = PlanStep(step_index=1, description="read", tool_name="file_reader", arguments_template={"file_path": "nonexistent_file_123.txt"})
    # Step 2 depends on step 1
    step2 = PlanStep(step_index=2, description="write", tool_name="file_writer", arguments_template={"file_path": "out.txt", "content": "$prev"}, dependencies=(1,))
    plan = Plan(plan_id="fail_test", goal="test", steps=(step1, step2))

    res = planner.execute_plan(plan)
    assert not res.success
    assert res.steps[0].status == PlanStepStatus.FAILED
    assert res.steps[1].status == PlanStepStatus.SKIPPED
