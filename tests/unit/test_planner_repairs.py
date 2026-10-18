from dataclasses import replace
from unittest.mock import Mock

import pytest

from little.actions.agentic_planner import AgenticPlanner
from little.actions.models import Plan, PlanStep, PlanStepStatus, ToolExecutionResult
from little.actions.planning_policy import AgenticPlanningPolicy


def step(index=1, expression="1+1", dependencies=()):
    return PlanStep(
        index, "calculate", "calculator", {"expression": expression}, dependencies
    )


@pytest.mark.parametrize(
    "goal", ["", "dance around", "calculate 1+1 then dance around"]
)
def test_incomplete_goal_is_not_success(goal):
    planner = AgenticPlanner()
    result = planner.plan_and_execute(goal)
    assert not result.success
    assert result.error
    assert result.diagnostics
    assert not any(s.result for s in result.steps)


def test_oversized_decomposition_is_not_silently_truncated():
    planner = AgenticPlanner(policy=replace(AgenticPlanningPolicy(), max_steps=1))
    plan = planner.decompose_goal("calculate 1+1 then calculate 2+2")
    assert len(plan.steps) == 2
    result = planner.execute_plan(plan)
    assert not result.success
    assert "max_steps" in result.error


@pytest.mark.parametrize(
    "steps",
    [(step(1), step(1)), (step(2), step(1)), (step(0),), (step(1, dependencies=(5,)),)],
)
def test_invalid_supplied_plan_rejected_before_dispatch(steps):
    executor = Mock()
    executor.execute.return_value = ToolExecutionResult(
        "calculator", {}, True, output="2"
    )
    result = AgenticPlanner(tool_executor=executor).execute_plan(
        Plan("bad", "goal", steps)
    )
    assert not result.success
    executor.execute.assert_not_called()


def test_supplied_plan_exceeding_budget_is_rejected():
    result = AgenticPlanner(
        policy=replace(AgenticPlanningPolicy(), max_steps=1)
    ).execute_plan(Plan("big", "goal", (step(1), step(2))))
    assert not result.success


def test_missing_variable_reference_rejected_before_side_effect():
    executor = Mock()
    plan = Plan("missing", "goal", (step(1, "$step_99 + 1"),))
    result = AgenticPlanner(tool_executor=executor).execute_plan(plan)
    assert not result.success
    assert "99" in result.error
    executor.execute.assert_not_called()


def test_interpolation_consumes_complete_tokens_once():
    planner = AgenticPlanner()
    args = planner._resolve_argument_variables(
        "file_writer",
        {"content": "$step_10.output and $step_1"},
        {1: "$step_10", 10: "ten"},
        10,
    )
    assert args["content"] == "ten and $step_10"


def test_missing_binding_raises_instead_of_inventing_empty_string():
    with pytest.raises(ValueError, match="99"):
        AgenticPlanner()._resolve_argument_variables(
            "calculator", {"expression": "$step_99.output"}, {}, 0
        )


def test_grounding_failure_is_partial_failure_and_skips_dependent():
    grounder = Mock()
    grounder.memory_store = object()
    grounder.ground.return_value = False
    result = AgenticPlanner(tool_grounder=grounder).execute_plan(
        Plan("ground", "goal", (step(1), step(2, "$prev + 1", (1,))))
    )
    assert not result.success
    assert result.steps[0].status == PlanStepStatus.FAILED
    assert result.steps[0].result.success  # Tool ran, but persistence failed.
    assert result.steps[1].status == PlanStepStatus.SKIPPED
    assert "ground" in result.error.lower()


def test_execution_without_memory_reports_provenance_unavailable():
    result = AgenticPlanner().execute_plan(Plan("plain", "goal", (step(1),)))
    assert result.success
    assert not result.provenance_available
    assert "I_ground_provenance" not in result.invariants_checked
    assert any("unavailable" in d for d in result.diagnostics)


def test_plan_reuse_resets_retries_and_preserves_previous_results():
    executor = Mock()
    fail = ToolExecutionResult(
        "calculator", {"expression": "2+2?"}, False, error="punctuation"
    )
    ok = ToolExecutionResult("calculator", {"expression": "2+2"}, True, output="4")
    executor.execute.side_effect = [fail, ok, fail, ok]
    planner = AgenticPlanner(tool_executor=executor)
    original = step(1, "2+2?")
    plan = Plan("reuse", "goal", (original,))
    first = planner.execute_plan(plan)
    second = planner.execute_plan(plan)
    assert first.success and second.success
    assert first.retries_used == second.retries_used == 1
    assert original.retries == 0
    assert original.status == PlanStepStatus.PENDING
    assert first.steps[0] is not second.steps[0]


def test_independent_branch_runs_after_failed_step():
    result = AgenticPlanner().execute_plan(
        Plan("branch", "goal", (step(1, "1/0"), step(2, "2+2")))
    )
    assert not result.success
    assert result.steps[1].status == PlanStepStatus.SUCCESS
    assert result.steps[1].result.output == 4


def test_quoted_conjunction_is_content_not_extra_goal():
    plan = AgenticPlanner().decompose_goal(
        'write to file note.txt: "first then second"'
    )
    assert len(plan.steps) == 1
    assert not plan.diagnostics


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_steps": 0},
        {"max_steps": True},
        {"max_retries_per_step": -1},
        {"enable_backtracking": "false"},
        {"step_timeout_seconds": 0},
    ],
)
def test_invalid_planning_policy_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        AgenticPlanningPolicy(**kwargs)


def test_partly_recognized_compound_goal_keeps_planner_route():
    assert AgenticPlanner().is_compound_goal("calculate 1+1 then dance around")


def test_read_between_two_writes_is_valid_resource_order(tmp_path):
    from little.actions.tool_executor import ToolExecutor
    from little.actions.tool_policy import ActionPolicy

    planner = AgenticPlanner(
        action_policy=ActionPolicy.default(workspace_root=tmp_path),
        tool_executor=ToolExecutor(workspace_root=tmp_path),
    )
    plan = Plan(
        "writes",
        "write read write",
        (
            PlanStep(
                1,
                "write",
                "file_writer",
                {"file_path": "value.txt", "content": "first"},
            ),
            PlanStep(2, "read", "file_reader", {"file_path": "value.txt"}, (1,)),
            PlanStep(
                3,
                "write",
                "file_writer",
                {"file_path": "value.txt", "content": "last"},
                (2,),
            ),
        ),
    )
    result = planner.execute_plan(plan)
    assert result.success
    assert result.steps[1].result.output == "first"
    assert (tmp_path / "value.txt").read_text() == "last"


def test_grounding_exception_is_reported_as_partial_failure():
    import sqlite3

    grounder = Mock()
    grounder.memory_store = object()
    grounder.ground.side_effect = sqlite3.OperationalError("database locked")
    result = AgenticPlanner(tool_grounder=grounder).execute_plan(
        Plan("ground", "goal", (step(1),))
    )
    assert not result.success
    assert any("database locked" in d for d in result.diagnostics)


def test_each_attempt_carries_step_timeout_and_confirmation():
    executor = Mock()
    executor.execute.return_value = ToolExecutionResult(
        "calculator", {}, True, output=2
    )
    policy = replace(AgenticPlanningPolicy(), step_timeout_seconds=0.25)
    supplied = replace(step(1), confirmed=True)
    result = AgenticPlanner(tool_executor=executor, policy=policy).execute_plan(
        Plan("bound", "goal", (supplied,))
    )
    assert result.success
    call = executor.execute.call_args
    assert call.args[0].confirmed is True
    assert call.kwargs["timeout_seconds"] == 0.25
