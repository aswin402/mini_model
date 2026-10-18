from little.actions import ToolCall, ToolExecutor
from little.procedural.code_sandbox import CodeSandbox
from little.procedural.math_cas import SymbolicCAS


def test_rational_is_exact_and_rejects_objects():
    cas = SymbolicCAS()
    assert cas.solve_rational_expression('(9007199254740993 / 1)+0').result == 9007199254740993
    assert cas.solve_rational_expression('Fraction.__new__.__globals__').status == 'ERROR'
    assert cas.solve_rational_expression('2**1000000000').status == 'ERROR'
    assert cas.solve_rational_expression('1e1000000000').status == 'ERROR'


def test_assertions_cannot_bypass_safety():
    result = CodeSandbox().run_assertions("import os\ndef f(): return 1", ['f() == 1'])
    assert not result.success
    result = CodeSandbox().run_assertions('def f(): return 1', ["open('/etc/passwd')"])
    assert not result.success


def test_tools_fail_closed(tmp_path):
    from dataclasses import replace

    from little.actions import ToolDefinitionsPolicy
    policy = ToolDefinitionsPolicy.default()
    policy.tools['file_writer'] = replace(policy.tools['file_writer'], requires_confirmation=True)
    executor = ToolExecutor(workspace_root=tmp_path, tool_policy=policy)
    for tool, args in [('python_repl', {'code': "import os\nprint(os.environ)"}), ('bash_executor', {'command': 'echo bad > owned'}), ('bash_executor', {'command': 'git reset --hard'}), ('file_writer', {'file_path': 'owned', 'content': 'bad'})]:
        assert not executor.execute(ToolCall(tool, args)).success
    assert not (tmp_path / 'owned').exists()
    assert not executor.execute(ToolCall('calculator', {'expression': '1', 'unexpected': True})).success
    assert not executor.execute(ToolCall('file_writer', {'file_path': 'owned', 'content': 'bad', 'append': 'false'}, confirmed=True)).success


def test_python_benign_imports_and_reflection_denial(tmp_path):
    executor = ToolExecutor(workspace_root=tmp_path)
    good = executor.execute(ToolCall('python_repl', {'code': "import hashlib\nimport itertools\nprint(hashlib.sha256(b'test').hexdigest())\nprint(list(itertools.chain([1], [2])))"}))
    assert good.success
    for code in ["print((1).__class__)", "import operator\nf = operator.attrgetter('__class__')", "from operator import attrgetter"]:
        assert not executor.execute(ToolCall('python_repl', {'code': code})).success


def test_resource_budgets_and_skill_denial():
    from dataclasses import replace

    from little.core.models import Skill
    from little.procedural.code_policy import CodeSynthesizerPolicy
    from little.procedural.runner import SkillRunner
    policy = CodeSynthesizerPolicy.default()
    sandbox = CodeSandbox(replace(policy, sandbox=replace(policy.sandbox, max_output_bytes=128)))
    assert not sandbox.execute_script("print('x' * 1000)").success
    assert not sandbox.execute_script("x = [0] * 100000000").success
    skill = Skill.create(name='ESCAPE', parameters=[], code_body='math.__dict__')
    assert not SkillRunner.execute(skill).success


def test_timezone_and_byte_read_limits(tmp_path):
    from datetime import datetime
    executor = ToolExecutor(workspace_root=tmp_path)
    result = executor.execute(ToolCall('system_time', {'timezone': 'Asia/Kolkata'}))
    assert result.success
    assert datetime.fromisoformat(result.output['iso']).utcoffset().total_seconds() == 19800
    assert not executor.execute(ToolCall('system_time', {'timezone': 'invalid/zone'})).success
    (tmp_path / 'text').write_text('é' * 1000)
    result = executor.execute(ToolCall('file_reader', {'file_path': 'text', 'max_bytes': 4}))
    assert result.output == 'éé'
    assert not executor.execute(ToolCall('file_reader', {'file_path': 'text', 'max_bytes': -1})).success


def test_global_timeout_clamps_tool_timeout(tmp_path):
    from dataclasses import replace

    from little.actions import ActionPolicy
    executor = ToolExecutor(workspace_root=tmp_path, action_policy=replace(ActionPolicy.default(workspace_root=tmp_path), max_execution_timeout_sec=0.05))
    result = executor.execute(ToolCall('python_repl', {'code': 'while True: pass'}))
    assert not result.success
    assert 'timed out' in result.error


def test_symbolic_skills_use_safe_parser():
    from little.procedural.runner import SkillRunner
    from little.procedural.skills import get_builtin_skills
    skills = {skill.name: skill for skill in get_builtin_skills()}
    result = SkillRunner.execute(skills['SOLVE_QUADRATIC'], equation='x^2 - 5x + 6 = 0')
    assert result.success, result.error
    assert result.result == [2, 3]
    result = SkillRunner.execute(skills['CALCULUS_DERIVATIVE'], expression='x^3 + sin(x)')
    assert result.success, result.error
    assert '3*x^2' in result.result
    result = SkillRunner.execute(skills['SIMPLIFY_EXPR'], expression='(x^2 - 1)/(x - 1)')
    assert result.success, result.error
    assert result.result == 'x + 1'
    result = SkillRunner.execute(skills['SIMPLIFY_EXPR'], expression="__import__('os').system('id')")
    assert not result.success


def test_per_call_timeout_is_local(tmp_path):
    executor = ToolExecutor(workspace_root=tmp_path)
    result = executor.execute(ToolCall('python_repl', {'code': 'while True: pass'}), timeout_seconds=0.05)
    assert not result.success
    assert 'timed out' in result.error
    assert executor.execute(ToolCall('python_repl', {'code': 'print(2)'})).success


def test_symbolic_facade_rejects_strings_and_format_reflection():
    from little.core.models import Skill
    from little.procedural.runner import SkillRunner
    for code in ["sympy.simplify(\"__import__('os').getcwd()\")", "'{0.__globals__}'.format(parse_expr)", "sympy.Symbol('x').subs('x', 'unsafe')"]:
        skill = Skill.create(name='UNSAFE_SYMBOLIC', parameters=[], code_body=code)
        assert not SkillRunner.execute(skill).success


def test_sandbox_recursion_and_int_digit_limits():
    sandbox = CodeSandbox()
    rec_res = sandbox.execute_script("def f(n):\n    return f(n + 1)\nf(0)")
    assert not rec_res.success
    digit_res = sandbox.execute_script("print(str(10**5000))")
    assert not digit_res.success


def test_prune_polluted_concepts_removes_artifacts_and_preserves_valid():
    from little.memory.store import MemoryStore
    store = MemoryStore(":memory:", seed_ontology=False)
    valid = store.create_concept("robin", category="bird")
    polluted1 = store.create_concept("test animal")
    store.create_concept("portable personal computer with a battery and screen")
    store.add_relation(polluted1.id, "is_a", valid.id)

    dry_names = store.prune_polluted_concepts(dry_run=True)
    assert set(dry_names) == {"portable personal computer with a battery and screen", "test animal"}
    assert store.get_concept("test animal") is not None

    removed = store.prune_polluted_concepts(dry_run=False)
    assert set(removed) == {"portable personal computer with a battery and screen", "test animal"}
    assert store.get_concept("test animal") is None
    assert store.get_concept("robin") is not None
    assert store.get_relations(subject_id=polluted1.id) == []


def test_bwrap_or_rlimit_sandbox_isolation():
    from little.procedural import restricted_execution
    prefix = restricted_execution._get_bwrap_prefix()
    assert isinstance(prefix, list)
    proc = restricted_execution.run("print(21 * 2)")
    assert proc.returncode == 0
    assert proc.stdout.strip() == "42"


def test_cli_interact_and_memory_prune(tmp_path):
    import subprocess
    import sys
    db_file = tmp_path / "cli_test.db"
    interact_proc = subprocess.run(
        [sys.executable, "-m", "little.main", "--db", str(db_file), "interact"],
        input="A kiwi is a bird.\nwhat is a kiwi?\nexit\n",
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert interact_proc.returncode == 0
    assert "A kiwi is a bird." in interact_proc.stdout

    prune_proc = subprocess.run(
        [sys.executable, "-m", "little.main", "--db", str(db_file), "memory", "prune", "--dry-run"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert prune_proc.returncode == 0
    assert "Matched Polluted Concepts: 0" in prune_proc.stdout

