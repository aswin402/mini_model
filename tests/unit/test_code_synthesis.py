"""Unit tests for Python AST Code Synthesis, Sandboxed Execution, and Socratic Auto-Repair."""

import ast
import pytest

from little.procedural.code_policy import CodeSynthesizerPolicy
from little.procedural.code_repair import SocraticCodeRepair
from little.procedural.code_sandbox import CodeSandbox
from little.procedural.code_synthesizer import ASTCodeSynthesizer, SynthesizedProgram


def test_ast_synthesis_all_primitives():
    """Verify that all configured algorithmic primitives synthesize valid ASTs and pass sandbox tests."""
    policy = CodeSynthesizerPolicy.default()
    sandbox = CodeSandbox(policy=policy)
    synthesizer = ASTCodeSynthesizer(policy=policy, sandbox=sandbox)

    for prim_name in policy.primitives.keys():
        prog = synthesizer.synthesize(prim_name, verify=True)
        assert prog.status == "VERIFIED"
        assert prog.tree is not None
        assert isinstance(prog.tree, ast.FunctionDef)
        assert prog.execution_hash is not None
        assert prog.execution_time_ms > 0
        assert len(prog.assertions) > 0


def test_ast_security_safety_verification():
    """Verify sandbox rejects forbidden AST nodes (Import, Exec) and dangerous names (open, eval)."""
    sandbox = CodeSandbox()

    # 1. Forbidden Import node
    safe, err = sandbox.verify_ast_safety("import os\nos.system('ls')")
    assert safe is False
    assert "Import" in err

    # 2. Forbidden ImportFrom node
    safe, err = sandbox.verify_ast_safety("from subprocess import Popen")
    assert safe is False
    assert "ImportFrom" in err

    # 3. Forbidden dangerous built-in: open
    safe, err = sandbox.verify_ast_safety("def read_file(): open('/etc/passwd')")
    assert safe is False
    assert "open" in err

    # 4. Forbidden dangerous built-in: eval
    safe, err = sandbox.verify_ast_safety("def evil(x): return eval(x)")
    assert safe is False
    assert "eval" in err

    # 5. Clean, safe computational AST
    safe, err = sandbox.verify_ast_safety("def add(a, b): return a + b")
    assert safe is True
    assert err is None


def test_sandbox_timeout_enforcement():
    """Verify sandbox enforces execution timeout on infinite loops without hanging."""
    # Use a small timeout policy
    policy = CodeSynthesizerPolicy.default()
    sandbox = CodeSandbox(policy=policy)

    infinite_loop_code = """
def run_forever():
    while True:
        pass
run_forever()
"""
    # Overwrite timeout for quick test
    res = sandbox.execute_script(infinite_loop_code)
    assert res.success is False
    assert res.timed_out is True
    assert "timed out" in res.error.lower()


def test_socratic_auto_repair_loop():
    """Verify Socratic Auto-Repair identifies failing test assertion, applies AST mutation, and verifies."""
    sandbox = CodeSandbox()
    policy = CodeSynthesizerPolicy.default()
    repairer = SocraticCodeRepair(sandbox=sandbox, policy=policy)

    # Broken implementation: factorial without base case for 0 (causes factorial(0) == 0 instead of 1)
    broken_code = """
def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    if n == 0:
        return 0
    return result
"""
    broken_tree = ast.parse(broken_code).body[0]
    assertions = (
        "factorial(0) == 1",
        "factorial(1) == 1",
        "factorial(5) == 120",
    )

    prog = SynthesizedProgram(
        name="factorial",
        params=("n",),
        docstring="Compute factorial.",
        tree=broken_tree,
        code=ast.unparse(broken_tree),
        assertions=assertions,
        proof_trace=["Initial broken code"],
    )

    # Initial test run must fail on factorial(0) == 1
    initial_res = sandbox.run_assertions(prog.code, prog.assertions)
    assert initial_res.success is False
    assert initial_res.failing_assertion == "factorial(0) == 1"

    # Run Socratic repair
    repaired = repairer.repair(prog, initial_res)
    assert repaired is True
    assert prog.status == "REPAIRED"
    assert "Socratic repair succeeded" in " ".join(prog.proof_trace)

    # Re-verify that repaired code passes all assertions
    final_res = sandbox.run_assertions(prog.code, prog.assertions)
    assert final_res.success is True
