"""HumanEval-style empirical evaluation benchmark for AST Code Synthesis in LITTLE."""

import time

from little.procedural.code_policy import CodeSynthesizerPolicy
from little.procedural.code_sandbox import CodeSandbox
from little.procedural.code_synthesizer import ASTCodeSynthesizer


def test_humaneval_style_algorithmic_benchmark():
    """Evaluate pass@1 accuracy, execution latency, and deterministic verification across algorithmic tasks."""
    policy = CodeSynthesizerPolicy.default()
    sandbox = CodeSandbox(policy=policy)
    synthesizer = ASTCodeSynthesizer(policy=policy, sandbox=sandbox)

    total_tasks = len(policy.primitives)
    passed_tasks = 0
    latencies_ms: list[float] = []

    for name, spec in policy.primitives.items():
        t0 = time.perf_counter()
        prog = synthesizer.synthesize(name, verify=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        if prog.status in ("VERIFIED", "REPAIRED"):
            passed_tasks += 1

    pass_at_1 = passed_tasks / total_tasks
    avg_latency_ms = sum(latencies_ms) / len(latencies_ms)

    print(f"\n--- AST Code Synthesis Benchmark Results ---")
    print(f"Tasks Evaluated: {total_tasks}")
    print(f"Tasks Passed:    {passed_tasks}")
    print(f"Pass@1 Accuracy: {pass_at_1 * 100.0:.1f}%")
    print(f"Avg Latency:     {avg_latency_ms:.2f} ms")

    # Invariants: 100% pass@1 on canonical algorithmic primitives and sub-100ms average latency
    assert pass_at_1 == 1.0
    assert avg_latency_ms < 100.0
