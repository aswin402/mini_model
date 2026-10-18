"""Empirical Benchmark Runner: LITTLE (Adaptive Neural Mesh) vs. Statistical SLMs vs. LLMs.

Evaluates 6 core real-world capabilities:
1. Natural Conversational English & Semantic Role Parsing
2. Mathematical Reasoning & Symbolic CAS
3. AST Python Code Synthesis & Sandboxed Execution
4. Typed Action Core & Pre-Dispatch Verified Tool Calling
5. Open-World Epistemic Honesty (0.0% Hallucination)
6. Lifelong Continual Learning (Zero Catastrophic Forgetting)
"""

from __future__ import annotations

import json
from pathlib import Path

from little.evaluation.comprehensive_matrix import (
    BenchmarkItemResult,
    ModalitySummary,
    SOTABenchmarkSuite,
)

__all__ = ["BenchmarkItemResult", "ModalitySummary", "SOTABenchmarkSuite"]


if __name__ == "__main__":
    suite = SOTABenchmarkSuite()
    out = suite.run_all()
    print("\n=================================================================")
    print("LITTLE (Adaptive Neural Mesh) Comprehensive Benchmark")
    print(f"Overall Accuracy: {out['overall_accuracy']}% ({out['total_passed']}/{out['total_items']})")
    print(f"Rust Acceleration: {out['is_rust_accelerated']}")
    print("=================================================================")
    for s in out["summaries"]:
        print(
            f"{s['modality']:<25} | Pass: {s['pass_rate']:>5.1f}% ({s['passed']}/{s['total']}) "
            f"| Latency Avg: {s['avg_latency_ms']:>6.2f} ms | P99: {s['p99_latency_ms']:>6.2f} ms"
        )

    out_dir = Path(__file__).resolve().parent
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved raw benchmark data to {out_dir / 'results.json'}")
