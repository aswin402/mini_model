"""Benchmark runner CLI for Experiment 011: Real-World Conversational Understanding."""

from __future__ import annotations

import json
from pathlib import Path

from little.evaluation.real_world_chat_matrix import RealWorldChatBenchmark

if __name__ == "__main__":
    benchmark = RealWorldChatBenchmark()
    report = benchmark.run()

    print("\n" + "=" * 78)
    print(f"LITTLE (MIVI) REAL-WORLD CONVERSATIONAL EVALUATION")
    print(f"Overall Pass Rate: {report['overall_accuracy']}% ({report['total_passed']}/{report['total_turns']} turns)")
    print(f"Latency: Avg {report['avg_latency_ms']} ms | P95 {report['p95_latency_ms']} ms | P99 {report['p99_latency_ms']} ms")
    print(f"Process RAM: Initial {report['initial_rss_mb']} MB -> Final {report['final_rss_mb']} MB (Budget: <500 MB)")
    print(f"Headroom Context Tokens: {report['active_headroom_tokens']} tokens (Reversible CCR)")
    print("=" * 78)

    for cat in report["categories"]:
        print(f"  - {cat['category']:<36} | Pass: {cat['pass_rate']:>5.1f}% ({cat['passed']}/{cat['total']}) | Latency Avg: {cat['avg_latency_ms']:>6.2f} ms")

    out_file = Path(__file__).resolve().parent / "results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Artifact] Saved detailed JSON execution trace to {out_file}\n")
