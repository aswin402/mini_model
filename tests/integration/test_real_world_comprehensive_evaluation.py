"""Automated Integration Test: Real-World Conversational Understanding & Knowledge Verification.

Ensures that the comprehensive real-world multi-domain dialogue suite maintains 100% pass rate
across all 11 core conversational modalities.
"""

from __future__ import annotations

import pytest

from little.evaluation.real_world_chat_matrix import RealWorldChatBenchmark


def test_real_world_chat_comprehensive_matrix() -> None:
    """Execute the full 44-turn real-world dialogue matrix and verify 100% accuracy."""
    benchmark = RealWorldChatBenchmark()
    report = benchmark.run()

    assert report["total_turns"] == 44, f"Expected 44 turns, got {report['total_turns']}"
    assert report["overall_accuracy"] == 100.0, f"Expected 100% pass rate, got {report['overall_accuracy']}% ({report['total_passed']}/{report['total_turns']})"

    # Verify each individual category achieved 100% pass rate
    for cat in report["categories"]:
        assert cat["pass_rate"] == 100.0, f"Category '{cat['category']}' failed with pass rate {cat['pass_rate']}%"

    # Verify memory constraints (<500 MB budget, process RAM < 100 MB)
    assert report["final_rss_mb"] < 250.0, f"Process RSS exceeded memory bound: {report['final_rss_mb']} MB"

    # Verify average latency < 100 ms
    assert report["avg_latency_ms"] < 100.0, f"Average latency too high: {report['avg_latency_ms']} ms"
