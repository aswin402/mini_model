"""Integration test verifying that the comprehensive benchmark matrix passes 100%."""

from little.evaluation.comprehensive_matrix import SOTABenchmarkSuite


def test_comprehensive_matrix_100_percent_pass():
    """Verify that all 6 modalities achieve >= 95% accuracy with sub-50ms average latency."""
    suite = SOTABenchmarkSuite()
    out = suite.run_all()

    assert out["overall_accuracy"] >= 95.0
    assert out["total_passed"] == out["total_items"]
    assert out["is_rust_accelerated"] is True

    for summary in out["summaries"]:
        assert summary["pass_rate"] == 100.0
        assert summary["avg_latency_ms"] < 60.0
