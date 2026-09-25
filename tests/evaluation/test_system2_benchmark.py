import json
from pathlib import Path

import pytest

from little.evaluation.system2_benchmark import (
    System2Case,
    System2Report,
    evaluate_cases,
    load_cases,
)


def test_load_system2_cases_reads_graphs_from_versioned_data():
    version, cases = load_cases(Path("data/benchmarks/system2_reasoning_cases.json"))

    assert version == "1"
    assert {case.case_id for case in cases} == {"direct", "multihop", "unknown"}
    assert cases[1].edges == (
        ("GALA_APPLE", "is_a", "APPLE"),
        ("APPLE", "is_a", "FRUIT"),
        ("FRUIT", "is_a", "PLANT"),
    )


def test_system2_evaluation_reports_verified_proofs_and_unknown_safety():
    _version, cases = load_cases(Path("data/benchmarks/system2_reasoning_cases.json"))

    report = evaluate_cases(cases)

    assert isinstance(report, System2Report)
    assert report.total == 3
    assert report.mode_accuracy == 1.0
    assert report.proof_accuracy == 1.0
    assert report.unknown_safety == 1.0
    assert report.errors == 0
    assert report.latency_ms >= 0.0
    assert report.p95_latency_ms >= report.latency_ms
    multihop = next(item for item in report.observations if item.case_id == "multihop")
    assert multihop.path == ("GALA_APPLE", "APPLE", "FRUIT", "PLANT")
    assert multihop.proof_valid is True
    assert all(gate in multihop.trace for gate in ("I_DAG", "I_MUTEX", "I_SORT", "I_GROUND"))


def test_load_system2_cases_rejects_duplicate_ids_and_bad_edges(tmp_path: Path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "system2_benchmark": {
                    "version": "1",
                    "cases": [
                        {
                            "id": "duplicate",
                            "subject": "A",
                            "predicate": "is_a",
                            "target": "B",
                            "edges": [["A", "is_a", "B"]],
                            "expected_mode": "FAST",
                            "expect_proof": False,
                            "expect_unknown": False,
                        },
                        {
                            "id": "duplicate",
                            "subject": "A",
                            "predicate": "is_a",
                            "target": "B",
                            "edges": [["A", "is_a"]],
                            "expected_mode": "FAST",
                            "expect_proof": False,
                            "expect_unknown": False,
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate|edge"):
        load_cases(path)
