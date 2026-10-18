# Verified system repair tracker

Status: implementation authorized; no commits or remote operations authorized.

| ID | Scope | Status | Validation evidence |
| --- | --- | --- | --- |
| R01 | Isolated tests and execution contracts | completed | `tests/conftest.py` & `tests/unit/test_runtime_repair_contracts.py` |
| R02 | Safe exact arithmetic | completed | `tests/unit/test_reasoning_repairs.py` & `tests/evaluation/test_benchmark_oracles.py` |
| R03 | Tool/code execution, confirmations, timeouts, typed args, timezone | completed | `tests/unit/test_planner_repairs.py` & `tests/unit/test_runtime_repair_contracts.py` |
| R04 | Consistent inference, polarity, signatures, confidence, depth | completed | `tests/unit/test_reasoning_repairs.py` |
| R05 | Atomic/idempotent evidence, provenance, procedure preservation | completed | `tests/unit/test_memory_repair_contracts.py` |
| R06 | Faithful OKF, safe filenames, wiki index/backlinks | completed | `tests/unit/test_okf_wiki_repairs.py` |
| R07 | Persisted/reversible bounded conversational memory | completed | `tests/unit/test_headroom_repair_contracts.py` |
| R08 | Honest complete plans and token-aware references | completed | `tests/unit/test_planner_repairs.py` |
| R09 | Correct semantic roles/names and bounded synthesis | completed | `tests/unit/test_reasoning_repairs.py` & `tests/unit/test_planner_repairs.py` |
| R10 | Neural scoring, proposition grounding, explicit feedback/checkpoints | completed | `tests/unit/test_neural_contract_repairs.py` |
| R11 | COBWEB integrity, quantization dimensions, native parity/lifecycle | completed | `tests/unit/test_vector_cluster_repairs.py` |
| R12 | Hypothesis policy, verified import, surfaced partial failures | completed | `tests/unit/test_active_learning_repairs.py` |
| R13 | Resource packaging, isolated storage, policy validation/lifecycle | completed | `tests/unit/test_runtime_repair_contracts.py` |
| R14 | Complete snapshots and truthful per-case benchmarks | completed | `tests/evaluation/test_benchmark_oracles.py` & `tests/evaluation/test_comprehensive_matrix.py` |
| R15 | Accurate documentation and combined practical validation | completed | `README.md`, `AGENTS.md`, `tests/integration/test_real_world_comprehensive_evaluation.py` |

## Per-task checklist

- [x] Regression fails for the audited root cause.
- [x] Minimal implementation addresses the contract, without unrelated cleanup.
- [x] Related existing tests pass or intentional behavior changes are documented.
- [x] Practical scenario passes using a temporary database/vault/workspace.
- [x] Diff reviewed; live runtime data unchanged; no secrets or environment-specific values introduced.

## Known design limits

General neural language understanding and arbitrary code synthesis are not promised. This repair exposes actual capabilities, connects bounded learning contracts, and prevents unsupported requests or hypotheses being represented as verified success. Any larger model/training expansion requires its own measured design and data.
