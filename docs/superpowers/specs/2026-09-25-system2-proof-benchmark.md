# System 2 Verified Proof Benchmark

## Goal

Measure local graph reasoning independently of any statistical language model.
The benchmark must show which route was selected, whether a multi-hop path was
verified by the invariant gates, and whether unsupported queries remain UNKNOWN.

## Design

- Graph edges and expected outcomes are stored in
  `data/benchmarks/system2_reasoning_cases.json`.
- Search depth and confidence values are loaded from
  `data/schemas/reasoning_policy.json`.
- Relation acyclicity is declared in `relation_types.json`, not in verifier
  Python code.
- Every verified multi-hop step exposes `I_DAG`, `I_MUTEX`, `I_SORT`, and
  `I_GROUND` in its inspectable trace.
- Direct relations use FAST mode; bounded multi-hop paths use THINKING mode;
  absent paths return UNKNOWN with no fabricated proof.
- `ThinkingController` now uses this verified engine for transitive graph
  predicates. If no path is found, the existing resolver still handles
  refutations and specialized queries, with the System 2 trace preserved.

## Execution

```bash
PYTHONPATH=src python3 experiments/008_system2_proof_benchmark/run.py
```

This evaluation is fully local and does not install, download, or emulate an
external model.
