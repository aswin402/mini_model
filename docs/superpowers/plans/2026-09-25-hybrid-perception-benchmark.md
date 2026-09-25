# Hybrid Perception Benchmark Plan

## Task 1: Add data-owned benchmark cases

- [x] Create a versioned JSON case file with statement, question, action, and
  unknown examples.
- [x] Validate IDs, expected routes, payload expectations, and UNKNOWN safety
  expectations before evaluation.

## Task 2: Implement reusable evaluation metrics

- [x] Evaluate any perception adapter with isolated in-memory stores.
- [x] Record route correctness, payload correctness, UNKNOWN safety, errors,
  median latency, p95 latency, and per-case observations.
- [x] Serialize reports as JSON-compatible data.

## Task 3: Add explicit runtime selection

- [x] Keep deterministic mode dependency-free.
- [x] Require an explicit model path/identifier or configured runtime policy for
  hybrid Laya mode.
- [x] Keep missing SDK/model failures visible; never fall back silently.

## Task 4: Verify and document

- [x] Add unit coverage for schema loading, malformed data, metrics, hybrid
  mismatch behavior, and explicit runtime configuration.
- [x] Run the deterministic benchmark, focused tests, full suite, and compileall.
- [x] Document the command boundary and the fact that no Laya checkpoint is
  exercised by the dependency-free baseline.
