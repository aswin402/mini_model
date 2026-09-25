# System 2 Proof Benchmark Plan

## Task 1: Move reasoning calibration into policy data

- [x] Add strict `ReasoningPolicy` loading for search depth and confidence.
- [x] Inject the policy into `DualSpeedInfillingEngine`.

## Task 2: Make proof traces auditable

- [x] Derive acyclic predicates from relation schema metadata.
- [x] Include all invariant-gate results for every verified proof step.
- [x] Keep existing direct and failure behavior compatible.

## Task 3: Add data-driven evaluation

- [x] Add direct, multi-hop, and UNKNOWN graph cases.
- [x] Measure mode accuracy, proof-gate coverage, UNKNOWN safety, errors, and
  latency.
- [x] Add a local CLI runner and documentation.

## Task 4: Connect the verified engine to query routing

- [x] Route transitive graph queries through the dual-speed engine by default.
- [x] Preserve the existing resolver for refutations and specialized queries
  when System 2 finds no path.
- [x] Add kernel-level coverage for the inspectable gate trace.

## Task 5: Verify

- [x] Run focused policy, dual-speed, invariant, registry, and benchmark tests.
- [x] Run the complete regression suite and compileall.
