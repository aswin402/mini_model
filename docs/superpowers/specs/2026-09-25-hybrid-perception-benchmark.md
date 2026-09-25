# Hybrid Perception Benchmark

## Goal

Measure the explicit perception boundary before an optional Laya checkpoint is
introduced into production experiments. The benchmark must compare the
deterministic parser with the hybrid adapter without embedding sentence rules,
model paths, or hidden fallbacks in Python.

## Contract

- Cases and expected outcomes live in
  `data/benchmarks/hybrid_perception_cases.json`.
- The evaluator accepts any `PerceptionAdapter` and records route accuracy,
  payload accuracy, UNKNOWN safety, adapter errors, median latency, and p95
  latency.
- Deterministic mode uses `DeterministicPerceptionAdapter` only.
- Hybrid mode requires an explicit model path/identifier or an explicit runtime
  policy containing one, then constructs `LayaSDKBackend` and
  `HybridPerceptionAdapter`.
- Missing SDKs, model metadata, or model paths fail clearly; they never select
  deterministic perception as a hidden replacement.
- The benchmark measures boundary behavior. It does not claim that an external
  Laya checkpoint was exercised unless the explicit hybrid command succeeds.

## Execution

```bash
PYTHONPATH=src python3 experiments/007_hybrid_perception_benchmark/run.py \
  --mode deterministic
```

```bash
PYTHONPATH=src python3 experiments/007_hybrid_perception_benchmark/run.py \
  --mode hybrid-laya \
  --model-id-or-path /explicit/path/to/checkpoint \
  --model-version checkpoint-identifier
```

The first command is dependency-free. The second requires an explicitly
installed Laya SDK and a caller-selected model.
