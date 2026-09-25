# Hybrid Perception Benchmark

This benchmark compares the deterministic parser boundary with the explicit
hybrid boundary. Cases and expected outcomes are stored in
`data/benchmarks/hybrid_perception_cases.json`; no sentence rules are embedded
in the runner.

Run the deterministic baseline:

```bash
PYTHONPATH=src python3 experiments/007_hybrid_perception_benchmark/run.py \
  --mode deterministic
```

Run hybrid mode only after explicitly installing/configuring the optional Laya
SDK and selecting a model or local checkpoint:

```bash
PYTHONPATH=src python3 experiments/007_hybrid_perception_benchmark/run.py \
  --mode hybrid-laya \
  --model-id-or-path /explicit/path/to/checkpoint \
  --model-version checkpoint-identifier
```

`both` runs the deterministic baseline and hybrid mode, but still requires the
explicit Laya model path. Missing SDKs or model metadata fail clearly; the
runner never silently falls back to deterministic perception.

The report measures route accuracy, payload accuracy, UNKNOWN safety, adapter
errors, median latency, and p95 latency. It does not claim that a Laya model
was exercised unless the explicit hybrid command succeeds.
