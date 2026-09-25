# System 2 Proof Benchmark

This benchmark evaluates the local verified graph-reasoning engine using
versioned graph cases from `data/benchmarks/system2_reasoning_cases.json`.

Run it with:

```bash
PYTHONPATH=src python3 experiments/008_system2_proof_benchmark/run.py
```

It measures mode selection, proof-gate coverage, UNKNOWN safety, errors, and
latency. Multi-hop traces must expose the four invariant gates before they are
counted as verified proofs. No external model, network access, or checkpoint is
required.
