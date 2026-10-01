"""Real-world Developer Audit & Performance/Resource Benchmark.

Tests:
1. End-to-end cognitive interaction across 15 real-world multi-domain queries
2. System 1 Laya Decision Head latency & calibration under real queries
3. GLM Dual-Speed Infilling Engine (Fast mode vs Thinking meeting-in-the-middle)
4. Epistemic Uncertainty & OWA strict zero-hallucination verification
5. Detailed Resource Profiling: RAM RSS, P50/P95/P99 latency, DB stats
"""

from __future__ import annotations

import os
import resource
import time
from typing import Any

from little.core.kernel import CognitiveKernel
from little.inference.dual_speed import DualSpeedInfillingEngine
from little.inference.invariant_gates import DeepSeekInvariantVerifier
from little.memory.store import MemoryStore
from little.neural.decision_head import LayaDecisionHead


def get_process_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On Linux maxrss is in kilobytes
    return usage.ru_maxrss / 1024.0


def run_developer_audit() -> dict[str, Any]:
    print("=" * 70)
    print("STARTING LITTLE COGNITIVE ARCHITECTURE REAL-WORLD AUDIT")
    print("=" * 70)

    start_rss = get_process_memory_mb()
    print(f"Initial Process RSS: {start_rss:.2f} MB")

    db_path = "data/little.db"
    store = MemoryStore(db_path)
    kernel = CognitiveKernel(store)
    verifier = DeepSeekInvariantVerifier(store)
    infilling_engine = DualSpeedInfillingEngine(store, verifier)
    decision_head = LayaDecisionHead(memory=store)

    post_init_rss = get_process_memory_mb()
    print(f"Post-Initialization RSS: {post_init_rss:.2f} MB (+{post_init_rss - start_rss:.2f} MB)")

    # 1. REAL-WORLD COMPLEX CHAT QUERIES
    real_world_queries = [
        # Domain 1: Multi-hop Taxonomy & Mereology
        ("is a sparrow an animal?", "taxonomic reasoning"),
        ("what is a cpu part of?", "mereological query"),
        ("does a dog have a tail?", "attribute reasoning"),
        
        # Domain 2: Open-World Assumption (OWA) & Epistemic Uncertainty
        ("is a flurblegorp a type of vehicle?", "unknown concept"),
        ("what is the mass of quazilopia?", "unknown entity attribute"),
        ("can a rock think?", "cross-domain negative / disjoint"),

        # Domain 3: Continuous Dynamic Physics & Sensory Grounding
        ("What happens to apple over 10 days at 25 Celsius?", "continuous decay"),
        ("What happens if I peel an apple?", "discrete procedural skill"),
        ("observe sensor: [color: red, mass: 180g]", "multimodal grounding"),

        # Domain 4: Procedural Math & Exact Calculation
        ("calculate 250 * 4 + 180", "arithmetic CAS"),
        ("what is 15% of 800?", "percentage computation"),

        # Domain 5: Non-Autoregressive System 1 Gating
        ("Hello, who created you?", "social dialogue"),
        ("run dream cycle 1", "curiosity dreaming"),
        ("what are your knowledge gaps?", "epistemic self-reflection"),
        ("is linux an operating system?", "truth unit evaluation"),
    ]

    query_latencies = []
    query_results = []

    print("\n--- 1. REAL-WORLD USER QUERIES TEST ---")
    for q, desc in real_world_queries:
        t0 = time.perf_counter()
        reply = kernel.chat(q)
        lat = (time.perf_counter() - t0) * 1000.0
        query_latencies.append(lat)
        text_preview = reply[:80].replace("\n", " ")
        print(f"[{lat:6.1f} ms] [{desc:26s}] {q[:35]:35s} -> {text_preview}")
        query_results.append({
            "query": q,
            "desc": desc,
            "latency_ms": lat,
            "response": reply,
        })

    # 2. GLM DUAL-SPEED INFILLING LATENCY & COLLISION
    print("\n--- 2. GLM DUAL-SPEED INFILLING BENCHMARK ---")
    infill_tests = [
        ("object fast", lambda: infilling_engine.infill_object("sparrow", "is_a")),
        ("subject fast", lambda: infilling_engine.infill_subject("bird", "is_a")),
        ("path collision", lambda: infilling_engine.infill_path("sparrow", "animal", max_depth=4)),
        ("transitive thinking", lambda: infilling_engine.query("sparrow", "animal", "is_a")),
        ("disconnected query", lambda: infilling_engine.query("sparrow", "linux", "is_a")),
    ]
    infill_metrics = []
    for label, fn in infill_tests:
        t0 = time.perf_counter()
        res = fn()
        lat = (time.perf_counter() - t0) * 1000.0
        mode = res.mode if hasattr(res, "mode") else (res[0].mode if res else "NONE")
        print(f"[{lat:6.2f} ms] Infilling [{label:20s}] -> Mode: {mode}")
        infill_metrics.append({"test": label, "latency_ms": lat, "mode": mode})

    # 3. LAYA SYSTEM 1 PERCEPTION & CALIBRATION BENCHMARK
    print("\n--- 3. LAYA DECISION HEAD LATENCY & CALIBRATION ---")
    decision_tests = [
        "calculate 500 / 25",
        "simulate apple decay over 5 days",
        "def quicksort(arr):",
        "Hello my friend!",
        "what are your knowledge gaps?",
        "Linux is a computer operating system",
        "A flurblegorp on quazilopia",
    ]
    dec_latencies = []
    for text in decision_tests:
        t0 = time.perf_counter()
        dec = decision_head.classify_intent(text)
        lat = (time.perf_counter() - t0) * 1000.0
        dec_latencies.append(lat)
        print(f"[{lat:5.2f} ms] Intent: {dec.choice:10s} (conf: {dec.confidence:.2f}, H_norm: {dec.normalized_entropy:.2f}, fast: {dec.is_fast_path}) <- '{text[:35]}'")

    # 4. RESOURCE CONSUMPTION PROFILING
    final_rss = get_process_memory_mb()
    query_latencies.sort()
    n = len(query_latencies)
    p50 = query_latencies[int(n * 0.50)]
    p95 = query_latencies[int(n * 0.95)]
    p99 = query_latencies[-1]
    avg_lat = sum(query_latencies) / n

    dec_latencies.sort()
    m = len(dec_latencies)
    dec_p50 = dec_latencies[int(m * 0.50)]
    dec_p95 = dec_latencies[int(m * 0.95)]

    print("\n--- 4. RESOURCE CONSUMPTION & LATENCY PROFILING ---")
    print(f"Peak Process RSS:        {final_rss:.2f} MB")
    print(f"Net Memory Allocated:    +{final_rss - start_rss:.2f} MB")
    print(f"Cognitive Query Avg:     {avg_lat:.2f} ms")
    print(f"Cognitive Query P50:     {p50:.2f} ms")
    print(f"Cognitive Query P95:     {p95:.2f} ms")
    print(f"Cognitive Query P99:     {p99:.2f} ms")
    print(f"Laya Decision Head P50:  {dec_p50:.2f} ms")
    print(f"Laya Decision Head P95:  {dec_p95:.2f} ms")
    print("=" * 70)

    return {
        "final_rss_mb": final_rss,
        "delta_rss_mb": final_rss - start_rss,
        "avg_latency_ms": avg_lat,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "laya_p50_ms": dec_p50,
        "queries": query_results,
    }


if __name__ == "__main__":
    run_developer_audit()
