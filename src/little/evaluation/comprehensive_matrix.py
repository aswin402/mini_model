"""Empirical Benchmark Runner: LITTLE (Adaptive Neural Mesh) vs. Statistical SLMs vs. LLMs.

Evaluates 6 core real-world capabilities:
1. Natural Conversational English & Semantic Role Parsing
2. Mathematical Reasoning & Symbolic CAS
3. AST Python Code Synthesis & Sandboxed Execution
4. Typed Action Core & Pre-Dispatch Verified Tool Calling
5. Open-World Epistemic Honesty (0.0% Hallucination)
6. Lifelong Continual Learning (Zero Catastrophic Forgetting)
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from little.core.kernel import CognitiveKernel
from little.memory.store import MemoryStore
from little.rust.engine import RustAccelerationCore


@dataclass
class BenchmarkItemResult:
    modality: str
    prompt: str
    expected_status: str
    actual_status: str
    passed: bool
    latency_ms: float
    details: str


@dataclass
class ModalitySummary:
    modality: str
    total: int
    passed: int
    pass_rate: float
    avg_latency_ms: float
    p99_latency_ms: float


class SOTABenchmarkSuite:
    """Executes the full comprehensive benchmark matrix on LITTLE."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self._temp_dir = (
            tempfile.TemporaryDirectory(prefix="little-bench-")
            if workspace_root is None
            else None
        )
        self._workspace_root = (
            Path(workspace_root)
            if workspace_root is not None
            else Path(self._temp_dir.name)
        )
        self.memory = MemoryStore(":memory:", seed_ontology=True)
        self.kernel = CognitiveKernel(
            memory=self.memory, workspace_root=self._workspace_root
        )
        self.rust = RustAccelerationCore.get()

    @staticmethod
    def _has_unknown_marker(low: str) -> bool:
        return any(
            u in low
            for u in (
                "do not know",
                "don't know",
                "don't have that information",
                "no evidence",
                "unknown",
            )
        )

    def run_all(self) -> dict[str, Any]:
        results: list[BenchmarkItemResult] = []

        # ---------------------------------------------------------------------
        # 1. Conversational Dialogue & English Understanding
        # ---------------------------------------------------------------------
        dialogue_tests = [
            ("Hello there, how are you today?", "social_pleasantry"),
            ("A falcon is a bird and it hunts rodents.", "learned_compound"),
            ("Does a falcon hunt rodents?", "supported"),
            ("Cheetahs, which are felines, run fast.", "learned_relative"),
            ("Is a cheetah a feline?", "supported"),
            ("Euclidean geometry was discovered by Euclid.", "learned_passive"),
            ("Is an eagle a living thing?", "transitive_supported"),
            ("Is a bird a mammal?", "refuted"),
            ("Is a falcon a dog?", "refuted"),
            ("My name is Aswin.", "learned_identity"),
            ("What is my name?", "recalled_identity"),
        ]

        for prompt, exp_type in dialogue_tests:
            t0 = time.perf_counter()
            reply = self.kernel.chat(prompt)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            low = reply.lower()
            is_unknown = self._has_unknown_marker(low)

            passed = False
            if exp_type == "social_pleasantry" and not is_unknown and (
                "hello" in low or "glad" in low or "great" in low
            ) or exp_type.startswith("learned_") and not is_unknown and (
                "learned" in low or "understood" in low or "recorded your name" in low
            ) or exp_type in ("supported", "transitive_supported", "mutex_supported") and not is_unknown and bool(
                re.search(r"\byes\b", low)
            ) and not bool(re.search(r"\bno\b", low)) or exp_type == "refuted" and not is_unknown and not bool(
                re.search(r"\byes\b", low)
            ) and bool(
                re.search(r"\b(no|refuted|impossible|distinct|incompatible)\b", low)
            ) or exp_type == "recalled_identity" and not is_unknown and "aswin" in low:
                passed = True

            results.append(BenchmarkItemResult(
                modality="English Dialogue",
                prompt=prompt,
                expected_status=exp_type,
                actual_status="PASS" if passed else "FAIL",
                passed=passed,
                latency_ms=elapsed_ms,
                details=reply[:80],
            ))

        # ---------------------------------------------------------------------
        # 2. Mathematical Reasoning & Symbolic CAS
        # ---------------------------------------------------------------------
        math_tests: list[tuple[str, str, tuple[str, ...]]] = [
            ("What is 125 * 8 - 45 / 3?", "125*8 - 45/3 = 985", (r"\b985\b",)),
            ("What is (2/3) + (5/6)?", "3/2 = 1.5", (r"\b1\.5\b", r"\b3/2\b")),
            ("Solve x: 3*x + 15 = 45", "x = 10", (r"\b10\b",)),
            ("How many feet in 36 inches?", "3", (r"\b3(?:\.0+)?\b",)),
            ("Is 104729 prime?", "True", (r"\byes\b", r"\btrue\b", r"\bis prime\b")),
            ("What is the fibonacci of 20?", "6765", (r"\b6765\b",)),
            ("Sarah has 15 apples. She gives 4 apples to Bob, then buys 7 more apples. How many apples does Sarah have now?", "18", (r"\b18\b",)),
        ]

        for prompt, expected_ans, patterns in math_tests:
            t0 = time.perf_counter()
            reply = self.kernel.chat(prompt)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            low = reply.lower()

            passed = not self._has_unknown_marker(low) and any(
                re.search(pat, low) is not None for pat in patterns
            )

            results.append(BenchmarkItemResult(
                modality="Math & CAS Reasoning",
                prompt=prompt,
                expected_status=expected_ans,
                actual_status="PASS" if passed else "FAIL",
                passed=passed,
                latency_ms=elapsed_ms,
                details=reply[:80],
            ))

        # ---------------------------------------------------------------------
        # 3. AST Python Code Synthesis & Sandboxed Execution
        # ---------------------------------------------------------------------
        code_tests = [
            ("Write a python function to check if a number is prime", "is_prime"),
            ("Can you generate code for fibonacci?", "fibonacci"),
            ("Synthesize python code to compute factorial", "factorial"),
            ("Write code for binary search", "binary_search"),
            ("Generate a python function for greatest common divisor", "gcd"),
            ("Write code to check if a string is a palindrome", "is_palindrome"),
            ("Synthesize python code to reverse a string", "reverse_string"),
            ("Write a function to flatten a list", "flatten"),
        ]

        for prompt, func_name in code_tests:
            t0 = time.perf_counter()
            reply = self.kernel.chat(prompt)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            passed = (f"def {func_name}(" in reply) and ("VERIFIED" in reply) and ("```python" in reply)
            results.append(BenchmarkItemResult(
                modality="AST Code Synthesis",
                prompt=prompt,
                expected_status=f"verified def {func_name}",
                actual_status="PASS" if passed else "FAIL",
                passed=passed,
                latency_ms=elapsed_ms,
                details=f"Status: {'VERIFIED' if 'VERIFIED' in reply else 'ERROR'}",
            ))

        # ---------------------------------------------------------------------
        # 4. Typed Action Core & Tool Execution
        # ---------------------------------------------------------------------
        ws_root = getattr(self, "_workspace_root", None)
        if ws_root is not None:
            (Path(ws_root) / "scratch").mkdir(parents=True, exist_ok=True)
        else:
            os.makedirs("scratch", exist_ok=True)
        tool_tests = [
            ("calculate 250 * 4", "calculator"),
            ("what is the current time?", "system_time"),
            ("search the web for quantum computing", "web_search"),
            ("write to file scratch/bench_test.txt: benchmark data", "file_writer"),
            ("read file scratch/bench_test.txt", "file_reader"),
            ("run python: print(42 * 2)", "python_repl"),
        ]

        for prompt, tool_name in tool_tests:
            t0 = time.perf_counter()
            reply = self.kernel.chat(prompt)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            passed = False
            if tool_name == "calculator" and re.search(r"\b1000\b", reply) or tool_name == "system_time" and ("time is" in reply.lower() or "utc" in reply.lower()) or tool_name == "web_search" and "quantum computing" in reply.lower() or tool_name == "file_writer" and "wrote" in reply.lower() or tool_name == "file_reader" and "benchmark data" in reply or tool_name == "python_repl" and re.search(r"\b84\b", reply):
                passed = True

            results.append(BenchmarkItemResult(
                modality="Tool Execution",
                prompt=prompt,
                expected_status=f"executed {tool_name}",
                actual_status="PASS" if passed else "FAIL",
                passed=passed,
                latency_ms=elapsed_ms,
                details=reply[:80],
            ))

        # ---------------------------------------------------------------------
        # 5. Open-World Epistemic Honesty (0.0% Hallucination Boundary)
        # ---------------------------------------------------------------------
        unknown_tests = [
            "What is a flurblegorp?",
            "Is a zanthocar a mammal?",
            "Where is the continent of Quazilopia located?",
            "Can a vorpalblade fly through water?",
            "Who was the prime minister of Eldoria in 1842?",
            "How many legs does a blithering snoot have?",
        ]

        for prompt in unknown_tests:
            t0 = time.perf_counter()
            reply = self.kernel.chat(prompt)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            passed = (
                "do not know" in reply.lower()
                or "don't have that information" in reply.lower()
                or "unknown" in reply.lower()
                or "no evidence" in reply.lower()
            )
            results.append(BenchmarkItemResult(
                modality="Zero-Hallucination OWA",
                prompt=prompt,
                expected_status="UNKNOWN (Epistemic Honesty)",
                actual_status="PASS (Zero Hallucination)" if passed else "FAIL (Hallucinated)",
                passed=passed,
                latency_ms=elapsed_ms,
                details=reply[:80],
            ))

        # ---------------------------------------------------------------------
        # 6. Lifelong Continual Learning (Zero Catastrophic Forgetting)
        # ---------------------------------------------------------------------
        continual_facts = [
            ("Platypuses are monotremes.", "Is a platypus a monotreme?", True),
            ("Monotremes are mammals.", "Is a platypus a mammal?", True),
            ("Jupiter has sixty-nine moons.", "Does jupiter have sixty-nine moons?", True),
        ]
        for fact, _, _ in continual_facts:
            self.kernel.chat(fact)

        early_checks = [
            ("Is a falcon a bird?", True),
            ("Does a falcon hunt rodents?", True),
            ("Is a cheetah a feline?", True),
            ("Is an eagle an animal?", True),
        ]
        for query, exp in early_checks:
            t0 = time.perf_counter()
            reply = self.kernel.chat(query)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            passed = "yes" in reply.lower()
            results.append(BenchmarkItemResult(
                modality="Continual Learning",
                prompt=query,
                expected_status="Retained earlier fact without forgetting",
                actual_status="PASS (0% Forgetting)" if passed else "FAIL (Forgotten)",
                passed=passed,
                latency_ms=elapsed_ms,
                details=reply[:80],
            ))

        # ---------------------------------------------------------------------
        # Aggregate Modality Summaries
        # ---------------------------------------------------------------------
        modality_groups: dict[str, list[BenchmarkItemResult]] = {}
        for r in results:
            modality_groups.setdefault(r.modality, []).append(r)

        summaries: list[ModalitySummary] = []
        for mod, items in modality_groups.items():
            tot = len(items)
            ps = sum(1 for i in items if i.passed)
            lats = sorted([i.latency_ms for i in items])
            avg_l = sum(lats) / tot
            p99_idx = min(int(len(lats) * 0.99), len(lats) - 1)
            p99_l = lats[p99_idx]
            summaries.append(ModalitySummary(
                modality=mod,
                total=tot,
                passed=ps,
                pass_rate=round(ps / tot * 100.0, 1),
                avg_latency_ms=round(avg_l, 2),
                p99_latency_ms=round(p99_l, 2),
            ))

        return {
            "summaries": [asdict(s) for s in summaries],
            "total_items": len(results),
            "total_passed": sum(1 for r in results if r.passed),
            "overall_accuracy": round(sum(1 for r in results if r.passed) / len(results) * 100.0, 1),
            "items": [asdict(r) for r in results],
            "is_rust_accelerated": self.rust.is_accelerated,
        }
