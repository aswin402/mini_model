"""Real-World Conversational Understanding & Multi-Domain Dialogue Benchmark Matrix.

Evaluates LITTLE (MIVI) across 44 real-world interactive dialogue turns:
1. Casual Human Chit-Chat & Social Protocols
2. Slang & Colloquial Phrasing
3. Multi-Domain Deductive Reasoning (CS, Astronomy, Biology, Geography, Physics)
4. Invariant Refutation & Mutual Exclusivity Refutations
5. Exact Symbolic CAS & Mathematical Problem Solving
6. Deterministic AST Python Code Synthesis & Sandboxed Execution
7. Typed Tool Calling & Sandboxed Environmental I/O
8. Living Wiki & TurboQuant 4-Bit/1-Bit Semantic Retrieval
9. Open-World Assumption (OWA) Zero-Hallucination Honesty
10. Epistemic Self-Study & Curiosity Dreaming (The Cognitive Dreamer)
11. Continuous Dynamic Physics & Multimodal Sensory Grounding

Profiles latency (ms), Headroom context token compression, and process RSS memory (< 50 MB).
"""

from __future__ import annotations

import re
import resource
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from little.core.kernel import CognitiveKernel
from little.memory.store import MemoryStore
from little.rust.engine import RustAccelerationCore


@dataclass
class TurnResult:
    turn_index: int
    category: str
    user_input: str
    model_output: str
    expected_status: str
    passed: bool
    latency_ms: float
    headroom_tokens: int
    rss_mb: float


@dataclass
class CategorySummary:
    category: str
    total: int
    passed: int
    pass_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


class RealWorldChatBenchmark:
    """Executes the 44-turn real-world conversational matrix against CognitiveKernel."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        self._temp_dir = (
            tempfile.TemporaryDirectory(prefix="little-rw-bench-")
            if workspace_root is None
            else None
        )
        self._workspace_root = (
            Path(workspace_root)
            if workspace_root is not None
            else Path(self._temp_dir.name)
        )
        (self._workspace_root / "scratch").mkdir(parents=True, exist_ok=True)
        source_vault = Path("data/knowledge_vault")
        target_vault = self._workspace_root / "data" / "knowledge_vault"
        if source_vault.is_dir() and not target_vault.exists():
            shutil.copytree(source_vault, target_vault)

        if db_path is None:
            import sqlite3

            default_db = Path("data/little.db")
            self.memory = MemoryStore(
                ":memory:", seed_ontology=not default_db.is_file()
            )
            if default_db.is_file():
                src_conn = sqlite3.connect(default_db)
                try:
                    src_conn.backup(self.memory._conn)
                finally:
                    src_conn.close()
                self.memory._init_db(seed_ontology=False)
                self.memory.prune_polluted_concepts()
        else:
            self.memory = MemoryStore(db_path)

        self.kernel = CognitiveKernel(
            memory=self.memory, workspace_root=self._workspace_root
        )
        self.rust = RustAccelerationCore.get()

    def _get_rss_mb(self) -> float:
        """Process resident set size in megabytes."""
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = float(line.split()[1])
                        return round(kb / 1024.0, 2)
        except OSError:
            pass
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
        return round(usage / divisor, 2)

    @staticmethod
    def _is_refuted_reply(r: str) -> bool:
        low = r.lower()
        if any(
            u in low
            for u in (
                "do not know",
                "don't know",
                "don't have that information",
                "no evidence",
                "unknown",
            )
        ):
            return False
        if re.search(r"\byes\b", low):
            return False
        return bool(
            re.search(r"\b(no|not|distinct|refuted|impossible|incompatible)\b", low)
        )

    def run(self) -> dict[str, Any]:
        turns = [
            # -------------------------------------------------------------
            # Category 1: Casual Human Conversation & Social Protocols
            # -------------------------------------------------------------
            ("Casual Conversation", "Hello there! How are you today?", "social_greeting",
             lambda r: any(w in r.lower() for w in ("hello", "operating smoothly", "assist you", "ready to reason"))),
            ("Casual Conversation", "My name is Aswin.", "learned_identity",
             lambda r: "aswin" in r.lower() and any(w in r.lower() for w in ("recorded", "meet you", "memory"))),
            ("Casual Conversation", "What is my name?", "recalled_identity",
             lambda r: "aswin" in r.lower()),
            ("Casual Conversation", "Thanks for your help, I appreciate it.", "social_gratitude",
             lambda r: any(w in r.lower() for w in ("welcome", "glad", "anytime", "pleasure"))),

            # -------------------------------------------------------------
            # Category 2: Slang & Colloquial Phrasing
            # -------------------------------------------------------------
            ("Slang & Colloquial", "whats up bro can u tell me if python is a programming language", "slang_compound",
             lambda r: "yes" in r.lower() and "python" in r.lower() and "programming language" in r.lower()),
            ("Slang & Colloquial", "can u tell me if a dolphin has a heart?", "slang_polite",
             lambda r: "yes" in r.lower() and "heart" in r.lower()),
            ("Slang & Colloquial", "can an eagle fly?", "colloquial_capability",
             lambda r: "yes" in r.lower() and "fly" in r.lower()),

            # -------------------------------------------------------------
            # Category 3: Multi-Domain Deductive Reasoning
            # -------------------------------------------------------------
            ("Multi-Domain Deductive Reasoning", "Is Python a software tool?", "cs_multi_hop",
             lambda r: "yes" in r.lower() and "software tool" in r.lower()),
            ("Multi-Domain Deductive Reasoning", "Is Linux an operating system?", "cs_os",
             lambda r: "yes" in r.lower() and "operating system" in r.lower()),
            ("Multi-Domain Deductive Reasoning", "Is binary search an algorithm?", "cs_algorithm",
             lambda r: "yes" in r.lower() and "algorithm" in r.lower()),
            ("Multi-Domain Deductive Reasoning", "Does a computer have a CPU?", "cs_hardware_mereology",
             lambda r: "yes" in r.lower() and "cpu" in r.lower()),
            ("Multi-Domain Deductive Reasoning", "Does Europa orbit Jupiter?", "astronomy_orbit",
             lambda r: "yes" in r.lower() and "europa" in r.lower() and "jupiter" in r.lower()),
            ("Multi-Domain Deductive Reasoning", "Is Betelgeuse a star?", "astronomy_star",
             lambda r: "yes" in r.lower() and "star" in r.lower()),
            ("Multi-Domain Deductive Reasoning", "Is a whale a vertebrate?", "biology_taxonomy",
             lambda r: "yes" in r.lower() and "vertebrate" in r.lower()),
            ("Multi-Domain Deductive Reasoning", "Does an elephant have a brain?", "biology_mereology",
             lambda r: "yes" in r.lower() and "brain" in r.lower()),
            ("Multi-Domain Deductive Reasoning", "Is Paris located in Europe?", "geography_transitive",
             lambda r: "yes" in r.lower() and "europe" in r.lower()),
            ("Multi-Domain Deductive Reasoning", "Is a hammer an artifact?", "physics_tools_transitive",
             lambda r: "yes" in r.lower() and "artifact" in r.lower()),

            # -------------------------------------------------------------
            # Category 4: Invariant Refutation & Mutual Exclusivity
            # -------------------------------------------------------------
            ("Invariant Refutation", "Is a reptile an amphibian?", "refuted_mutex_vertebrate",
             self._is_refuted_reply),
            ("Invariant Refutation", "Is a star a planet?", "refuted_mutex_celestial",
             self._is_refuted_reply),
            ("Invariant Refutation", "Is a falcon a dog?", "refuted_disjoint_family",
             self._is_refuted_reply),
            ("Invariant Refutation", "Is a solid a gas?", "refuted_mutex_state_of_matter",
             self._is_refuted_reply),

            # -------------------------------------------------------------
            # Category 5: Exact CAS & Symbolic Math
            # -------------------------------------------------------------
            ("Exact Math & CAS", "What is 45 * 12 + 180 / 6?", "compound_cas_arithmetic",
             lambda r: bool(re.search(r"\b570\b", r))),
            ("Exact Math & CAS", "What is (3/4) + (2/5)?", "fraction_cas_arithmetic",
             lambda r: bool(re.search(r"\b(1\.15|23/20)\b", r))),
            ("Exact Math & CAS", "Sarah has 15 apples. She gives 4 apples to Bob, then buys 7 more apples. How many apples does Sarah have now?", "word_problem_proportional",
             lambda r: bool(re.search(r"\b18\b", r))),
            ("Exact Math & CAS", "Convert 36 inches to feet", "unit_conversion_graph",
             lambda r: bool(re.search(r"\b3(?:\.0+)?\b", r))),
            ("Exact Math & CAS", "Is 104729 prime?", "symbolic_primality_test",
             lambda r: bool(re.search(r"\b(yes|true)\b", r.lower())) and "not prime" not in r.lower()),

            # -------------------------------------------------------------
            # Category 6: Deterministic AST Code Synthesis
            # -------------------------------------------------------------
            ("AST Code Synthesis", "Write a python function to check if a string is a palindrome", "ast_palindrome",
             lambda r: "def is_palindrome(" in r and "VERIFIED" in r and "```python" in r),
            ("AST Code Synthesis", "Can you generate code for fibonacci?", "ast_fibonacci",
             lambda r: "def fibonacci(" in r and "VERIFIED" in r and "```python" in r),
            ("AST Code Synthesis", "Synthesize python code to check if a number is prime", "ast_is_prime",
             lambda r: "def is_prime(" in r and "VERIFIED" in r and "```python" in r),

            # -------------------------------------------------------------
            # Category 7: Typed Tool Calling & Environmental Sandbox
            # -------------------------------------------------------------
            ("Typed Tool Calling", "calculate 250 * 4", "calculator_tool",
             lambda r: "1000" in r),
            ("Typed Tool Calling", "what is the current time?", "system_time_tool",
             lambda r: any(w in r.lower() for w in ("utc", "current time is", "2026"))),
            ("Typed Tool Calling", "write to file scratch/test_chat_env.txt: Verified real world agent execution", "file_writer_tool",
             lambda r: any(w in r.lower() for w in ("wrote", "successfully", "bytes"))),

            # -------------------------------------------------------------
            # Category 8: Living Wiki & TurboQuant Semantic Retrieval
            # -------------------------------------------------------------
            ("Living Wiki & TurboQuant", "search wiki for python", "turboquant_semantic_search",
             lambda r: "relevant" in r.lower() and "python" in r.lower()),
            ("Living Wiki & TurboQuant", "what links to animal?", "living_wiki_backlinks",
             lambda r: "link" in r.lower() and any(w in r.lower() for w in ("bird", "mammal", "fish"))),
            ("Living Wiki & TurboQuant", "show wiki page for airplane", "okf_page_summary",
             lambda r: "airplane" in r.lower() and any(w in r.lower() for w in ("aircraft", "vehicle"))),

            # -------------------------------------------------------------
            # Category 9: Open-World Epistemic Honesty (0.0% Hallucination)
            # -------------------------------------------------------------
            ("Zero-Hallucination OWA", "What is a flurblegorp?", "unknown_entity",
             lambda r: any(w in r.lower() for w in ("do not know", "don't have that information", "unknown"))),
            ("Zero-Hallucination OWA", "Is a zanthocar a mammal?", "unknown_predicate",
             lambda r: any(w in r.lower() for w in ("do not know", "don't have that information", "unknown"))),
            ("Zero-Hallucination OWA", "Where is the continent of Quazilopia located?", "unknown_location",
             lambda r: any(w in r.lower() for w in ("do not know", "don't have that information", "unknown"))),

            # -------------------------------------------------------------
            # Category 10: Epistemic Self-Study & Curiosity Dreaming
            # -------------------------------------------------------------
            ("Epistemic Self-Study & Dreaming", "run dream cycle", "dream_cycle_execution",
             lambda r: any(w in r.lower() for w in ("dream", "step", "assimilated", "frontier", "entropy"))),
            ("Epistemic Self-Study & Dreaming", "what are your knowledge gaps", "knowledge_gaps_inquiry",
             lambda r: any(w in r.lower() for w in ("knowledge gap", "radial entropy", "missing axes"))),
            ("Epistemic Self-Study & Dreaming", "study concept hawk", "autonomous_curiosity_study",
             lambda r: "hawk" in r.lower() and any(w in r.lower() for w in ("curiosity study", "verified", "assimilated", "radial axis"))),

            # -------------------------------------------------------------
            # Category 11: Continuous Dynamic Physics & Sensory Grounding
            # -------------------------------------------------------------
            ("Dynamic Physics & Sensory Grounding", "What happens to apple over 10 days?", "continuous_ode_decay_simulation",
             lambda r: "apple" in r.lower() and "freshness" in r.lower() and any(w in r.lower() for w in ("decay", "simulation", "milestone", "oxidation"))),
            ("Dynamic Physics & Sensory Grounding", "What happens if I peel an apple?", "procedural_action_jump_peel",
             lambda r: "peel" in r.lower() and any(w in r.lower() for w in ("mass", "skin", "flesh", "conserved", "qpt"))),
            ("Dynamic Physics & Sensory Grounding", "observe sensor: [color: red, mass: 180g]", "multimodal_sensory_grounding",
             lambda r: "apple" in r.lower() and any(w in r.lower() for w in ("sensory", "grounded", "episodic", "anchored"))),
        ]

        item_results: list[TurnResult] = []

        initial_rss = self._get_rss_mb()

        for idx, (cat, user_msg, exp_tag, verifier_fn) in enumerate(turns, 1):
            t0 = time.perf_counter()
            reply = self.kernel.chat(user_msg)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            passed = bool(verifier_fn(reply))
            hd_tokens = self.kernel.headroom.stats().active_tokens
            current_rss = self._get_rss_mb()

            item_results.append(
                TurnResult(
                    turn_index=idx,
                    category=cat,
                    user_input=user_msg,
                    model_output=reply,
                    expected_status=exp_tag,
                    passed=passed,
                    latency_ms=round(elapsed_ms, 2),
                    headroom_tokens=hd_tokens,
                    rss_mb=current_rss,
                )
            )

        final_rss = self._get_rss_mb()

        # Group and compute category summaries
        cat_map: dict[str, list[TurnResult]] = {}
        for r in item_results:
            cat_map.setdefault(r.category, []).append(r)

        summaries: list[CategorySummary] = []
        for c_name, items in cat_map.items():
            tot = len(items)
            ps = sum(1 for i in items if i.passed)
            lats = sorted([i.latency_ms for i in items])
            avg_lat = sum(lats) / tot
            p95_idx = min(int(len(lats) * 0.95), len(lats) - 1)
            p99_idx = min(int(len(lats) * 0.99), len(lats) - 1)

            summaries.append(
                CategorySummary(
                    category=c_name,
                    total=tot,
                    passed=ps,
                    pass_rate=round(ps / tot * 100.0, 1),
                    avg_latency_ms=round(avg_lat, 2),
                    p95_latency_ms=round(lats[p95_idx], 2),
                    p99_latency_ms=round(lats[p99_idx], 2),
                )
            )

        all_lats = sorted([r.latency_ms for r in item_results])
        total_items = len(item_results)
        total_passed = sum(1 for r in item_results if r.passed)
        overall_acc = round(total_passed / total_items * 100.0, 1)
        avg_lat_all = round(sum(all_lats) / total_items, 2)
        p95_all = round(all_lats[min(int(total_items * 0.95), total_items - 1)], 2)
        p99_all = round(all_lats[min(int(total_items * 0.99), total_items - 1)], 2)

        return {
            "total_turns": total_items,
            "total_passed": total_passed,
            "overall_accuracy": overall_acc,
            "avg_latency_ms": avg_lat_all,
            "p95_latency_ms": p95_all,
            "p99_latency_ms": p99_all,
            "initial_rss_mb": initial_rss,
            "final_rss_mb": final_rss,
            "active_headroom_tokens": self.kernel.headroom.stats().active_tokens,
            "is_rust_accelerated": self.rust.is_accelerated,
            "categories": [asdict(s) for s in summaries],
            "items": [asdict(r) for r in item_results],
        }
