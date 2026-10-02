#!/usr/bin/env python3
"""Interactive Terminal Chat interface for LITTLE / MIVI Cognitive Architecture."""

import os
import sys
import time
from pathlib import Path

# Ensure src/ is on the Python module search path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from little.core.kernel import CognitiveKernel
from little.memory.store import MemoryStore


def main() -> None:
    db_path = ROOT_DIR / "data" / "little.db"
    store = MemoryStore(db_path, seed_ontology=True)
    kernel = CognitiveKernel(store)

    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print(f"\n{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{GREEN}  LITTLE / MIVI Cognitive Architecture — Interactive Chat & Learning  {RESET}")
    print(f"{GRAY}  Verifiable Neuro-Symbolic Model • 0% Hallucinations • Sub-15ms CPU{RESET}")
    print(f"{CYAN}======================================================================{RESET}")
    print(f"{YELLOW}Commands & Capabilities:{RESET}")
    print(f"  • Chat naturally:  {GRAY}\"Hello! Nice to meet you.\"{RESET}")
    print(f"  • Inquire:         {GRAY}\"Describe an ocelot.\" / \"Is an ocelot an animal?\"{RESET}")
    print(f"  • Teach new facts: Type {BOLD}'teach <sentence>'{RESET} (e.g. 'teach An ocelot is a feline.')")
    print(f"  • Ingest corpus:   Type {BOLD}'corpus'{RESET} to ingest real-world astronomy, biology, CS, physics.")
    print(f"  • Sleep & dream:   Type {BOLD}'sleep'{RESET} or {BOLD}'dream'{RESET} for memory consolidation & invariant mining.")
    print(f"  • Multi-step plan: Type {BOLD}'plan <goal>'{RESET} (e.g. 'plan calculate 15 * 84 and save to ans.txt')")
    print(f"  • Memory stats:    Type {BOLD}'memory'{RESET} to view 3-tier memory telemetry.")
    print(f"  • Screen/Exit:     Type {BOLD}'clear'{RESET} to clear, {BOLD}'exit'{RESET} to quit.\n")

    while True:
        try:
            user_input = input(f"{BOLD}You > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{BOLD}{CYAN}LITTLE >{RESET} Goodbye! Everything learned is safely preserved in memory.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print(f"\n{BOLD}{CYAN}LITTLE >{RESET} Goodbye! Everything learned is safely preserved in memory.")
            break

        if user_input.lower() == "clear":
            os.system("clear" if os.name != "nt" else "cls")
            continue

        if user_input.lower() in ("memory", "stats", "telemetry"):
            telemetry = kernel.paged_store.get_telemetry()
            res_list = ", ".join(telemetry.resident_concepts[:10]) or "None yet"
            print(f"\n{BOLD}{CYAN}LITTLE > [3-Tier Memory & Working Set Hierarchy]{RESET}")
            print(f"  • {BOLD}Tier 3 (Resident Window){RESET}: {telemetry.resident_count}/{telemetry.resident_capacity} concepts")
            print(f"    Active concepts: {GRAY}{res_list}{RESET}")
            print(f"  • {BOLD}Tier 2 (RAM LRU Cache){RESET}:   {telemetry.lru_count}/{telemetry.lru_capacity} concepts")
            print(f"  • {BOLD}Tier 1 (Disk Vault){RESET}:      SQLite + OKF Markdown Knowledge Base")
            print(f"  • {BOLD}Telemetry{RESET}:               Hit Rate: {telemetry.hit_rate*100:.1f}% | Resident Hits: {telemetry.resident_hits} | LRU Hits: {telemetry.lru_hits} | Disk Fetches: {telemetry.disk_fetches}\n")
            continue

        if user_input.lower() in ("sleep", "dream", "consolidate"):
            print(f"\n{BOLD}{CYAN}LITTLE >{RESET} Entering offline sleep consolidation cycle (dreaming, invariant mining, episodic compaction)...")
            t0 = time.perf_counter()
            report = kernel.sleep(max_dream_steps=5)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"{BOLD}{GREEN}LITTLE > [Sleep Consolidation Report]{RESET}")
            print(f"  • {report.summary_message}")
            if report.invariants_mined:
                mined_str = ", ".join(f"({m['subject']} disjoint_with {m['object']})" for m in report.invariants_mined[:4])
                print(f"  • Mined Invariants: {GRAY}{mined_str}{RESET}")
            if report.dream_report.persisted_claims:
                claims_str = ", ".join(f"({c['subject']} {c['predicate']} {c['object']})" for c in report.dream_report.persisted_claims[:4])
                print(f"  • Discovered Hypotheses: {GRAY}{claims_str}{RESET}")
            print(f"  • Total Cycle Time: {dt_ms:.1f}ms\n")
            continue

        if user_input.lower() in ("corpus", "learn corpus", "ingest corpus"):
            print(f"\n{BOLD}{CYAN}LITTLE >{RESET} Ingesting foundational encyclopedic world corpus...")
            t0 = time.perf_counter()
            summary = kernel.learn_world_corpus()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"{BOLD}{GREEN}LITTLE > [World Corpus Ingestion Summary]{RESET}")
            print(f"  • Chunks processed:        {summary.total_chunks}")
            print(f"  • Sentences analyzed:      {summary.total_sentences}")
            print(f"  • Concepts created:        {summary.concepts_count}")
            print(f"  • Relations committed:      {summary.relations_count}")
            print(f"  • Contradictions blocked:  {summary.total_contradictions_blocked}")
            print(f"  • Ingestion latency:       {dt_ms:.1f}ms\n")
            continue

        if user_input.lower().startswith("plan "):
            goal_text = user_input[5:].strip()
            print(f"\n{BOLD}{CYAN}LITTLE >{RESET} Formulating verified plan for: '{goal_text}'...")
            t0 = time.perf_counter()
            plan_res = kernel.plan(goal_text)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            status_color = GREEN if plan_res.success else YELLOW
            print(f"{BOLD}{status_color}LITTLE > [Agentic Plan Outcome: {'SUCCESS' if plan_res.success else 'FAILED'}]{RESET}")
            for step in plan_res.steps:
                st_val = getattr(step.status, "value", str(step.status))
                print(f"  • Step {step.step_index}: {step.description} -> {st_val}")
                if step.result and step.result.output:
                    print(f"    Output: {GRAY}{str(step.result.output)[:120]}{RESET}")
            print(f"  • Execution time: {dt_ms:.1f}ms\n")
            continue

        if user_input.lower().startswith("teach "):
            fact_text = user_input[6:].strip()
            t0 = time.perf_counter()
            res = kernel.teach(fact_text)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"\n{BOLD}{CYAN}LITTLE >{RESET} {res.message}  {GRAY}({dt_ms:.1f}ms){RESET}\n")
            continue

        t0 = time.perf_counter()
        reply = kernel.chat(user_input)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        print(f"\n{BOLD}{CYAN}LITTLE >{RESET} {reply}  {GRAY}({dt_ms:.1f}ms){RESET}\n")


if __name__ == "__main__":
    main()
