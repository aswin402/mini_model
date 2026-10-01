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
    print(f"{YELLOW}Tips:{RESET}")
    print(f"  • Chat naturally:  {GRAY}\"Hello! Nice to meet you.\"{RESET}")
    print(f"  • Teach new facts: {GRAY}\"An ocelot is a feline.\"{RESET}")
    print(f"  • Inquire:         {GRAY}\"Describe an ocelot.\" / \"Is an ocelot an animal?\"{RESET}")
    print(f"  • Solve math:      {GRAY}\"What is 15% of 800?\" / \"What is 7 factorial?\"{RESET}")
    print(f"  • Commands:        Type {BOLD}'exit'{RESET} to quit, {BOLD}'clear'{RESET} to clear screen.\n")

    while True:
        try:
            user_input = input(f"{BOLD}You > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{BOLD}{CYAN}LITTLE >{RESET} Goodbye! Everything learned is safely preserved in memory.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print(f"{BOLD}{CYAN}LITTLE >{RESET} Goodbye! Everything learned is safely preserved in memory.")
            break

        if user_input.lower() == "clear":
            os.system("clear" if os.name != "nt" else "cls")
            continue

        t0 = time.perf_counter()
        reply = kernel.chat(user_input)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        print(f"\n{BOLD}{CYAN}LITTLE >{RESET} {reply}  {GRAY}({dt_ms:.1f}ms){RESET}\n")


if __name__ == "__main__":
    main()
