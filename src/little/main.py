"""CLI entrypoint for the LITTLE cognitive architecture.

Supports commands:
  little init
  little learn "<statement>"
  little ask "<question>"
  little inspect concept <name>
  little memory list
  little export
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from little.active.inquisitor import ActiveInquisitor
from little.core.kernel import CognitiveKernel
from little.core.models import UpdateType
from little.core.runtime_paths import RuntimePaths
from little.knowledge.policy import LanguagePolicy
from little.language.parser import LearningEngine
from little.memory.store import MemoryStore


def default_db_path() -> Path:
    """Return the configured persistent-memory path."""
    return RuntimePaths.default().database


def get_engine(
    db_path: Path | None = None, seed_ontology: bool = True
) -> tuple[MemoryStore, LearningEngine]:
    store = MemoryStore(db_path or default_db_path(), seed_ontology=seed_ontology)
    engine = LearningEngine(store)
    return store, engine


def get_kernel(
    db_path: Path | None = None, seed_ontology: bool = True
) -> tuple[MemoryStore, CognitiveKernel]:
    store = MemoryStore(db_path or default_db_path(), seed_ontology=seed_ontology)
    return store, CognitiveKernel(store)


def cmd_init(args: argparse.Namespace) -> None:
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with MemoryStore(db_path, seed_ontology=True):
        pass
    print(f"✓ Initialized LITTLE persistent memory store at: {db_path}")


def cmd_learn(args: argparse.Namespace) -> None:
    store, kernel = get_kernel(Path(args.db))
    try:
        statement = " ".join(args.statement)
        outcome = kernel.process(statement)
        result = outcome.learning
        if result is None:
            inference = outcome.inference
            if outcome.frame.intent == "question":
                status = "NO_OP"
                message = (
                    "Input was recognized as an inquiry or calculation, not a "
                    f"declarative statement: '{statement}'"
                )
            else:
                status = inference.status.value if inference is not None else "UNKNOWN"
                message = (
                    inference.verbalize()
                    if inference is not None
                    else "No structured learning result was produced."
                )
            print("\n[Learning Event: NO_OP]")
            print(f'Input:       "{statement}"')
            print(f"Status:      {status}: {message}\n")
            return
        print(f"\n[Learning Event: {result.update_type.value}]")
        print(f'Input:       "{statement}"')
        if result.concepts_created:
            print(f"New Concepts: {', '.join(result.concepts_created)}")
        if result.relations_created:
            print(f"New Relations: {', '.join(result.relations_created)}")
        print(f"Status:      {result.message}\n")
    finally:
        store.close()


def cmd_ask(args: argparse.Namespace) -> None:
    store, kernel = get_kernel(Path(args.db))
    try:
        question = " ".join(args.question)
        outcome = kernel.process(question, allow_learning=False)
        res = outcome.inference
        if res is None:
            print('\n[Inference Result: UNKNOWN]')
            print(f'Question:    "{question}"')
            print("Response:    Input was routed to the learning pipeline.")
            print("Answer:      None")
            print("Confidence:  0.0%\n")
            return
        if getattr(args, "thinking", False):
            for step in res.trace:
                print(f"\n{step}" if step == res.trace[0] else step)
        print(f"\n[Inference Result: {res.status.value}]")
        print(f'Question:    "{question}"')
        print(f"Response:    {res.verbalize()}")
        print(f"Answer:      {res.answer}")
        print(f"Confidence:  {res.confidence * 100:.1f}%")
        if res.evidence:
            print(f"Evidence:    {', '.join(res.evidence)}")
        if args.verbose and res.trace:
            print("\nReasoning Trace:")
            for step in res.trace:
                print(f"  ↳ {step}")
        print()
    finally:
        store.close()


def cmd_inspect(args: argparse.Namespace) -> None:
    store, _ = get_engine(Path(args.db))
    try:
        target = args.target.strip().lower()
        if args.type == "concept":
            concept = store.get_concept(target)
            if not concept:
                print(f"Concept '{target}' not found in memory.")
                return
            print(f"\n[CONCEPT: {concept.name.upper()}]")
            print(f"ID:          {concept.id}")
            print(f"Category:    {concept.category or 'None'}")
            print(
                f"Aliases:     {', '.join(concept.aliases) if concept.aliases else 'None'}"
            )
            print(f"Attributes:  {json.dumps(concept.attributes, indent=2)}")
            print(f"Confidence:  {concept.confidence}")
            print(f"Status:      {concept.status.value}")

            # Outgoing relations
            out_rels = store.get_relations(subject_id=concept.id)
            if out_rels:
                print("\nOutgoing Relations:")
                for r in out_rels:
                    obj = store.get_concept(r.object_id)
                    obj_name = obj.name if obj else r.object_id
                    print(
                        f"  ↳ {concept.name} --[{r.predicate}]--> {obj_name} (conf: {r.confidence})"
                    )

            # Incoming relations
            in_rels = store.get_relations(object_id=concept.id)
            if in_rels:
                print("\nIncoming Relations:")
                for r in in_rels:
                    subj = store.get_concept(r.subject_id)
                    subj_name = subj.name if subj else r.subject_id
                    print(
                        f"  ↳ {subj_name} --[{r.predicate}]--> {concept.name} (conf: {r.confidence})"
                    )
            print()
    finally:
        store.close()


def cmd_memory_list(args: argparse.Namespace) -> None:
    store, _ = get_engine(Path(args.db))
    try:
        concepts = store.list_concepts()
        relations = store.get_relations()
        print(f"\n[LITTLE Persistent Memory Overview: {Path(args.db)}]")
        print(f"Total Concepts:  {len(concepts)}")
        print(f"Total Relations: {len(relations)}\n")

        if concepts:
            print("Registered Concepts:")
            for c in concepts[:30]:
                print(
                    f"  • {c.name.ljust(16)} (category: {c.category or 'N/A'}, status: {c.status.value})"
                )
            if len(concepts) > 30:
                print(f"  ... and {len(concepts) - 30} more")

        if relations:
            print("\nSemantic Relations:")
            for r in relations[:30]:
                s = store.get_concept(r.subject_id)
                o = store.get_concept(r.object_id)
                s_name = s.name if s else r.subject_id
                o_name = o.name if o else r.object_id
                print(
                    f"  ↳ ({s_name}, {r.predicate}, {o_name})  [conf: {r.confidence:.2f}, w+: {r.weight_positive}, w-: {r.weight_negative}]"
                )
            if len(relations) > 30:
                print(f"  ... and {len(relations) - 30} more")
        print()
    finally:
        store.close()


def cmd_export(args: argparse.Namespace) -> None:
    store, _ = get_engine(Path(args.db))
    try:
        state = store.export_state()
        out_path = Path(args.output) if args.output else None
        data = json.dumps(state, indent=2)
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(data, encoding="utf-8")
            print(f"✓ Exported cognitive memory state to: {out_path}")
        else:
            print(data)
    finally:
        store.close()


def cmd_import(args: argparse.Namespace) -> None:
    store = MemoryStore(Path(args.db), seed_ontology=True)
    try:
        from little.knowledge.importer import KnowledgeImporter

        importer = KnowledgeImporter(store)
        print("\n🚀 LITTLE Knowledge Ingestion Pipeline")
        print("---------------------------------------")

        def progress(parsed: int, stored: int) -> None:
            print(
                f"  ↳ Processed {stored:,} relations ({parsed:,} triples parsed)...",
                end="\r",
                flush=True,
            )

        dataset = getattr(args, "dataset", None)
        file_arg = getattr(args, "file", None)
        min_w = getattr(args, "min_weight", 1.0)
        limit = getattr(args, "limit", None)
        fmt = getattr(args, "format", None)

        if dataset in ("commonsense", "world"):
            print("📦 Loading curated commonsense world-knowledge bundle...")
            stats = importer.import_commonsense_bundle(progress_cb=progress)
        elif dataset in ("large", "world100k", "extended"):
            target_n = limit or 100_000
            print(
                f"📦 Streaming {target_n:,} curated large-scale commonsense ontology triples..."
            )
            stats = importer.import_large_scale_ontology(
                target_count=target_n, progress_cb=progress
            )
        elif file_arg:
            path = Path(file_arg)
            print(
                f"📂 Ingesting knowledge assertions from: {path} (min_weight: {min_w})..."
            )
            if fmt == "conceptnet" or (
                not fmt and path.suffix.lower() in (".csv", ".tsv")
            ):
                stats = importer.import_conceptnet_file(
                    path,
                    min_weight=min_w,
                    max_triples=limit,
                    progress_cb=progress,
                )
            elif fmt in ("json", "jsonl") or (
                not fmt and path.suffix.lower() in (".json", ".jsonl")
            ):
                stats = importer.import_json_file(
                    path,
                    min_weight=min_w,
                    max_triples=limit,
                    progress_cb=progress,
                )
            else:
                stats = importer.import_tsv_file(
                    path,
                    min_weight=min_w,
                    max_triples=limit,
                    progress_cb=progress,
                )
        else:
            print("❌ Error: Specify either --dataset commonsense or --file <path>")
            return

        print()
        print(f"\n✓ Ingestion complete! {stats.summary()}\n")
    finally:
        store.close()


def cmd_skills_list(args: argparse.Namespace) -> None:
    store, _ = get_engine(Path(args.db))
    try:
        skills = store.list_skills()
        print(f"\n[LITTLE Procedural Skills: {len(skills)} registered]")
        for s in skills:
            params = ", ".join(s.parameters)
            print(f"  • {s.name}({params})")
            if s.description:
                print(f"    ↳ {s.description}")
            print(f"    Code: {s.code_body.strip().replace(chr(10), '; ')}")
        print()
    finally:
        store.close()


def cmd_interact(args: argparse.Namespace) -> None:
    store, kernel = get_kernel(Path(args.db))
    policy = LanguagePolicy.default()
    inquisitor = ActiveInquisitor(store, kernel.learner, policy=policy)
    print("=" * 68)
    print(" LITTLE Cognitive Architecture — Continuous Learning Interactive REPL")
    print(" Commands: 'help' for examples, 'memory' for stored facts, 'quit' to exit.")
    print(" Try: facts, questions, math, slicing, or physical dynamics!")
    print("=" * 68 + "\n")

    help_cmds = set(policy.help_commands)

    try:
        while True:
            try:
                user_input = input("LITTLE> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            clean_lower = user_input.lower().strip()

            if clean_lower in policy.exit_commands:
                print("Goodbye!")
                break

            if clean_lower in policy.clear_commands:
                import os

                os.system("clear" if os.name != "nt" else "cls")
                continue

            if clean_lower in help_cmds:
                print("""
========================================================================
 LITTLE Cognitive Architecture — Interactive Capabilities
========================================================================

1. 📚 Teach Facts (declarative statements):
   • "A dog is an animal."
   • "An animal is a living thing."
   • "An animal is not a vehicle."
   • "The apple is red."
   • "Alice has a dog."

2. ❓ Ask Questions (transitive deduction & properties):
   • "Is a dog an animal?"
   • "Is a dog a living thing?"       (Multi-hop deductive reasoning)
   • "Is a dog a vehicle?"           (Disjoint refutation -> False)
   • "What is an apple?"             (Knowledge graph summary)
   • "What color is the apple?"

3. 🔪 Physical Transformations & Continuous Dynamics (CfC ODEs):
   • "Slice an apple into 4 pieces."
   • "Is an apple slice part of an apple?"
   • "What color is the apple slice after 2 hours?" (Browns over continuous time)
   • "Is the apple slice fresh after 30 minutes?"

4. ⚡ Exact Python Algorithmic Skills (0.45 ms, 100% precision):
   • "What is 123 + 456?"            / "Calculate 50 * 25"
   • "What is the factorial of 10?"
   • "What is the fibonacci of 25?"
   • "Is 104729 prime?"
   • "What is the reverse of 'antigravity'?"
   • "Is 'racecar' a palindrome?"

5. 🔍 Inspection & REPL Utilities:
   • "memory" or "concepts"          (List all stored concepts & relations)
   • "inspect <concept>"             (Inspect properties & edges, e.g. 'inspect apple')
   • "skills"                        (List registered procedural Python skills)
   • "who are you"                   (Self-identity & architecture explanation)
   • "clear"                         (Clear terminal screen)
   • "quit" or "exit"                (Save and exit)
========================================================================
""")
                continue

            if clean_lower in policy.memory_commands:
                concepts = store.list_concepts()
                relations = store.get_relations()
                print(
                    f"\n[Persistent Memory: {len(concepts)} Concepts, {len(relations)} Relations]"
                )
                if concepts:
                    c_names = [c.name for c in concepts]
                    print(f"Concepts:  {', '.join(c_names)}")
                if relations:
                    print("Relations:")
                    for r in relations[:20]:
                        s = store.get_concept(r.subject_id)
                        o = store.get_concept(r.object_id)
                        s_name = s.name if s else r.subject_id
                        o_name = o.name if o else r.object_id
                        print(
                            f"  ↳ ({s_name} {r.predicate} {o_name}) [conf: {r.confidence:.2f}]"
                        )
                    if len(relations) > 20:
                        print(f"  ... and {len(relations) - 20} more")
                print()
                continue

            if clean_lower.startswith("inspect "):
                target = clean_lower[8:].strip()
                concept = store.get_concept(target)
                if not concept:
                    print(f"\nConcept '{target}' not found in memory.\n")
                    continue
                print(f"\n[CONCEPT: {concept.name.upper()}]")
                print(f"ID:          {concept.id}")
                print(f"Category:    {concept.category or 'None'}")
                print(f"Attributes:  {json.dumps(concept.attributes, indent=2)}")
                print(f"Confidence:  {concept.confidence}")
                out_rels = store.get_relations(subject_id=concept.id)
                if out_rels:
                    print("Outgoing Relations:")
                    for r in out_rels:
                        obj = store.get_concept(r.object_id)
                        obj_name = obj.name if obj else r.object_id
                        print(
                            f"  ↳ {concept.name} --[{r.predicate}]--> {obj_name} (conf: {r.confidence:.2f})"
                        )
                in_rels = store.get_relations(object_id=concept.id)
                if in_rels:
                    print("Incoming Relations:")
                    for r in in_rels:
                        subj = store.get_concept(r.subject_id)
                        subj_name = subj.name if subj else r.subject_id
                        print(
                            f"  ↳ {subj_name} --[{r.predicate}]--> {concept.name} (conf: {r.confidence:.2f})"
                        )
                print()
                continue

            if clean_lower in policy.skills_commands:
                print("\n[Registered Procedural Skills]")
                for s in store.list_skills():
                    print(f"  • {s.name}({', '.join(s.parameters)}) - {s.description}")
                print()
                continue

            # Semantic input always enters through the cognitive kernel.
            outcome = kernel.process(user_input)
            if outcome.inference is not None:
                res = outcome.inference
                if getattr(args, "thinking", False):
                    print(
                        f"\n🧭 Route: {outcome.mode.value} — "
                        f"{res.trace[1] if len(res.trace) > 1 else ''}"
                    )
                    for step in res.trace:
                        print(f"   ↳ {step}")
                print(f"\n💬 LITTLE: {res.verbalize()}")
                print(
                    f"   ↳ [Status: {res.status.value} | Confidence: {res.confidence * 100:.1f}%]"
                )
                if res.evidence:
                    print(f"   ↳ Evidence: {', '.join(res.evidence)}")

                # Check if unknown and prompt active clarification
                if res.is_unknown:
                    parsed = outcome.frame.parsed_query or next(
                        (
                            query
                            for query in outcome.frame.parsed_queries
                            if query is not None
                        ),
                        None,
                    )
                    if parsed:
                        s, p, o = parsed
                        prompt = inquisitor.inspect_uncertainty(res, s, p, o)
                        if prompt:
                            print(f"\n🤔 LITTLE: {prompt.question_for_user}")
                            try:
                                resp = input("Your Answer> ").strip()
                                if resp:
                                    resp_lower = resp.lower().strip()
                                    if resp_lower in policy.cancel_words:
                                        print("💬 LITTLE: Understood, skipped.\n")
                                        continue
                                    response_outcome = kernel.process(
                                        resp, allow_learning=False
                                    )
                                    if response_outcome.frame.intent == "question":
                                        re_q = response_outcome.inference
                                        if re_q is None:
                                            continue
                                        print(f"\n💬 LITTLE: {re_q.verbalize()}")
                                        print(
                                            f"   ↳ [Status: {re_q.status.value} | Confidence: {re_q.confidence * 100:.1f}%]"
                                        )
                                        if re_q.evidence:
                                            print(
                                                f"   ↳ Evidence: {', '.join(re_q.evidence)}"
                                            )
                                        print()
                                        continue

                                    statement = inquisitor.candidate_statement(prompt, resp)
                                    if statement is None:
                                        print("💬 LITTLE: I could not use that response.\n")
                                        continue
                                    accepted = kernel.process(statement)
                                    learn_res = accepted.learning

                                    if learn_res is not None and learn_res.update_type.value != "NO_OP":
                                        updated = kernel.process(user_input)
                                        re_res = updated.inference
                                        if re_res is None:
                                            re_res = res
                                        gain = inquisitor.calculate_information_gain(
                                            res.confidence, re_res.confidence
                                        )
                                        print(
                                            f"💬 LITTLE: Thank you! {learn_res.message}"
                                        )
                                        print(
                                            f"   ↳ New belief: {re_res.status.value} (Conf: {re_res.confidence * 100:.1f}%, Info Gain: {gain:.2f} bits)\n"
                                        )
                                    else:
                                        print(
                                            f"💬 LITTLE: {learn_res.message if learn_res else 'I could not use that response.'}\n"
                                        )
                            except (EOFError, KeyboardInterrupt):
                                break
                print()
            elif outcome.learning is not None:
                res = outcome.learning
                if res.update_type == UpdateType.NO_OP:
                    print(
                        f"\n💬 LITTLE: I could not extract structured relations from: '{user_input}'"
                    )
                    print(
                        "   💡 Tip: Try phrasing as a fact (e.g. 'A dog is an animal', 'An apple is red') or action ('Slice apple into 4 pieces'). Type 'help' for examples.\n"
                    )
                else:
                    print(f"\n💬 LITTLE: Understood! {res.message}")
                    print(f"   ↳ [Update: {res.update_type.value}]\n")
    finally:
        store.close()


def build_parser() -> argparse.ArgumentParser:
    db_parent = argparse.ArgumentParser(add_help=False)
    db_parent.add_argument(
        "--db",
        default=argparse.SUPPRESS,
        help="Path to SQLite persistent database (default: data/little.db)",
    )

    parser = argparse.ArgumentParser(
        prog="little",
        description="LITTLE — Continual Concept Learning and Cognitive Architecture CLI",
    )
    parser.add_argument(
        "--db",
        default=str(default_db_path()),
        help="Path to SQLite persistent database (default: data/little.db)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # init
    p_init = subparsers.add_parser(
        "init", parents=[db_parent], help="Initialize local database"
    )
    p_init.set_defaults(func=cmd_init)

    # learn
    p_learn = subparsers.add_parser(
        "learn", parents=[db_parent], help="Teach LITTLE a fact or observation"
    )
    p_learn.add_argument("statement", nargs="+", help="Natural language statement")
    p_learn.set_defaults(func=cmd_learn)

    # ask
    p_ask = subparsers.add_parser(
        "ask", parents=[db_parent], help="Ask LITTLE a question"
    )
    p_ask.add_argument("question", nargs="+", help="Natural language question")
    p_ask.add_argument(
        "-v", "--verbose", action="store_true", help="Display reasoning trace"
    )
    p_ask.add_argument(
        "--thinking",
        action="store_true",
        help="Display GLM bidirectional frontier collision trace",
    )
    p_ask.set_defaults(func=cmd_ask)

    # inspect
    p_inspect = subparsers.add_parser(
        "inspect", parents=[db_parent], help="Inspect an internal concept or entity"
    )
    p_inspect.add_argument("type", choices=["concept", "entity"], help="Item type")
    p_inspect.add_argument("target", help="Name or ID to inspect")
    p_inspect.set_defaults(func=cmd_inspect)

    # memory list
    p_mem = subparsers.add_parser(
        "memory", parents=[db_parent], help="Memory management commands"
    )
    p_mem_sub = p_mem.add_subparsers(dest="memory_cmd")
    p_mem_list = p_mem_sub.add_parser(
        "list", parents=[db_parent], help="List stored concepts and relations"
    )
    p_mem_list.set_defaults(func=cmd_memory_list)

    # skills list
    p_skills = subparsers.add_parser(
        "skills", parents=[db_parent], help="Procedural skills commands"
    )
    p_skills_sub = p_skills.add_subparsers(dest="skills_cmd")
    p_skills_list = p_skills_sub.add_parser(
        "list", parents=[db_parent], help="List procedural skills"
    )
    p_skills_list.set_defaults(func=cmd_skills_list)

    # interact / chat
    p_chat = subparsers.add_parser(
        "chat", parents=[db_parent], help="Interactive learning REPL"
    )
    p_chat.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Thinking Mode inspectable traces",
    )
    p_chat.set_defaults(func=cmd_interact)
    p_interact = subparsers.add_parser(
        "interact", parents=[db_parent], help="Interactive learning REPL"
    )
    p_interact.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Thinking Mode inspectable traces",
    )
    p_interact.set_defaults(func=cmd_interact)

    # export
    p_export = subparsers.add_parser(
        "export", parents=[db_parent], help="Export memory state to JSON"
    )
    p_export.add_argument("-o", "--output", help="Destination JSON file path")
    p_export.set_defaults(func=cmd_export)

    # import
    p_import = subparsers.add_parser(
        "import",
        parents=[db_parent],
        help="Import world-knowledge triples into memory",
    )
    p_import.add_argument(
        "--file",
        "-f",
        default=None,
        help="Path to knowledge dataset file (CSV/TSV/JSON)",
    )
    p_import.add_argument(
        "--dataset",
        choices=["commonsense", "world", "large", "world100k", "extended"],
        default=None,
        help="Built-in curated knowledge dataset (commonsense, world, large, world100k)",
    )
    p_import.add_argument(
        "--format",
        choices=["conceptnet", "tsv", "json", "jsonl"],
        default=None,
        help="File format override",
    )
    p_import.add_argument(
        "--min-weight",
        type=float,
        default=1.0,
        help="Minimum assertion confidence weight (default: 1.0)",
    )
    p_import.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum triples to import",
    )
    p_import.set_defaults(func=cmd_import)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
