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
from little.language.parser import LearningEngine, SimpleParser
from little.memory.store import MemoryStore

DEFAULT_DB_PATH = Path("data/little.db")


def get_engine(db_path: Path = DEFAULT_DB_PATH) -> tuple[MemoryStore, LearningEngine]:
    store = MemoryStore(db_path)
    engine = LearningEngine(store)
    return store, engine


def cmd_init(args: argparse.Namespace) -> None:
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with MemoryStore(db_path):
        pass
    print(f"✓ Initialized LITTLE persistent memory store at: {db_path}")


def cmd_learn(args: argparse.Namespace) -> None:
    store, engine = get_engine(Path(args.db))
    try:
        statement = " ".join(args.statement)
        result = engine.learn(statement)
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
    store, engine = get_engine(Path(args.db))
    try:
        question = " ".join(args.question)
        res = engine.ask(question)
        print(f"\n[Inference Result: {res.status.value}]")
        print(f'Question:    "{question}"')
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
    store, engine = get_engine(Path(args.db))
    inquisitor = ActiveInquisitor(store, engine)
    print("=" * 65)
    print(" LITTLE Cognitive Architecture — Continuous Learning REPL")
    print(" Commands: 'quit' to exit, 'skills' to list procedural skills.")
    print(" Try: math, slicing, facts, or asking questions!")
    print("=" * 65 + "\n")

    try:
        while True:
            try:
                user_input = input("LITTLE> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", ":q"):
                print("Goodbye!")
                break

            if user_input.lower() == "skills":
                for s in store.list_skills():
                    print(f"  • {s.name}({', '.join(s.parameters)}) - {s.description}")
                continue

            # Determine whether input is question or statement/action
            is_question = (
                user_input.endswith("?")
                or user_input.lower().startswith(
                    ("what", "is", "are", "does", "can", "how", "calculate")
                )
            )

            if is_question:
                res = engine.ask(user_input)
                print(f"\n[{res.status.value}] Answer: {res.answer} (Confidence: {res.confidence * 100:.1f}%)")
                if res.evidence:
                    print(f"Evidence: {', '.join(res.evidence)}")

                # Check if unknown and prompt active clarification
                if res.is_unknown:
                    parsed = SimpleParser.parse_question(user_input)
                    if parsed:
                        s, p, o = parsed
                        prompt = inquisitor.inspect_uncertainty(res, s, p, o)
                        if prompt:
                            print(f"\n🤔 [Curiosity Question] {prompt.question_for_user}")
                            try:
                                resp = input("Your Answer> ").strip()
                                if resp:
                                    learn_res = inquisitor.resolve_response(prompt, resp)
                                    re_res = engine.ask(user_input)
                                    gain = inquisitor.calculate_information_gain(
                                        res.confidence, re_res.confidence
                                    )
                                    print(f"✓ Learned: {learn_res.message}")
                                    print(f"  New belief: {re_res.status.value} (Conf: {re_res.confidence*100:.1f}%, Info Gain: {gain:.2f} bits)\n")
                            except (EOFError, KeyboardInterrupt):
                                break
                print()
            else:
                # Statement or Action
                res = engine.learn(user_input)
                print(f"\n[Learned: {res.update_type.value}] {res.message}\n")
    finally:
        store.close()


def main() -> None:
    db_parent = argparse.ArgumentParser(add_help=False)
    db_parent.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help="Path to SQLite persistent database (default: data/little.db)",
    )

    parser = argparse.ArgumentParser(
        prog="little",
        parents=[db_parent],
        description="LITTLE — Continual Concept Learning and Cognitive Architecture CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # init
    p_init = subparsers.add_parser("init", parents=[db_parent], help="Initialize local database")
    p_init.set_defaults(func=cmd_init)

    # learn
    p_learn = subparsers.add_parser("learn", parents=[db_parent], help="Teach LITTLE a fact or observation")
    p_learn.add_argument("statement", nargs="+", help="Natural language statement")
    p_learn.set_defaults(func=cmd_learn)

    # ask
    p_ask = subparsers.add_parser("ask", parents=[db_parent], help="Ask LITTLE a question")
    p_ask.add_argument("question", nargs="+", help="Natural language question")
    p_ask.add_argument(
        "-v", "--verbose", action="store_true", help="Display reasoning trace"
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
    p_mem = subparsers.add_parser("memory", parents=[db_parent], help="Memory management commands")
    p_mem_sub = p_mem.add_subparsers(dest="memory_cmd")
    p_mem_list = p_mem_sub.add_parser("list", parents=[db_parent], help="List stored concepts and relations")
    p_mem_list.set_defaults(func=cmd_memory_list)

    # skills list
    p_skills = subparsers.add_parser("skills", parents=[db_parent], help="Procedural skills commands")
    p_skills_sub = p_skills.add_subparsers(dest="skills_cmd")
    p_skills_list = p_skills_sub.add_parser("list", parents=[db_parent], help="List procedural skills")
    p_skills_list.set_defaults(func=cmd_skills_list)

    # interact / chat
    p_chat = subparsers.add_parser("chat", parents=[db_parent], help="Interactive learning REPL")
    p_chat.set_defaults(func=cmd_interact)
    p_interact = subparsers.add_parser("interact", parents=[db_parent], help="Interactive learning REPL")
    p_interact.set_defaults(func=cmd_interact)

    # export
    p_export = subparsers.add_parser("export", parents=[db_parent], help="Export memory state to JSON")
    p_export.add_argument("-o", "--output", help="Destination JSON file path")
    p_export.set_defaults(func=cmd_export)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()

