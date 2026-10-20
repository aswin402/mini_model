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


def cmd_memory_prune(args: argparse.Namespace) -> None:
    store = MemoryStore(Path(args.db), seed_ontology=False)
    try:
        dry_run = bool(getattr(args, "dry_run", False))
        matched = store.prune_polluted_concepts(dry_run=dry_run)
        mode = "Dry-Run Preview" if dry_run else "Pruned"
        print(f"\n[LITTLE Memory Hygiene — {mode}: {Path(args.db)}]")
        print(f"Matched Polluted Concepts: {len(matched)}")
        for name in matched:
            print(f"  • {name}")
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


def cmd_curious(args: argparse.Namespace) -> None:
    store, _ = get_kernel(Path(args.db), seed_ontology=False)
    try:
        from little.active.self_study import AutonomousSelfStudyEngine
        study_engine = AutonomousSelfStudyEngine(store)
        entropies = study_engine.compute_all_knot_entropies()
        print("\n🧠 [360° Concept Knot Spoke Entropy]")
        print("Scanned radial axes: Taxonomy, Mereology, Dynamics, Invariants, Procedural, Episodic\n")
        concept_filter = getattr(args, "concept", None)
        top_k = getattr(args, "top", 10)
        displayed = 0
        for concept_name, entropy in entropies.items():
            if concept_filter and concept_filter.lower() not in concept_name.lower():
                continue
            if displayed >= top_k:
                break
            missing = study_engine.identify_missing_axes(concept_name)
            bar = "█" * int(entropy * 10) + "░" * (10 - int(entropy * 10))
            print(f"  • {concept_name:<20} H={entropy:.2f} [{bar}] Missing: {', '.join(missing)}")
            displayed += 1
        print(f"\nTotal concepts indexed: {len(entropies)}\n")
    finally:
        store.close()


def cmd_study(args: argparse.Namespace) -> None:
    store, _ = get_kernel(Path(args.db), seed_ontology=False)
    try:
        from little.active.self_study import AutonomousSelfStudyEngine
        study_engine = AutonomousSelfStudyEngine(store)
        steps = getattr(args, "steps", 5)
        target = getattr(args, "target", None)
        verbose = getattr(args, "verbose", False)

        print("\n🔄 [Autonomous Self-Study Engine]")
        print(f"Starting closed-loop curiosity cycle (max_steps={steps}, target={target or 'auto-entropy'})...\n")

        if target:
            results = []
            res = study_engine.study_step(target_concept=target)
            if res:
                results.append(res)
        else:
            results = study_engine.study_cycle(max_steps=steps)

        if not results:
            print("No analogical hypotheses could be formulated from current hypernym DAG.\n")
            return

        for idx, r in enumerate(results, 1):
            status_symbol = "✓" if r.status == "PERSISTED" else "✗"
            print(f"[{idx}] {status_symbol} Target: {r.target_concept} | Axis: {r.missing_axis} | Status: {r.status}")
            if r.hypothesis:
                hyp_str = f"({r.hypothesis.get('subject')} {r.hypothesis.get('predicate')} {r.hypothesis.get('object')})"
                donor_str = f" (via donor '{r.hypothesis.get('donor')}')" if r.hypothesis.get("donor") else ""
                print(f"    ↳ Hypothesis: {hyp_str}{donor_str}")
            if verbose and r.trace:
                for t in r.trace:
                    print(f"       • {t}")
        persisted_count = sum(1 for r in results if r.status == "PERSISTED")
        print(f"\nCompleted self-study cycle: {persisted_count}/{len(results)} hypotheses verified and persisted live to SQLite.\n")
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

            if clean_lower in policy.curious_commands or any(
                clean_lower.startswith(f"{cmd} ") for cmd in policy.curious_commands
            ):
                from little.active.self_study import AutonomousSelfStudyEngine

                study_engine = AutonomousSelfStudyEngine(store)
                parts = clean_lower.split(maxsplit=1)
                concept_filter = parts[1].strip() if len(parts) > 1 else None
                entropies = study_engine.compute_all_knot_entropies()
                print("\n🧠 [360° Concept Knot Spoke Entropy]")
                displayed = 0
                for c_name, ent in entropies.items():
                    if concept_filter and concept_filter not in c_name.lower():
                        continue
                    if displayed >= 10:
                        break
                    missing = study_engine.identify_missing_axes(c_name)
                    bar = "█" * int(ent * 10) + "░" * (10 - int(ent * 10))
                    print(
                        f"  • {c_name:<18} H={ent:.2f} [{bar}] Missing: {', '.join(missing)}"
                    )
                    displayed += 1
                print()
                continue

            if clean_lower in policy.study_commands or any(
                clean_lower.startswith(f"{cmd} ") for cmd in policy.study_commands
            ):
                from little.active.self_study import AutonomousSelfStudyEngine

                study_engine = AutonomousSelfStudyEngine(store)
                parts = clean_lower.split(maxsplit=1)
                target = parts[1].strip() if len(parts) > 1 else None
                print(
                    f"\n🔄 Running autonomous self-study cycle (target={target or 'auto-entropy'})..."
                )
                if target:
                    res = study_engine.study_step(target_concept=target)
                    results = [res] if res else []
                else:
                    results = study_engine.study_cycle(max_steps=3)
                if not results:
                    print(
                        "  No analogical hypotheses could be formulated from current hypernym DAG.\n"
                    )
                for r in results:
                    status_symbol = "✓" if r.status == "PERSISTED" else "✗"
                    print(
                        f"  {status_symbol} [{r.status}] {r.target_concept} ({r.missing_axis})"
                    )
                    if r.hypothesis:
                        print(
                            f"     ↳ ({r.hypothesis.get('subject')} {r.hypothesis.get('predicate')} {r.hypothesis.get('object')})"
                        )
                print()
                continue

            # Check for pure social intents (greetings, pleasantries, identity)
            social_reply = kernel.dialogue_surface.handle_social(user_input)
            if social_reply is not None:
                print(f"\n💬 LITTLE: {social_reply}\n")
                continue

            # Decompose mixed utterance if leading greeting is present
            greeting_word, remainder = kernel.dialogue_surface.decompose_utterance(user_input)
            if greeting_word:
                rem_social = kernel.dialogue_surface.handle_social(remainder)
                if rem_social is not None:
                    print(f"\n💬 LITTLE: {greeting_word.capitalize()}! {rem_social}\n")
                    continue

            # Check action router for tool execution intents
            tool_call = kernel.action_router.route(remainder)
            if tool_call is not None:
                exec_result = kernel.tool_executor.execute(tool_call)
                kernel.tool_grounder.ground(exec_result)
                if getattr(args, "thinking", False):
                    print(
                        f"\n🧭 Route: TOOL_ACTION ({tool_call.tool_name}) [SHA: {exec_result.verifiable_hash}]"
                    )
                    print(f"   ↳ Arguments: {tool_call.arguments}")
                    print(f"   ↳ Elapsed: {exec_result.execution_time_ms:.2f}ms")
                body = kernel.dialogue_surface.verbalize_tool_result(exec_result)
                reply = kernel.dialogue_surface.compose_reply(greeting_word, body)
                print(f"\n💬 LITTLE: {reply}\n")
                continue

            # Semantic input enters through the cognitive kernel.
            outcome = kernel.process(remainder)
            if outcome.inference is not None:
                res = outcome.inference
                if getattr(args, "thinking", False):
                    print(
                        f"\n🧭 Route: {outcome.mode.value} — "
                        f"{res.trace[1] if len(res.trace) > 1 else ''}"
                    )
                    for step in res.trace:
                        print(f"   ↳ {step}")
                resolved_subj = (
                    outcome.frame.parsed_query[0]
                    if outcome.frame.parsed_query and isinstance(outcome.frame.parsed_query, tuple)
                    else kernel.learner.dialogue.get_salient_subject()
                )
                verbalized = kernel.dialogue_surface.verbalize_inference(
                    res,
                    query=remainder,
                    resolved_subject=resolved_subj,
                    parsed_query=outcome.frame.parsed_query,
                )
                final_reply = kernel.dialogue_surface.compose_reply(greeting_word, verbalized)
                print(f"\n💬 LITTLE: {final_reply}")
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
                    verbalized = kernel.dialogue_surface.verbalize_learning(res)
                    final_reply = kernel.dialogue_surface.compose_reply(greeting_word, verbalized)
                    print(f"\n💬 LITTLE: {final_reply}")
                    print(f"   ↳ [Update: {res.update_type.value}]\n")
                    if (
                        getattr(args, "auto_study", False)
                        and res.concepts_created
                    ):
                        from little.active.self_study import AutonomousSelfStudyEngine

                        study_engine = AutonomousSelfStudyEngine(store)
                        for created_concept in res.concepts_created:
                            study_res = study_engine.study_step(
                                target_concept=created_concept
                            )
                            if study_res and study_res.status == "PERSISTED":
                                hyp = study_res.hypothesis or {}
                                print(
                                    f"   🔄 [Auto-Study] Formulated & persisted spoke for '{created_concept}': "
                                    f"({hyp.get('subject')} {hyp.get('predicate')} {hyp.get('object')})\n"
                                )
    finally:
        store.close()


def cmd_tool(args: argparse.Namespace) -> None:
    from little.actions import (
        ActionPolicy,
        ToolCall,
        ToolDefinitionsPolicy,
        ToolExecutor,
    )

    tool_policy = ToolDefinitionsPolicy.default()
    action_policy = ActionPolicy.default()
    executor = ToolExecutor(tool_policy=tool_policy, action_policy=action_policy)

    if args.action == "list":
        print("\n🔧 Registered Typed Tools:")
        print(f"{'Tool Name':<16} {'Return Type':<12} {'Sandboxed':<10} {'Description'}")
        print("-" * 75)
        for name, tdef in tool_policy.tools.items():
            sandboxed = "Yes" if tdef.sandboxed else "No"
            print(f"{name:<16} {tdef.return_type:<12} {sandboxed:<10} {tdef.description}")
        print()
    elif args.action == "exec":
        tool_name = args.name
        if tool_name not in tool_policy.tools:
            print(f"❌ Unknown tool '{tool_name}'. Available: {list(tool_policy.tools.keys())}")
            sys.exit(1)
        raw_args = {}
        if getattr(args, "args", None):
            try:
                raw_args = json.loads(args.args)
            except json.JSONDecodeError as ex:
                print(f"❌ Invalid JSON arguments: {ex}")
                sys.exit(1)
        call = ToolCall(tool_name=tool_name, arguments=raw_args)
        result = executor.execute(call)
        print(f"\n⚙️  Tool Execution: {tool_name}")
        print(f"   ↳ Success: {'✓' if result.success else '✗'}")
        print(f"   ↳ Elapsed: {result.execution_time_ms:.2f}ms")
        print(f"   ↳ SHA-256: {result.verifiable_hash}")
        if result.success:
            print(f"   ↳ Output:  {result.output}\n")
        else:
            print(f"   ↳ Error:   {result.error}\n")


def cmd_status(args: argparse.Namespace) -> None:
    store, kernel = get_kernel(Path(args.db), seed_ontology=False)
    try:
        snap = kernel.get_telemetry_snapshot()
        if getattr(args, "json", False):
            print(json.dumps(snap, indent=2))
            return
        mem = snap["memory"]
        paged = snap["paged_store"]
        plast = snap["plasticity"]
        brakes = snap["growth_brakes"]
        canary = snap["canary"]
        mlm = snap.get("mlm_spiderweb", {})
        print(f"\n📊 [LITTLE Cognitive Telemetry Dashboard: {Path(args.db)}]")
        print("-" * 68)
        print(
            f"Memory Store:      {mem['total_concepts']} concepts | "
            f"{mem['total_relations']} relations | epoch={mem['concept_mutation_epoch']}"
        )
        print(
            f"Paged Store:       {paged['active_count']} resident | {paged['lru_count']} LRU | "
            f"reuses={paged['slot_reuses']} | dirty_inv={paged['dirty_invalidations']} | "
            f"hit_rate={paged['hit_rate'] * 100:.1f}%"
        )
        print(
            f"Plasticity:        lr={plast['effective_lr']:.4f} (base={plast['base_lr']:.4f}) | "
            f"e_slow={plast['slow_effect_size']:+.3f} | e_fast={plast['fast_effect_size']:+.3f} | "
            f"trunk={plast['trunk_damping']}x"
        )
        print(
            f"Spider-Web MLM:    {mlm.get('total_mlu_nodes', 0)} MLUs | "
            f"{mlm.get('subcore_hubs', 0)} Sub-Cores | "
            f"{mlm.get('total_360_spoke_edges', 0)} 360° spoke edges | "
            f"mitosis={mlm.get('mitosis_promotions', 0)}"
        )
        print(
            f"Growth Brakes:     {'OPEN (Allowed)' if brakes['allowed'] else 'ENGAGED (Blocked)'} "
            f"[room={brakes['room']}, used={brakes['used']}, earning={brakes['earning']}, "
            f"fits={brakes['fits']}, honest={brakes['honest']}]"
        )
        print(
            f"Held-Out Canary:   {'PASS' if canary['passed'] else 'FAIL'} "
            f"({canary['checks_run']} checks, {canary['violations']} violations)\n"
        )
    finally:
        kernel.close()
        store.close()


def cmd_mlu(args: argparse.Namespace) -> None:
    action = getattr(args, "mlu_action", "status")
    if action == "benchmark":
        from little.mlu.spiderweb_net import SpiderWebMLM

        bench = SpiderWebMLM.run_benchmark_suite()
        print(json.dumps(bench, indent=2))
        return
    if action == "vs-llm":
        from little.mlu.spiderweb_net import SpiderWebMLM

        scorecard = SpiderWebMLM.run_vs_llm_scorecard()
        print(json.dumps(scorecard, indent=2))
        return

    store, kernel = get_kernel(Path(args.db), seed_ontology=False)
    try:
        if action == "train":
            text = " ".join(args.text)
            rep = kernel.mlm.train_on_text(text)
            print(
                f"🕸️  [360° Spider-Web MLM Trained] {rep.sentences_processed} sentences | "
                f"{rep.relations_verified} verified spoke edges, {rep.relations_refuted} refuted | "
                f"surprise={rep.average_surprise:.3f} | plasticity={rep.plasticity_multiplier:.3f} | "
                f"MLUs={rep.total_mlu_nodes} ({rep.active_subcores} Sub-Cores)"
            )
            if rep.mitosis_promotions:
                print(f"    ↳ Fractal Mitosis Promotions -> Sub-Core Hubs: {', '.join(rep.mitosis_promotions)}")
        elif action == "ask":
            q = " ".join(args.question)
            res = kernel.mlm.query_360(q)
            print(f"[{res.status}] {res.answer}")
            print(f"  ↳ {res.inspectable_trace}")
        elif action == "ingest":
            rep_dict = kernel.mlm.ingest_corpus_path(
                args.path, epochs=int(getattr(args, "epochs", 2))
            )
            print(
                f"📚 [360° Spider-Web Corpus Ingestion] {rep_dict['files_processed']} file(s) x {rep_dict['epochs']} epoch(s) | "
                f"{rep_dict['sentences_processed']} sentences | {rep_dict['relations_verified']} verified spoke links | "
                f"epoch_surprises={rep_dict['epoch_surprises']}"
            )
            if rep_dict["mitosis_promotions"]:
                print(
                    f"    ↳ Promoted 360° Sub-Core Hubs: {', '.join(rep_dict['mitosis_promotions'])}"
                )
        elif action == "bootstrap":
            rep_dict = kernel.mlm.bootstrap_curriculum(
                epochs=int(getattr(args, "epochs", 2))
            )
            print(
                f"🌍 [360° Spider-Web Curriculum Bootstrap] {rep_dict['files_processed']} file(s) x {rep_dict['epochs']} epoch(s) | "
                f"{rep_dict['relations_verified']} verified spoke links | "
                f"MLUs={rep_dict['telemetry']['total_mlu_nodes']}"
            )
        elif action == "consolidate":
            cons_dict = kernel.mlm.consolidate()
            rc = cons_dict["rust_consolidation"]
            print(
                f"🌙 [360° Spider-Web Sleep-Cycle Consolidation] "
                f"replayed_edges={rc.get('replayed_edges', 0)} | "
                f"pruned_transitions={rc.get('pruned_transitions', 0)} | "
                f"replay_surprise={rc.get('average_replay_surprise', 0.0):.4f} | "
                f"rust_nodes={cons_dict['rust_arena_nodes']}"
            )
        elif action == "compare":
            cmp_dict = kernel.mlm.compare_360(args.concept_a, args.concept_b)
            print(json.dumps(cmp_dict, indent=2))
        elif action == "curiosity":
            cur_dict = kernel.mlm.discover_curiosity()
            print(json.dumps(cur_dict, indent=2))
        elif action == "causal":
            causal_dict = kernel.mlm.simulate_causal_360(
                args.concept, max_hops=int(getattr(args, "hops", 3))
            )
            print(json.dumps(causal_dict, indent=2))
        elif action == "self-study":
            step_dict = kernel.mlm.autonomous_curiosity_step()
            print(json.dumps(step_dict, indent=2))
        elif action == "counterfactual":
            cf_dict = kernel.mlm.intervene_counterfactual_360(
                args.root_concept,
                args.suppressed_concept,
                max_hops=int(getattr(args, "hops", 3)),
            )
            print(json.dumps(cf_dict, indent=2))
        elif action == "bridge":
            br_dict = kernel.mlm.find_bridge_360(
                args.concept_a,
                args.concept_b,
                max_hops=int(getattr(args, "hops", 4)),
            )
            print(json.dumps(br_dict, indent=2))
        elif action == "graph":
            gr_dict = kernel.mlm.export_topology_graph()
            print(json.dumps(gr_dict, indent=2))
        elif action == "clusters":
            cl_list = kernel.mlm.discover_clusters_360()
            print(json.dumps(cl_list, indent=2))
        elif action == "revise":
            rev_dict = kernel.mlm.revise_belief_360(
                args.subject,
                args.predicate,
                args.object_node,
                verified=not bool(getattr(args, "refute", False)),
                evidence_weight=float(getattr(args, "weight", 3.0)),
            )
            print(json.dumps(rev_dict, indent=2))
        elif action == "deliberate":
            delib_dict = kernel.mlm.deliberate_360(
                args.concept, max_hops=int(getattr(args, "hops", 3))
            )
            print(json.dumps(delib_dict, indent=2))
        elif action == "plan":
            plan_dict = kernel.mlm.plan_goal_360(
                args.goal_concept,
                start_concept=str(getattr(args, "start", "") or ""),
                max_steps=int(getattr(args, "steps", 5)),
            )
            print(json.dumps(plan_dict, indent=2))
        elif action == "order":
            order_dict = kernel.mlm.evaluate_order_360(
                args.predicate,
                start_concept=str(getattr(args, "start", "") or ""),
                max_hops=int(getattr(args, "hops", 6)),
            )
            print(json.dumps(order_dict, indent=2))
        elif action == "summary":
            sum_dict = kernel.mlm.summarize_domain_360(
                domain_concept=str(getattr(args, "domain", "") or ""),
                max_items=int(getattr(args, "top", 8)),
            )
            print(json.dumps(sum_dict, indent=2))
        elif action == "diagnose":
            diag_dict = kernel.mlm.diagnose_abductive_360(
                args.observations,
                max_hops=int(getattr(args, "hops", 3)),
                top_k=int(getattr(args, "top", 5)),
            )
            print(json.dumps(diag_dict, indent=2))
        elif action == "essay":
            essay_dict = kernel.mlm.compose_discourse_360(
                args.concept,
                max_hops=int(getattr(args, "hops", 3)),
            )
            print(json.dumps(essay_dict, indent=2))
        elif action == "chat":
            print("🕸️  [360° Spider-Web MLU Interactive Chat] Type 'exit' or 'quit' to stop.\n")
            while True:
                try:
                    user_in = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not user_in or user_in.lower() in ("exit", "quit"):
                    break
                turn = kernel.mlm.chat_turn_360(user_in, persist=True)
                print(f"MLU [{turn['mode']}]: {turn['reply']}")
                print(f"  ↳ {turn['trace']}\n")
        else:
            tel = kernel.mlm.get_telemetry()
            print(json.dumps(tel, indent=2))
    finally:
        kernel.close()
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

    # memory list / prune
    p_mem = subparsers.add_parser(
        "memory", parents=[db_parent], help="Memory management commands"
    )
    p_mem_sub = p_mem.add_subparsers(dest="memory_cmd")
    p_mem_list = p_mem_sub.add_parser(
        "list", parents=[db_parent], help="List stored concepts and relations"
    )
    p_mem_list.set_defaults(func=cmd_memory_list)
    p_mem_prune = p_mem_sub.add_parser(
        "prune", parents=[db_parent], help="Prune polluted test/benchmark concepts"
    )
    p_mem_prune.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview polluted concepts without deleting them",
    )
    p_mem_prune.set_defaults(func=cmd_memory_prune)

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
    p_chat.add_argument(
        "--auto-study",
        action="store_true",
        help="Run background self-study after learning new facts",
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
    p_interact.add_argument(
        "--auto-study",
        action="store_true",
        help="Run background self-study after learning new facts",
    )
    p_interact.set_defaults(func=cmd_interact)

    # curious
    p_curious = subparsers.add_parser(
        "curious",
        parents=[db_parent],
        help="Analyze 360° concept knot spoke entropy across memory",
    )
    p_curious.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of highest-entropy concepts to display",
    )
    p_curious.add_argument(
        "--concept",
        default=None,
        help="Filter by concept name",
    )
    p_curious.set_defaults(func=cmd_curious)

    # study
    p_study = subparsers.add_parser(
        "study",
        parents=[db_parent],
        help="Run autonomous closed-loop curiosity self-study cycle",
    )
    p_study.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Maximum curiosity study steps to execute",
    )
    p_study.add_argument(
        "--target",
        default=None,
        help="Target a specific concept for self-study",
    )
    p_study.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display verification gate trace",
    )
    p_study.set_defaults(func=cmd_study)

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

    # tool
    p_tool = subparsers.add_parser(
        "tool",
        parents=[db_parent],
        help="Inspect and execute typed tools and action schemas",
    )
    p_tool_subs = p_tool.add_subparsers(dest="action", required=True)
    p_tool_list = p_tool_subs.add_parser("list", help="List registered typed tools")
    p_tool_list.set_defaults(func=cmd_tool)
    p_tool_exec = p_tool_subs.add_parser("exec", help="Execute a typed tool")
    p_tool_exec.add_argument("name", help="Name of the tool to execute")
    p_tool_exec.add_argument("--args", default="{}", help="JSON dictionary of tool arguments")
    p_tool_exec.set_defaults(func=cmd_tool)

    # status
    p_status = subparsers.add_parser(
        "status",
        parents=[db_parent],
        help="Display cognitive telemetry dashboard (paging, plasticity, growth brakes, canary)",
    )
    p_status.add_argument(
        "--json",
        action="store_true",
        help="Output telemetry snapshot as JSON",
    )
    p_status.set_defaults(func=cmd_status)

    # mlu (360° Spider-Web Mini Learning Model)
    p_mlu = subparsers.add_parser(
        "mlu",
        parents=[db_parent],
        help="Train, query, and inspect the 360° Spider-Web Mini Learning Model (MLM/MLU)",
    )
    p_mlu_subs = p_mlu.add_subparsers(dest="mlu_action", required=True)
    p_mlu_train = p_mlu_subs.add_parser(
        "train", parents=[db_parent], help="Train 360° Spider-Web MLM on raw text"
    )
    p_mlu_train.add_argument("text", nargs="+", help="Text stream to learn")
    p_mlu_train.set_defaults(func=cmd_mlu)
    p_mlu_ask = p_mlu_subs.add_parser(
        "ask", parents=[db_parent], help="Query 360° Spider-Web MLM"
    )
    p_mlu_ask.add_argument("question", nargs="+", help="Question or concept to query")
    p_mlu_ask.set_defaults(func=cmd_mlu)
    p_mlu_status = p_mlu_subs.add_parser(
        "status", parents=[db_parent], help="Display 360° Spider-Web MLM topology JSON"
    )
    p_mlu_status.set_defaults(func=cmd_mlu)
    p_mlu_ingest = p_mlu_subs.add_parser(
        "ingest",
        parents=[db_parent],
        help="Stream-train 360° Spider-Web MLM on a file or directory of Markdown/text files",
    )
    p_mlu_ingest.add_argument("path", help="File or directory path to ingest")
    p_mlu_ingest.add_argument(
        "--epochs", type=int, default=2, help="Number of training epochs (default: 2)"
    )
    p_mlu_ingest.set_defaults(func=cmd_mlu)
    p_mlu_consolidate = p_mlu_subs.add_parser(
        "consolidate",
        parents=[db_parent],
        help="Run Sleep-Cycle Consolidation (spoke replay + transition pruning) on the 360° Arena",
    )
    p_mlu_consolidate.set_defaults(func=cmd_mlu)
    p_mlu_bootstrap = p_mlu_subs.add_parser(
        "bootstrap",
        parents=[db_parent],
        help="Pre-train 360° Spider-Web MLM on the foundational world curriculum",
    )
    p_mlu_bootstrap.add_argument(
        "--epochs", type=int, default=2, help="Number of training epochs (default: 2)"
    )
    p_mlu_bootstrap.set_defaults(func=cmd_mlu)
    p_mlu_benchmark = p_mlu_subs.add_parser(
        "benchmark",
        parents=[db_parent],
        help="Run the isolated 5-axis 360° Spider-Web MLM & Rust Arena benchmark suite",
    )
    p_mlu_benchmark.set_defaults(func=cmd_mlu)
    p_mlu_compare = p_mlu_subs.add_parser(
        "compare",
        parents=[db_parent],
        help="Run native Rust 360° bidirectional wave intersection between two concepts",
    )
    p_mlu_compare.add_argument("concept_a", help="First concept")
    p_mlu_compare.add_argument("concept_b", help="Second concept")
    p_mlu_compare.set_defaults(func=cmd_mlu)
    p_mlu_curiosity = p_mlu_subs.add_parser(
        "curiosity",
        parents=[db_parent],
        help="Discover invariant-safe analogical hypotheses and epistemic curiosity gaps",
    )
    p_mlu_curiosity.set_defaults(func=cmd_mlu)
    p_mlu_causal = p_mlu_subs.add_parser(
        "causal",
        parents=[db_parent],
        help="Run native Rust 360° causal & counterfactual wave simulation from a root concept",
    )
    p_mlu_causal.add_argument("concept", help="Root concept to propagate causal wave from")
    p_mlu_causal.add_argument(
        "--hops", type=int, default=3, help="Maximum causal wave hops (default: 3)"
    )
    p_mlu_causal.set_defaults(func=cmd_mlu)
    p_mlu_self_study = p_mlu_subs.add_parser(
        "self-study",
        parents=[db_parent],
        help="Run active curiosity-driven autonomous hypothesis verification & promotion",
    )
    p_mlu_self_study.set_defaults(func=cmd_mlu)
    p_mlu_cf = p_mlu_subs.add_parser(
        "counterfactual",
        parents=[db_parent],
        help="Run native Rust 360° counterfactual intervention wave",
    )
    p_mlu_cf.add_argument("root_concept", help="Root concept to propagate wave from")
    p_mlu_cf.add_argument("suppressed_concept", help="Concept to suppress in virtual wave")
    p_mlu_cf.add_argument(
        "--hops", type=int, default=3, help="Maximum wave hops (default: 3)"
    )
    p_mlu_cf.set_defaults(func=cmd_mlu)
    p_mlu_bridge = p_mlu_subs.add_parser(
        "bridge",
        parents=[db_parent],
        help="Find the shortest 360° relational path bridge between two concepts",
    )
    p_mlu_bridge.add_argument("concept_a", help="Starting concept")
    p_mlu_bridge.add_argument("concept_b", help="Target concept")
    p_mlu_bridge.add_argument(
        "--hops", type=int, default=4, help="Maximum bridge hops (default: 4)"
    )
    p_mlu_bridge.set_defaults(func=cmd_mlu)
    p_mlu_graph = p_mlu_subs.add_parser(
        "graph",
        parents=[db_parent],
        help="Export live 360° Spider-Web Arena topology and Mermaid diagram",
    )
    p_mlu_graph.set_defaults(func=cmd_mlu)
    p_mlu_clusters = p_mlu_subs.add_parser(
        "clusters",
        parents=[db_parent],
        help="Discover emergent 64-D concept clusters sharing spoke attractors",
    )
    p_mlu_clusters.set_defaults(func=cmd_mlu)
    p_mlu_revise = p_mlu_subs.add_parser(
        "revise",
        parents=[db_parent],
        help="Apply explicit non-monotonic belief revision to a spoke synapse",
    )
    p_mlu_revise.add_argument("subject", help="Subject concept")
    p_mlu_revise.add_argument("predicate", help="Relational predicate")
    p_mlu_revise.add_argument("object_node", help="Target concept")
    p_mlu_revise.add_argument(
        "--refute", action="store_true", help="Apply refuting (negative) evidence"
    )
    p_mlu_revise.add_argument(
        "--weight", type=float, default=3.0, help="Evidence weight (default: 3.0)"
    )
    p_mlu_revise.set_defaults(func=cmd_mlu)
    p_mlu_delib = p_mlu_subs.add_parser(
        "deliberate",
        parents=[db_parent],
        help="Run native Rust 5-wave System-2 Deliberative Thinking over a concept",
    )
    p_mlu_delib.add_argument("concept", help="Concept to deliberate on")
    p_mlu_delib.add_argument(
        "--hops", type=int, default=3, help="Maximum deliberation hops (default: 3)"
    )
    p_mlu_delib.set_defaults(func=cmd_mlu)
    p_mlu_plan = p_mlu_subs.add_parser(
        "plan",
        parents=[db_parent],
        help="Run native Rust 360° Goal-Directed Backward-Chaining & Action Planner",
    )
    p_mlu_plan.add_argument("goal_concept", help="Target goal concept to achieve")
    p_mlu_plan.add_argument(
        "--start", default="", help="Optional starting concept/resource"
    )
    p_mlu_plan.add_argument(
        "--steps", type=int, default=5, help="Maximum planning steps (default: 5)"
    )
    p_mlu_plan.set_defaults(func=cmd_mlu)
    p_mlu_order = p_mlu_subs.add_parser(
        "order",
        parents=[db_parent],
        help="Run native Rust 360° Transitive Poset & Quantitative Order Wave",
    )
    p_mlu_order.add_argument("predicate", help="Relational predicate to order along")
    p_mlu_order.add_argument(
        "--start", default="", help="Optional starting concept"
    )
    p_mlu_order.add_argument(
        "--hops", type=int, default=6, help="Maximum chain hops (default: 6)"
    )
    p_mlu_order.set_defaults(func=cmd_mlu)
    p_mlu_summary = p_mlu_subs.add_parser(
        "summary",
        parents=[db_parent],
        help="Run native Rust 360° Domain & Sub-Core Knowledge Summarizer",
    )
    p_mlu_summary.add_argument(
        "domain", nargs="?", default="", help="Optional sub-core or concept domain"
    )
    p_mlu_summary.add_argument(
        "--top", type=int, default=8, help="Maximum items per section (default: 8)"
    )
    p_mlu_summary.set_defaults(func=cmd_mlu)
    p_mlu_diagnose = p_mlu_subs.add_parser(
        "diagnose",
        parents=[db_parent],
        help="Run native Rust 360° Abductive Diagnostic & Root-Cause Engine",
    )
    p_mlu_diagnose.add_argument(
        "observations", nargs="+", help="Observed symptoms, properties, or effects"
    )
    p_mlu_diagnose.add_argument(
        "--hops", type=int, default=3, help="Maximum reverse abductive hops (default: 3)"
    )
    p_mlu_diagnose.add_argument(
        "--top", type=int, default=5, help="Maximum candidate explanations (default: 5)"
    )
    p_mlu_diagnose.set_defaults(func=cmd_mlu)
    p_mlu_essay = p_mlu_subs.add_parser(
        "essay",
        parents=[db_parent],
        help="Run native Rust 360° Multi-Paragraph Discourse & Essay Composer",
    )
    p_mlu_essay.add_argument("concept", help="Topic or concept to compose discourse on")
    p_mlu_essay.add_argument(
        "--hops", type=int, default=3, help="Maximum 360° wave hops (default: 3)"
    )
    p_mlu_essay.set_defaults(func=cmd_mlu)
    p_mlu_vs_llm = p_mlu_subs.add_parser(
        "vs-llm",
        parents=[db_parent],
        help="Run the head-to-head 360° Spider-Web MLU vs. Standard LLM / Transformer real-world scorecard",
    )
    p_mlu_vs_llm.set_defaults(func=cmd_mlu)
    p_mlu_chat = p_mlu_subs.add_parser(
        "chat",
        parents=[db_parent],
        help="Interactive 360° Spider-Web MLU conversational REPL (live learning + deliberation + inquiry)",
    )
    p_mlu_chat.set_defaults(func=cmd_mlu)

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
