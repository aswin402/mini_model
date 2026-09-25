from argparse import Namespace
from pathlib import Path

from little.main import cmd_ask, cmd_interact, cmd_learn, get_kernel
from little.memory.store import MemoryStore


def test_get_kernel_uses_the_requested_database(tmp_path: Path):
    db_path = tmp_path / "kernel.db"
    store, kernel = get_kernel(db_path, seed_ontology=False)
    try:
        result = kernel.process("A falcon is a bird.")
        assert result.learning is not None
        assert store.find_relation_by_names("falcon", "is_a", "bird") is not None
        assert db_path.exists()
    finally:
        store.close()


def test_kernel_command_path_does_not_write_for_questions(tmp_path: Path):
    db_path = tmp_path / "kernel.db"
    store, kernel = get_kernel(db_path, seed_ontology=False)
    try:
        kernel.process("A falcon is a bird.")
        before = store.count_relations()
        outcome = kernel.process("Is a falcon a bird?")
        assert outcome.inference is not None
        assert outcome.inference.answer is True
        assert store.count_relations() == before
    finally:
        store.close()


def test_cmd_learn_question_is_a_no_op_without_an_extra_experience(
    tmp_path: Path, capsys
):
    db_path = tmp_path / "learn-question.db"
    with MemoryStore(db_path, seed_ontology=True) as store:
        before_relations = store.count_relations()
        before_experiences = store._conn.execute(
            "SELECT COUNT(*) FROM experiences"
        ).fetchone()[0]

    cmd_learn(
        Namespace(db=str(db_path), statement=["Is", "a", "falcon", "a", "bird?"])
    )

    output = capsys.readouterr().out
    assert "[Learning Event: NO_OP]" in output
    assert "recognized as an inquiry or calculation" in output
    with MemoryStore(db_path, seed_ontology=False) as store:
        assert store.count_relations() == before_relations
        assert (
            store._conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
            == before_experiences
        )


def test_cmd_ask_non_question_is_explicit_unknown_without_writing(
    tmp_path: Path, capsys
):
    db_path = tmp_path / "ask-statement.db"
    with MemoryStore(db_path, seed_ontology=True) as store:
        before_relations = store.count_relations()
        before_experiences = store._conn.execute(
            "SELECT COUNT(*) FROM experiences"
        ).fetchone()[0]

    cmd_ask(
        Namespace(
            db=str(db_path),
            question=["A", "zorp", "is", "a", "flibbertigibbet."],
            thinking=False,
            verbose=False,
        )
    )

    output = capsys.readouterr().out
    assert "[Inference Result: UNKNOWN]" in output
    with MemoryStore(db_path, seed_ontology=False) as store:
        assert store.count_relations() == before_relations
        assert (
            store._conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
            == before_experiences
        )
        assert store.find_relation_by_names("zorp", "is_a", "flibbertigibbet") is None


def test_repl_routes_clarification_candidate_back_through_kernel(
    tmp_path: Path, capsys, monkeypatch
):
    db_path = tmp_path / "clarification.db"
    responses = iter(["Is a pear a fruit?", "Yes", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(responses))

    cmd_interact(Namespace(db=str(db_path), thinking=False))

    output = capsys.readouterr().out
    assert "Thank you!" in output
    with MemoryStore(db_path, seed_ontology=False) as store:
        assert store.find_relation_by_names("pear", "is_a", "fruit") is not None


def test_repl_does_not_reparse_normal_question_after_kernel_perception(
    tmp_path: Path, capsys, monkeypatch
):
    db_path = tmp_path / "repl-one-pass.db"
    responses = iter(["Is a pear a fruit?", "skip", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(responses))

    def fail_if_reparsed(*_args, **_kwargs):
        raise AssertionError("REPL reparsed a question after kernel perception")

    monkeypatch.setattr(
        "little.language.parser.SimpleParser.parse_question", fail_if_reparsed
    )

    cmd_interact(Namespace(db=str(db_path), thinking=False))

    capsys.readouterr()
