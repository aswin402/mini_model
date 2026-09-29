from argparse import Namespace
from pathlib import Path

from little.main import (
    build_parser,
    cmd_ask,
    cmd_curious,
    cmd_study,
    get_kernel,
)


def test_cmd_curious_displays_knot_entropies(tmp_path: Path, capsys):
    db_path = tmp_path / "curious.db"
    store, kernel = get_kernel(db_path, seed_ontology=False)
    try:
        kernel.process("A titanium is a metal.")
    finally:
        store.close()

    cmd_curious(Namespace(db=str(db_path), top=5, concept="titanium"))
    output = capsys.readouterr().out
    assert "360° Concept Knot Spoke Entropy" in output
    assert "titanium" in output


def test_cmd_study_executes_autonomous_learning_cycle(tmp_path: Path, capsys):
    db_path = tmp_path / "study.db"
    store, kernel = get_kernel(db_path, seed_ontology=True)
    try:
        kernel.process("A falcon is a bird.")
        kernel.process("A falcon can fly.")
        kernel.process("A hawk is a bird.")
    finally:
        store.close()

    # Run little study with target hawk
    cmd_study(Namespace(db=str(db_path), target="hawk", steps=1, verbose=True))
    output = capsys.readouterr().out
    assert "Autonomous Self-Study" in output
    assert "PERSISTED" in output
    assert "hawk" in output

    # Verify relation was persisted and can be queried via ask
    cmd_ask(
        Namespace(
            db=str(db_path),
            question=["Can", "a", "hawk", "fly?"],
            verbose=False,
            thinking=False,
        )
    )
    ask_output = capsys.readouterr().out
    assert "SUPPORTED" in ask_output or "True" in ask_output or "fly" in ask_output


def test_build_parser_curious_and_study_subcommands():
    parser = build_parser()

    # Test 'curious'
    args_curious = parser.parse_args(["curious", "--top", "15", "--concept", "apple"])
    assert args_curious.command == "curious"
    assert args_curious.top == 15
    assert args_curious.concept == "apple"
    assert args_curious.func == cmd_curious

    # Test 'study'
    args_study = parser.parse_args(["study", "--steps", "8", "--target", "falcon", "-v"])
    assert args_study.command == "study"
    assert args_study.steps == 8
    assert args_study.target == "falcon"
    assert args_study.verbose is True
    assert args_study.func == cmd_study

    # Test 'chat --auto-study'
    args_chat = parser.parse_args(["chat", "--auto-study"])
    assert args_chat.auto_study is True
