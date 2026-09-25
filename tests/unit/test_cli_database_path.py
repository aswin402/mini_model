from argparse import Namespace
from pathlib import Path

from little.language.parser import LearningEngine
from little.main import build_parser, cmd_ask, cmd_interact
from little.memory.store import MemoryStore


def test_global_db_path_is_not_overwritten_by_subparser_default(tmp_path: Path):
    path = tmp_path / "custom.db"
    parser = build_parser()

    args = parser.parse_args(["--db", str(path), "init"])

    assert args.db == str(path)


def test_subcommand_db_path_is_supported(tmp_path: Path):
    path = tmp_path / "custom.db"
    parser = build_parser()

    args = parser.parse_args(["init", "--db", str(path)])

    assert args.db == str(path)


def test_thinking_ask_uses_controller_route_trace(tmp_path: Path, capsys):
    db_path = tmp_path / "thinking.db"
    with MemoryStore(db_path) as memory:
        LearningEngine(memory).learn("An apple is a fruit.")

    cmd_ask(
        Namespace(
            db=str(db_path),
            question=["Is", "an", "apple", "a", "fruit?"],
            thinking=True,
            verbose=False,
        )
    )

    output = capsys.readouterr().out
    assert "Controller route: FAST." in output


def test_thinking_chat_uses_controller_route(tmp_path: Path, capsys, monkeypatch):
    db_path = tmp_path / "chat-thinking.db"
    with MemoryStore(db_path) as memory:
        LearningEngine(memory).learn("An apple is a fruit.")

    responses = iter(["Is an apple a fruit?", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(responses))

    cmd_interact(Namespace(db=str(db_path), thinking=True))

    output = capsys.readouterr().out
    assert "Route: FAST" in output
