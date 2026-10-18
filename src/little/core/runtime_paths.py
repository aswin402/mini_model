"""Versioned runtime resource paths loaded from project data."""

from __future__ import annotations

import importlib.resources
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimePaths:
    """Locations for mutable runtime state and versioned project resources."""

    database: Path
    construction_pack: Path
    knowledge_pack: Path
    ontology_pack: Path
    concept_fixture_directory: Path

    @property
    def schema_directory(self) -> Path:
        """Return the directory containing the versioned runtime schema."""
        return self.construction_pack.parent

    @property
    def world_corpus_directory(self) -> Path:
        """Immutable corpus belongs beside the knowledge pack, independent of state."""
        return self.knowledge_pack.parent / "world_corpus"

    @classmethod
    def load(cls, directory: str | Path) -> RuntimePaths:
        directory = Path(directory)
        path = directory / "runtime_paths.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.runtime_paths.v1":
            raise ValueError(f"{path} has an unsupported runtime-path format")

        values = payload.get("runtime_paths")
        if not isinstance(values, dict):
            raise TypeError(f"{path} must contain a 'runtime_paths' object")

        required = (
            "database",
            "construction_pack",
            "knowledge_pack",
            "ontology_pack",
            "concept_fixture_directory",
        )
        missing = [key for key in required if key not in values]
        if missing:
            raise ValueError(f"{path} is missing runtime paths: {missing}")

        def resource(key: str) -> Path:
            import os
            value = values[key]
            if not isinstance(value, str) or not value.strip():
                raise TypeError(f"{path} runtime path {key!r} must be a string")
            return Path(os.path.normpath(directory / value))

        database = values["database"]
        if not isinstance(database, str) or not database.strip():
            raise TypeError(f"{path} runtime path 'database' must be a string")

        return cls(
            database=Path(database),
            construction_pack=resource("construction_pack"),
            knowledge_pack=resource("knowledge_pack"),
            ontology_pack=resource("ontology_pack"),
            concept_fixture_directory=resource("concept_fixture_directory"),
        )

    @classmethod
    def default(cls) -> RuntimePaths:
        module = Path(__file__).resolve()
        root = module.parents[3]
        source_package = root / "src" / "little"
        if (root / "pyproject.toml").is_file() and module.parent.parent == source_package and (root / "data" / "schemas" / "runtime_paths.json").is_file():
            return cls.load(root / "data" / "schemas")
        package = importlib.resources.files("little")
        schema = package.joinpath("_resources", "schemas")
        # Standard wheel installers unpack package resources into actual directories.
        # Path-based consumers intentionally require this installed layout.
        if not isinstance(schema, Path):
            raise RuntimeError("LITTLE resources require an unpacked wheel installation")
        return cls.load(schema)
