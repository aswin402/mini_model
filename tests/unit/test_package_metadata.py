import tomllib
from pathlib import Path

import little


def test_package_version_matches_project_metadata():
    project = tomllib.loads(
        Path("pyproject.toml").read_text(encoding="utf-8")
    )

    assert little.__version__ == project["project"]["version"]
