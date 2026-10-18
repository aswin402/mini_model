"""Keep test-generated knowledge away from the user's persistent vault."""

from dataclasses import replace

import pytest

from little.knowledge.living_wiki_policy import LivingWikiPolicy
from little.knowledge.okf_policy import OKFPolicy


@pytest.fixture(autouse=True)
def isolated_default_vault(tmp_path, monkeypatch):
    wiki = LivingWikiPolicy.default()
    okf = OKFPolicy.default()
    vault = str(tmp_path / "knowledge_vault")
    monkeypatch.setattr(
        LivingWikiPolicy, "default", classmethod(lambda cls: replace(wiki, vault_directory=vault))
    )
    monkeypatch.setattr(
        OKFPolicy, "default", classmethod(lambda cls: replace(okf, vault_directory=vault))
    )
