"""One injected bundle of versioned policies for a LITTLE runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from little.core.runtime_paths import RuntimePaths

from little.core.grammar_registry import GrammarRegistry
from little.core.models import Construction
from little.dynamics.profiles import DynamicsProfileRegistry
from little.dynamics.transformation_policy import TransformationPolicy
from little.inference.reasoning_policy import ReasoningPolicy
from little.knowledge.policy import LanguagePolicy
from little.language.parser_policy import ParserPolicy
from little.language.perception import PerceptionPolicy
from little.procedural.skill_policy import ProceduralSkillPolicy
from little.procedural.unit_conversion_policy import UnitConversionPolicy


@dataclass(frozen=True)
class RuntimePolicy:
    """Configuration boundary shared by the cognitive kernel's subsystems."""

    parser: ParserPolicy
    constructions: tuple[Construction, ...]
    language: LanguagePolicy
    perception: PerceptionPolicy
    reasoning: ReasoningPolicy
    units: UnitConversionPolicy
    dynamics: DynamicsProfileRegistry
    procedural: ProceduralSkillPolicy
    transformation: TransformationPolicy

    @property
    def semantic(self):
        """Return the semantic predicate policy used by the parser bundle."""
        return self.parser.semantic

    @classmethod
    def load(cls, directory: Path) -> RuntimePolicy:
        directory = Path(directory)
        parser = ParserPolicy.load(directory)
        return cls(
            parser=parser,
            constructions=tuple(GrammarRegistry.load(directory / "constructions.json")),
            language=LanguagePolicy.load(directory),
            perception=PerceptionPolicy.load(directory),
            reasoning=ReasoningPolicy.load(directory),
            units=UnitConversionPolicy.load(directory),
            dynamics=DynamicsProfileRegistry.load(directory),
            procedural=ProceduralSkillPolicy.load(directory),
            transformation=TransformationPolicy.load(directory),
        )

    @classmethod
    def default(cls) -> RuntimePolicy:
        return cls.load(RuntimePaths.default().schema_directory)
