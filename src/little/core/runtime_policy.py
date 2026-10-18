"""One injected bundle of versioned policies for a LITTLE runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from little.actions.planning_policy import AgenticPlanningPolicy
from little.actions.tool_policy import ActionPolicy, ToolDefinitionsPolicy
from little.active.self_study_policy import SelfStudyPolicy
from little.core.grammar_registry import GrammarRegistry
from little.core.models import Construction
from little.core.runtime_paths import RuntimePaths
from little.dynamics.profiles import DynamicsProfileRegistry
from little.dynamics.transformation_policy import TransformationPolicy
from little.inference.reasoning_policy import ReasoningPolicy
from little.knowledge.living_wiki_policy import LivingWikiPolicy
from little.knowledge.policy import LanguagePolicy
from little.language.dialogue_surface_policy import DialogueSurfacePolicy
from little.language.parser_policy import ParserPolicy
from little.language.perception import PerceptionPolicy
from little.memory.headroom_policy import HeadroomPolicy
from little.memory.memory_policy import MemoryPolicy
from little.neural.router_policy import NeuralRouterPolicy
from little.procedural.code_policy import CodeSynthesizerPolicy
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
    dialogue_surface: DialogueSurfacePolicy
    memory: MemoryPolicy | None = None
    headroom: HeadroomPolicy | None = None
    wiki: LivingWikiPolicy | None = None
    actions: ActionPolicy | None = None
    tools: ToolDefinitionsPolicy | None = None
    planning: AgenticPlanningPolicy | None = None
    neural: NeuralRouterPolicy | None = None
    code: CodeSynthesizerPolicy | None = None
    study: SelfStudyPolicy | None = None

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
            dialogue_surface=DialogueSurfacePolicy.load(directory),
            memory=MemoryPolicy.load(directory),
            headroom=HeadroomPolicy.load(directory),
            wiki=LivingWikiPolicy.load(directory),
            actions=ActionPolicy.load(directory),
            tools=ToolDefinitionsPolicy.load(directory),
            planning=AgenticPlanningPolicy.load(directory),
            neural=NeuralRouterPolicy.load(directory),
            code=CodeSynthesizerPolicy.from_dict(__import__("json").loads((directory / "code_synthesizer_policy.json").read_text())),
            study=SelfStudyPolicy.load(directory),
        )

    @classmethod
    def default(cls) -> RuntimePolicy:
        return cls.load(RuntimePaths.default().schema_directory)
