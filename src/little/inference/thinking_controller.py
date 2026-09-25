"""Cognitive routing boundary for fast lookup, deliberation, asking, and study.

The controller does not decide whether a claim is true.  It selects the least
expensive valid operation, then delegates graph truth evaluation to the
deterministic inference engine or to an injected query resolver for richer
handlers such as mathematics and temporal dynamics.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from little.core.contracts import ParsedQuery
from little.core.models import BeliefStatus, InferenceResult
from little.core.runtime_policy import RuntimePolicy
from little.inference.dual_speed import DualSpeedInfillingEngine, InfillingResult
from little.inference.engine import InferenceEngine
from little.inference.invariant_gates import DeepSeekInvariantVerifier
from little.knowledge.policy import LanguagePolicy
from little.language.parser import configured_parser
from little.memory.store import MemoryStore
from little.procedural.math_cas import UnitConversionGraph


class CognitiveMode(str, Enum):
    """The operation selected for a user input."""

    FAST = "FAST"
    THINK = "THINK"
    ASK = "ASK"
    STUDY = "STUDY"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class QueryPlan:
    """A normalized query and the controller's routing decision."""

    question: str
    mode: CognitiveMode
    subject: str | None = None
    predicate: str | None = None
    target: Any = None
    reason: str = ""


@dataclass(frozen=True)
class ThinkingResult:
    """A routed inference result with its inspectable query plan."""

    plan: QueryPlan
    inference: InferenceResult

    @property
    def mode(self) -> CognitiveMode:
        return self.plan.mode

    @property
    def reason(self) -> str:
        return self.plan.reason


Resolver = Callable[..., InferenceResult]


class _UnspecifiedParsedQuery:
    """Distinguish legacy parsing from an explicit missing candidate query."""


_UNSPECIFIED_QUERY = _UnspecifiedParsedQuery()


class ThinkingController:
    """Route inputs without allowing routing to bypass epistemic verification."""

    def __init__(
        self,
        memory: MemoryStore,
        resolver: Resolver | None = None,
        policy: LanguagePolicy | None = None,
        dual_speed: DualSpeedInfillingEngine | None = None,
        runtime_policy: RuntimePolicy | None = None,
    ) -> None:
        self.memory = memory
        explicit_runtime_policy = runtime_policy is not None
        self.runtime_policy = runtime_policy or RuntimePolicy.default()
        self.parser = configured_parser(
            self.runtime_policy.parser if explicit_runtime_policy else None,
            constructions=(
                self.runtime_policy.constructions
                if explicit_runtime_policy
                else None
            ),
            unit_conversions=UnitConversionGraph(self.runtime_policy.units),
        )
        self.registry = self.memory.ledger.registry
        self.inference = InferenceEngine(
            memory,
            registry=self.registry,
            semantic_policy=self.parser.POLICY.semantic,
            policy=self.runtime_policy.reasoning,
        )
        self.resolver = resolver
        self.policy = policy or self.runtime_policy.language
        self.dual_speed = dual_speed or DualSpeedInfillingEngine(
            memory,
            DeepSeekInvariantVerifier(memory, registry=self.registry),
            policy=self.runtime_policy.reasoning,
        )

    def _is_question_like(self, text: str) -> bool:
        clean = text.strip().lower()
        return clean.endswith("?") or clean.startswith(self.policy.question_prefixes)

    def _parse_question(self, text: str) -> tuple[str, str, Any] | None:
        catalog = list(self.parser.CONSTRUCTIONS) + self.memory.list_constructions()
        parsed = self.parser.CONSTRUCTION_ENGINE.parse_question_from_catalog_with_procedural(
            text, catalog
        )
        if parsed:
            return parsed
        known = {concept.name.lower() for concept in self.memory.list_concepts()}
        return self.parser.parse_question(text, known_concepts=known)

    def plan(
        self,
        question: str,
        parsed_query: ParsedQuery | None | _UnspecifiedParsedQuery = _UNSPECIFIED_QUERY,
        parsed_queries: list[ParsedQuery | None] | None = None,
        question_parts: list[str] | None = None,
    ) -> QueryPlan:
        """Normalize input and select a mode without performing inference."""
        clean_question = question.strip()
        payload_supplied = parsed_queries is not None or question_parts is not None
        parse_input = parsed_query is _UNSPECIFIED_QUERY and not payload_supplied
        if parsed_query is not _UNSPECIFIED_QUERY and parsed_query is not None:
            parsed = parsed_query
        elif parsed_queries:
            parsed = next((candidate for candidate in parsed_queries if candidate is not None), None)
        else:
            parsed = self._parse_question(clean_question) if parse_input else None

        if parsed is None and parse_input and not self._is_question_like(clean_question):
            statements = self.parser.parse_statement(clean_question)
            if statements:
                statement = statements[0]
                return QueryPlan(
                    question=clean_question,
                    mode=CognitiveMode.STUDY,
                    subject=statement.subject,
                    predicate=statement.predicate,
                    target=statement.object_,
                    reason="Declarative input is a learning candidate, not a question.",
                )

        if parsed is None:
            return QueryPlan(
                question=clean_question,
                mode=CognitiveMode.ASK,
                reason="Question pattern was not recognized; clarification is required.",
            )

        subject, predicate, target = parsed
        if predicate.startswith("__"):
            return QueryPlan(
                question=clean_question,
                mode=CognitiveMode.FAST,
                subject=subject,
                predicate=predicate,
                target=target,
                reason="Structured procedural or dialogue query has a direct handler.",
            )

        if target == "?" or subject == "?":
            return QueryPlan(
                question=clean_question,
                mode=CognitiveMode.THINK,
                subject=subject,
                predicate=predicate,
                target=target,
                reason="Answer requires relation enumeration or reverse graph search.",
            )

        direct = self.memory.find_relation_by_names(subject, predicate, target)
        if direct is not None:
            return QueryPlan(
                question=clean_question,
                mode=CognitiveMode.FAST,
                subject=subject,
                predicate=predicate,
                target=target,
                reason="A persisted relation matches the normalized query directly.",
            )

        return QueryPlan(
            question=clean_question,
            mode=CognitiveMode.THINK,
            subject=subject,
            predicate=predicate,
            target=target,
            reason="No direct relation matched; bounded graph reasoning is required.",
        )

    def _unknown_result(self, plan: QueryPlan, detail: str) -> InferenceResult:
        return InferenceResult(
            query=plan.question,
            status=BeliefStatus.UNKNOWN,
            answer=None,
            confidence=0.0,
            evidence=[],
            trace=[detail],
        )

    def _can_use_dual_speed(self, plan: QueryPlan) -> bool:
        if (
            plan.subject is None
            or plan.predicate is None
            or not isinstance(plan.target, str)
            or plan.target == "?"
        ):
            return False
        return plan.predicate.strip().lower() in self.dual_speed.verifier.registry.predicates(
            transitive=True
        )

    def _from_infilling(
        self, plan: QueryPlan, result: InfillingResult
    ) -> InferenceResult:
        path = " -> ".join(result.path)
        return InferenceResult(
            query=plan.question,
            status=BeliefStatus.SUPPORTED,
            answer=True,
            confidence=result.confidence,
            evidence=[f"Verified proof path: {path}"],
            trace=[
                f"System 2 proof path verified: {path}",
                f"Transitive path discovered: {path} (System 2 verified)",
                *result.inspectable_trace.splitlines(),
            ],
        )

    def _with_infilling_trace(
        self, inference: InferenceResult, result: InfillingResult
    ) -> InferenceResult:
        return replace(
            inference,
            trace=[
                *inference.trace,
                "System 2 search did not produce a verified path.",
                *result.inspectable_trace.splitlines(),
            ],
        )

    def _run_resolver(
        self,
        plan: QueryPlan,
        parsed_query: ParsedQuery | None | _UnspecifiedParsedQuery,
        parsed_queries: list[ParsedQuery | None] | None,
        question_parts: list[str] | None,
    ) -> InferenceResult:
        if self.resolver is None:
            return self._unknown_result(plan, "No specialized resolver is configured.")

        resolver = self.resolver
        payload = {
            "parsed_query": (
                parsed_query
                if parsed_query is not _UNSPECIFIED_QUERY
                else (
                    (plan.subject, plan.predicate, plan.target)
                    if plan.subject is not None and plan.predicate is not None
                    else None
                )
            ),
            "parsed_queries": parsed_queries,
            "question_parts": question_parts,
        }
        try:
            parameters = inspect.signature(resolver).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_var_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        supported_payload = (
            payload
            if accepts_var_kwargs
            else {
                name: value
                for name, value in payload.items()
                if name in parameters
                and parameters[name].kind
                in {
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                }
            }
        )
        if supported_payload:
            return resolver(plan.question, **supported_payload)
        return resolver(plan.question)

    def run(
        self,
        question: str,
        parsed_query: ParsedQuery | None | _UnspecifiedParsedQuery = _UNSPECIFIED_QUERY,
        parsed_queries: list[ParsedQuery | None] | None = None,
        question_parts: list[str] | None = None,
    ) -> ThinkingResult:
        """Route and execute a query while preserving an inspectable trace."""
        plan = self.plan(
            question,
            parsed_query=parsed_query,
            parsed_queries=parsed_queries,
            question_parts=question_parts,
        )

        if plan.mode is CognitiveMode.ASK:
            inference = self._unknown_result(
                plan, "Controller requested clarification before inference."
            )
        elif plan.mode is CognitiveMode.STUDY:
            inference = self._unknown_result(
                plan, "Controller routed declarative input to the study pipeline."
            )
        elif self._can_use_dual_speed(plan):
            infilling = self.dual_speed.query(
                plan.subject, plan.target, predicate=plan.predicate
            )
            if infilling.path:
                inference = self._from_infilling(plan, infilling)
            elif self.resolver is not None:
                inference = self._with_infilling_trace(
                    self._run_resolver(
                        plan, parsed_query, parsed_queries, question_parts
                    ),
                    infilling,
                )
            else:
                inference = self._with_infilling_trace(
                    self._unknown_result(
                        plan, "System 2 found no verified path; answer is UNKNOWN."
                    ),
                    infilling,
                )
        elif self.resolver is not None:
            inference = self._run_resolver(
                plan, parsed_query, parsed_queries, question_parts
            )
        elif (
            plan.subject is None
            or plan.predicate is None
            or plan.target is None
            or not isinstance(plan.target, str)
            or plan.target == "?"
            or plan.subject == "?"
        ):
            inference = self._unknown_result(
                plan, "This query requires a specialized resolver.")
        else:
            inference = self.inference.infer(
                plan.subject, plan.predicate, plan.target
            )

        trace = [
            f"Controller route: {plan.mode.value}.",
            f"Routing reason: {plan.reason}",
        ]
        trace.extend(inference.trace)
        return ThinkingResult(plan=plan, inference=replace(inference, trace=trace))
