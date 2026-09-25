"""Policy-controlled promotion of candidate evidence into semantic memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from little.core.contracts import (
    CandidateAction,
    CandidateClaim,
    CandidateFrame,
    CommitStatus,
    CommitDecision,
    EvidenceRecord,
    EvidenceStatus,
    SourceType,
    VerificationResult,
    VerificationStatus,
)
from little.core.models import LearningResult, UpdateType, generate_id
from little.knowledge.registry import ActionSchema, SchemaRegistry

if TYPE_CHECKING:
    from little.memory.store import MemoryStore


class EvidenceLedger:
    """Keep candidate evidence separate from accepted graph mutations."""

    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory
        self.registry = SchemaRegistry.default()
        self.transformation_policy = None
        self.dynamics_registry = None
        from little.inference.invariant_gates import DeepSeekInvariantVerifier

        self.verifier = DeepSeekInvariantVerifier(memory, registry=self.registry)
        self.executors = ActionExecutorRegistry()
        self.executors.register("slice_object", _execute_slice_object)

    def propose_relation(
        self,
        frame: CandidateFrame,
        claim: CandidateClaim,
        source_type: SourceType,
        source_reference: str,
        positive: bool = True,
    ) -> EvidenceRecord:
        evidence = EvidenceRecord.from_claim(
            claim,
            source_type=source_type,
            source_reference=source_reference,
            source_text=frame.source_text,
            positive=positive,
        )
        return self.memory.evidence.append(evidence)

    def commit_relation(
        self,
        evidence: EvidenceRecord,
        verification: VerificationResult,
        positive: bool | None = None,
    ) -> CommitDecision:
        if verification.status is VerificationStatus.PASSED:
            subject = self.memory.get_or_create_concept(evidence.subject_id)
            object_ = self.memory.get_or_create_concept(evidence.object_id or "")
            self.memory.add_relation(
                subject.id,
                evidence.predicate,
                object_.id,
                # An explicit call argument wins; otherwise preserve evidence polarity.
                positive=evidence.positive if positive is None else positive,
                source_experience_id=evidence.source_reference,
            )
            self.memory.evidence.update_status(
                evidence.evidence_id, EvidenceStatus.ACCEPTED
            )
        elif verification.status is VerificationStatus.UNKNOWN:
            self.memory.evidence.update_status(
                evidence.evidence_id, EvidenceStatus.CANDIDATE
            )
        else:
            self.memory.evidence.update_status(
                evidence.evidence_id, EvidenceStatus.REJECTED
            )

        decision = verification.to_commit_decision(evidence.evidence_id)
        self.memory.evidence.save_decision(decision)
        return decision

    def commit_property(
        self,
        evidence: EvidenceRecord,
        verification: VerificationResult,
        *,
        key: str,
        value: str,
        update_policy: str = "append",
        value_format: str = "preserve",
    ) -> CommitDecision:
        if verification.status is VerificationStatus.PASSED:
            if update_policy not in ("append", "replace"):
                raise ValueError(f"Unknown attribute update policy: {update_policy}")
            if value_format not in ("preserve", "capitalize"):
                raise ValueError(f"Unknown attribute value format: {value_format}")
            formatted_value = value.capitalize() if value_format == "capitalize" else value
            concept = self.memory.get_or_create_concept(evidence.subject_id)
            attributes = dict(concept.attributes)
            if update_policy == "replace":
                attributes[key] = formatted_value
            elif key in attributes:
                current = attributes[key]
                if isinstance(current, list):
                    if formatted_value not in current:
                        attributes[key] = [*current, formatted_value]
                elif current != formatted_value:
                    attributes[key] = [current, formatted_value]
            else:
                attributes[key] = formatted_value
            self.memory.update_concept_attributes(concept.id, attributes)
            status = EvidenceStatus.ACCEPTED
        elif verification.status is VerificationStatus.UNKNOWN:
            status = EvidenceStatus.CANDIDATE
        else:
            status = EvidenceStatus.REJECTED
        self.memory.evidence.update_status(evidence.evidence_id, status)
        decision = verification.to_commit_decision(evidence.evidence_id)
        self.memory.evidence.save_decision(decision)
        return decision

    def commit_action(
        self,
        frame: CandidateFrame,
        action: CandidateAction,
        *,
        source_type: SourceType = SourceType.USER,
    ) -> LearningResult:
        evidence = self.memory.evidence.append(
            EvidenceRecord(
                evidence_id=generate_id("evidence"),
                subject_id=action.arguments.get("object", "").strip().lower()
                or action.name.strip().lower(),
                predicate=f"action:{action.name.strip().lower()}",
                object_id=None,
                source_type=source_type,
                source_reference=f"frame:{frame.frame_id}",
                source_text=frame.source_text,
                extraction_confidence=action.probability,
            )
        )
        schema = self.registry.action(action.name)
        executor = self.executors.get(schema.executor) if schema else None
        arguments = (
            {key: str(value) for key, value in schema.defaults.items()}
            if schema
            else {}
        )
        arguments.update({key: str(value) for key, value in action.arguments.items()})
        if schema is None or executor is None:
            verification = VerificationResult(
                VerificationStatus.UNKNOWN,
                reasons=[f"Unregistered action or executor: {action.name}"],
            )
        else:
            errors = []
            for key, expected_type in schema.argument_types.items():
                value = str(arguments.get(key, "")).strip()
                if expected_type == "entity":
                    if not value:
                        errors.append(f"Missing entity argument: {key}")
                elif expected_type == "quantity":
                    try:
                        if int(value) < 1:
                            errors.append(f"Invalid quantity argument: {key}")
                    except ValueError:
                        errors.append(f"Invalid quantity argument: {key}")
                else:
                    errors.append(f"Unsupported argument type: {expected_type}")
            verification = VerificationResult(
                VerificationStatus.FAILED if errors else VerificationStatus.PASSED,
                reasons=errors,
            )

        status = {
            VerificationStatus.PASSED: EvidenceStatus.ACCEPTED,
            VerificationStatus.FAILED: EvidenceStatus.REJECTED,
            VerificationStatus.UNKNOWN: EvidenceStatus.CANDIDATE,
        }[verification.status]
        self.memory.evidence.update_status(evidence.evidence_id, status)
        decision = verification.to_commit_decision(evidence.evidence_id)
        self.memory.evidence.save_decision(decision)

        if verification.status is VerificationStatus.PASSED:
            try:
                execution = _coerce_execution_result(
                    executor(ActionExecutionContext(self.memory), frame, arguments)
                )
            except ActionExecutionViolation as exc:
                failure = VerificationResult(
                    VerificationStatus.FAILED,
                    reasons=[f"Action executor violated the semantic write boundary: {exc}"],
                )
                self.memory.evidence.update_status(
                    evidence.evidence_id, EvidenceStatus.REJECTED
                )
                self.memory.evidence.replace_decision(
                    failure.to_commit_decision(evidence.evidence_id)
                )
                exp = self.memory.add_experience(
                    input_text=frame.source_text, extracted_triples=[]
                )
                return LearningResult(
                    input_text=frame.source_text,
                    update_type=UpdateType.NO_OP,
                    experience_id=exp.id,
                    message=failure.reasons[0],
                )
            relation_candidates, effect_errors = self._render_effects(
                schema, arguments, execution.bindings, action.probability
            )
            accepted_relations: list[str] = []
            extracted_effects: list[dict[str, object]] = []
            seen_effects: set[tuple[str, str, str, bool]] = set()
            for reason in effect_errors:
                effect_evidence = self._propose_effect_failure(
                    frame, evidence, action.probability, source_type
                )
                effect_decision = self.commit_relation(
                    effect_evidence,
                    VerificationResult(
                        VerificationStatus.FAILED,
                        reasons=[reason],
                    ),
                )
                extracted_effects.append(
                    {
                        "predicate": "action_effect",
                        "is_negative": False,
                        "decision": effect_decision.status.value,
                        "reason": reason,
                    }
                )
            for claim in relation_candidates:
                effect_key = (
                    claim.subject,
                    claim.predicate,
                    claim.object,
                    claim.positive,
                )
                if effect_key in seen_effects:
                    continue
                seen_effects.add(effect_key)
                effect_evidence = self.propose_relation(
                    frame,
                    claim,
                    source_type=source_type,
                    source_reference=f"action:{evidence.evidence_id}",
                    positive=claim.positive,
                )
                effect_verification = self._verify_effect(claim)
                effect_decision = self.commit_relation(
                    effect_evidence,
                    effect_verification,
                    positive=claim.positive,
                )
                extracted_effects.append(
                    {
                        "subject": claim.subject,
                        "predicate": claim.predicate,
                        "object": claim.object,
                        "is_negative": not claim.positive,
                        "decision": effect_decision.status.value,
                    }
                )
                if effect_decision.status is CommitStatus.ACCEPTED:
                    accepted_relations.append(
                        f"({claim.subject} {claim.predicate} {claim.object})"
                    )

            exp = self.memory.add_experience(
                input_text=frame.source_text,
                extracted_triples=extracted_effects,
                source="action_execution",
            )
            base_result = execution.result
            return LearningResult(
                input_text=base_result.input_text,
                update_type=base_result.update_type,
                experience_id=exp.id,
                concepts_created=base_result.concepts_created,
                relations_created=accepted_relations,
                message=base_result.message,
            )
        exp = self.memory.add_experience(
            input_text=frame.source_text, extracted_triples=[]
        )
        return LearningResult(
            input_text=frame.source_text,
            update_type=UpdateType.NO_OP,
            experience_id=exp.id,
            message=verification.reasons[0],
        )

    def _render_effects(
        self,
        schema: ActionSchema,
        arguments: dict[str, str],
        bindings: dict[str, str],
        probability: float,
    ) -> tuple[list[CandidateClaim], list[str]]:
        """Resolve declarative relation effects against action result bindings."""
        values = {**arguments, **bindings}
        candidates: list[CandidateClaim] = []
        errors: list[str] = []
        for effect in schema.effects:
            if not isinstance(effect, dict):
                errors.append("Action schema effect must be an object")
                continue
            if effect.get("type", "relation") != "relation":
                errors.append(
                    f"Unsupported action schema effect type: {effect.get('type')}"
                )
                continue
            predicate_template = effect.get("predicate", effect.get("relation"))
            subject_template = effect.get("subject")
            object_template = effect.get("object")
            if not all(isinstance(value, str) for value in (
                predicate_template, subject_template, object_template
            )):
                errors.append("Action schema relation effect requires string subject, predicate, and object templates")
                continue
            try:
                subject = _resolve_effect_template(subject_template, values)
                predicate = _resolve_effect_template(predicate_template, values)
                object_ = _resolve_effect_template(object_template, values)
            except (KeyError, ValueError) as exc:
                errors.append(f"Could not resolve action schema effect: {exc}")
                continue
            positive = effect.get("positive", True)
            if isinstance(positive, str):
                positive = positive.strip().lower() not in {"0", "false", "no"}
            candidates.append(
                CandidateClaim(
                    subject=subject,
                    predicate=predicate,
                    object=object_,
                    probability=probability,
                    positive=bool(positive),
                )
            )
        return candidates, errors

    def _propose_effect_failure(
        self,
        frame: CandidateFrame,
        action_evidence: EvidenceRecord,
        probability: float,
        source_type: SourceType,
    ) -> EvidenceRecord:
        """Create an evidence record for an invalid schema effect."""
        return self.memory.evidence.append(
            EvidenceRecord(
                evidence_id=generate_id("evidence"),
                subject_id=action_evidence.subject_id,
                predicate="action_effect",
                object_id=None,
                source_type=source_type,
                source_reference=f"action:{action_evidence.evidence_id}",
                source_text=frame.source_text,
                extraction_confidence=probability,
            )
        )

    def _verify_effect(self, claim: CandidateClaim) -> VerificationResult:
        if self.registry.relation(claim.predicate) is None:
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                reasons=[f"Unregistered predicate: {claim.predicate}"],
            )
        # Tests and callers may replace the registry after construction.
        self.verifier.registry = self.registry
        return self.verifier.verify_relation(
            claim.subject, claim.predicate, claim.object
        ).to_verification_result()

    def reject(
        self, evidence: EvidenceRecord, reasons: list[str]
    ) -> CommitDecision:
        verification = VerificationResult(
            status=VerificationStatus.FAILED,
            checks={},
            reasons=list(reasons),
        )
        return self.commit_relation(evidence, verification)


@dataclass(frozen=True)
class ActionExecutionResult:
    """Executor output: a public learning result plus data bindings for effects."""

    result: LearningResult
    bindings: dict[str, str] = field(default_factory=dict)


ActionExecutorOutput = LearningResult | ActionExecutionResult
ActionExecutor = Callable[["ActionExecutionContext", CandidateFrame, dict[str, str]], ActionExecutorOutput]


class ActionExecutionViolation(RuntimeError):
    """Raised when an action executor attempts a semantic write."""


class ActionExecutionContext:
    """Narrow API for physical action executors.

    The context deliberately exposes no raw store, SQLite connection, evidence
    store, or generic attribute lookup. Semantic effects must be returned from
    the executor and committed from the action schema.
    """

    __slots__ = (
        "__store",
        "__relation_names",
        "__transformation_policy",
        "__dynamics_registry",
    )

    def __init__(self, memory: MemoryStore) -> None:
        self.__store = memory
        self.__transformation_policy = getattr(
            memory.ledger, "transformation_policy", None
        )
        self.__dynamics_registry = getattr(
            memory.ledger, "dynamics_registry", None
        )
        self.__relation_names = frozenset(
            memory.ledger.registry.predicates()
        )

    @property
    def transformation_policy(self):
        return self.__transformation_policy

    @property
    def dynamics_registry(self):
        return self.__dynamics_registry

    def count_relations(self) -> int:
        return self.__store.count_relations()

    def get_concept(self, id_or_name: str):
        return self.__store.get_concept(id_or_name)

    def get_entity(self, id_or_name: str):
        return self.__store.get_entity(id_or_name)

    def get_relations(self, **kwargs):
        return self.__store.get_relations(**kwargs)

    def list_concepts(self):
        return self.__store.list_concepts()

    def create_concept(self, name: str, category=None, aliases=None, attributes=None, confidence=None):
        attrs = dict(attributes or {})
        self.__reject_semantic_metadata(attrs, "concept attributes")
        return self.__store.create_concept(
            name,
            category=category,
            aliases=aliases,
            attributes=attrs,
            confidence=confidence,
        )

    def create_entity(self, name: str, concept_id: str, properties=None):
        props = dict(properties or {})
        self.__reject_semantic_metadata(props, "entity properties")
        return self.__store.create_entity(
            name, concept_id=concept_id, properties=props
        )

    def __reject_semantic_metadata(self, values: dict, label: str) -> None:
        semantic_keys = sorted(
            {
                str(key).strip().lower()
                for key in values
                if str(key).strip().lower() in self.__relation_names
            }
        )
        if semantic_keys:
            raise ActionExecutionViolation(
                f"physical {label} cannot declare registered relations: "
                + ", ".join(semantic_keys)
            )

    def list_evidence_by_source(self, source_type: str):
        return self.__store.evidence.list_by_source(source_type)

    def get_commit_status(self, evidence_id: str) -> str | None:
        row = self.__store._conn.execute(
            "SELECT status FROM commit_decisions WHERE evidence_id = ? LIMIT 1;",
            (evidence_id,),
        ).fetchone()
        return row["status"] if row else None

    def add_relation(self, *args, **kwargs):
        raise ActionExecutionViolation("semantic relation writes are not allowed")

    def add_experience(self, *args, **kwargs):
        raise ActionExecutionViolation("experience writes are not allowed")

    def update_concept_attributes(self, *args, **kwargs):
        raise ActionExecutionViolation("semantic attribute updates are not allowed")

    def bulk_import_triples(self, *args, **kwargs):
        raise ActionExecutionViolation("semantic triple imports are not allowed")


def _coerce_execution_result(output: ActionExecutorOutput) -> ActionExecutionResult:
    if isinstance(output, ActionExecutionResult):
        return output
    return ActionExecutionResult(result=output)


def _resolve_effect_template(template: str, values: dict[str, str]) -> str:
    """Resolve ``{binding}`` and ``$binding`` references in schema effects."""
    if template.startswith("$"):
        key = template[1:]
        if key not in values:
            raise KeyError(key)
        return str(values[key]).strip().lower()
    try:
        return template.format(**values).strip().lower()
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"invalid action effect template: {template}") from exc


class ActionExecutorRegistry:
    """Map schema executor names to action implementations."""

    def __init__(self) -> None:
        self._executors: dict[str, ActionExecutor] = {}

    def register(self, name: str, executor: ActionExecutor) -> None:
        self._executors[name] = executor

    def get(self, name: str | None) -> ActionExecutor | None:
        return self._executors.get(name) if name else None


def _execute_slice_object(
    memory: MemoryStore, frame: CandidateFrame, arguments: dict[str, str]
) -> ActionExecutionResult:
    from little.dynamics.transformations import TransformationEngine

    result = TransformationEngine.slice_object(
        memory,
        object_name=str(arguments["object"]).strip(),
        num_pieces=int(arguments["count"]),
        commit_relations=False,
        record_experience=False,
        policy=memory.transformation_policy,
        profile_registry=memory.dynamics_registry,
    )
    return ActionExecutionResult(
        result=LearningResult(
            input_text=frame.source_text,
            update_type=UpdateType.NEW_ENTITY,
            experience_id="",
            concepts_created=[result.slice_concept],
            relations_created=[],
            message=result.message,
        ),
        bindings={"result": result.slice_concept},
    )
