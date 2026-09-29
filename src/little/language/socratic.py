"""Socratic Explanation Engine — Multi-hop deductive syllogisms and invariant proofs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from little.core.models import BeliefStatus
from little.language.semantic_predicate_policy import SemanticPredicatePolicy
from little.language.socratic_policy import SocraticPolicy

if TYPE_CHECKING:
    from little.inference.engine import InferenceEngine
    from little.memory.store import MemoryStore


@dataclass
class SocraticExplanation:
    """Formal syllogistic proof and natural language pedagogical explanation."""

    query: str
    status: BeliefStatus
    premises: list[str] = field(default_factory=list)
    conclusion: str = ""
    verbalized: str = ""
    trace: list[str] = field(default_factory=list)


class SocraticReasoner:
    """Constructs formal deductive proofs and invariant explanations without token hallucination."""

    def __init__(
        self,
        memory: MemoryStore,
        inference: InferenceEngine,
        policy: SocraticPolicy | None = None,
        semantic: SemanticPredicatePolicy | None = None,
    ) -> None:
        self.memory = memory
        self.inference = inference
        self.policy = policy or SocraticPolicy.default()
        self.semantic = semantic or SemanticPredicatePolicy.default()

    @staticmethod
    def _article(word: str, capitalize: bool = False) -> str:
        art = "an" if word and word[0].lower() in "aeiou" else "a"
        return art.capitalize() if capitalize else art

    def explain_why_is_a(self, subject: str, target: str) -> SocraticExplanation:
        """Formulate a step-by-step deductive syllogism explaining why subject is target."""
        clean_s = subject.strip().lower()
        clean_t = target.strip().lower()
        res = self.inference.infer(
            subject=clean_s,
            predicate=self.semantic.taxonomy,
            target=clean_t,
        )

        query = f"Why is a {clean_s} a {clean_t}?"

        if res.status == BeliefStatus.SUPPORTED:
            trans_step = next(
                (t for t in res.trace if "Transitive path discovered" in t), None
            )
            premises: list[str] = []
            if trans_step:
                path_str = trans_step.split(": ")[1].split(" (")[0]
                nodes = [n.strip() for n in path_str.split("->")]
                for i in range(len(nodes) - 1):
                    s_w = nodes[i]
                    t_w = nodes[i + 1]
                    s_art = self._article(s_w, capitalize=True)
                    t_art = self._article(t_w, capitalize=False)
                    p_text = self.policy.premise_template.format(
                        index=i + 1,
                        subject=s_w,
                        target=t_w,
                        subject_article=s_art,
                        target_article=t_art,
                    )
                    premises.append(p_text)
                first_art = self._article(nodes[0], capitalize=False)
                last_art = self._article(nodes[-1], capitalize=False)
                conclusion = self.policy.deduction_template.format(
                    subject=nodes[0],
                    target=nodes[-1],
                    subject_article=first_art,
                    target_article=last_art,
                )
            elif res.evidence:
                premises.append(f"Grounded axiom: {res.evidence[0]}")
                first_art = self._article(clean_s, capitalize=False)
                last_art = self._article(clean_t, capitalize=False)
                conclusion = f"Therefore, {first_art} {clean_s} is {last_art} {clean_t}."
            else:
                premises.append(
                    f"Verified direct relation: ({clean_s}, is_a, {clean_t})"
                )
                first_art = self._article(clean_s, capitalize=False)
                last_art = self._article(clean_t, capitalize=False)
                conclusion = f"Therefore, {first_art} {clean_s} is {last_art} {clean_t}."

            verbalized_lines = [self.policy.syllogism_header]
            verbalized_lines.extend(f"  • {p}" for p in premises)
            verbalized_lines.append(f"  ↳ {conclusion}")
            verbalized = "\n".join(verbalized_lines)

            return SocraticExplanation(
                query=query,
                status=BeliefStatus.SUPPORTED,
                premises=premises,
                conclusion=conclusion,
                verbalized=verbalized,
                trace=res.trace,
            )

        if res.status == BeliefStatus.REFUTED:
            return self.explain_why_not(clean_s, clean_t)

        # UNKNOWN
        msg = self.policy.unknown_explanation_template.format(
            subject=clean_s, target=clean_t
        )
        return SocraticExplanation(
            query=query,
            status=BeliefStatus.UNKNOWN,
            premises=[],
            conclusion=msg,
            verbalized=msg,
            trace=res.trace,
        )

    def explain_why_not(self, subject: str, target: str) -> SocraticExplanation:
        """Explain why subject cannot be target by isolating the exact disjoint invariant constraint."""
        clean_s = subject.strip().lower()
        clean_t = target.strip().lower()
        res = self.inference.infer(
            subject=clean_s,
            predicate=self.semantic.taxonomy,
            target=clean_t,
        )

        query = f"Why is a {clean_s} not a {clean_t}?"

        disj_step = next(
            (t for t in res.trace if "Disjoint constraint triggered" in t), None
        )
        premises: list[str] = []
        if disj_step:
            reason = disj_step.split(": ")[1]
            premises.append(f"Disjoint constraint: {reason}")
            # Try parsing ancestors
            s_c = self.memory.get_concept(clean_s)
            t_c = self.memory.get_concept(clean_t)
            s_anc = [clean_s] + (
                self.inference.get_ancestor_ids(s_c.id, self.semantic.taxonomy)
                if s_c
                else []
            )
            t_anc = [clean_t] + (
                self.inference.get_ancestor_ids(t_c.id, self.semantic.taxonomy)
                if t_c
                else []
            )
            # Find disjoint edge
            disj_pair = None
            for sa in s_anc:
                for ta in t_anc:
                    r = self.memory.get_relations(
                        subject_id=sa,
                        predicate=self.semantic.disjoint,
                        object_id=ta,
                    )
                    if not r:
                        r = self.memory.get_relations(
                            subject_id=ta,
                            predicate=self.semantic.disjoint,
                            object_id=sa,
                        )
                    if r and r[0].weight_positive > r[0].weight_negative:
                        disj_pair = (sa, ta)
                        break
                if disj_pair:
                    break

            if disj_pair:
                s_art = self._article(clean_s, capitalize=False)
                sc_art = self._article(disj_pair[0], capitalize=False)
                conclusion = self.policy.refutation_disjoint_template.format(
                    subject=clean_s,
                    subject_article=s_art,
                    subject_category=disj_pair[0],
                    subject_category_article=sc_art,
                    target_category=disj_pair[1],
                )
            else:
                conclusion = f"Because {reason}."
        else:
            conclusion = f"Because a {clean_s} and {clean_t} are mutually exclusive in memory."

        verbalized = f"Refutation Explanation:\n  ↳ {conclusion}"
        return SocraticExplanation(
            query=query,
            status=BeliefStatus.REFUTED,
            premises=premises,
            conclusion=conclusion,
            verbalized=verbalized,
            trace=res.trace,
        )

    def explain_capability(self, subject: str, capability: str) -> SocraticExplanation:
        """Explain why subject has or does not have capability, tracing hypernym inheritance."""
        clean_s = subject.strip().lower()
        clean_cap = capability.strip().lower()
        res = self.inference.infer(
            subject=clean_s,
            predicate=self.semantic.capability,
            target=clean_cap,
        )

        query = f"Why can a {clean_s} {clean_cap}?"

        if res.status == BeliefStatus.SUPPORTED:
            inh_step = next(
                (
                    t
                    for t in res.trace
                    if "Inherited relation from ancestor" in t or "Inherited" in t
                ),
                None,
            )
            premises: list[str] = []
            if inh_step and len(res.evidence) >= 2:
                p1 = res.evidence[0].split()
                anc = p1[2] if len(p1) >= 3 else "its parent class"
                s_art = self._article(clean_s, capitalize=True)
                anc_art = self._article(anc, capitalize=False)
                premises.append(f"Taxonomic premise: {s_art} {clean_s} is {anc_art} {anc}.")
                premises.append(
                    f"Inherited capability: {self._article(anc, capitalize=True)} {anc} can {clean_cap}."
                )
                conclusion = self.policy.capability_inheritance_template.format(
                    subject=clean_s,
                    subject_article=self._article(clean_s, capitalize=False),
                    ancestor=anc,
                    ancestor_article=anc_art,
                    capability=clean_cap,
                )
            else:
                s_art = self._article(clean_s, capitalize=True)
                premises.append(
                    f"Direct observation: {s_art} {clean_s} can {clean_cap}."
                )
                conclusion = f"Because {self._article(clean_s, capitalize=False)} {clean_s} has verified direct capability to {clean_cap}."

            verbalized_lines = [self.policy.syllogism_header]
            verbalized_lines.extend(f"  • {p}" for p in premises)
            verbalized_lines.append(f"  ↳ {conclusion}")
            verbalized = "\n".join(verbalized_lines)

            return SocraticExplanation(
                query=query,
                status=BeliefStatus.SUPPORTED,
                premises=premises,
                conclusion=conclusion,
                verbalized=verbalized,
                trace=res.trace,
            )

        # UNKNOWN or REFUTED
        msg = f"I cannot deduce why a {clean_s} would {clean_cap} based on current knowledge."
        return SocraticExplanation(
            query=query,
            status=res.status,
            premises=[],
            conclusion=msg,
            verbalized=msg,
            trace=res.trace,
        )

    def explain_property(self, subject: str, prop: str) -> SocraticExplanation:
        """Explain why subject possesses property prop."""
        clean_s = subject.strip().lower()
        clean_p = prop.strip().lower()
        res = self.inference.infer(
            subject=clean_s,
            predicate=self.semantic.property,
            target=clean_p,
        )

        query = f"Why is a {clean_s} {clean_p}?"

        if res.status == BeliefStatus.SUPPORTED:
            premises: list[str] = []
            if len(res.evidence) >= 2:
                p1 = res.evidence[0].split()
                anc = p1[2] if len(p1) >= 3 else "its category"
                premises.append(f"Taxonomic premise: A {clean_s} is a {anc}.")
                premises.append(f"Category property: A {anc} has property {clean_p}.")
                conclusion = self.policy.property_inheritance_template.format(
                    subject=clean_s, ancestor=anc, property=clean_p
                )
            else:
                premises.append(f"Direct observation: A {clean_s} is {clean_p}.")
                conclusion = f"Because a {clean_s} is directly verified as {clean_p} in memory."

            verbalized_lines = [self.policy.syllogism_header]
            verbalized_lines.extend(f"  • {p}" for p in premises)
            verbalized_lines.append(f"  ↳ {conclusion}")
            verbalized = "\n".join(verbalized_lines)

            return SocraticExplanation(
                query=query,
                status=BeliefStatus.SUPPORTED,
                premises=premises,
                conclusion=conclusion,
                verbalized=verbalized,
                trace=res.trace,
            )

        msg = f"I do not have verified premises explaining why a {clean_s} is {clean_p}."
        return SocraticExplanation(
            query=query,
            status=res.status,
            premises=[],
            conclusion=msg,
            verbalized=msg,
            trace=res.trace,
        )
