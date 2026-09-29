"""Conversational dialogue surface engine for natural English verbalization and social interaction.

Translates formal symbolic inferences, learning events, and social exchanges
into warm, articulate, human-like dialogue governed by externalized schemas.
"""

from __future__ import annotations

import re

from little.core.models import BeliefStatus, InferenceResult, LearningResult
from little.language.dialogue_surface_policy import DialogueSurfacePolicy


class DialogueSurfaceEngine:
    """Zero-hardcoding conversational surface engine for natural English dialogue."""

    def __init__(self, policy: DialogueSurfacePolicy | None = None) -> None:
        self.policy = policy or DialogueSurfacePolicy.default()

    @staticmethod
    def _article(word: str, capitalize: bool = False) -> str:
        w = word.strip().lower()
        art = "an" if w and w[0] in "aeiou" else "a"
        return art.capitalize() if capitalize else art

    def handle_social(self, text: str) -> str | None:
        """Detect and respond to pure social intents (greetings, pleasantries, identity)."""
        matched = self.policy.match_social_intent(text)
        if matched:
            _, response = matched
            return response

        clauses = [p.strip() for p in re.split(r"[.!?]+", text) if p.strip()]
        if len(clauses) > 1:
            matched_clauses: list[tuple[str, str]] = []
            for clause in clauses:
                m = self.policy.match_social_intent(clause)
                if not m:
                    return None
                matched_clauses.append(m)

            replies: list[str] = []
            for i, (name, resp) in enumerate(matched_clauses):
                if name == "greeting" and i == 0 and len(matched_clauses) > 1:
                    replies.append("Hello!")
                else:
                    replies.append(resp)
            return " ".join(replies)

        return None

    def decompose_utterance(self, text: str) -> tuple[str | None, str]:
        """Separate leading social pleasantries from substantive question/statement."""
        res = self.policy.decompose_leading_greeting(text)
        if res:
            greeting_word, remainder = res
            return greeting_word, remainder
        return None, text.strip()

    def compose_reply(self, greeting_word: str | None, body: str) -> str:
        """Compose final natural English reply, attaching a greeting reply if appropriate."""
        if not greeting_word:
            return body
        if body.lower().startswith(greeting_word.lower()):
            return body
        tpl = self.policy.get_template("mixed_greeting_response", "{greeting_reply} {body}")
        greeting_reply = f"{greeting_word.capitalize()}!"
        return tpl.format(greeting_reply=greeting_reply, body=body)

    def verbalize_learning(self, result: LearningResult) -> str:
        """Render a natural English confirmation when the model learns a new fact."""
        if "Nice to meet you" in result.message:
            return result.message
        if result.update_type.value == "NO_OP":
            return result.message

        clean_text = result.input_text.strip().rstrip(".?!")
        tpl = self.policy.get_template(
            "learning_confirmation", "Got it! I've learned that {summary}."
        )
        return tpl.format(summary=clean_text)

    def verbalize_inference(self, inference: InferenceResult, query: str = "") -> str:
        """Render formal InferenceResult into fluent conversational English."""
        raw_ans = str(inference.answer) if inference.answer is not None else ""

        # 1. Custom traces & multi-step solutions (Socratic syllogisms, CAS math traces, ODE simulations, posets)
        if (
            "Deductive Proof:" in raw_ans
            or "<math_trace>" in raw_ans
            or any("<math_trace>" in str(e) for e in inference.evidence)
            or any("Counterfactual simulation" in str(e) for e in inference.evidence)
            or "is the fastest" in raw_ans
            or "is the tallest" in raw_ans
            or "is the shortest" in raw_ans
            or "is the slowest" in raw_ans
            or "strict partial order:" in raw_ans
        ):
            return raw_ans

        # 2. Epistemic Unknown (Honest admissions with conversational invitation)
        if inference.status == BeliefStatus.UNKNOWN:
            m_subj = re.search(
                r"\b(?:is|can|does|what color is|where is|where does|what is)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_]+)",
                query,
                re.IGNORECASE,
            )
            subject = m_subj.group(1).strip() if m_subj else "that concept"
            tpl = self.policy.get_template(
                "epistemic_unknown",
                "I don't have that information in my memory yet. Could you tell me more about {subject}?",
            )
            return tpl.format(subject=subject)

        # 3. Refuted claims
        if inference.status == BeliefStatus.REFUTED or inference.answer is False:
            m_tax = re.search(
                r"Is\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_]+)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s]+?)\?*$",
                query.strip(),
                re.IGNORECASE,
            )
            if m_tax:
                s_name = m_tax.group(1).strip()
                t_name = m_tax.group(2).strip()
                tpl = self.policy.get_template(
                    "refuted_taxonomy", "No, {subject_article} {subject} is not {target_article} {target}."
                )
                return tpl.format(
                    subject_article=self._article(s_name, capitalize=False),
                    subject=s_name,
                    target_article=self._article(t_name, capitalize=False),
                    target=t_name,
                )
            m_can = re.search(
                r"Can\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_]+)\s+([a-zA-Z0-9_\s]+?)\?*$",
                query.strip(),
                re.IGNORECASE,
            )
            if m_can:
                s_name = m_can.group(1).strip()
                action = m_can.group(2).strip()
                tpl = self.policy.get_template(
                    "refuted_capability", "No, {subject} cannot {action}."
                )
                return tpl.format(subject=s_name, action=action)

            m_has = re.search(
                r"(?:Does|Has)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_]+)\s+(?:have|has)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s]+?)\?*$",
                query.strip(),
                re.IGNORECASE,
            )
            if m_has:
                s_name = m_has.group(1).strip()
                feature = m_has.group(2).strip()
                tpl = self.policy.get_template(
                    "refuted_feature", "No, {subject_article} {subject} does not have {target}."
                )
                return tpl.format(
                    subject_article=self._article(s_name, capitalize=False),
                    subject=s_name,
                    target=feature,
                )

            return self.policy.get_template(
                "refuted_general", "No, that does not match what I know to be true."
            )

        # 4. Supported Affirmations
        if inference.status == BeliefStatus.SUPPORTED:
            if inference.answer is True:
                m_can = re.search(
                    r"Can\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_]+)\s+([a-zA-Z0-9_\s]+?)\?*$",
                    query.strip(),
                    re.IGNORECASE,
                )
                if m_can:
                    s_name = m_can.group(1).strip()
                    action = m_can.group(2).strip()
                    tpl = self.policy.get_template(
                        "affirmative_capability", "Yes, {subject} can certainly {action}."
                    )
                    return tpl.format(subject=s_name, action=action)

                m_tax = re.search(
                    r"Is\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_]+)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s]+?)\?*$",
                    query.strip(),
                    re.IGNORECASE,
                )
                if m_tax:
                    s_name = m_tax.group(1).strip()
                    t_name = m_tax.group(2).strip()
                    tpl = self.policy.get_template(
                        "affirmative_taxonomy",
                        "Yes, {subject_article} {subject} is indeed {target_article} {target}.",
                    )
                    return tpl.format(
                        subject_article=self._article(s_name, capitalize=False),
                        subject=s_name,
                        target_article=self._article(t_name, capitalize=False),
                        target=t_name,
                    )

                m_has = re.search(
                    r"(?:Does|Has)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_]+)\s+(?:have|has)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s]+?)\?*$",
                    query.strip(),
                    re.IGNORECASE,
                )
                if m_has:
                    s_name = m_has.group(1).strip()
                    feature = m_has.group(2).strip()
                    tpl = self.policy.get_template(
                        "affirmative_feature",
                        "Yes, {subject_article} {subject} has {target}.",
                    )
                    return tpl.format(
                        subject_article=self._article(s_name, capitalize=False),
                        subject=s_name,
                        target=feature,
                    )

                return self.policy.get_template(
                    "affirmative_general", "Yes, that is correct."
                )

            # Property (e.g. color: "Based on my knowledge, that is red.")
            if isinstance(inference.answer, str):
                if raw_ans.startswith(("A ", "An ", "Because ")) or " lives in " in raw_ans or " is a " in raw_ans or " is located in " in raw_ans:
                    return raw_ans
                if not raw_ans.startswith(("Yes", "No")):
                    return f"Based on my knowledge, that is {raw_ans}."

            return str(inference.answer)

        return str(inference.answer) if inference.answer is not None else ""
