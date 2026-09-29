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
        w = word.strip()
        if not w:
            return ""
        if w[0].isupper():
            return ""
        w_lower = w.lower()
        art = "an" if w_lower[0] in "aeiou" else "a"
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

    def _format_conjoined(self, items: list[str], conj: str = "and") -> str:
        """Join a list of items into a natural English conjoined phrase."""
        clean = [item.strip() for item in items if item.strip()]
        if not clean:
            return ""
        if len(clean) == 1:
            return clean[0]
        if len(clean) == 2:
            return f"{clean[0]} {conj} {clean[1]}"
        sep = self.policy.get_template("conjoined_separator", ", and ") if conj == "and" else f", {conj} "
        return f"{', '.join(clean[:-1])}{sep}{clean[-1]}"

    def verbalize_concept_definition(self, text: str) -> str:
        """Render a raw semi-colon separated definition into warm, flowing English prose."""
        parts = [p.strip().rstrip(".") for p in text.split(";") if p.strip()]
        if not parts:
            return text

        first = parts[0]
        m = re.match(r"^([a-zA-Z0-9_\s-]+?)\s+(is(?:\s+a)?\s+.+)$", first, re.IGNORECASE)
        if not m:
            return text

        subj = m.group(1).strip()
        subj_clean = subj.lower()
        subj_art = self._article(subj_clean, capitalize=True)
        sentences: list[str] = []

        tax_raw = m.group(2).strip().rstrip(".")
        raw_items = [re.sub(r"^(?:is\s+(?:a\s+)?)", "", item).strip() for item in tax_raw.split(",")]
        tax_items = [f"{self._article(item)} {item}" for item in raw_items if item]
        if tax_items:
            tax_str = self._format_conjoined(tax_items)
            tpl_tax = self.policy.get_template("definition_taxonomy", "{subject_article} {subject} is {taxonomy}.")
            sentences.append(tpl_tax.format(subject_article=subj_art, subject=subj_clean, taxonomy=tax_str))

        for part in parts[1:]:
            p_strip = part.strip()
            if p_strip.startswith("has "):
                items = [it.strip() for it in p_strip[4:].split(",") if it.strip()]
                if items:
                    f_str = self._format_conjoined(items)
                    tpl_has = self.policy.get_template("definition_has", "It has {features}.")
                    sentences.append(tpl_has.format(features=f_str))
            elif p_strip.startswith("is part of "):
                target = p_strip[11:].strip()
                tpl_po = self.policy.get_template("definition_part_of", "It is part of {target_article} {target}.")
                sentences.append(tpl_po.format(target_article=self._article(target), target=target))
            elif p_strip.startswith("consists of parts:"):
                items = [it.strip() for it in p_strip[18:].split(",") if it.strip()]
                if items:
                    f_str = self._format_conjoined(items)
                    tpl_parts = self.policy.get_template("definition_parts", "It consists of {parts}.")
                    sentences.append(tpl_parts.format(parts=f_str))
            elif p_strip.startswith(("is not a ", "is not ")):
                raw_disj = re.sub(r"^is\s+not\s+(?:a\s+)?", "", p_strip).strip()
                items = [it.strip() for it in raw_disj.split(",") if it.strip()]
                if items:
                    f_str = self._format_conjoined(items)
                    tpl_disj = self.policy.get_template("definition_disjoint", "It is distinct from {disjoint}.")
                    sentences.append(tpl_disj.format(disjoint=f_str))
            elif p_strip.startswith("attributes:"):
                tpl_attr = self.policy.get_template("definition_attributes", "Its attributes include {attributes}.")
                sentences.append(tpl_attr.format(attributes=p_strip[11:].strip()))

        return " ".join(sentences)

    def verbalize_inference(
        self,
        inference: InferenceResult,
        query: str = "",
        resolved_subject: str | None = None,
        parsed_query: tuple[str, str, Any] | None = None,
    ) -> str:
        """Render formal InferenceResult into fluent conversational English."""
        if isinstance(inference.answer, (list, tuple)):
            clean_list = [str(x).strip() for x in inference.answer if x is not None and str(x).strip()]
            unique_list = list(dict.fromkeys(clean_list))
            if len(unique_list) == 1:
                raw_ans = unique_list[0]
            else:
                return "\n\n".join(unique_list)
        else:
            raw_ans = str(inference.answer) if inference.answer is not None else ""

        pronouns = {"it", "they", "this", "that", "he", "she"}

        def _clean(text: str) -> str:
            res = re.sub(r" +", " ", text)
            res = res.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
            return res.strip()

        def _restore_casing(name: str) -> str:
            m = re.search(rf"\b{re.escape(name)}\b", query, re.IGNORECASE)
            if m and m.group(0)[0].isupper():
                return m.group(0)
            return name

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

        # 1b. Concept definitions (knowledge graph summaries and definition retrievals)
        if (
            not raw_ans.startswith("Deductive Proof:")
            and not raw_ans.startswith("Yes")
            and not raw_ans.startswith("No")
            and (
                ";" in raw_ans
                or any("Retrieved concept definition for" in str(e) for e in inference.evidence)
            )
        ):
            verbalized_def = self.verbalize_concept_definition(raw_ans)
            if verbalized_def:
                return _clean(verbalized_def)

        # 2. Epistemic Unknown (Honest admissions with conversational invitation)
        if inference.status == BeliefStatus.UNKNOWN:
            subject = None
            if parsed_query and len(parsed_query) >= 1 and parsed_query[0]:
                subject = str(parsed_query[0]).strip()
            if not subject:
                m_subj = re.search(
                    r"\b(?:is|can|does|what color is|where is|where does|what is)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s]+?)(?:\s+(?:a|an|the|have|in|on|at|can|fly|swim|live|color)|\?|$)",
                    query,
                    re.IGNORECASE,
                )
                subject = m_subj.group(1).strip() if m_subj else "that concept"
            if subject.lower() in pronouns and resolved_subject:
                subject = resolved_subject
            tpl = self.policy.get_template(
                "epistemic_unknown",
                "I don't have that information in my memory yet. Could you tell me more about {subject}?",
            )
            return _clean(tpl.format(subject=subject))

        # 3. Refuted claims
        if inference.status == BeliefStatus.REFUTED or inference.answer is False:
            if parsed_query and len(parsed_query) == 3 and parsed_query[1] in {"is_a", "subclass_of", "isa"}:
                s_name = _restore_casing(str(parsed_query[0]).strip())
                t_name = _restore_casing(str(parsed_query[2]).strip())
                if s_name.lower() in pronouns and resolved_subject:
                    s_name = _restore_casing(resolved_subject)
                tpl = self.policy.get_template(
                    "refuted_taxonomy", "No, {subject_article} {subject} is not {target_article} {target}."
                )
                return _clean(tpl.format(
                    subject_article=self._article(s_name, capitalize=False),
                    subject=s_name,
                    target_article=self._article(t_name, capitalize=False),
                    target=t_name,
                ))

            if parsed_query and len(parsed_query) == 3 and parsed_query[1] in {"can", "capable_of"}:
                s_name = _restore_casing(str(parsed_query[0]).strip())
                action = str(parsed_query[2]).strip()
                if s_name.lower() in pronouns and resolved_subject:
                    s_name = _restore_casing(resolved_subject)
                tpl = self.policy.get_template(
                    "refuted_capability", "No, {subject} cannot {action}."
                )
                return _clean(tpl.format(subject=s_name, action=action))

            if parsed_query and len(parsed_query) == 3 and parsed_query[1] in {"has", "has_feature", "has_part"}:
                s_name = _restore_casing(str(parsed_query[0]).strip())
                m_has = re.search(
                    r"(?:Does|Has)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s]+?)\s+(?:have|has)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s]+?)\?*$",
                    query.strip(),
                    re.IGNORECASE,
                )
                raw_feature = m_has.group(2).strip() if m_has else str(parsed_query[2]).strip()
                feature = _restore_casing(raw_feature)
                if s_name.lower() in pronouns and resolved_subject:
                    s_name = _restore_casing(resolved_subject)
                tpl = self.policy.get_template(
                    "refuted_feature", "No, {subject_article} {subject} does not have {target}."
                )
                return _clean(tpl.format(
                    subject_article=self._article(s_name, capitalize=False),
                    subject=s_name,
                    target=feature,
                ))

            m_tax = re.search(
                r"Is\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s]+?)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s]+?)\?*$",
                query.strip(),
                re.IGNORECASE,
            )
            if m_tax:
                s_name = m_tax.group(1).strip()
                t_name = m_tax.group(2).strip()
                if s_name.lower() in pronouns and resolved_subject:
                    s_name = resolved_subject
                tpl = self.policy.get_template(
                    "refuted_taxonomy", "No, {subject_article} {subject} is not {target_article} {target}."
                )
                return _clean(tpl.format(
                    subject_article=self._article(s_name, capitalize=False),
                    subject=s_name,
                    target_article=self._article(t_name, capitalize=False),
                    target=t_name,
                ))
            m_can = re.search(
                r"Can\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s]+?)\s+([a-zA-Z0-9_\s]+?)\?*$",
                query.strip(),
                re.IGNORECASE,
            )
            if m_can:
                s_name = m_can.group(1).strip()
                action = m_can.group(2).strip()
                if s_name.lower() in pronouns and resolved_subject:
                    s_name = resolved_subject
                tpl = self.policy.get_template(
                    "refuted_capability", "No, {subject} cannot {action}."
                )
                return _clean(tpl.format(subject=s_name, action=action))

            m_has = re.search(
                r"(?:Does|Has)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s]+?)\s+(?:have|has)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s]+?)\?*$",
                query.strip(),
                re.IGNORECASE,
            )
            if m_has:
                s_name = m_has.group(1).strip()
                feature = m_has.group(2).strip()
                if s_name.lower() in pronouns and resolved_subject:
                    s_name = resolved_subject
                tpl = self.policy.get_template(
                    "refuted_feature", "No, {subject_article} {subject} does not have {target}."
                )
                return _clean(tpl.format(
                    subject_article=self._article(s_name, capitalize=False),
                    subject=s_name,
                    target=feature,
                ))

            return _clean(self.policy.get_template(
                "refuted_general", "No, that does not match what I know to be true."
            ))

        # 4. Supported Affirmations
        if inference.status == BeliefStatus.SUPPORTED:
            if inference.answer is True:
                if parsed_query and len(parsed_query) == 3 and parsed_query[1] in {"is_a", "subclass_of", "isa"}:
                    s_name = _restore_casing(str(parsed_query[0]).strip())
                    t_name = _restore_casing(str(parsed_query[2]).strip())
                    if s_name.lower() in pronouns and resolved_subject:
                        s_name = _restore_casing(resolved_subject)
                    tpl = self.policy.get_template(
                        "affirmative_taxonomy",
                        "Yes, {subject_article} {subject} is indeed {target_article} {target}.",
                    )
                    return _clean(tpl.format(
                        subject_article=self._article(s_name, capitalize=False),
                        subject=s_name,
                        target_article=self._article(t_name, capitalize=False),
                        target=t_name,
                    ))

                if parsed_query and len(parsed_query) == 3 and parsed_query[1] in {"can", "capable_of"}:
                    s_name = _restore_casing(str(parsed_query[0]).strip())
                    action = str(parsed_query[2]).strip()
                    if s_name.lower() in pronouns and resolved_subject:
                        s_name = _restore_casing(resolved_subject)
                    tpl = self.policy.get_template(
                        "affirmative_capability", "Yes, {subject} can certainly {action}."
                    )
                    return _clean(tpl.format(subject=s_name, action=action))

                if parsed_query and len(parsed_query) == 3 and parsed_query[1] in {"has", "has_feature", "has_part"}:
                    s_name = _restore_casing(str(parsed_query[0]).strip())
                    m_has = re.search(
                        r"(?:Does|Has)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s]+?)\s+(?:have|has)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s]+?)\?*$",
                        query.strip(),
                        re.IGNORECASE,
                    )
                    raw_feature = m_has.group(2).strip() if m_has else str(parsed_query[2]).strip()
                    feature = _restore_casing(raw_feature)
                    if s_name.lower() in pronouns and resolved_subject:
                        s_name = _restore_casing(resolved_subject)
                    tpl = self.policy.get_template(
                        "affirmative_feature",
                        "Yes, {subject_article} {subject} has {target}.",
                    )
                    return _clean(tpl.format(
                        subject_article=self._article(s_name, capitalize=False),
                        subject=s_name,
                        target=feature,
                    ))

                if parsed_query and len(parsed_query) == 3 and parsed_query[1] not in {"is_a", "subclass_of", "isa", "can", "capable_of", "has", "has_feature", "has_part"}:
                    s_name = _restore_casing(str(parsed_query[0]).strip())
                    pred = str(parsed_query[1]).strip()
                    t_name = _restore_casing(str(parsed_query[2]).strip())
                    if s_name.lower() in pronouns and resolved_subject:
                        s_name = _restore_casing(resolved_subject)
                    pred_3sg = f"{pred}s" if not pred.endswith(("s", "x", "z", "ch", "sh")) else pred
                    tpl = self.policy.get_template("affirmative_relation", "Yes, {subject} {predicate} {target}.")
                    return _clean(tpl.format(subject=s_name, predicate=pred_3sg, target=t_name))

                m_can = re.search(
                    r"Can\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s]+?)\s+([a-zA-Z0-9_\s]+?)\?*$",
                    query.strip(),
                    re.IGNORECASE,
                )
                if m_can:
                    s_name = m_can.group(1).strip()
                    action = m_can.group(2).strip()
                    if s_name.lower() in pronouns and resolved_subject:
                        s_name = resolved_subject
                    tpl = self.policy.get_template(
                        "affirmative_capability", "Yes, {subject} can certainly {action}."
                    )
                    return _clean(tpl.format(subject=s_name, action=action))

                m_tax = re.search(
                    r"Is\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s]+?)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s]+?)\?*$",
                    query.strip(),
                    re.IGNORECASE,
                )
                if m_tax:
                    s_name = m_tax.group(1).strip()
                    t_name = m_tax.group(2).strip()
                    if s_name.lower() in pronouns and resolved_subject:
                        s_name = resolved_subject
                    tpl = self.policy.get_template(
                        "affirmative_taxonomy",
                        "Yes, {subject_article} {subject} is indeed {target_article} {target}.",
                    )
                    return _clean(tpl.format(
                        subject_article=self._article(s_name, capitalize=False),
                        subject=s_name,
                        target_article=self._article(t_name, capitalize=False),
                        target=t_name,
                    ))

                m_has = re.search(
                    r"(?:Does|Has)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s]+?)\s+(?:have|has)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\s]+?)\?*$",
                    query.strip(),
                    re.IGNORECASE,
                )
                if m_has:
                    s_name = m_has.group(1).strip()
                    feature = m_has.group(2).strip()
                    if s_name.lower() in pronouns and resolved_subject:
                        s_name = resolved_subject
                    tpl = self.policy.get_template(
                        "affirmative_feature",
                        "Yes, {subject_article} {subject} has {target}.",
                    )
                    return _clean(tpl.format(
                        subject_article=self._article(s_name, capitalize=False),
                        subject=s_name,
                        target=feature,
                    ))

                return _clean(self.policy.get_template(
                    "affirmative_general", "Yes, that is correct."
                ))

            # Property / Textual answers
            if isinstance(inference.answer, str):
                if (
                    raw_ans.startswith(("A ", "An ", "Because ", "Here is ", "Your ", "You are ", "I am "))
                    or " lives in " in raw_ans
                    or " is a " in raw_ans
                    or " is located in " in raw_ans
                    or raw_ans.endswith((".", "!"))
                ):
                    return _clean(raw_ans)
                if not raw_ans.startswith(("Yes", "No")):
                    return _clean(f"Based on my knowledge, that is {raw_ans}.")

            return _clean(raw_ans)

        return _clean(raw_ans) if raw_ans else ""
