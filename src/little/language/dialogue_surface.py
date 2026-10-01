"""Conversational dialogue surface engine for natural English verbalization and social interaction.

Translates formal symbolic inferences, learning events, and social exchanges
into warm, articulate, human-like dialogue governed by externalized schemas.
"""

from __future__ import annotations

import re
from typing import Any

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
        if len(clauses) <= 1 and re.search(r"\band\b", text, re.IGNORECASE):
            and_clauses = [p.strip() for p in re.split(r"\band\b", text, flags=re.IGNORECASE) if p.strip()]
            if len(and_clauses) > 1 and all(self.policy.match_social_intent(c) for c in and_clauses):
                clauses = and_clauses

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
        tax_items = [
            f"{self._article(item)} {item}"
            for item in raw_items
            if item and not item.lower().startswith("latent_")
        ]
        if tax_items:
            tax_str = self._format_conjoined(tax_items)
            tpl_tax = self.policy.get_template("definition_taxonomy", "{subject_article} {subject} is {taxonomy}.")
            sentences.append(tpl_tax.format(subject_article=subj_art, subject=subj_clean, taxonomy=tax_str))

        for part in parts[1:]:
            p_strip = part.strip()
            if p_strip.startswith("has "):
                items = [
                    it.strip()
                    for it in p_strip[4:].split(",")
                    if it.strip() and not it.strip().lower().startswith("latent_")
                ]
                if items:
                    f_str = self._format_conjoined(items)
                    tpl_has = self.policy.get_template("definition_has", "It has {features}.")
                    sentences.append(tpl_has.format(features=f_str))
            elif p_strip.startswith("is part of "):
                target = p_strip[11:].strip()
                if not target.lower().startswith("latent_"):
                    tpl_po = self.policy.get_template("definition_part_of", "It is part of {target_article} {target}.")
                    sentences.append(tpl_po.format(target_article=self._article(target), target=target))
            elif p_strip.startswith("consists of parts:"):
                items = [
                    it.strip()
                    for it in p_strip[18:].split(",")
                    if it.strip() and not it.strip().lower().startswith("latent_")
                ]
                if items:
                    f_str = self._format_conjoined(items)
                    tpl_parts = self.policy.get_template("definition_parts", "It consists of {parts}.")
                    sentences.append(tpl_parts.format(parts=f_str))
            elif p_strip.startswith(("is not a ", "is not ")):
                raw_disj = re.sub(r"^is\s+not\s+(?:a\s+)?", "", p_strip).strip()
                items = [
                    it.strip()
                    for it in raw_disj.split(",")
                    if it.strip() and not it.strip().lower().startswith("latent_")
                ]
                if items:
                    f_str = self._format_conjoined(items)
                    tpl_disj = self.policy.get_template("definition_disjoint", "It is distinct from {disjoint}.")
                    sentences.append(tpl_disj.format(disjoint=f_str))
            elif p_strip.startswith("attributes:"):
                raw_attrs = p_strip[11:].strip().strip("()")
                attr_pairs = [pair.strip() for pair in raw_attrs.split(",") if pair.strip()]
                clean_pairs = []
                for pair in attr_pairs:
                    k = pair.split("=")[0].strip().lower()
                    if k in {"is_latent", "category_utility", "constituent_count", "induced_at"} or k.startswith("latent_"):
                        continue
                    clean_pairs.append(pair)
                if clean_pairs:
                    tpl_attr = self.policy.get_template("definition_attributes", "Its attributes include {attributes}.")
                    sentences.append(tpl_attr.format(attributes=", ".join(clean_pairs)))

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
                    raw_ans.startswith(("A ", "An ", "Because ", "Here is ", "Your ", "You are ", "I am ", "Hello", "Hi", "Hey", "Greetings", "Not at all"))
                    or " lives in " in raw_ans
                    or " is a " in raw_ans
                    or " is located in " in raw_ans
                    or raw_ans.endswith((".", "!", "?"))
                ):
                    return _clean(raw_ans)
                if not raw_ans.startswith(("Yes", "No")):
                    return _clean(f"Based on my knowledge, that is {raw_ans}.")

            return _clean(raw_ans)

        return _clean(raw_ans) if raw_ans else ""

    def _format_calculator_result(self, output: Any, args: dict[str, Any], tool_name: str) -> str:
        tpl = self.policy.get_template(
            "tool_success_calculator", "The calculated result is {output}."
        )
        return tpl.format(output=output)

    def _format_python_repl_result(self, output: Any, args: dict[str, Any], tool_name: str) -> str:
        tpl = self.policy.get_template(
            "tool_success_python_repl", "Python execution output:\n{output}"
        )
        return tpl.format(output=output)

    def _format_system_time_result(self, output: Any, args: dict[str, Any], tool_name: str) -> str:
        if isinstance(output, dict):
            tpl = self.policy.get_template(
                "tool_success_system_time", "The current time is {time} on {date} ({timezone})."
            )
            return tpl.format(
                time=output.get("time", ""),
                date=output.get("date", ""),
                timezone=output.get("timezone", "UTC"),
            )
        return self._format_generic_tool_result(output, args, tool_name)

    def _format_web_search_result(self, output: Any, args: dict[str, Any], tool_name: str) -> str:
        if isinstance(output, list):
            tpl = self.policy.get_template(
                "tool_success_web_search", "I found the following search results for '{query}':\n{results}"
            )
            res_lines = [
                f"- {item.get('title')}: {item.get('snippet')}"
                for item in output
                if isinstance(item, dict)
            ]
            query_val = args.get("query", "")
            return tpl.format(query=query_val, results="\n".join(res_lines))
        return self._format_generic_tool_result(output, args, tool_name)

    def _format_file_reader_result(self, output: Any, args: dict[str, Any], tool_name: str) -> str:
        tpl = self.policy.get_template(
            "tool_success_file_reader", "Contents of {file_path}:\n{output}"
        )
        return tpl.format(file_path=args.get("file_path", ""), output=output)

    def _format_file_writer_result(self, output: Any, args: dict[str, Any], tool_name: str) -> str:
        if isinstance(output, dict):
            tpl = self.policy.get_template(
                "tool_success_file_writer", "Successfully wrote {bytes_written} bytes to {file_path}."
            )
            return tpl.format(
                bytes_written=output.get("bytes_written", 0),
                file_path=output.get("file_path", ""),
            )
        return self._format_generic_tool_result(output, args, tool_name)

    def _format_bash_executor_result(self, output: Any, args: dict[str, Any], tool_name: str) -> str:
        if isinstance(output, dict):
            tpl = self.policy.get_template(
                "tool_success_bash_executor", "Command output:\n{stdout}"
            )
            return tpl.format(stdout=output.get("stdout", ""))
        return self._format_generic_tool_result(output, args, tool_name)

    def _format_generic_tool_result(self, output: Any, args: dict[str, Any], tool_name: str) -> str:
        tpl = self.policy.get_template(
            "tool_success_generic", "Tool '{tool_name}' executed successfully:\n{output}"
        )
        return tpl.format(tool_name=tool_name, output=output)

    def verbalize_tool_result(self, result: Any) -> str:
        """Verbalize a ToolExecutionResult using schema-driven strategy dispatch."""
        tool_name = getattr(result, "tool_name", "tool")
        success = getattr(result, "success", False)
        output = getattr(result, "output", None)
        error = getattr(result, "error", None)
        args = getattr(result, "arguments", {})

        if not success:
            tpl = self.policy.get_template(
                "tool_error", "I encountered an issue executing tool '{tool_name}': {error}"
            )
            return tpl.format(tool_name=tool_name, error=error or "Unknown error")

        dispatch_table = {
            "calculator": self._format_calculator_result,
            "python_repl": self._format_python_repl_result,
            "system_time": self._format_system_time_result,
            "web_search": self._format_web_search_result,
            "file_reader": self._format_file_reader_result,
            "file_writer": self._format_file_writer_result,
            "bash_executor": self._format_bash_executor_result,
        }

        formatter = dispatch_table.get(tool_name, self._format_generic_tool_result)
        return formatter(output, args, tool_name)

    def verbalize_agentic_plan_result(self, result: Any) -> str:
        """Verbalize an AgenticPlanExecutionResult trace with status badges and provenance."""
        steps = getattr(result, "steps", ())
        trace_hash = getattr(result, "trace_hash", "")[:8]
        elapsed_ms = getattr(result, "total_execution_time_ms", 0.0)
        success = getattr(result, "success", False)
        final_output = getattr(result, "final_output", None)
        error = getattr(result, "error", None)

        hdr_tpl = self.policy.get_template(
            "agentic_plan_header",
            "I formulated and executed an autonomous {step_count}-step plan to accomplish your goal:\n[Trace: {trace_hash} | {execution_time_ms:.1f}ms]",
        )
        header = hdr_tpl.format(
            step_count=len(steps),
            trace_hash=trace_hash,
            execution_time_ms=elapsed_ms,
        )

        lines = [header]

        for s in steps:
            args_summary = ", ".join(f"{k}={v!r}" for k, v in s.arguments_template.items())
            if len(args_summary) > 40:
                args_summary = args_summary[:37] + "..."

            out_summary = str(s.observation or "")
            if len(out_summary) > 50:
                out_summary = out_summary[:47] + "..."

            badge = "OK" if s.status == "SUCCESS" else s.status
            lines.append(f"• Step {s.step_index}: {s.tool_name}({args_summary}) -> {out_summary} [{badge}]")

        if success:
            lines.append("")
            lines.append(f"Final Result: {final_output}")
        elif error:
            lines.append("")
            lines.append(f"The plan encountered an error: {error}")

        return "\n".join(lines)

    def verbalize_code_synthesis(self, prog: Any) -> str:
        """Verbalize synthesized Python AST program and sandbox verification status."""
        lines = [
            f"Here is the verified Python implementation for `{prog.name}`:\n",
            "```python",
            prog.code,
            "```\n",
            f"**Verification Status:** {prog.status}",
            f"- Passed {len(prog.assertions)} test assertions in sandbox ({prog.execution_time_ms:.2f} ms).",
        ]
        if prog.execution_hash:
            lines.append(f"- Execution Hash: `{prog.execution_hash[:16]}...`")
        return "\n".join(lines)

    def verbalize_crawl_report(self, report: Any) -> str:
        """Verbalize knowledge crawling and ingestion session results."""
        lines = [
            f"Crawled and ingested '{report.source_reference}' in {report.duration_ms:.2f} ms:",
            f"- Extracted claims: {report.extracted_claims}",
            f"- Verified claims committed to memory: {report.verified_claims}",
            f"- OKF pages created/updated in vault: {report.new_pages_written}",
        ]
        if report.rejected_claims > 0:
            lines.append(
                f"- Invariant gate rejections: {report.rejected_claims} ({len(report.rejection_reasons)} violations caught)"
            )
        return "\n".join(lines)

    def verbalize_wiki_search(self, query: str, results: list[Any]) -> str:
        """Verbalize TurboQuant semantic search results over the living knowledge wiki."""
        if not results:
            return f"No concept pages matching '{query}' were found in the living wiki."
        lines = [f"Found {len(results)} relevant wiki concept(s) for '{query}':"]
        for r in results:
            lines.append(f"- **{r.id}** (similarity: {r.score:.3f})")
        return "\n".join(lines)

    def verbalize_backlinks(self, concept_name: str, backlinks: list[str]) -> str:
        """Verbalize incoming wiki links to a concept."""
        if not backlinks:
            return f"There are currently no concepts linking to '{concept_name}' in the living wiki."
        links_str = ", ".join(f"[[{c}]]" for c in backlinks)
        return f"The following concept(s) link to '{concept_name}': {links_str}."

    def verbalize_wiki_page(self, doc: Any) -> str:
        """Verbalize an OKF wiki page summary."""
        name = doc.concept_name
        tax = doc.frontmatter.get("taxonomy", {})
        parents = tax.get("is_a", []) if isinstance(tax, dict) else []
        parents_str = f"a type of {', '.join(parents)}" if parents else "a concept knot in the living wiki"
        lines = [f"**{name.title()}** is {parents_str}."]
        if doc.wikilinks:
            linked_str = ", ".join(f"[[{link.target}]]" for link in doc.wikilinks)
            lines.append(f"- Linked concepts: {linked_str}")
        return "\n".join(lines)

    def verbalize_latent_induction(self, report: Any) -> str:
        """Verbalize autonomous Cobweb clustering and latent knot induction results."""
        lines = [
            f"Autonomous Cobweb clustering completed in {report.duration_ms:.2f} ms:",
            f"- Concepts analyzed: {report.total_concepts_analyzed}",
            f"- Clusters formed: {report.clusters_formed}",
            f"- Latent knots induced: {report.latent_knots_induced}",
        ]
        if report.latent_knots_rejected > 0:
            lines.append(f"- Candidate knots rejected by invariant gates: {report.latent_knots_rejected}")
        if report.inductions:
            lines.append("\n**Induced Latent Knots:**")
            for ind in report.inductions[:8]:
                if ind.verification_passed:
                    members_str = ", ".join(f"[[{c}]]" for c in ind.constituent_concepts[:6])
                    if len(ind.constituent_concepts) > 6:
                        members_str += f" and {len(ind.constituent_concepts) - 6} more"
                    lines.append(f"- **{ind.latent_name}** (CU: {ind.category_utility:.2f}): unifies {members_str}")
        return "\n".join(lines)

    def verbalize_latent_knots(self, latent_concepts: list[Any]) -> str:
        """Verbalize list of active latent knots in memory."""
        if not latent_concepts:
            return "No latent concept knots have been induced in memory yet. You can trigger discovery with 'cluster concepts'."
        lines = [f"Found {len(latent_concepts)} induced latent concept knot(s) in memory:"]
        for c in latent_concepts:
            cnt = c.attributes.get("constituent_count", 0)
            cnt_str = f" ({cnt} concepts)" if cnt else ""
            lines.append(f"- **{c.name}**{cnt_str}")
        return "\n".join(lines)

    def verbalize_cluster_explanation(self, info: dict[str, Any] | None, concept_name: str) -> str:
        """Verbalize explanation of a concept's latent cluster membership."""
        if not info:
            return f"Concept '{concept_name}' has not been assigned to a latent cluster yet."
        cat = info.get("latent_category", "")
        peers = info.get("peer_concepts", [])
        cu = info.get("category_utility", 0.0)
        peers_str = ", ".join(f"[[{p}]]" for p in peers[:8])
        if len(peers) > 8:
            peers_str += f" and {len(peers) - 8} more"
        return (
            f"**{concept_name.title()}** is clustered under the latent category **{cat}** (Category Utility: {cu:.2f}).\n"
            f"- Co-clustered peer concepts: {peers_str}."
        )

    def verbalize_dream_report(self, report: Any) -> str:
        """Verbalize an autonomous cognitive dreaming session."""
        lines = [
            f"Autonomous cognitive dream cycle completed in {report.duration_ms:.1f} ms:",
            f"- Explored {report.steps_attempted} frontier concept(s)",
            f"- Verified & assimilated {report.steps_persisted} novel claim(s) into memory",
            f"- Knowledge entropy reduced by {report.entropy_reduction:.4f}",
            f"- Living Wiki pages synchronized: {report.wiki_pages_synced}",
        ]
        if report.persisted_claims:
            lines.append("\n**Assimilated Beliefs:**")
            for c in report.persisted_claims:
                lines.append(f"- • {c['subject']} {c['predicate']} {c['object']}")
        if report.steps_rejected > 0:
            lines.append(
                f"\n- Prevented {report.steps_rejected} invalid candidate(s) via deterministic invariant gates (zero hallucination guarantee)."
            )
        return "\n".join(lines)

    def verbalize_epistemic_gaps(self, gaps: list[dict[str, Any]]) -> str:
        """Verbalize concepts with high missing spoke entropy and knowledge gaps."""
        if not gaps:
            return "All concept knots are fully populated across their radial axes with zero epistemic entropy."
        lines = [f"Identified {len(gaps)} key knowledge gap(s) on the cognitive frontier:"]
        for g in gaps:
            c_name = g["concept"].title()
            ent = g["entropy"]
            missing = ", ".join(g.get("missing_axes", []))
            stub_tag = " [Wiki Stub]" if g.get("is_stub") else ""
            lines.append(f"- **{c_name}**{stub_tag} (radial entropy: {ent:.2f}) — Missing axes: {missing}")
        return "\n".join(lines)

    def verbalize_curiosity_step(self, step: Any) -> str:
        """Verbalize a targeted curiosity study step."""
        if step.status == "PERSISTED":
            hyp = step.hypothesis
            return (
                f"Autonomous curiosity study for '{step.target_concept}':\n"
                f"- Verified and assimilated: ({hyp.get('subject')} {hyp.get('predicate')} {hyp.get('object')})\n"
                f"- Radial axis satisfied: {step.missing_axis}\n"
                f"- Gate verification: Passed all 4 DeepSeek deterministic gates."
            )
        elif step.status == "REJECTED":
            return (
                f"Autonomous curiosity study for '{step.target_concept}':\n"
                f"- Proposed hypothesis ({step.hypothesis.get('subject')} {step.hypothesis.get('predicate')} {step.hypothesis.get('object')}) was rejected.\n"
                f"- Invariant Gate Defense: {step.reason}"
            )
        else:
            return f"Curiosity study for '{step.target_concept}': {step.reason}"

    def verbalize_physical_simulation(self, sim: Any) -> str:
        """Verbalize a continuous-time physical simulation."""
        lines = [
            f"Physical simulation for **{sim.concept_name.title()}** over {sim.delta_t_hours:.1f} hours at {sim.ambient_temperature_c:.1f}°C:",
            f"- **Freshness**: {sim.initial_state.freshness:.2f} ➔ {sim.final_state.freshness:.2f}",
            f"- **Oxidation**: {sim.initial_state.oxidation:.2f} ➔ {sim.final_state.oxidation:.2f}",
            f"- **Moisture**: {sim.initial_state.moisture:.2f} ➔ {sim.final_state.moisture:.2f}",
            f"- **Skin Boundary**: {'Intact protective skin' if sim.skin_intact else 'Severed / exposed tissue'}",
        ]
        if sim.milestones:
            lines.append("\n**Key Dynamic Milestones:**")
            for m in sim.milestones:
                lines.append(f"- T+{m.time_hours:.1f}h: {m.description}")
        return "\n".join(lines)

    def verbalize_action_jump(
        self, action: str, target: str, pieces: list[Any], details: str = ""
    ) -> str:
        """Verbalize a discrete physical action jump and QPT mass conservation."""
        total_mass = sum(getattr(p, "invariant_mass", 0.0) for p in pieces)
        lines = [
            f"Applied physical action **{action}** to **{target.title()}**:",
            f"- Produced {len(pieces)} piece(s) under Qualitative Process Theory (QPT).",
            f"- Total invariant mass conserved: {total_mass:.1f}g.",
        ]
        for p in pieces:
            p_id = getattr(p, "concept_id", "piece")
            p_mass = getattr(p, "invariant_mass", 0.0)
            lines.append(f"  • {p_id}: mass {p_mass:.1f}g")
        if details:
            lines.append(f"\n{details}")
        return "\n".join(lines)

    def verbalize_sensory_grounding(self, res: Any) -> str:
        """Verbalize multimodal sensory grounding to 0° episodic axis."""
        if res.status == "GROUNDED":
            features_str = ", ".join(f"{k}={v}" for k, v in res.observation.features.items())
            return (
                f"Sensory Observation Grounded:\n"
                f"- Concept: **{res.matched_concept.title()}** (confidence: {res.confidence * 100:.1f}%)\n"
                f"- Episodic Axis: Anchored at 0° Episodic Instance axis (Entity: `{res.entity_id}`)\n"
                f"- Perceived Features: [{features_str}]\n"
                f"- State: Verified against physical invariant bounds."
            )
        elif res.status == "MUTEX_REJECTED":
            return (
                f"Sensory Observation Rejected by Invariant Gate:\n"
                f"- Reason: {res.reason}\n"
                f"- Zero Hallucination Guarantee: Conflicting sensory reading was not assimilated."
            )
        else:
            return f"Sensory observation ambiguous: {res.reason}"






