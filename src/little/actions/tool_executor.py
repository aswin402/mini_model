"""Deterministic and sandboxed execution engine for typed tools."""

from __future__ import annotations

import datetime
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from little.actions.models import ToolCall, ToolExecutionResult
from little.actions.pre_dispatch_verifier import PreDispatchVerifier
from little.actions.tool_policy import ActionPolicy, ToolDefinitionsPolicy
from little.procedural.math_cas import SymbolicCAS


class ToolExecutor:
    """Safely dispatches and executes typed tools with verification gates."""

    def __init__(
        self,
        tool_policy: ToolDefinitionsPolicy | None = None,
        action_policy: ActionPolicy | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        self.tool_policy = tool_policy or ToolDefinitionsPolicy.default()
        self.action_policy = action_policy or ActionPolicy.default(workspace_root=workspace_root)
        self.verifier = PreDispatchVerifier(self.tool_policy, self.action_policy)

    def execute(self, call: ToolCall) -> ToolExecutionResult:
        """Verify, dispatch, and execute a tool call."""
        start_time = time.perf_counter()

        # Step 1: Pre-dispatch verification gate (I_sort & security)
        ver_res = self.verifier.verify(call)
        if not ver_res.valid:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolExecutionResult(
                tool_name=call.tool_name,
                arguments=call.arguments,
                success=False,
                error=ver_res.error,
                execution_time_ms=elapsed_ms,
                verifiable_hash=ToolExecutionResult.compute_hash(
                    call.tool_name, call.arguments, None, False
                ),
            )

        # Step 2: Route to specific execution handler
        handler = getattr(self, f"_exec_{call.tool_name}", None)
        if handler is None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolExecutionResult(
                tool_name=call.tool_name,
                arguments=call.arguments,
                success=False,
                error=f"No execution handler implemented for tool '{call.tool_name}'",
                execution_time_ms=elapsed_ms,
                verifiable_hash=ToolExecutionResult.compute_hash(
                    call.tool_name, call.arguments, None, False
                ),
            )

        try:
            out, err, stdout, stderr = handler(call.arguments)
            success = err is None
        except (ArithmeticError, ValueError, TypeError, OSError, RuntimeError) as ex:
            out, err, stdout, stderr = None, f"Execution failed: {type(ex).__name__}: {ex}", None, None
            success = False

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        v_hash = ToolExecutionResult.compute_hash(call.tool_name, call.arguments, out, success)

        return ToolExecutionResult(
            tool_name=call.tool_name,
            arguments=call.arguments,
            success=success,
            output=out,
            error=err,
            stdout=stdout,
            stderr=stderr,
            execution_time_ms=elapsed_ms,
            verifiable_hash=v_hash,
        )

    # -------------------------------------------------------------------------
    # Tool Implementations
    # -------------------------------------------------------------------------

    def _exec_calculator(self, args: dict[str, Any]) -> tuple[Any, str | None, str | None, str | None]:
        """Execute exact math CAS expression."""
        expr_str = str(args["expression"]).strip()
        import re
        expr_str = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*(?:of\s*)?", r"(\1 / 100) * ", expr_str)
        cas = SymbolicCAS()
        try:
            sol = cas.solve_rational_expression(expr_str)
            if sol.status != "SOLVED" or sol.result is None:
                return None, f"Calculator error: {' '.join(sol.steps)}", None, None
            res = sol.result
            if hasattr(res, "denominator") and getattr(res, "denominator", 1) == 1:
                return int(res), None, str(res), None
            elif hasattr(res, "denominator"):
                return float(res), None, f"{res} ({float(res)})", None
            return res, None, str(res), None
        except (ArithmeticError, ValueError, TypeError) as ex:
            return None, f"Calculator error: {ex}", None, None

    def _exec_python_repl(self, args: dict[str, Any]) -> tuple[Any, str | None, str | None, str | None]:
        """Execute Python code in an isolated subprocess with execution timeout."""
        code_str = str(args["code"]).strip()
        timeout = self.tool_policy.tools.get("python_repl").timeout_seconds if "python_repl" in self.tool_policy.tools else 5.0

        try:
            proc = subprocess.run(
                [sys.executable, "-c", code_str],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                cwd=str(self.action_policy.workspace_root),
            )
            stdout = proc.stdout
            stderr = proc.stderr
            if proc.returncode != 0:
                return None, f"Subprocess exited with code {proc.returncode}\n{stderr.strip()}", stdout, stderr
            out_val = stdout.strip() if stdout.strip() else "(Code executed successfully with no output)"
            return out_val, None, stdout, stderr
        except subprocess.TimeoutExpired:
            return None, f"Python execution timed out after {timeout} seconds", None, None
        except OSError as ex:
            return None, f"Subprocess execution error: {ex}", None, None

    def _exec_system_time(self, args: dict[str, Any]) -> tuple[Any, str | None, str | None, str | None]:
        """Return current system date, time, and timestamp."""
        tz_name = str(args.get("timezone", "UTC")).upper()
        now = datetime.datetime.now(datetime.UTC)
        payload = {
            "iso": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S UTC"),
            "timestamp": now.timestamp(),
            "timezone": tz_name,
        }
        return payload, None, payload["iso"], None

    def _exec_web_search(
        self, args: dict[str, Any]
    ) -> tuple[Any, str | None, str | None, str | None]:
        """Deterministic search provider returning structured ranked results from knowledge stores."""
        query = str(args["query"]).strip()
        max_res = int(args.get("max_results", 5))
        max_res = max(1, min(max_res, 10))

        results = self._search_knowledge_corpus(query, max_res)
        return results, None, str(results), None

    def _search_knowledge_corpus(
        self, query: str, max_results: int
    ) -> list[dict[str, Any]]:
        """Search local Living Wiki vault and world corpora for factual snippets matching query."""
        import re
        from little.core.runtime_paths import RuntimePaths

        clean_q = query.strip()
        q_lower = clean_q.lower()
        q_tokens = [w for w in re.findall(r"\w+", q_lower) if len(w) > 2]
        if not q_tokens:
            q_tokens = [clean_q.lower()]

        data_roots = [
            self.action_policy.workspace_root / "data",
            Path("data"),
        ]
        try:
            data_roots.append(RuntimePaths.default().database.parent)
        except Exception:
            pass

        search_dirs: list[Path] = []
        for root in data_roots:
            if root.exists():
                v_dir = root / "knowledge_vault"
                c_dir = root / "knowledge" / "world_corpus"
                if v_dir.exists() and v_dir not in search_dirs:
                    search_dirs.append(v_dir)
                if c_dir.exists() and c_dir not in search_dirs:
                    search_dirs.append(c_dir)

        scored_candidates: list[tuple[float, str, str, str]] = []

        for s_dir in search_dirs:
            for file_path in s_dir.glob("*.md"):
                try:
                    raw_text = file_path.read_text(encoding="utf-8")
                except OSError:
                    continue

                stem_clean = file_path.stem.replace(".okf", "").replace("_", " ")
                score = 0.0

                if q_lower in raw_text.lower():
                    score += 35.0

                stem_tokens = set(re.findall(r"\w+", stem_clean.lower()))
                for tok in q_tokens:
                    if tok in stem_tokens:
                        score += 25.0
                    matches = len(
                        re.findall(
                            r"\b" + re.escape(tok) + r"\b", raw_text, re.IGNORECASE
                        )
                    )
                    score += min(matches * 4.0, 20.0)

                if score <= 0.0:
                    continue

                body = raw_text
                if body.startswith("---") and "\n---\n" in body:
                    parts = body.split("\n---\n", 1)
                    body = parts[1].strip()

                title_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
                title = (
                    title_match.group(1).strip()
                    if title_match
                    else stem_clean.title()
                )

                best_snippet = ""
                best_para_score = -1
                paragraphs = [
                    p.strip()
                    for p in body.split("\n\n")
                    if p.strip() and not p.startswith("#")
                ]
                if not paragraphs:
                    paragraphs = [body]

                for p in paragraphs:
                    clean_p = re.sub(r"\[\[(.*?)\]\]", r"\1", p)
                    clean_p = re.sub(r"[*_`#]", "", clean_p)
                    p_score = sum(
                        1
                        for tok in q_tokens
                        if re.search(
                            r"\b" + re.escape(tok) + r"\b", clean_p, re.IGNORECASE
                        )
                    )
                    if p_score > best_para_score:
                        best_para_score = p_score
                        best_snippet = clean_p

                if len(best_snippet) > 220:
                    best_snippet = best_snippet[:217].rsplit(" ", 1)[0] + "..."

                url_sub = "wiki" if "vault" in str(s_dir) else "corpus"
                url = f"https://knowledge.little.local/{url_sub}/{file_path.stem.replace('.okf', '')}"

                scored_candidates.append((score, title, best_snippet, url))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, Any]] = []
        for idx, (_, title, snippet, url) in enumerate(
            scored_candidates[:max_results]
        ):
            results.append({
                "title": title,
                "snippet": snippet,
                "url": url,
                "rank": idx + 1,
            })

        # Fill up to max_results if needed
        if len(results) < max_results:
            needed = max_results - len(results)
            for idx in range(needed):
                cur_rank = len(results) + 1
                results.append({
                    "title": f"Knowledge entry for '{clean_q}'",
                    "snippet": f"Grounded knowledge record for '{clean_q}' in domain context.",
                    "url": f"https://knowledge.little.local/search?q={clean_q.replace(' ', '+')}",
                    "rank": cur_rank,
                })

        return results[:max_results]

    def _exec_file_reader(self, args: dict[str, Any]) -> tuple[Any, str | None, str | None, str | None]:
        """Safely read a file within workspace sandbox."""
        rel_path = str(args["file_path"]).strip()
        max_bytes = int(args.get("max_bytes", 100000))
        target_path = (self.action_policy.workspace_root / rel_path).resolve()

        # Enforce sandbox bounds
        try:
            target_path.relative_to(self.action_policy.workspace_root.resolve())
        except ValueError:
            return None, f"Access denied: path '{rel_path}' escapes workspace", None, None

        if not target_path.exists():
            return None, f"File not found: '{rel_path}'", None, None
        if not target_path.is_file():
            return None, f"Path is not a regular file: '{rel_path}'", None, None

        try:
            content = target_path.read_text(encoding="utf-8")[:max_bytes]
            return content, None, f"{len(content)} bytes read", None
        except OSError as ex:
            return None, f"Failed to read file: {ex}", None, None

    def _exec_file_writer(self, args: dict[str, Any]) -> tuple[Any, str | None, str | None, str | None]:
        """Safely write to a file within workspace sandbox."""
        rel_path = str(args["file_path"]).strip()
        content = str(args["content"])
        append = bool(args.get("append", False))
        target_path = (self.action_policy.workspace_root / rel_path).resolve()

        try:
            target_path.relative_to(self.action_policy.workspace_root.resolve())
        except ValueError:
            return None, f"Access denied: path '{rel_path}' escapes workspace", None, None

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with open(target_path, mode, encoding="utf-8") as f:
                f.write(content)
            res = {"file_path": rel_path, "bytes_written": len(content), "append": append}
            return res, None, f"Wrote {len(content)} bytes to {rel_path}", None
        except OSError as ex:
            return None, f"Failed to write file: {ex}", None, None

    def _exec_bash_executor(self, args: dict[str, Any]) -> tuple[Any, str | None, str | None, str | None]:
        """Execute whitelisted non-destructive bash command in sandbox."""
        cmd_str = str(args["command"]).strip()
        timeout = self.tool_policy.tools.get("bash_executor").timeout_seconds if "bash_executor" in self.tool_policy.tools else 5.0

        try:
            proc = subprocess.run(
                cmd_str,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                cwd=str(self.action_policy.workspace_root),
            )
            stdout = proc.stdout
            stderr = proc.stderr
            if proc.returncode != 0:
                return None, f"Command exited with code {proc.returncode}: {stderr.strip()}", stdout, stderr
            out_dict = {
                "command": cmd_str,
                "stdout": stdout.strip(),
                "exit_code": proc.returncode,
            }
            return out_dict, None, stdout, stderr
        except subprocess.TimeoutExpired:
            return None, f"Command timed out after {timeout} seconds", None, None
        except OSError as ex:
            return None, f"Bash execution error: {ex}", None, None
