"""Abstract Syntax Tree (AST) Python Code Synthesizer for verified procedural generation."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from little.procedural.code_policy import CodeSynthesizerPolicy, PrimitiveSpec
from little.procedural.code_sandbox import CodeSandbox


@dataclass
class SynthesizedProgram:
    name: str
    params: tuple[str, ...]
    docstring: str
    tree: ast.FunctionDef
    code: str
    status: str = "SYNTHESIZED"
    assertions: tuple[str, ...] = ()
    proof_trace: list[str] = field(default_factory=list)
    execution_hash: str | None = None
    execution_time_ms: float = 0.0


class ASTCodeSynthesizer:
    """Synthesizes verified Python Abstract Syntax Trees for computational and algorithmic tasks."""

    def __init__(
        self,
        policy: CodeSynthesizerPolicy | None = None,
        sandbox: CodeSandbox | None = None,
    ) -> None:
        self.policy = policy or CodeSynthesizerPolicy.default()
        self.sandbox = sandbox or CodeSandbox(policy=self.policy)

    def match_primitive(self, text: str) -> str | None:
        """Match natural language request to an available algorithmic primitive dynamically from CodeSynthesizerPolicy."""
        low = text.lower()
        allowed_words = self.policy.allowed_request_vocabulary
        tokens: list[str] = []
        curr: list[str] = []
        for ch in low:
            if ("a" <= ch <= "z") or ("0" <= ch <= "9") or ch == "_":
                curr.append(ch)
            elif curr:
                tokens.append("".join(curr))
                curr = []
        if curr:
            tokens.append("".join(curr))

        if any(token not in allowed_words for token in tokens) or "%" in low:
            return None

        tok_set = set(tokens)
        norm_words = " ".join(tokens)
        matches: list[str] = []
        for prim_key, spec in self.policy.primitives.items():
            triggers = set(spec.trigger_tokens) | {prim_key.lower(), spec.name.lower()}
            has_token_hit = bool(tok_set & triggers)
            has_phrase_hit = any(phrase in norm_words for phrase in spec.trigger_phrases)
            if has_token_hit or has_phrase_hit:
                matches.append(prim_key)
        return matches[0] if len(matches) == 1 else None

    def synthesize_from_request(
        self, text: str, verify: bool = True
    ) -> SynthesizedProgram | None:
        """Attempt to synthesize a verified program from a natural language request."""
        prim = self.match_primitive(text)
        if prim is None:
            return None
        return self.synthesize(prim, verify=verify)

    def synthesize(
        self, primitive_name: str, verify: bool = True
    ) -> SynthesizedProgram:
        """Synthesize an algorithmic primitive by name and optionally verify via sandbox."""
        prim_clean = primitive_name.strip().lower()
        if prim_clean not in self.policy.primitives:
            raise KeyError(f"Unknown algorithmic primitive '{primitive_name}'")

        spec = self.policy.primitives[prim_clean]
        builder = getattr(self, f"_build_{spec.name}", None)
        if builder is None:
            raise NotImplementedError(
                f"No AST generator implemented for primitive '{spec.name}'"
            )

        tree = builder(spec)
        code = ast.unparse(tree)
        trace = [
            f"Synthesized AST for function '{spec.name}' with params {spec.params}",
            f"Generated source code ({len(code)} chars)",
        ]

        prog = SynthesizedProgram(
            name=spec.name,
            params=spec.params,
            docstring=spec.docstring,
            tree=tree,
            code=code,
            assertions=spec.assertions,
            proof_trace=trace,
        )

        if verify and spec.assertions:
            self.verify_and_repair(prog)

        return prog

    def verify_and_repair(self, prog: SynthesizedProgram) -> bool:
        """Verify program against its assertions in sandbox, invoking Socratic repair on failure."""
        res = self.sandbox.run_assertions(prog.code, prog.assertions)
        prog.execution_time_ms = res.execution_time_ms
        prog.execution_hash = res.verifiable_hash

        if res.success:
            prog.status = "VERIFIED"
            prog.proof_trace.append(
                f"All {len(prog.assertions)} assertions verified successfully in sandbox ({res.execution_time_ms:.2f} ms)"
            )
            return True

        prog.proof_trace.append(f"Verification failed: {res.error}")
        from little.procedural.code_repair import SocraticCodeRepair

        repairer = SocraticCodeRepair(sandbox=self.sandbox, policy=self.policy)
        repaired = repairer.repair(prog, res)
        return repaired

    # -------------------------------------------------------------------------
    # Deterministic AST Builders for Algorithmic Primitives
    # -------------------------------------------------------------------------

    def _build_factorial(self, spec: PrimitiveSpec) -> ast.FunctionDef:
        """def factorial(n): ..."""
        # if n < 0: raise ValueError("n must be non-negative")
        # if n <= 1: return 1
        # result = 1
        # for i in range(2, n + 1): result *= i
        # return result
        body = [
            ast.Expr(value=ast.Constant(value=spec.docstring)),
            ast.If(
                test=ast.Compare(
                    left=ast.Name(id="n", ctx=ast.Load()),
                    ops=[ast.Lt()],
                    comparators=[ast.Constant(value=0)],
                ),
                body=[
                    ast.Raise(
                        exc=ast.Call(
                            func=ast.Name(id="ValueError", ctx=ast.Load()),
                            args=[ast.Constant(value="n must be non-negative")],
                            keywords=[],
                        )
                    )
                ],
                orelse=[],
            ),
            ast.If(
                test=ast.Compare(
                    left=ast.Name(id="n", ctx=ast.Load()),
                    ops=[ast.LtE()],
                    comparators=[ast.Constant(value=1)],
                ),
                body=[ast.Return(value=ast.Constant(value=1))],
                orelse=[],
            ),
            ast.Assign(
                targets=[ast.Name(id="result", ctx=ast.Store())],
                value=ast.Constant(value=1),
            ),
            ast.For(
                target=ast.Name(id="i", ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[
                        ast.Constant(value=2),
                        ast.BinOp(
                            left=ast.Name(id="n", ctx=ast.Load()),
                            op=ast.Add(),
                            right=ast.Constant(value=1),
                        ),
                    ],
                    keywords=[],
                ),
                body=[
                    ast.AugAssign(
                        target=ast.Name(id="result", ctx=ast.Store()),
                        op=ast.Mult(),
                        value=ast.Name(id="i", ctx=ast.Load()),
                    )
                ],
                orelse=[],
            ),
            ast.Return(value=ast.Name(id="result", ctx=ast.Load())),
        ]
        return self._make_func(spec.name, list(spec.params), body)

    def _build_fibonacci(self, spec: PrimitiveSpec) -> ast.FunctionDef:
        """def fibonacci(n): ..."""
        # if n < 0: raise ValueError(...)
        # if n == 0: return 0
        # if n == 1: return 1
        # a, b = 0, 1
        # for _ in range(2, n + 1): a, b = b, a + b
        # return b
        body = [
            ast.Expr(value=ast.Constant(value=spec.docstring)),
            ast.If(
                test=ast.Compare(
                    left=ast.Name(id="n", ctx=ast.Load()),
                    ops=[ast.Lt()],
                    comparators=[ast.Constant(value=0)],
                ),
                body=[
                    ast.Raise(
                        exc=ast.Call(
                            func=ast.Name(id="ValueError", ctx=ast.Load()),
                            args=[ast.Constant(value="n must be non-negative")],
                            keywords=[],
                        )
                    )
                ],
                orelse=[],
            ),
            ast.If(
                test=ast.Compare(
                    left=ast.Name(id="n", ctx=ast.Load()),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value=0)],
                ),
                body=[ast.Return(value=ast.Constant(value=0))],
                orelse=[],
            ),
            ast.If(
                test=ast.Compare(
                    left=ast.Name(id="n", ctx=ast.Load()),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value=1)],
                ),
                body=[ast.Return(value=ast.Constant(value=1))],
                orelse=[],
            ),
            ast.Assign(
                targets=[
                    ast.Tuple(
                        elts=[
                            ast.Name(id="a", ctx=ast.Store()),
                            ast.Name(id="b", ctx=ast.Store()),
                        ],
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Tuple(
                    elts=[ast.Constant(value=0), ast.Constant(value=1)],
                    ctx=ast.Load(),
                ),
            ),
            ast.For(
                target=ast.Name(id="_", ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[
                        ast.Constant(value=2),
                        ast.BinOp(
                            left=ast.Name(id="n", ctx=ast.Load()),
                            op=ast.Add(),
                            right=ast.Constant(value=1),
                        ),
                    ],
                    keywords=[],
                ),
                body=[
                    ast.Assign(
                        targets=[
                            ast.Tuple(
                                elts=[
                                    ast.Name(id="a", ctx=ast.Store()),
                                    ast.Name(id="b", ctx=ast.Store()),
                                ],
                                ctx=ast.Store(),
                            )
                        ],
                        value=ast.Tuple(
                            elts=[
                                ast.Name(id="b", ctx=ast.Load()),
                                ast.BinOp(
                                    left=ast.Name(id="a", ctx=ast.Load()),
                                    op=ast.Add(),
                                    right=ast.Name(id="b", ctx=ast.Load()),
                                ),
                            ],
                            ctx=ast.Load(),
                        ),
                    )
                ],
                orelse=[],
            ),
            ast.Return(value=ast.Name(id="b", ctx=ast.Load())),
        ]
        return self._make_func(spec.name, list(spec.params), body)

    def _build_is_prime(self, spec: PrimitiveSpec) -> ast.FunctionDef:
        """def is_prime(n): ..."""
        # if n <= 1: return False
        # if n <= 3: return True
        # if n % 2 == 0 or n % 3 == 0: return False
        # i = 5
        # while i * i <= n:
        #     if n % i == 0 or n % (i + 2) == 0: return False
        #     i += 6
        # return True
        body = [
            ast.Expr(value=ast.Constant(value=spec.docstring)),
            ast.If(
                test=ast.Compare(
                    left=ast.Name(id="n", ctx=ast.Load()),
                    ops=[ast.LtE()],
                    comparators=[ast.Constant(value=1)],
                ),
                body=[ast.Return(value=ast.Constant(value=False))],
                orelse=[],
            ),
            ast.If(
                test=ast.Compare(
                    left=ast.Name(id="n", ctx=ast.Load()),
                    ops=[ast.LtE()],
                    comparators=[ast.Constant(value=3)],
                ),
                body=[ast.Return(value=ast.Constant(value=True))],
                orelse=[],
            ),
            ast.If(
                test=ast.BoolOp(
                    op=ast.Or(),
                    values=[
                        ast.Compare(
                            left=ast.BinOp(
                                left=ast.Name(id="n", ctx=ast.Load()),
                                op=ast.Mod(),
                                right=ast.Constant(value=2),
                            ),
                            ops=[ast.Eq()],
                            comparators=[ast.Constant(value=0)],
                        ),
                        ast.Compare(
                            left=ast.BinOp(
                                left=ast.Name(id="n", ctx=ast.Load()),
                                op=ast.Mod(),
                                right=ast.Constant(value=3),
                            ),
                            ops=[ast.Eq()],
                            comparators=[ast.Constant(value=0)],
                        ),
                    ],
                ),
                body=[ast.Return(value=ast.Constant(value=False))],
                orelse=[],
            ),
            ast.Assign(
                targets=[ast.Name(id="i", ctx=ast.Store())],
                value=ast.Constant(value=5),
            ),
            ast.While(
                test=ast.Compare(
                    left=ast.BinOp(
                        left=ast.Name(id="i", ctx=ast.Load()),
                        op=ast.Mult(),
                        right=ast.Name(id="i", ctx=ast.Load()),
                    ),
                    ops=[ast.LtE()],
                    comparators=[ast.Name(id="n", ctx=ast.Load())],
                ),
                body=[
                    ast.If(
                        test=ast.BoolOp(
                            op=ast.Or(),
                            values=[
                                ast.Compare(
                                    left=ast.BinOp(
                                        left=ast.Name(id="n", ctx=ast.Load()),
                                        op=ast.Mod(),
                                        right=ast.Name(id="i", ctx=ast.Load()),
                                    ),
                                    ops=[ast.Eq()],
                                    comparators=[ast.Constant(value=0)],
                                ),
                                ast.Compare(
                                    left=ast.BinOp(
                                        left=ast.Name(id="n", ctx=ast.Load()),
                                        op=ast.Mod(),
                                        right=ast.BinOp(
                                            left=ast.Name(id="i", ctx=ast.Load()),
                                            op=ast.Add(),
                                            right=ast.Constant(value=2),
                                        ),
                                    ),
                                    ops=[ast.Eq()],
                                    comparators=[ast.Constant(value=0)],
                                ),
                            ],
                        ),
                        body=[ast.Return(value=ast.Constant(value=False))],
                        orelse=[],
                    ),
                    ast.AugAssign(
                        target=ast.Name(id="i", ctx=ast.Store()),
                        op=ast.Add(),
                        value=ast.Constant(value=6),
                    ),
                ],
                orelse=[],
            ),
            ast.Return(value=ast.Constant(value=True)),
        ]
        return self._make_func(spec.name, list(spec.params), body)

    def _build_reverse_string(self, spec: PrimitiveSpec) -> ast.FunctionDef:
        """def reverse_string(s): return s[::-1]"""
        body = [
            ast.Expr(value=ast.Constant(value=spec.docstring)),
            ast.Return(
                value=ast.Subscript(
                    value=ast.Name(id="s", ctx=ast.Load()),
                    slice=ast.Slice(
                        lower=None,
                        upper=None,
                        step=ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1)),
                    ),
                    ctx=ast.Load(),
                )
            ),
        ]
        return self._make_func(spec.name, list(spec.params), body)

    def _build_is_palindrome(self, spec: PrimitiveSpec) -> ast.FunctionDef:
        """def is_palindrome(s): cleaned = ''.join(c.lower() for c in s if c.isalnum()); return cleaned == cleaned[::-1]"""
        body = [
            ast.Expr(value=ast.Constant(value=spec.docstring)),
            ast.Assign(
                targets=[ast.Name(id="cleaned", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Constant(value=""),
                        attr="join",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.GeneratorExp(
                            elt=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id="c", ctx=ast.Load()),
                                    attr="lower",
                                    ctx=ast.Load(),
                                ),
                                args=[],
                                keywords=[],
                            ),
                            generators=[
                                ast.comprehension(
                                    target=ast.Name(id="c", ctx=ast.Store()),
                                    iter=ast.Name(id="s", ctx=ast.Load()),
                                    ifs=[
                                        ast.Call(
                                            func=ast.Attribute(
                                                value=ast.Name(id="c", ctx=ast.Load()),
                                                attr="isalnum",
                                                ctx=ast.Load(),
                                            ),
                                            args=[],
                                            keywords=[],
                                        )
                                    ],
                                    is_async=0,
                                )
                            ],
                        )
                    ],
                    keywords=[],
                ),
            ),
            ast.Return(
                value=ast.Compare(
                    left=ast.Name(id="cleaned", ctx=ast.Load()),
                    ops=[ast.Eq()],
                    comparators=[
                        ast.Subscript(
                            value=ast.Name(id="cleaned", ctx=ast.Load()),
                            slice=ast.Slice(
                                lower=None,
                                upper=None,
                                step=ast.UnaryOp(
                                    op=ast.USub(), operand=ast.Constant(value=1)
                                ),
                            ),
                            ctx=ast.Load(),
                        )
                    ],
                )
            ),
        ]
        return self._make_func(spec.name, list(spec.params), body)

    def _build_gcd(self, spec: PrimitiveSpec) -> ast.FunctionDef:
        """def gcd(a, b): while b != 0: a, b = b, a % b; return abs(a)"""
        body = [
            ast.Expr(value=ast.Constant(value=spec.docstring)),
            ast.While(
                test=ast.Compare(
                    left=ast.Name(id="b", ctx=ast.Load()),
                    ops=[ast.NotEq()],
                    comparators=[ast.Constant(value=0)],
                ),
                body=[
                    ast.Assign(
                        targets=[
                            ast.Tuple(
                                elts=[
                                    ast.Name(id="a", ctx=ast.Store()),
                                    ast.Name(id="b", ctx=ast.Store()),
                                ],
                                ctx=ast.Store(),
                            )
                        ],
                        value=ast.Tuple(
                            elts=[
                                ast.Name(id="b", ctx=ast.Load()),
                                ast.BinOp(
                                    left=ast.Name(id="a", ctx=ast.Load()),
                                    op=ast.Mod(),
                                    right=ast.Name(id="b", ctx=ast.Load()),
                                ),
                            ],
                            ctx=ast.Load(),
                        ),
                    )
                ],
                orelse=[],
            ),
            ast.Return(
                value=ast.Call(
                    func=ast.Name(id="abs", ctx=ast.Load()),
                    args=[ast.Name(id="a", ctx=ast.Load())],
                    keywords=[],
                )
            ),
        ]
        return self._make_func(spec.name, list(spec.params), body)

    def _build_sum_list(self, spec: PrimitiveSpec) -> ast.FunctionDef:
        """def sum_list(numbers): total = 0; for n in numbers: total += n; return total"""
        body = [
            ast.Expr(value=ast.Constant(value=spec.docstring)),
            ast.Assign(
                targets=[ast.Name(id="total", ctx=ast.Store())],
                value=ast.Constant(value=0),
            ),
            ast.For(
                target=ast.Name(id="n", ctx=ast.Store()),
                iter=ast.Name(id="numbers", ctx=ast.Load()),
                body=[
                    ast.AugAssign(
                        target=ast.Name(id="total", ctx=ast.Store()),
                        op=ast.Add(),
                        value=ast.Name(id="n", ctx=ast.Load()),
                    )
                ],
                orelse=[],
            ),
            ast.Return(value=ast.Name(id="total", ctx=ast.Load())),
        ]
        return self._make_func(spec.name, list(spec.params), body)

    def _build_max_element(self, spec: PrimitiveSpec) -> ast.FunctionDef:
        """def max_element(lst): if not lst: raise ValueError(...); m = lst[0]; for x in lst[1:]: if x > m: m = x; return m"""
        body = [
            ast.Expr(value=ast.Constant(value=spec.docstring)),
            ast.If(
                test=ast.UnaryOp(
                    op=ast.Not(),
                    operand=ast.Name(id="lst", ctx=ast.Load()),
                ),
                body=[
                    ast.Raise(
                        exc=ast.Call(
                            func=ast.Name(id="ValueError", ctx=ast.Load()),
                            args=[ast.Constant(value="list cannot be empty")],
                            keywords=[],
                        )
                    )
                ],
                orelse=[],
            ),
            ast.Assign(
                targets=[ast.Name(id="m", ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Name(id="lst", ctx=ast.Load()),
                    slice=ast.Constant(value=0),
                    ctx=ast.Load(),
                ),
            ),
            ast.For(
                target=ast.Name(id="x", ctx=ast.Store()),
                iter=ast.Subscript(
                    value=ast.Name(id="lst", ctx=ast.Load()),
                    slice=ast.Slice(lower=ast.Constant(value=1), upper=None, step=None),
                    ctx=ast.Load(),
                ),
                body=[
                    ast.If(
                        test=ast.Compare(
                            left=ast.Name(id="x", ctx=ast.Load()),
                            ops=[ast.Gt()],
                            comparators=[ast.Name(id="m", ctx=ast.Load())],
                        ),
                        body=[
                            ast.Assign(
                                targets=[ast.Name(id="m", ctx=ast.Store())],
                                value=ast.Name(id="x", ctx=ast.Load()),
                            )
                        ],
                        orelse=[],
                    )
                ],
                orelse=[],
            ),
            ast.Return(value=ast.Name(id="m", ctx=ast.Load())),
        ]
        return self._make_func(spec.name, list(spec.params), body)

    def _build_binary_search(self, spec: PrimitiveSpec) -> ast.FunctionDef:
        """def binary_search(arr, target): low, high = 0, len(arr) - 1; while low <= high: ..."""
        body = [
            ast.Expr(value=ast.Constant(value=spec.docstring)),
            ast.Assign(
                targets=[
                    ast.Tuple(
                        elts=[
                            ast.Name(id="low", ctx=ast.Store()),
                            ast.Name(id="high", ctx=ast.Store()),
                        ],
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Tuple(
                    elts=[
                        ast.Constant(value=0),
                        ast.BinOp(
                            left=ast.Call(
                                func=ast.Name(id="len", ctx=ast.Load()),
                                args=[ast.Name(id="arr", ctx=ast.Load())],
                                keywords=[],
                            ),
                            op=ast.Sub(),
                            right=ast.Constant(value=1),
                        ),
                    ],
                    ctx=ast.Load(),
                ),
            ),
            ast.While(
                test=ast.Compare(
                    left=ast.Name(id="low", ctx=ast.Load()),
                    ops=[ast.LtE()],
                    comparators=[ast.Name(id="high", ctx=ast.Load())],
                ),
                body=[
                    ast.Assign(
                        targets=[ast.Name(id="mid", ctx=ast.Store())],
                        value=ast.BinOp(
                            left=ast.BinOp(
                                left=ast.Name(id="low", ctx=ast.Load()),
                                op=ast.Add(),
                                right=ast.Name(id="high", ctx=ast.Load()),
                            ),
                            op=ast.FloorDiv(),
                            right=ast.Constant(value=2),
                        ),
                    ),
                    ast.If(
                        test=ast.Compare(
                            left=ast.Subscript(
                                value=ast.Name(id="arr", ctx=ast.Load()),
                                slice=ast.Name(id="mid", ctx=ast.Load()),
                                ctx=ast.Load(),
                            ),
                            ops=[ast.Eq()],
                            comparators=[ast.Name(id="target", ctx=ast.Load())],
                        ),
                        body=[ast.Return(value=ast.Name(id="mid", ctx=ast.Load()))],
                        orelse=[
                            ast.If(
                                test=ast.Compare(
                                    left=ast.Subscript(
                                        value=ast.Name(id="arr", ctx=ast.Load()),
                                        slice=ast.Name(id="mid", ctx=ast.Load()),
                                        ctx=ast.Load(),
                                    ),
                                    ops=[ast.Lt()],
                                    comparators=[ast.Name(id="target", ctx=ast.Load())],
                                ),
                                body=[
                                    ast.Assign(
                                        targets=[ast.Name(id="low", ctx=ast.Store())],
                                        value=ast.BinOp(
                                            left=ast.Name(id="mid", ctx=ast.Load()),
                                            op=ast.Add(),
                                            right=ast.Constant(value=1),
                                        ),
                                    )
                                ],
                                orelse=[
                                    ast.Assign(
                                        targets=[ast.Name(id="high", ctx=ast.Store())],
                                        value=ast.BinOp(
                                            left=ast.Name(id="mid", ctx=ast.Load()),
                                            op=ast.Sub(),
                                            right=ast.Constant(value=1),
                                        ),
                                    )
                                ],
                            )
                        ],
                    ),
                ],
                orelse=[],
            ),
            ast.Return(value=ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1))),
        ]
        return self._make_func(spec.name, list(spec.params), body)

    def _build_flatten(self, spec: PrimitiveSpec) -> ast.FunctionDef:
        """def flatten(nested_list): return [item for sublist in nested_list for item in sublist]"""
        body = [
            ast.Expr(value=ast.Constant(value=spec.docstring)),
            ast.Return(
                value=ast.ListComp(
                    elt=ast.Name(id="item", ctx=ast.Load()),
                    generators=[
                        ast.comprehension(
                            target=ast.Name(id="sublist", ctx=ast.Store()),
                            iter=ast.Name(id="nested_list", ctx=ast.Load()),
                            ifs=[],
                            is_async=0,
                        ),
                        ast.comprehension(
                            target=ast.Name(id="item", ctx=ast.Store()),
                            iter=ast.Name(id="sublist", ctx=ast.Load()),
                            ifs=[],
                            is_async=0,
                        ),
                    ],
                )
            ),
        ]
        return self._make_func(spec.name, list(spec.params), body)

    def _make_func(
        self, name: str, params: list[str], body: list[ast.stmt]
    ) -> ast.FunctionDef:
        args = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=p) for p in params],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )
        func = ast.FunctionDef(
            name=name,
            args=args,
            body=body,
            decorator_list=[],
            returns=None,
        )
        ast.fix_missing_locations(func)
        return func
