"""Restricted computational Python in a resource-bounded child, not OS isolation."""
from __future__ import annotations

import ast
import inspect
import subprocess
import sys
import tempfile

SAFE_BUILTINS = (['repr', 'abs', 'all', 'any', 'bool', 'bytes', 'dict', 'enumerate', 'filter', 'float', 'int', 'len', 'list', 'map', 'max', 'min', 'range', 'reversed', 'round', 'set', 'sorted', 'str', 'sum', 'tuple', 'zip', 'print', 'isinstance', 'type', 'Exception', 'ValueError', 'ZeroDivisionError', 'AssertionError'])
SAFE_ATTRIBUTES = {'sqrt', 'isqrt', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh', 'exp', 'log', 'log10', 'log2', 'ceil', 'floor', 'trunc', 'fabs', 'factorial', 'gcd', 'lcm', 'isclose', 'isfinite', 'isnan', 'isinf', 'degrees', 'radians', 'pi', 'e', 'tau', 'inf', 'nan', 'pow', 'prod', 'fsum', 'copysign', 'hypot', 'comb', 'perm', 'remainder', 'modf', 'frexp', 'ldexp', 'dist', 'sha256', 'sha512', 'sha1', 'md5', 'digest', 'hexdigest', 'update', 'copy', 'chain', 'combinations', 'permutations', 'product', 'repeat', 'cycle', 'count', 'islice', 'zip_longest', 'accumulate', 'starmap', 'takewhile', 'dropwhile', 'groupby', 'tee', 'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'neg', 'pos', 'parse', 'walk', 'Expression', 'Constant', 'Num', 'BinOp', 'UnaryOp', 'Add', 'Sub', 'Mult', 'Div', 'FloorDiv', 'Mod', 'Pow', 'USub', 'UAdd', 'body', 'left', 'right', 'operand', 'op', 'value', 'n', 'id', 'Symbol', 'solve', 'diff', 'simplify', 'is_real', 'is_integer', 'is_Integer', 'append', 'extend', 'pop', 'remove', 'sort', 'reverse', 'index', 'keys', 'values', 'items', 'get', 'setdefault', 'split', 'rsplit', 'strip', 'lstrip', 'rstrip', 'replace', 'join', 'lower', 'upper', 'title', 'capitalize', 'casefold', 'startswith', 'endswith', 'isdigit', 'isalpha', 'isalnum', 'encode', 'decode', 'bit_length', 'real', 'imag', 'numerator', 'denominator'}
SAFE_MODULES = {'math', 'hashlib', 'itertools', 'operator', 'ast', 'sympy'}
# Positive syntax policy; reflection, class creation, async and exception traceback access excluded.
SAFE_NODES = {'Module', 'Expression', 'Expr', 'Constant', 'Name', 'Load', 'Store', 'Del', 'Assign', 'AnnAssign', 'AugAssign', 'FunctionDef', 'arguments', 'arg', 'Return', 'Raise', 'If', 'IfExp', 'For', 'While', 'Break', 'Continue', 'Pass', 'List', 'Tuple', 'Set', 'Dict', 'ListComp', 'SetComp', 'DictComp', 'GeneratorExp', 'comprehension', 'BinOp', 'UnaryOp', 'BoolOp', 'Compare', 'Call', 'keyword', 'Attribute', 'Subscript', 'Slice', 'JoinedStr', 'FormattedValue', 'Assert', 'Import', 'ImportFrom', 'alias', 'Add', 'Sub', 'Mult', 'Div', 'FloorDiv', 'Mod', 'Pow', 'UAdd', 'USub', 'Not', 'And', 'Or', 'Eq', 'NotEq', 'Lt', 'LtE', 'Gt', 'GtE', 'In', 'NotIn', 'Is', 'IsNot', 'BitAnd', 'BitOr', 'BitXor', 'LShift', 'RShift'}


def validate(code: str) -> tuple[bool, str | None]:
    if len(code) > 100000:
        return False, 'Code size limit exceeded'
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError) as exc:
        return False, f'SyntaxError: {exc}'
    nodes = list(ast.walk(tree))
    if len(nodes) > 10000:
        return False, 'AST size limit exceeded'
    for node in nodes:
        kind = type(node).__name__
        if kind not in SAFE_NODES:
            return False, f'Forbidden AST node {kind}'
        if isinstance(node, ast.Name) and (node.id.startswith('__') or node.id in {'open', 'eval', 'exec', 'compile', 'globals', 'locals', 'getattr', 'setattr', 'vars', 'input', 'help', 'breakpoint', 'attrgetter', 'methodcaller'}):
            return False, f'Forbidden identifier {node.id}'
        if isinstance(node, ast.Attribute) and node.attr not in SAFE_ATTRIBUTES:
            return False, f'Forbidden attribute {node.attr}'
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and any(a.name.startswith('_') or a.name in {'attrgetter', 'methodcaller'} for a in node.names):
                return False, 'Forbidden import capability'
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            if getattr(node, 'level', 0) or any(name not in SAFE_MODULES for name in names):
                return False, f'Forbidden {kind}: imports restricted to computational modules'
    return True, None


_BWRAP_PREFIX: list[str] | None = None


def _get_bwrap_prefix() -> list[str]:
    global _BWRAP_PREFIX
    if _BWRAP_PREFIX is not None:
        return _BWRAP_PREFIX
    import shutil
    bwrap = shutil.which('bwrap')
    if not bwrap:
        _BWRAP_PREFIX = []
        return _BWRAP_PREFIX
    candidate = [
        bwrap,
        '--ro-bind', '/', '/',
        '--tmpfs', '/tmp',
        '--unshare-net',
        '--unshare-ipc',
        '--unshare-pid',
        '--die-with-parent',
        '--chdir', '/tmp',
    ]
    try:
        probe = subprocess.run(
            [*candidate, sys.executable, '-I', '-B', '-c', 'pass'],
            capture_output=True,
            text=True,
            timeout=2,
            close_fds=True,
            check=False,
        )
        _BWRAP_PREFIX = candidate if probe.returncode == 0 else []
    except OSError:
        _BWRAP_PREFIX = []
    return _BWRAP_PREFIX


def run(code: str, *, timeout: float = 3, memory_mb: int = 64, output_bytes: int = 65536, builtins: list[str] | None = None, modules: list[str] | None = None) -> subprocess.CompletedProcess:
    safe, error = validate(code)
    if not safe:
        return subprocess.CompletedProcess([], 1, '', error)
    names = [n for n in (builtins if builtins is not None else SAFE_BUILTINS) if n in SAFE_BUILTINS]
    mods = [
        m
        for m in (modules or [])
        if m in SAFE_MODULES and (m != 'sympy' or 'sympy' in code or 'parse_expr' in code)
    ]
    cpu_seconds = max(1, int(timeout) + 1)
    harness = '''import builtins, importlib, os, resource, sys
resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))
resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT, OUTPUT))
resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
class BoundedOutput:
    def __init__(self, target): self.target, self.remaining = target, OUTPUT
    def write(self, value):
        self.remaining -= len(value.encode('utf-8'))
        if self.remaining < 0: raise ValueError('Output budget exceeded')
        return self.target.write(value)
    def flush(self): self.target.flush()
sys.stdout = BoundedOutput(sys.stdout)
sys.stderr = BoundedOutput(sys.stderr)
def permitted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name not in MODULES: raise ImportError('Import forbidden')
    if name in {'ast', 'sympy', 'operator'}: raise ImportError('Module is only available through a restricted skill facade')
    return importlib.import_module(name)
from types import SimpleNamespace
facades = {}
for module_name, members in {
    'sympy': ['Symbol', 'solve', 'diff', 'simplify', 'sin', 'cos', 'sqrt', 'pi'],
    'operator': ['add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'pow', 'neg', 'pos'],
    'ast': ['parse', 'walk', 'Expression', 'Constant', 'Num', 'BinOp', 'UnaryOp', 'Add', 'Sub', 'Mult', 'Div', 'FloorDiv', 'Mod', 'Pow', 'USub', 'UAdd'],
}.items():
    if module_name in PRELOAD:
        real = importlib.import_module(module_name)
        facades[module_name] = SimpleNamespace(**{name: getattr(real, name) for name in members})
if 'sympy' in facades:
    symbolic = importlib.import_module('sympy')
    def symbol(name):
        if not isinstance(name, str) or not name.isalpha() or len(name) > 16: raise ValueError('Invalid symbolic name')
        return symbolic.Symbol(name)
    def symbolic_operation(name):
        def apply(*args):
            if not args or not all(isinstance(arg, symbolic.Basic) for arg in args): raise ValueError('Symbolic operations require parsed expressions')
            return getattr(symbolic, name)(*args)
        return apply
    facades['sympy'] = SimpleNamespace(Symbol=symbol, **{name: symbolic_operation(name) for name in ['solve', 'diff', 'simplify']})
namespace = {'__builtins__' : {name: getattr(builtins, name) for name in NAMES}}
namespace['__builtins__']['__import__'] = permitted_import
for name in PRELOAD: namespace[name] = facades.get(name) or importlib.import_module(name)
if 'sympy' in PRELOAD:
    exec(SYMBOLIC_HELPER, globals())
    namespace['parse_expr'] = symbolic_expression
    namespace['sympy_transformations'] = ()
compiled_code = compile(CODE, '<restricted>', 'exec')
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(4300)
sys.setrecursionlimit(200)
try:
    os.chmod(os.getcwd(), 0o500)
except OSError:
    pass
if 'sympy' not in PRELOAD:
    for mod_name in MODULES - {'ast', 'sympy', 'operator'}:
        importlib.import_module(mod_name)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (4, 4))
    except (ValueError, OSError):
        pass
exec(compiled_code, namespace, namespace)
'''
    prefix = f'MEMORY={int(memory_mb)*1024*1024!r}\nOUTPUT={int(output_bytes)!r}\nCPU_SECONDS={cpu_seconds!r}\nMODULES={SAFE_MODULES!r}\nNAMES={names!r}\nPRELOAD={mods!r}\nCODE={code!r}\nSYMBOLIC_HELPER={inspect.getsource(symbolic_expression)!r}\n'
    import os
    bwrap_prefix = _get_bwrap_prefix()
    with tempfile.TemporaryDirectory(prefix='little-exec-') as directory:
        try:
            return subprocess.run(
                [*bwrap_prefix, sys.executable, '-I', '-B', '-c', prefix + harness],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=directory,
                env={'PYTHONHASHSEED': '0', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1'},
                close_fds=True,
                start_new_session=True,
                check=False,
            )
        finally:
            try:
                os.chmod(directory, 0o700)
            except OSError:
                pass


def symbolic_expression(text, transformations=()):
    """Construct bounded algebra directly from syntax nodes, without Python evaluation."""
    import ast
    import re

    import sympy
    text = str(text).strip().replace('^', '**')
    if len(text) > 4096:
        raise ValueError('Symbolic expression size limit exceeded')
    text = re.sub(r'(\d|\))\s*([A-Za-z(])', r'\1*\2', text)
    tree = ast.parse(text, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 256:
        raise ValueError('Symbolic complexity limit exceeded')
    functions = {name: getattr(sympy, name) for name in ('sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'Abs')}
    def build(node, depth=0):
        if depth > 32:
            raise ValueError('Symbolic depth limit exceeded')
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            literal = ast.get_source_segment(text, node)
            exponent = re.search(r"[eE]([+-]?\d+)$", literal)
            if exponent and abs(int(exponent.group(1))) > 1000:
                raise ValueError('Numeric literal exponent limit exceeded')
            return sympy.Rational(literal)
        if isinstance(node, ast.Name) and node.id.isalpha() and len(node.id) <= 16:
            return {'pi': sympy.pi, 'E': sympy.E}.get(node.id, sympy.Symbol(node.id))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return build(node.operand, depth + 1) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions and len(node.args) == 1 and not node.keywords:
            return functions[node.func.id](build(node.args[0], depth + 1))
        if isinstance(node, ast.BinOp):
            left, right = build(node.left, depth + 1), build(node.right, depth + 1)
            if isinstance(node.op, ast.Pow):
                if not right.is_Integer or abs(right) > 100:
                    raise ValueError('Symbolic exponent limit exceeded')
                return left ** right
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
        raise ValueError('Unsupported symbolic syntax')
    return build(tree.body)
