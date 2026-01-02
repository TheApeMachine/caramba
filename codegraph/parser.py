from __future__ import annotations

import ast
import subprocess
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class Node:
    label: str
    id: str
    props: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Edge:
    rel: str
    src: str
    dst: str
    props: dict[str, Any]


def _module_name_from_rel(rel: str) -> str:
    rel = rel.replace("\\", "/")
    if rel.endswith(".py"):
        rel = rel[: -len(".py")]
    # Treat __init__ as the package itself.
    if rel.endswith("/__init__"):
        rel = rel[: -len("/__init__")]
    core = rel.replace("/", ".").strip(".")
    # This repo's import root is `caramba` (package-dir points at repo root).
    return ("caramba" if not core else f"caramba.{core}").strip(".")


def _qualname(module: str, *parts: str) -> str:
    return ".".join([module, *[p for p in parts if p]]).strip(".")


def _node_id(kind: str, qualname: str) -> str:
    return f"py:{kind}:{qualname}"

def _symbolref_id(scope_module: str, qualname: str) -> str:
    """Symbol reference node id, scoped to the *caller module*.

    We scope unresolved symbols to the caller module to avoid collisions across files
    (important for incremental sync + file-scoped deletes).
    """
    return f"py:symbolref:{scope_module}:{qualname}"

def _is_internal_qual(qualname: str) -> bool:
    return str(qualname or "").startswith("caramba.")


_DEFAULT_EXTERNAL_ALLOWLIST = ("builtins.",)


def _include_external() -> bool:
    # Set to 1 to keep external (stdlib/third-party) calls in the graph.
    return (os.getenv("CARAMBA_CODEGRAPH_INCLUDE_EXTERNAL") or "").strip() in {"1", "true", "yes", "on"}


_BUILTIN_CALL_NAMES = {
    # Extremely common noise / non-actionable for code understanding.
    "len",
    "range",
    "print",
    "str",
    "int",
    "float",
    "bool",
    "dict",
    "list",
    "set",
    "tuple",
    "sorted",
    "min",
    "max",
    "sum",
    "any",
    "all",
    "enumerate",
    "zip",
    "map",
    "filter",
    # Exceptions (constructors); usually noise for call graph purposes.
    "Exception",
    "ValueError",
    "TypeError",
    "RuntimeError",
    "KeyError",
    "IndexError",
    "AssertionError",
    "NotImplementedError",
}


def _iter_py_files(repo_root: Path, files: list[str] | None) -> list[Path]:
    if files:
        return [repo_root / f for f in files]
    # Full scan: prefer git-tracked files (respects .gitignore), otherwise fallback.
    try:
        proc = subprocess.run(
            ["git", "ls-files", "*.py"],
            cwd=str(repo_root),
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0:
            rels = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
            return [repo_root / r for r in rels]
    except Exception:
        pass
    return sorted(repo_root.rglob("*.py"))


def _safe_parse(text: str, *, filename: str) -> ast.AST | None:
    try:
        return ast.parse(text, filename=filename)
    except SyntaxError:
        return None


def _import_targets(node: ast.AST) -> Iterable[str]:
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name:
                yield alias.name
    elif isinstance(node, ast.ImportFrom):
        mod = node.module or ""
        if mod:
            yield mod


def _expr_name(expr: ast.AST) -> str:
    """Best-effort string name for an expression (for calls/bases)."""
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return f"{_expr_name(expr.value)}.{expr.attr}".strip(".")
    if isinstance(expr, ast.Subscript):
        return _expr_name(expr.value)
    if isinstance(expr, ast.Call):
        return _expr_name(expr.func)
    if isinstance(expr, ast.Constant):
        return str(expr.value)
    return expr.__class__.__name__

def _module_package(module: str) -> list[str]:
    parts = [p for p in (module or "").split(".") if p]
    return parts[:-1]  # parent package


def _resolve_relative(module: str, *, level: int, modname: str | None) -> str | None:
    """Resolve `from .x import y` style imports to absolute-ish module names."""
    # Absolute import (`from x import y`): no package prefixing.
    if level == 0:
        return modname
    pkg = _module_package(module)
    if level <= 0:
        base = pkg
    else:
        base = pkg[: max(0, len(pkg) - (level - 1))]
    if not base and not modname:
        return None
    if modname:
        return ".".join([*base, modname])
    return ".".join(base) if base else None


class _CallCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def visit_Call(self, node: ast.Call) -> Any:  # noqa: N802
        self.calls.append(_expr_name(node.func))
        self.generic_visit(node)


class _TypeHintCollector(ast.NodeVisitor):
    """Best-effort local type bindings from simple assignments.

    We only use deterministic, syntax-local signals:
    - x = SomeClass(...)
    - self.x = SomeClass(...)
    - x = module_alias.SomeClass(...)
    - x = imported_symbol(...) where imported_symbol resolved to module.Thing
    """

    def __init__(
        self,
        *,
        module: str,
        local_classes: set[str],
        import_mod_alias: dict[str, str],
        import_symbol_alias: dict[str, str],
    ) -> None:
        self.module = module
        self.local_classes = local_classes
        self.import_mod_alias = import_mod_alias
        self.import_symbol_alias = import_symbol_alias
        self.local_var_types: dict[str, str] = {}
        self.self_attr_types: dict[str, str] = {}

    def _resolve_ctor(self, func_expr: ast.AST) -> str | None:
        name = _expr_name(func_expr).strip()
        if not name:
            return None

        # Local class constructor
        if "." not in name and name in self.local_classes:
            return _qualname(self.module, name)

        # imported symbol constructor: Foo(...) where Foo -> pkg.mod.Foo
        if "." not in name and name in self.import_symbol_alias:
            return self.import_symbol_alias[name]

        # module alias: m.Foo(...)
        if "." in name:
            head, tail = name.split(".", 1)
            if head in self.import_mod_alias:
                return f"{self.import_mod_alias[head]}.{tail}"
            if head in self.import_symbol_alias:
                return f"{self.import_symbol_alias[head]}.{tail}"
        return None

    def visit_Assign(self, node: ast.Assign) -> Any:  # noqa: N802
        # Only handle simple x = Call(...)
        if isinstance(node.value, ast.Call):
            ctor = self._resolve_ctor(node.value.func)
            if ctor:
                for t in node.targets:
                    # x = ...
                    if isinstance(t, ast.Name) and t.id:
                        self.local_var_types[t.id] = ctor
                    # self.x = ...
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                        if t.attr:
                            self.self_attr_types[t.attr] = ctor
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:  # noqa: N802
        # x: T = Call(...)
        if node.value and isinstance(node.value, ast.Call):
            ctor = self._resolve_ctor(node.value.func)
            if ctor:
                t = node.target
                if isinstance(t, ast.Name) and t.id:
                    self.local_var_types[t.id] = ctor
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                    if t.attr:
                        self.self_attr_types[t.attr] = ctor
        self.generic_visit(node)


def parse_python_file(repo_root: Path, file_path: Path) -> tuple[list[Node], list[Edge]]:
    """Parse one python file into nodes/edges (deterministic, best-effort)."""
    rel = str(file_path.relative_to(repo_root)).replace("\\", "/")
    module = _module_name_from_rel(rel)

    try:
        txt = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        txt = ""
    tree = _safe_parse(txt, filename=rel)
    if tree is None:
        return [], []

    nodes: list[Node] = []
    edges: list[Edge] = []

    mod_id = _node_id("module", module)
    nodes.append(Node("Module", mod_id, {"name": module, "file": rel}))

    # Import bindings for basic cross-module resolution.
    # - import x.y as z => z -> "x.y"
    # - from x.y import a as b => b -> "x.y.a"
    import_mod_alias: dict[str, str] = {}
    import_symbol_alias: dict[str, str] = {}

    # Imports
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for alias in n.names:
                if not alias.name:
                    continue
                tgt = str(alias.name)
                imp_id = _node_id("module", tgt)
                nodes.append(Node("Module", imp_id, {"name": tgt}))
                edges.append(Edge("IMPORTS", mod_id, imp_id, {"file": rel}))
                bind = alias.asname or tgt.split(".", 1)[0]
                if bind:
                    import_mod_alias[bind] = tgt
        elif isinstance(n, ast.ImportFrom):
            resolved_mod = _resolve_relative(module, level=int(getattr(n, "level", 0) or 0), modname=n.module)
            if not resolved_mod:
                continue
            imp_id = _node_id("module", resolved_mod)
            nodes.append(Node("Module", imp_id, {"name": resolved_mod}))
            edges.append(Edge("IMPORTS", mod_id, imp_id, {"file": rel, "from": True}))
            for alias in n.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                if not local:
                    continue
                import_symbol_alias[local] = f"{resolved_mod}.{alias.name}"

    # Definitions + basic calls
    top_level_funcs: set[str] = set()
    class_methods: dict[str, set[str]] = {}
    local_classes: set[str] = set()

    for stmt in getattr(tree, "body", []) or []:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top_level_funcs.add(stmt.name)
        if isinstance(stmt, ast.ClassDef):
            local_classes.add(stmt.name)
            class_methods[stmt.name] = set(
                s.name for s in stmt.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))
            )

    # Top-level defs
    for stmt in getattr(tree, "body", []) or []:
        if isinstance(stmt, ast.ClassDef):
            qn = _qualname(module, stmt.name)
            cid = _node_id("class", qn)
            bases = [_expr_name(b) for b in (stmt.bases or [])]
            nodes.append(
                Node(
                    "Class",
                    cid,
                    {
                        "name": stmt.name,
                        "qualname": qn,
                        "file": rel,
                        "lineno": int(getattr(stmt, "lineno", 0) or 0),
                        "bases": bases,
                    },
                )
            )
            edges.append(Edge("DEFINES", mod_id, cid, {"file": rel}))
            for b in bases:
                if not b:
                    continue
                # Resolve base through imports when possible.
                base_qual = import_symbol_alias.get(b, b)
                if "." not in base_qual and base_qual in local_classes:
                    base_qual = _qualname(module, base_qual)

                # Prefer internal inheritance edges; external bases are optional and usually noisy.
                if not _is_internal_qual(base_qual) and not _include_external():
                    continue

                # Leave as "symbol-like" id; a repo-level pass can rewrite to an actual class node if present.
                bid = _symbolref_id(module, base_qual)
                nodes.append(Node("Symbol", bid, {"name": b, "qualname": base_qual, "file": rel}))
                edges.append(Edge("INHERITS", cid, bid, {"file": rel, "base": base_qual}))

            # Methods
            for s in stmt.body:
                if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                mqn = _qualname(module, stmt.name, s.name)
                mid = _node_id("method", mqn)
                nodes.append(
                    Node(
                        "Method",
                        mid,
                        {
                            "name": s.name,
                            "qualname": mqn,
                            "file": rel,
                            "lineno": int(getattr(s, "lineno", 0) or 0),
                            "class": qn,
                        },
                    )
                )
                edges.append(Edge("OWNS", cid, mid, {"file": rel}))

                # Collect local type bindings for better call resolution.
                th = _TypeHintCollector(
                    module=module,
                    local_classes=local_classes,
                    import_mod_alias=import_mod_alias,
                    import_symbol_alias=import_symbol_alias,
                )
                th.visit(s)

                cc = _CallCollector()
                cc.visit(s)
                for callee in cc.calls:
                    # Resolve a few common patterns:
                    # - self.method -> method in same class
                    # - ClassName.method -> method in same module/class
                    # - module_alias.func -> imported module function
                    # - imported_symbol(...) -> imported function/class constructor
                    resolved_callee_qual: str | None = None

                    if callee.startswith("self."):
                        parts = callee.split(".")
                        if len(parts) == 2:
                            meth = parts[1]
                            if meth in class_methods.get(stmt.name, set()):
                                callee_qual = _qualname(module, stmt.name, meth)
                                dst = _node_id("method", callee_qual)
                                edges.append(
                                    Edge(
                                        "CALLS",
                                        mid,
                                        dst,
                                        {"file": rel, "callee": callee, "callee_qual": callee_qual},
                                    )
                                )
                                continue
                        # Resolve self.attr.method(...) using inferred self attr types.
                        if len(parts) >= 3:
                            attr = parts[1]
                            rest = ".".join(parts[2:])
                            owner = th.self_attr_types.get(attr)
                            if owner:
                                resolved_callee_qual = f"{owner}.{rest}"
                                sym_id = _symbolref_id(module, resolved_callee_qual)
                                nodes.append(
                                    Node(
                                        "Symbol",
                                        sym_id,
                                        {"name": callee, "qualname": resolved_callee_qual, "file": rel},
                                    )
                                )
                                edges.append(
                                    Edge(
                                        "CALLS",
                                        mid,
                                        sym_id,
                                        {"file": rel, "callee": callee, "callee_qual": resolved_callee_qual},
                                    )
                                )
                                continue

                    if "." in callee:
                        head, tail = callee.split(".", 1)
                        if head in local_classes and tail in class_methods.get(head, set()):
                            callee_qual = _qualname(module, head, tail)
                            dst = _node_id("method", callee_qual)
                            edges.append(Edge("CALLS", mid, dst, {"file": rel, "callee": callee, "callee_qual": callee_qual}))
                            continue
                        if head in import_mod_alias:
                            resolved_callee_qual = f"{import_mod_alias[head]}.{tail}"
                        elif head in import_symbol_alias:
                            # Imported symbol then attribute (rare, but keep).
                            resolved_callee_qual = f"{import_symbol_alias[head]}.{tail}"
                        elif head in th.local_var_types:
                            # Variable instance method call: x.foo -> <TypeOfX>.foo
                            resolved_callee_qual = f"{th.local_var_types[head]}.{tail}"
                        else:
                            # Unknown attribute chain (often string/list operations like `target.strip()`).
                            # Drop by default to keep the graph high-signal.
                            continue
                    else:
                        if callee in top_level_funcs:
                            resolved_callee_qual = _qualname(module, callee)
                        elif callee in local_classes:
                            # Constructor call
                            resolved_callee_qual = _qualname(module, callee)
                        elif callee in import_symbol_alias:
                            resolved_callee_qual = import_symbol_alias[callee]
                        else:
                            # Unknown bare call; usually builtin or dynamic. Keep only if explicitly requested.
                            if callee in _BUILTIN_CALL_NAMES:
                                continue
                            continue

                    if resolved_callee_qual:
                        # Prefer internal calls; optionally keep external calls if enabled.
                        if not _is_internal_qual(resolved_callee_qual) and not _include_external():
                            # Allow a tiny external allowlist (builtins.*) if it ever appears qualified.
                            if not any(str(resolved_callee_qual).startswith(p) for p in _DEFAULT_EXTERNAL_ALLOWLIST):
                                continue
                        # Use a symbol id for now; repo pass will rewrite to concrete def ids when available.
                        sym_id = _symbolref_id(module, resolved_callee_qual)
                        nodes.append(
                            Node(
                                "Symbol",
                                sym_id,
                                {"name": callee, "qualname": resolved_callee_qual, "file": rel},
                            )
                        )
                        edges.append(Edge("CALLS", mid, sym_id, {"file": rel, "callee": callee, "callee_qual": resolved_callee_qual}))
                    else:
                        sym_qn = _qualname(module, callee)
                        sym_id = _symbolref_id(module, sym_qn)
                        nodes.append(Node("Symbol", sym_id, {"name": callee, "qualname": sym_qn, "file": rel}))
                        edges.append(Edge("CALLS", mid, sym_id, {"file": rel, "callee": callee, "callee_qual": sym_qn, "unresolved": True}))

        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qn = _qualname(module, stmt.name)
            fid = _node_id("function", qn)
            nodes.append(
                Node(
                    "Function",
                    fid,
                    {
                        "name": stmt.name,
                        "qualname": qn,
                        "file": rel,
                        "lineno": int(getattr(stmt, "lineno", 0) or 0),
                    },
                )
            )
            edges.append(Edge("DEFINES", mod_id, fid, {"file": rel}))

            th2 = _TypeHintCollector(
                module=module,
                local_classes=local_classes,
                import_mod_alias=import_mod_alias,
                import_symbol_alias=import_symbol_alias,
            )
            th2.visit(stmt)

            cc = _CallCollector()
            cc.visit(stmt)
            for callee in cc.calls:
                callee_qual: str | None = None
                if "." in callee:
                    head, tail = callee.split(".", 1)
                    if head in import_mod_alias:
                        callee_qual = f"{import_mod_alias[head]}.{tail}"
                    elif head in import_symbol_alias:
                        callee_qual = f"{import_symbol_alias[head]}.{tail}"
                    elif head in th2.local_var_types:
                        callee_qual = f"{th2.local_var_types[head]}.{tail}"
                    else:
                        continue
                else:
                    if callee in top_level_funcs:
                        callee_qual = _qualname(module, callee)
                    elif callee in local_classes:
                        callee_qual = _qualname(module, callee)
                    elif callee in import_symbol_alias:
                        callee_qual = import_symbol_alias[callee]
                    else:
                        if callee in _BUILTIN_CALL_NAMES:
                            continue
                        continue

                if callee_qual:
                    if not _is_internal_qual(callee_qual) and not _include_external():
                        if not any(str(callee_qual).startswith(p) for p in _DEFAULT_EXTERNAL_ALLOWLIST):
                            continue
                    sym_id = _symbolref_id(module, callee_qual)
                    nodes.append(Node("Symbol", sym_id, {"name": callee, "qualname": callee_qual, "file": rel}))
                    edges.append(Edge("CALLS", fid, sym_id, {"file": rel, "callee": callee, "callee_qual": callee_qual}))
                else:
                    sym_qn = _qualname(module, callee)
                    sym_id = _symbolref_id(module, sym_qn)
                    nodes.append(Node("Symbol", sym_id, {"name": callee, "qualname": sym_qn, "file": rel}))
                    edges.append(Edge("CALLS", fid, sym_id, {"file": rel, "callee": callee, "callee_qual": sym_qn, "unresolved": True}))

    return nodes, edges


def parse_repo(repo_root: str | Path, *, files: list[str] | None = None) -> tuple[list[Node], list[Edge]]:
    root = Path(repo_root).resolve()
    all_nodes: list[Node] = []
    all_edges: list[Edge] = []

    for p in _iter_py_files(root, files):
        if not p.exists() or not p.is_file():
            continue
        nodes, edges = parse_python_file(root, p)
        all_nodes.extend(nodes)
        all_edges.extend(edges)

    # Repo-level definition index: qualname -> strongest known node id.
    # This lets us rewrite CALLS/INHERITS edges that currently point at Symbol ids.
    def_index: dict[str, str] = {}
    for n in all_nodes:
        qn = str(n.props.get("qualname", "") or "")
        if not qn:
            continue
        # Prefer concrete defs over symbols.
        if n.label in {"Function", "Method", "Class", "Module"}:
            def_index[qn] = n.id

    rewritten_edges: list[Edge] = []
    for e in all_edges:
        if e.rel in {"CALLS", "INHERITS"} and e.dst.startswith("py:symbolref:"):
            qn = str(e.props.get("callee_qual") or e.props.get("base") or "")
            if qn and qn in def_index:
                rewritten_edges.append(Edge(e.rel, e.src, def_index[qn], dict(e.props)))
                continue
        rewritten_edges.append(e)
    all_edges = rewritten_edges

    # Deduplicate by id/rel (prefer first occurrence, deterministic order)
    seen_nodes: set[str] = set()
    uniq_nodes: list[Node] = []
    for n in all_nodes:
        if n.id in seen_nodes:
            continue
        seen_nodes.add(n.id)
        uniq_nodes.append(n)

    seen_edges: set[tuple[str, str, str, str]] = set()
    uniq_edges: list[Edge] = []
    for e in all_edges:
        k = (e.rel, e.src, e.dst, str(e.props.get("file", "")))
        if k in seen_edges:
            continue
        seen_edges.add(k)
        uniq_edges.append(e)

    return uniq_nodes, uniq_edges

