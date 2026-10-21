"""Scan the library's internal imports against ``apeiron.layers``.

``scan(root)`` returns ``(modules, crossings, unplaced)``: every library
module (dotted name -> level), every internal import that breaks the level
rules as ``"importer -> imported"``, and the modules no level is given to.
Imports inside functions count too: they are dependencies all the same.
Tests, scratch and the scanner itself are not library modules.
"""
import ast
from pathlib import Path

from apeiron.layers import RE_EXPORTS, level_of, may_import

SKIP_DIRS = {"tests", ".git", "__pycache__", "data", "docs", ".idea", "venv", ".venv"}
#: Not library modules: the suite and Gaia containers (docstrings only), the map and scanner.
SKIP_MODULES = {"apeiron", "apeiron.gaia", "apeiron.layers", "apeiron.scan", "__init__"}


def library_modules(root):
    """``{dotted module name: path}`` for the library's Python files."""
    root = Path(root)
    found = {}
    for path in root.rglob("*.py"):
        parts = path.relative_to(root).with_suffix("").parts
        if any(part in SKIP_DIRS or part.startswith(".") for part in parts[:-1]):
            continue
        if parts[0] in SKIP_DIRS:
            continue
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        if name and name not in SKIP_MODULES:
            found[name] = path
    return found


def _resolve(target, modules):
    """The library module an import of ``target`` reaches, or ``None`` (external)."""
    parts = target.split(".")
    for end in range(len(parts), 0, -1):
        candidate = ".".join(parts[:end])
        if candidate in modules:
            return RE_EXPORTS.get(candidate, candidate)
    return None


def imports_of(path, module, modules):
    """Library modules imported by the file at ``path`` (any depth in the file)."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    return _imports_in(tree, path, module, modules)


def scan(root):
    modules = library_modules(root)
    levels = {name: level_of(name) for name in modules}
    unplaced = sorted(name for name, level in levels.items() if level is None)
    crossings = set()
    for name, path in modules.items():
        importer = levels[name]
        if importer is None:
            continue
        for imported in imports_of(path, name, modules):
            target = levels.get(imported)
            if target is not None and not may_import(importer, target):
                crossings.add(f"{name} -> {imported}")
    return levels, sorted(crossings), unplaced


def import_graph(root, module_level_only=False):
    """``{module: set of library modules it imports}``. With
    ``module_level_only``, only imports run at load time (module body, incl.
    ``if``/``try`` blocks), not those inside functions or classes."""
    modules = library_modules(root)
    graph = {}
    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        if module_level_only:
            nodes, stack = [], list(tree.body)
            while stack:
                node = stack.pop()
                nodes.append(node)
                if isinstance(node, (ast.If, ast.Try)):
                    stack.extend(node.body + node.orelse + getattr(node, "finalbody", [])
                                 + [n for h in getattr(node, "handlers", []) for n in h.body])
            # Only the import statements themselves: walking a def or class
            # found here would reach the imports inside it.
            body = ast.Module(body=[n for n in nodes if isinstance(n, (ast.Import, ast.ImportFrom))],
                              type_ignores=[])
        else:
            body = tree
        graph[name] = _imports_in(body, path, name, modules)
    return graph


def _imports_in(tree, path, module, modules):
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found = set()
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".") if package else []
                anchor = anchor[:len(anchor) - (node.level - 1)] if node.level > 1 else anchor
                base = ".".join([*anchor, base] if base else anchor)
            targets = [base] + [f"{base}.{alias.name}" for alias in node.names]
        for target in targets:
            resolved = _resolve(target, modules)
            if resolved and resolved != module:
                found.add(resolved)
    return found


def cycles(root, module_level_only=False):
    """Import cycles as sorted groups of modules (strongly connected
    components with more than one module), sorted."""
    graph = import_graph(root, module_level_only)
    index, low, stack, on_stack, groups, counter = {}, {}, [], set(), [], [0]

    def visit(node):
        index[node] = low[node] = counter[0]
        counter[0] += 1
        stack.append(node)
        on_stack.add(node)
        for child in graph.get(node, ()):
            if child not in index:
                visit(child)
                low[node] = min(low[node], low[child])
            elif child in on_stack:
                low[node] = min(low[node], index[child])
        if low[node] == index[node]:
            group = []
            while True:
                child = stack.pop()
                on_stack.discard(child)
                group.append(child)
                if child == node:
                    break
            if len(group) > 1:
                groups.append(sorted(group))

    import sys
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 10000))
    try:
        for node in graph:
            if node not in index:
                visit(node)
    finally:
        sys.setrecursionlimit(limit)
    return sorted(groups)
