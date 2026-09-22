# Vendored unchanged from Agentic_Knowledge_Base/scripts/vendi_changes.py.
# Source snapshot SHA256: 4e9b2eab595acb7b9f5841d3b32d82c7fe943583ea2b413a1ada4b7991d464fd
"""Build bounded, source-grounded parent/child evidence without executing code."""

import ast
import difflib
import hashlib


VERSION = "vendi-change-packet-v2"

_SHORT_STATEMENT_LINES = 80
_CONTROL_BLOCKS = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With,
                   ast.AsyncWith, ast.Try, ast.Match)
_COLLECTION_MUTATORS = {"append", "extend", "insert", "update", "add", "discard",
                        "remove", "pop", "clear", "setdefault"}


def _raw_line(line):
    return line.removesuffix("\n").removesuffix("\r")


def _span(node):
    return range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1)


def _window(line, size, radius=6):
    return set(range(max(1, line - radius), min(size, line + radius) + 1))


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _root_name(node):
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _module_statements(tree):
    """Include module control-flow bodies, but not function/class local bindings."""
    if isinstance(tree, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return
    if isinstance(tree, ast.stmt):
        yield tree
    for child in ast.iter_child_nodes(tree):
        yield from _module_statements(child)


def _written_names(statement):
    names = set()
    for node in ast.walk(statement):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names.update(part.id for part in ast.walk(target)
                             if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store))
                root = _root_name(target)
                if root:
                    names.add(root)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr in _COLLECTION_MUTATORS):
            root = _root_name(node.func.value)
            if root:
                names.add(root)
    return names


def _request_statement_context(info, line, priority):
    # Preserve complete short calls/assignments and control blocks. A large loop
    # does not pull in its whole body, nor does a function/class declaration.
    for statement in info["statements"]:
        if line in _span(statement) and len(_span(statement)) <= _SHORT_STATEMENT_LINES:
            info["requests"].append((priority, set(_span(statement))))


def _source_info(code, lines, changed):
    """Locate enclosing definitions; syntax failures fall back to line windows."""
    info = {"lines": lines, "changed": changed, "tree": None,
            "definitions": [], "statements": [], "module_variables": set(),
            "symbols": set(), "requests": [], "limitation": ""}
    for line in sorted(changed):
        info["requests"].append((0, _window(line, len(lines))))
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError) as exc:
        info["limitation"] = f"AST unavailable ({type(exc).__name__}); line windows only"
        return info
    info["tree"] = tree
    info["statements"] = [node for node in ast.walk(tree)
                          if isinstance(node, _CONTROL_BLOCKS +
                                        (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr))]
    for line in sorted(changed):
        _request_statement_context(info, line, 0)
    # Seed only names written by changed short module statements. Looking up
    # their uses is bounded to one construction step below, not recursive dataflow.
    for statement in _module_statements(tree):
        if (len(_span(statement)) <= _SHORT_STATEMENT_LINES
                and changed.intersection(_span(statement))):
            info["module_variables"].update(_written_names(statement))
    definitions = [node for node in ast.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    info["definitions"] = definitions
    defined_names = {node.name for node in definitions}
    touched_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
                     and node.lineno in changed}
    info["symbols"].update(touched_names & defined_names)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Call):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id in touched_names for target in targets):
                name = _call_name(node.value.func)
                if name in defined_names:
                    info["symbols"].add(name)
    for line in sorted(changed):
        enclosing = [node for node in definitions if line in _span(node)]
        if not enclosing:
            continue
        # Prefer the smallest definition; do not include an entire large class.
        nearest = min(enclosing, key=lambda node: len(_span(node)))
        info["symbols"].add(nearest.name)
        info["symbols"].update(node.name for node in enclosing if isinstance(node, ast.ClassDef))
        if len(_span(nearest)) <= 80:
            info["requests"].append((2, set(_span(nearest))))
        else:
            info["requests"].append((2, _window(nearest.lineno, len(lines), 2)))
            info["requests"].append((2, _window(line, len(lines), 12)))
    return info


def _add_references(info, symbols):
    tree = info["tree"]
    if tree is None:
        return
    for definition in info["definitions"]:
        if definition.name in symbols:
            numbers = (set(_span(definition)) if len(_span(definition)) <= 80 else
                       _window(definition.lineno, len(info["lines"]), 3))
            info["requests"].append((2, numbers))
    # One simple alias step also catches FooSampler(...) -> sampler -> DataLoader.
    # This is deliberately not a static call graph or a proof of execution.
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Call):
            if _call_name(node.value.func) in symbols:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                aliases.update(target.id for target in targets if isinstance(target, ast.Name))
    references = symbols | aliases
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
            name = node.id if isinstance(node, ast.Name) else node.attr
            if name not in references:
                continue
            info["requests"].append((1, _window(node.lineno, len(info["lines"]))))
            _request_statement_context(info, node.lineno, 1)
            enclosing = [definition for definition in info["definitions"]
                         if node.lineno in _span(definition)]
            if enclosing:
                nearest = min(enclosing, key=lambda definition: len(_span(definition)))
                info["requests"].append((1, _window(nearest.lineno, len(info["lines"]), 2)))


def _add_module_variable_references(info):
    """Connect changed module assignments/containers to one consuming constructor."""
    if info["tree"] is None or not info["module_variables"]:
        return
    seeds = info["module_variables"]
    aliases = set()
    # For example, changed group assignment -> AdamW(groups) -> optimizer.step.
    # Do not follow aliases again, or add unrelated names from retrieved windows.
    for statement in _module_statements(info["tree"]):
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        if not isinstance(statement.value, ast.Call):
            continue
        uses = {node.id for node in ast.walk(statement.value)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        if uses & seeds:
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            aliases.update(target.id for target in targets if isinstance(target, ast.Name))
    for node in ast.walk(info["tree"]):
        if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                and node.id in seeds | aliases):
            info["requests"].append((1, _window(node.lineno, len(info["lines"]))))
            _request_statement_context(info, node.lineno, 1)


def build_change_packet(parent_code: str, child_code: str, max_chars: int = 96000) -> dict:
    """Return complete diff plus bounded static context, with exact source refs.

    ``complete`` guarantees that *all diff hunks* are sent, not that all program
    dependencies or runtime behavior have been established. Omitted context is
    reported separately. A diff that cannot fit is unavailable rather than
    silently truncated. ``evidence_lines`` contains only lines actually sent in
    ``diff`` or ``context``. Raw strings are parsed/read, never imported/executed.
    """
    if not isinstance(parent_code, str) or not isinstance(child_code, str):
        raise TypeError("parent_code and child_code must be strings")
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    packet = {
        "version": VERSION, "status": "complete", "reason": "",
        "identical": parent_code == child_code, "diff": "", "context": "",
        "ast_equal": None, "evidence_lines": {}, "changed_refs": [], "usage_refs": [],
        "source_hashes": {"parent": hashlib.sha256(parent_code.encode()).hexdigest(),
                          "child": hashlib.sha256(child_code.encode()).hexdigest()},
        "omitted_context_lines": 0, "omitted_context_chars": 0,
        "limitations": ["Static source evidence only; references do not prove runtime execution."],
    }
    if packet["identical"]:
        packet["ast_equal"] = True
        return packet
    # Preserve line endings when matching so a final-newline-only change is not
    # mistaken for identical raw source. Evidence values omit the line separator.
    originals = {"PARENT": parent_code.splitlines(keepends=True),
                 "CHILD": child_code.splitlines(keepends=True)}
    lines = {side: [_raw_line(line) for line in raw] for side, raw in originals.items()}
    changed = {"PARENT": set(), "CHILD": set()}
    evidence = {}
    diff_parts = []

    def emit(side, index, marker):
        ref = f"{side}:{index + 1}"
        raw = lines[side][index]
        evidence[ref] = raw
        ending = originals[side][index]
        note = " [no newline]" if not ending.endswith(("\n", "\r")) else ""
        if ending.endswith("\r\n"):
            note = " [CRLF]"
        diff_parts.append(f"{marker} {ref}{note}: {raw}\n")

    matcher = difflib.SequenceMatcher(None, originals["PARENT"], originals["CHILD"], autojunk=False)
    for group in matcher.get_grouped_opcodes(n=3):
        first, last = group[0], group[-1]
        diff_parts.append(f"@@ PARENT {first[1] + 1}:{last[2]} CHILD {first[3] + 1}:{last[4]} @@\n")
        for tag, start_p, end_p, start_c, end_c in group:
            if tag == "equal":
                # Both sides are shown so either exact reference can be cited.
                for p, c in zip(range(start_p, end_p), range(start_c, end_c)):
                    emit("PARENT", p, " ")
                    emit("CHILD", c, " ")
            else:
                for side, start, end in (("PARENT", start_p, end_p), ("CHILD", start_c, end_c)):
                    for index in range(start, end):
                        changed[side].add(index + 1)
                        emit(side, index, "-" if side == "PARENT" else "+")
    diff = "".join(diff_parts)
    infos = {side: _source_info(code, lines[side], changed[side])
             for side, code in (("PARENT", parent_code), ("CHILD", child_code))}
    if all(info["tree"] is not None for info in infos.values()):
        packet["ast_equal"] = (ast.dump(infos["PARENT"]["tree"], include_attributes=False) ==
                               ast.dump(infos["CHILD"]["tree"], include_attributes=False))
    for side, info in infos.items():
        if info["limitation"]:
            packet["limitations"].append(f"{side}: {info['limitation']}")
    if len(diff) > max_chars:
        packet.update(status="insufficient_evidence", reason="complete_diff_exceeds_budget",
                      required_diff_chars=len(diff), omitted_diff_chars=len(diff))
        return packet
    packet["diff"] = diff
    packet["evidence_lines"] = evidence
    packet["changed_refs"] = [f"{side}:{line}" for side in ("PARENT", "CHILD")
                              for line in sorted(changed[side])]

    symbols = set().union(*(info["symbols"] for info in infos.values()))
    for info in infos.values():
        _add_references(info, symbols)
        _add_module_variable_references(info)
    requests = sorted((priority, side, sorted(numbers)) for side, info in infos.items()
                      for priority, numbers in info["requests"])
    context_parts = []
    remaining = max_chars - len(diff)
    requested = set()
    for _, side, numbers in requests:
        requested.update(f"{side}:{number}" for number in numbers)
        fresh = [(f"{side}:{number}", lines[side][number - 1]) for number in numbers
                 if f"{side}:{number}" not in evidence]
        text = "".join(f"{ref}: {raw}\n" for ref, raw in fresh)
        if len(text) <= remaining:
            context_parts.append(text)
            evidence.update(fresh)
            remaining -= len(text)
    packet["context"] = "".join(context_parts)
    omitted = requested - evidence.keys()
    packet["omitted_context_lines"] = len(omitted)
    packet["omitted_context_chars"] = sum(
        len(ref) + len(lines[ref.split(":")[0]][int(ref.split(":")[1]) - 1]) + 3
        for ref in omitted)
    if omitted:
        packet["limitations"].append("Some related source context omitted to fit the character budget.")
    usage_refs = set()
    for side, info in infos.items():
        if info["tree"] is None:
            continue
        declarations = {node.lineno for node in info["definitions"]}
        for node in ast.walk(info["tree"]):
            is_usage = isinstance(node, ast.Call) or (
                isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load))
            if is_usage and node.lineno not in declarations:
                ref = f"{side}:{node.lineno}"
                if ref in evidence:
                    usage_refs.add(ref)
    packet["usage_refs"] = sorted(usage_refs, key=lambda ref: (ref.split(":")[0], int(ref.split(":")[1])))
    return packet
