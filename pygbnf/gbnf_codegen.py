"""
pygbnf.gbnf_codegen — Compile an AST into a GBNF grammar string.

This module walks the :class:`~pygbnf.grammar.Grammar` object and emits a
well-formatted, llama.cpp-compatible GBNF grammar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from .nodes import (
    Alternative,
    CharacterClass,
    Group,
    Literal,
    Node,
    Optional_,
    Repeat,
    RuleReference,
    Sequence,
    TokenReference,
)

if TYPE_CHECKING:
    from .grammar import Grammar


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compile_grammar(grammar: "Grammar", *, optimize: bool = True) -> str:
    """Compile *grammar* into a GBNF string.

    Parameters
    ----------
    grammar : Grammar
        A fully-built grammar container.
    optimize : bool
        If *True*, run optimisation passes before code-generation.
    """
    rules = dict(grammar._rules)

    if optimize:
        from .optimizations import optimize_rules
        rules = optimize_rules(rules)

    start = grammar._start
    order = list(grammar._rule_order)

    lines: List[str] = []

    # Emit root alias first if a start rule is set
    # Emit root alias first if a start rule is set and is not already "root"
    if start:
        if start not in rules:
            raise ValueError(f"Start rule {start!r} not found in grammar")
        if start != "root":
            lines.append(f"root ::= {start}")
            lines.append("")

    # Emit all rules in declaration order — start rule first if it's "root"
    if start == "root" and "root" in order:
        ordered = ["root"] + [n for n in order if n != "root"]
    else:
        ordered = order

    for name in ordered:
        node = rules.get(name)
        if node is None:
            continue
        body = _emit_rule_body(node)
        lines.append(f"{name} ::= {body}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Emitters — each returns a GBNF fragment string
# ---------------------------------------------------------------------------

def _emit(node: Node, *, parent_is_seq: bool = False) -> str:
    """Recursively emit GBNF for *node*."""
    if isinstance(node, Literal):
        return _emit_literal(node)
    if isinstance(node, CharacterClass):
        return _emit_char_class(node)
    if isinstance(node, Sequence):
        return _emit_sequence(node)
    if isinstance(node, Alternative):
        return _emit_alternative(node, wrap=parent_is_seq)
    if isinstance(node, Repeat):
        return _emit_repeat(node)
    if isinstance(node, RuleReference):
        return node.name
    if isinstance(node, TokenReference):
        return _emit_token(node)
    if isinstance(node, Group):
        return _emit_group(node)
    if isinstance(node, Optional_):
        inner = _emit(node.child)
        return f"{_maybe_wrap(node.child, inner)}?"
    raise TypeError(f"Unknown node type: {type(node)!r}")


def _emit_rule_body(node: Node) -> str:
    """Emit the right-hand side of a rule, with pretty multi-line
    formatting for alternatives.

    GBNF requires ``|`` to appear *before* the newline so the parser
    knows the rule continues.  Format::

        rule ::= alt1 |
          alt2 |
          alt3
    """
    if isinstance(node, Alternative) and len(node.alternatives) > 1:
        parts = [_emit(a) for a in node.alternatives]
        # Multi-line if any alternative is long or there are ≥ 3
        if len(parts) >= 3 or any(len(p) > 60 for p in parts):
            # Trailing " |" on every line except the last
            lines = [p + " |" for p in parts[:-1]]
            lines.append(parts[-1])
            return ("\n  ").join(lines)
        return " | ".join(parts)
    return _emit(node)


# ---------------------------------------------------------------------------
# Leaf / sub-emitters
# ---------------------------------------------------------------------------

def _emit_literal(node: Literal) -> str:
    if not node.value:
        return '""'
    return '"' + _escape_literal(node.value) + '"'


def _escape_literal(s: str) -> str:
    """Escape special characters inside a GBNF double-quoted literal."""
    out: List[str] = []
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20:
            out.append(f"\\x{ord(ch):02X}")
        else:
            out.append(ch)
    return "".join(out)


def _emit_char_class(node: CharacterClass) -> str:
    pattern = node.pattern
    negated = node.negated

    # Allow users to embed ``^`` at the start of the pattern string to mean
    # negation, even with ``negated=False``.
    if not negated and pattern.startswith("^"):
        negated = True
        pattern = pattern[1:]

    prefix = "^" if negated else ""
    inner = _escape_char_class(pattern)
    return f"[{prefix}{inner}]"


def _escape_char_class(pattern: str) -> str:
    """Escape characters for use inside ``[...]``.

    The pattern may already contain GBNF escape sequences (like ``\\n``) or
    range expressions (like ``a-z``).  We detect these and pass them through
    verbatim, only escaping raw special characters.
    """
    # If the whole pattern looks pre-formatted (ranges, escape seqs), pass
    # through as-is.  This covers ``0-9``, ``a-zA-Z_``, ``^"\\\\``, etc.
    if _is_preformatted(pattern):
        return pattern

    # Otherwise escape character-by-character.
    # Dashes are collected separately and appended at the end of the class
    # so that they are unambiguous (``[+-]`` not ``[+\-]``).
    # Note: ``\-`` is NOT supported by all llama.cpp versions — placing the
    # dash at the end of the class is universally compatible.
    out: List[str] = []
    has_dash = False
    for ch in pattern:
        if ch == "]":
            out.append("\\]")
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "^":
            out.append("\\^")
        elif ch == "-":
            has_dash = True
        elif ord(ch) < 0x20:
            out.append(f"\\x{ord(ch):02X}")
        else:
            out.append(ch)
    if has_dash:
        out.append("-")
    return "".join(out)


def _is_preformatted(pattern: str) -> bool:
    """Heuristic: does *pattern* already contain range dashes or escape
    sequences that should not be double-escaped?"""
    # Contains backslash-escape sequences (already formatted)
    if "\\" in pattern:
        return True
    # Contains range dashes (X-Y where X and Y are not at edges)
    for i in range(1, len(pattern) - 1):
        if pattern[i] == "-":
            return True
    return False


def _emit_sequence(node: Sequence) -> str:
    parts = [_emit(c, parent_is_seq=True) for c in node.children]
    return " ".join(parts)


def _emit_alternative(node: Alternative, wrap: bool = False) -> str:
    parts = [_emit(a) for a in node.alternatives]
    body = " | ".join(parts)
    if wrap:
        return f"({body})"
    return body


def _emit_repeat(node: Repeat) -> str:
    inner = _emit(node.child)
    inner_wrapped = _maybe_wrap(node.child, inner)

    if node.min == 0 and node.max is None:
        return f"{inner_wrapped}*"
    if node.min == 1 and node.max is None:
        return f"{inner_wrapped}+"
    if node.min == 0 and node.max == 1:
        return f"{inner_wrapped}?"
    if node.max is None:
        return f"{inner_wrapped}{{{node.min},}}"
    if node.min == node.max:
        return f"{inner_wrapped}{{{node.min}}}"
    return f"{inner_wrapped}{{{node.min},{node.max}}}"


def _emit_token(node: TokenReference) -> str:
    neg = "!" if node.negated else ""
    if isinstance(node.value, int):
        return f"{neg}<[{node.value}]>"
    return f"{neg}<{node.value}>"


def _emit_group(node: Group) -> str:
    inner = _emit(node.child)
    return f"({inner})"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _maybe_wrap(node: Node, rendered: str) -> str:
    """Wrap *rendered* in parentheses if the node is compound (sequence or
    alternative) so that a postfix operator applies to the whole group."""
    if isinstance(node, (Sequence, Alternative)):
        return f"({rendered})"
    return rendered
