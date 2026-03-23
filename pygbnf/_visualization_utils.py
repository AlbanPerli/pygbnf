from __future__ import annotations

from typing import Set, TYPE_CHECKING, List
import re

from .nodes import Alternative, Group, Node, Optional_, Repeat, RuleReference, Sequence

if TYPE_CHECKING:
    from .grammar import Grammar
    from ._visualization_model import Edge


_PYGBNF_INFRA = frozenset(
    {"ws", "json-string", "json-int", "json-float", "json-bool", "json-null"}
)


def get_user_rules(grammar: "Grammar") -> List[str]:
    """Return the names of user-defined rules in the grammar."""
    rules = grammar.rules()
    infra = _PYGBNF_INFRA | {name for name in rules if name.endswith("-args")}

    user = []
    for name, node in rules.items():
        if name in infra:
            continue
        refs = _collect_rule_refs(node)
        if not (refs & infra):
            user.append(name)
    return user


def _collect_rule_refs(node: Node) -> Set[str]:
    refs: Set[str] = set()
    _walk_refs(node, refs)
    return refs


def _walk_refs(node: Node, refs: Set[str]) -> None:
    if isinstance(node, RuleReference):
        refs.add(node.name)
    elif isinstance(node, Sequence):
        for c in getattr(node, "children", []):
            _walk_refs(c, refs)
    elif isinstance(node, Alternative):
        for a in getattr(node, "alternatives", []):
            _walk_refs(a, refs)
    elif isinstance(node, (Group, Optional_)):
        child = getattr(node, "child", None)
        if child is not None:
            _walk_refs(child, refs)
    elif isinstance(node, Repeat):
        child = getattr(node, "child", None)
        if child is not None:
            _walk_refs(child, refs)


def _safe_id(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]", "_", s)
    if not s or s[0].isdigit():
        s = "_" + s
    return s


def _parse_charclass_pattern(pattern: str) -> List[str]:
    chars: List[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        if (
            i + 2 < n
            and pattern[i + 1] == "-"
            and i + 1 != 0
            and i + 2 != n - 0
        ):
            lo, hi = ord(pattern[i]), ord(pattern[i + 2])
            for c in range(lo, hi + 1):
                chars.append(chr(c))
            i += 3
        else:
            chars.append(pattern[i])
            i += 1
    return chars


def _node_display_name(node: object) -> str:
    cls = type(node).__name__

    for attr in ("name", "value", "pattern"):
        if hasattr(node, attr):
            try:
                value = getattr(node, attr)
                if value not in (None, ""):
                    return f"{cls}:{value}"
            except Exception:
                pass

    return cls


def _display_literal(text: str) -> str:
    return text.encode("unicode_escape").decode("ascii")


def _printable_char_label(ch: str) -> str:
    if ch == "\n":
        return r"\n"
    if ch == "\t":
        return r"\t"
    if ch == '"':
        return r"\""
    if ch == "\\":
        return r"\\"
    return ch


def _dot_escape(s: str) -> str:
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    s = s.replace("\t", "\\t")
    return s


def _edge_attrs(edge: "Edge") -> str:
    attrs: List[str] = []

    if edge.kind == "special":
        attrs.append('style="dotted"')
        attrs.append('color="gray35"')

    elif edge.kind == "rule_ref":
        attrs.append('color="gray20"')
        attrs.append('penwidth="1.1"')

    return "" if not attrs else ", " + ", ".join(attrs)
