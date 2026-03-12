"""
pygbnf.visualization — Export grammars as NFA diagrams in DOT / SVG format.

This module uses a Thompson construction to convert grammar rules into
NFA graphs, then renders them as Graphviz DOT strings or SVG files.

Quick start::

    import pygbnf as cfg
    from pygbnf.visualization import write_grammar_svg

    g = cfg.Grammar()
    # ... define rules ...
    write_grammar_svg(g, "my_grammar.svg")

Public API
----------
- :func:`grammar_rule_to_nfa_dot` — single rule → DOT string
- :func:`grammar_to_nfa_dot` — multiple rules → DOT string
- :func:`write_rule_dot` — single rule → ``.dot`` file
- :func:`write_grammar_dot` — grammar → ``.dot`` file
- :func:`render_dot_to_svg` — ``.dot`` → ``.svg`` (requires Graphviz)
- :func:`write_grammar_svg` — grammar → ``.svg`` (all-in-one)
- :func:`get_user_rules` — detect user-defined (non-infrastructure) rules
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
import re
import shutil
import subprocess

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

# TYPE_CHECKING import to avoid circular dependency at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .grammar import Grammar


# ============================================================
# Model
# ============================================================

@dataclass
class Edge:
    src: int
    dst: int
    label: Optional[str] = None   # None = epsilon
    kind: str = "normal"          # normal | special | rule_ref


@dataclass
class RuleLink:
    src_state: int
    dst_state: int
    target_rule: str


@dataclass
class Fragment:
    start: int
    accepts: Set[int]


@dataclass
class NFA:
    start: int
    accepts: Set[int]
    edges: List[Edge] = field(default_factory=list)
    rule_links: List[RuleLink] = field(default_factory=list)

    def all_states(self) -> Set[int]:
        states = {self.start} | set(self.accepts)
        for e in self.edges:
            states.add(e.src)
            states.add(e.dst)
        for rl in self.rule_links:
            states.add(rl.src_state)
            states.add(rl.dst_state)
        return states

    def to_dot(self, name: str = "NFA") -> str:
        lines = [
            f'digraph "{_dot_escape(name)}" {{',
            "  rankdir=LR;",
            '  node [shape=circle, fontsize=10];',
            '  edge [fontsize=9];',
        ]

        for state in sorted(self.all_states()):
            if state == self.start and state in self.accepts:
                attrs = 'shape=doublecircle, penwidth=3.0'
            elif state == self.start:
                attrs = 'shape=circle, penwidth=3.0'
            elif state in self.accepts:
                attrs = 'shape=doublecircle'
            else:
                attrs = 'shape=circle'
            lines.append(f'  s{state} [{attrs}, label="{state}"];')

        for e in self.edges:
            attrs = _edge_attrs(e)
            label = "ε" if e.label is None else _dot_escape(e.label)
            lines.append(f'  s{e.src} -> s{e.dst} [label="{label}"{attrs}];')

        lines.append("}")
        return "\n".join(lines)


# ============================================================
# Errors
# ============================================================

class RegularSubsetError(ValueError):
    pass


# ============================================================
# State factory
# ============================================================

class StateFactory:
    def __init__(self) -> None:
        self._next = 0

    def new(self) -> int:
        s = self._next
        self._next += 1
        return s


# ============================================================
# Builder
# ============================================================

class ThompsonBuilder:
    def __init__(
        self,
        rules: Dict[str, Node],
        *,
        compact_literals: bool = True,
        allow_unknown_nodes_as_special: bool = True,
        expand_rules: Optional[Set[str]] = None,
        expand_charclasses: bool = True,
        max_expand_depth: Optional[int] = None,
    ) -> None:
        self.rules = rules
        self.compact_literals = compact_literals
        self.allow_unknown_nodes_as_special = allow_unknown_nodes_as_special
        self.expand_rules = expand_rules
        self.expand_charclasses = expand_charclasses
        self.max_expand_depth = max_expand_depth
        self.sf = StateFactory()
        self.edges: List[Edge] = []
        self.rule_links: List[RuleLink] = []
        self._expand_stack: List[str] = []

    def build_rule(self, rule_name: str) -> NFA:
        if rule_name not in self.rules:
            raise KeyError(f"Unknown rule: {rule_name!r}")
        frag = self._build(self.rules[rule_name])
        nfa = NFA(
            start=frag.start,
            accepts=set(frag.accepts),
            edges=list(self.edges),
            rule_links=list(self.rule_links),
        )
        return _simplify_nfa(nfa)

    def _build(self, node: Node) -> Fragment:
        if isinstance(node, RuleReference):
            return self._build_rule_ref(node.name)

        if isinstance(node, Literal):
            return self._build_literal(node.value)

        if isinstance(node, CharacterClass):
            return self._build_charclass(
                pattern=getattr(node, "pattern", ""),
                negated=getattr(node, "negated", False),
            )

        if isinstance(node, Group):
            child = getattr(node, "child", None)
            if child is None:
                raise RegularSubsetError("Group sans child")
            return self._build(child)

        if isinstance(node, Sequence):
            children = list(getattr(node, "children", []))
            if not children:
                return self._build_epsilon()

            frag = self._build(children[0])
            for child in children[1:]:
                right = self._build(child)
                for a in frag.accepts:
                    self.edges.append(Edge(a, right.start, None))
                frag = Fragment(start=frag.start, accepts=set(right.accepts))
            return frag

        if isinstance(node, Alternative):
            alts = list(getattr(node, "alternatives", []))
            start = self.sf.new()
            end = self.sf.new()
            for alt in alts:
                f = self._build(alt)
                self.edges.append(Edge(start, f.start, None))
                for a in f.accepts:
                    self.edges.append(Edge(a, end, None))
            return Fragment(start=start, accepts={end})

        if isinstance(node, Optional_):
            child = getattr(node, "child", None)
            if child is None:
                raise RegularSubsetError("Optional_ sans child")
            inner = self._build(child)
            start = self.sf.new()
            end = self.sf.new()
            self.edges.append(Edge(start, inner.start, None))
            self.edges.append(Edge(start, end, None))
            for a in inner.accepts:
                self.edges.append(Edge(a, end, None))
            return Fragment(start=start, accepts={end})

        if isinstance(node, Repeat):
            return self._build_repeat(node)

        if isinstance(node, TokenReference):
            return self._build_special(f"<token:{_node_display_name(node)}>")

        if self.allow_unknown_nodes_as_special:
            return self._build_special(f"<{_node_display_name(node)}>")

        raise RegularSubsetError(f"Unsupported node type: {type(node).__name__}")

    def _build_rule_ref(self, name: str) -> Fragment:
        if name not in self.rules:
            raise RegularSubsetError(f"Unknown rule reference: {name}")

        depth_ok = (
            self.max_expand_depth is None
            or len(self._expand_stack) < self.max_expand_depth
        )
        if (
            self.expand_rules is not None
            and name in self.expand_rules
            and name not in self._expand_stack
            and depth_ok
        ):
            self._expand_stack.append(name)
            try:
                return self._build(self.rules[name])
            finally:
                self._expand_stack.pop()

        s = self.sf.new()
        t = self.sf.new()

        self.edges.append(Edge(s, t, f"<{name}>", kind="rule_ref"))
        self.rule_links.append(RuleLink(src_state=s, dst_state=t, target_rule=name))
        return Fragment(start=s, accepts={t})

    def _build_special(self, label: str) -> Fragment:
        s = self.sf.new()
        t = self.sf.new()
        self.edges.append(Edge(s, t, label, kind="special"))
        return Fragment(start=s, accepts={t})

    def _build_epsilon(self) -> Fragment:
        s = self.sf.new()
        t = self.sf.new()
        self.edges.append(Edge(s, t, None))
        return Fragment(start=s, accepts={t})

    def _build_literal(self, text: str) -> Fragment:
        if text == "":
            return self._build_epsilon()

        if self.compact_literals:
            s = self.sf.new()
            t = self.sf.new()
            self.edges.append(Edge(s, t, _display_literal(text)))
            return Fragment(start=s, accepts={t})

        start = self.sf.new()
        current = start
        for ch in text:
            nxt = self.sf.new()
            self.edges.append(Edge(current, nxt, _printable_char_label(ch)))
            current = nxt
        return Fragment(start=start, accepts={current})

    def _build_charclass(self, pattern: str, negated: bool) -> Fragment:
        if self.expand_charclasses and not negated:
            chars = _parse_charclass_pattern(pattern)
            if len(chars) > 1:
                start = self.sf.new()
                end = self.sf.new()
                for ch in chars:
                    self.edges.append(
                        Edge(start, end, _printable_char_label(ch))
                    )
                return Fragment(start=start, accepts={end})

        s = self.sf.new()
        t = self.sf.new()
        label = f"[^{pattern}]" if negated else f"[{pattern}]"
        self.edges.append(Edge(s, t, label))
        return Fragment(start=s, accepts={t})

    def _build_repeat(self, node: Repeat) -> Fragment:
        child = getattr(node, "child", None)
        if child is None:
            raise RegularSubsetError("Repeat sans child")

        m = getattr(node, "min", None)
        n = getattr(node, "max", None)

        if m is None:
            m = 0

        if m == 0 and n == 1:
            return self._build(Optional_(child=child))

        if m == 0 and n is None:
            inner = self._build(child)
            start = self.sf.new()
            end = self.sf.new()
            self.edges.append(Edge(start, end, None))
            self.edges.append(Edge(start, inner.start, None))
            for a in inner.accepts:
                self.edges.append(Edge(a, inner.start, None))
                self.edges.append(Edge(a, end, None))
            return Fragment(start=start, accepts={end})

        if m == 1 and n is None:
            first = self._build(child)
            end = self.sf.new()
            for a in first.accepts:
                self.edges.append(Edge(a, first.start, None))
                self.edges.append(Edge(a, end, None))
            return Fragment(start=first.start, accepts={end})

        if n is not None and n < m:
            raise RegularSubsetError(f"Invalid Repeat bounds: min={m}, max={n}")

        if m == 0:
            frag = self._build_epsilon()
        else:
            frag: Optional[Fragment] = None
            for _ in range(m):
                part = self._build(child)
                if frag is None:
                    frag = part
                else:
                    for a in frag.accepts:
                        self.edges.append(Edge(a, part.start, None))
                    frag = Fragment(start=frag.start, accepts=set(part.accepts))
            assert frag is not None

        if n is None:
            raise RegularSubsetError("Unexpected Repeat case")

        optional_count = n - m
        result = frag
        for _ in range(optional_count):
            part = self._build(Optional_(child=child))
            for a in result.accepts:
                self.edges.append(Edge(a, part.start, None))
            result = Fragment(start=result.start, accepts=set(part.accepts))

        return result


# ============================================================
# Epsilon simplification
# ============================================================

def _simplify_nfa(nfa: NFA) -> NFA:
    """Remove unnecessary epsilon transitions by contracting pass-through states.

    A state is contractable when it is neither start nor accept and serves
    only as an epsilon relay — i.e. it has exactly one incoming epsilon and
    no other incoming edges, OR exactly one outgoing epsilon and no other
    outgoing edges.  In both cases the state can be merged away.

    The pass is repeated until no more contractions are possible.
    """
    edges = list(nfa.edges)
    start = nfa.start
    accepts = set(nfa.accepts)
    rule_links = list(nfa.rule_links)

    # Build a union-find to track merged states
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: int, b: int, keep: int) -> None:
        """Merge a and b, keeping `keep` as the representative."""
        a, b = find(a), find(b)
        if a == b:
            return
        discard = b if keep == a else a
        parent[discard] = keep

    protected = {start} | accepts

    changed = True
    while changed:
        changed = False

        # Rewrite edges through union-find
        for e in edges:
            e.src = find(e.src)
            e.dst = find(e.dst)
        for rl in rule_links:
            rl.src_state = find(rl.src_state)
            rl.dst_state = find(rl.dst_state)

        # Remove self-loops on epsilon
        edges = [e for e in edges if not (e.label is None and e.src == e.dst)]

        # Index
        all_states: Set[int] = {start} | accepts
        for e in edges:
            all_states.add(e.src)
            all_states.add(e.dst)

        out_eps: Dict[int, List[Edge]] = {s: [] for s in all_states}
        out_lab: Dict[int, List[Edge]] = {s: [] for s in all_states}
        in_eps: Dict[int, List[Edge]] = {s: [] for s in all_states}
        in_lab: Dict[int, List[Edge]] = {s: [] for s in all_states}

        for e in edges:
            if e.label is None:
                out_eps[e.src].append(e)
                in_eps[e.dst].append(e)
            else:
                out_lab[e.src].append(e)
                in_lab[e.dst].append(e)

        for s in all_states:
            if s in protected:
                continue

            # Case 1: s has exactly one outgoing epsilon, no outgoing labelled
            # edges, and is not a target of a rule_link.
            # Merge s into the epsilon target — all incoming edges of s now
            # point to the target.
            if (
                len(out_eps[s]) == 1
                and not out_lab[s]
            ):
                target = out_eps[s][0].dst
                if target != s and find(target) not in protected | {find(s)}:
                    # Safe: merge s → target (keep target)
                    union(s, target, keep=find(target))
                    changed = True
                    break
                elif target != s:
                    # target is protected — merge target direction reversed:
                    # redirect all of s's incoming to target, remove the epsilon
                    union(s, target, keep=find(target))
                    changed = True
                    break

            # Case 2: s has exactly one incoming epsilon, no incoming labelled
            # edges.  Merge s into the epsilon source.
            if (
                len(in_eps[s]) == 1
                and not in_lab[s]
            ):
                source = in_eps[s][0].src
                if source != s:
                    union(s, source, keep=find(source))
                    changed = True
                    break

    # Final rewrite
    for e in edges:
        e.src = find(e.src)
        e.dst = find(e.dst)
    for rl in rule_links:
        rl.src_state = find(rl.src_state)
        rl.dst_state = find(rl.dst_state)
    start = find(start)
    accepts = {find(a) for a in accepts}

    # Remove epsilon self-loops and deduplicate
    seen: Set[tuple] = set()
    clean: List[Edge] = []
    for e in edges:
        if e.label is None and e.src == e.dst:
            continue
        key = (e.src, e.dst, e.label, e.kind)
        if key not in seen:
            seen.add(key)
            clean.append(e)

    # Renumber states contiguously
    live: Set[int] = {start} | accepts
    for e in clean:
        live.add(e.src)
        live.add(e.dst)
    for rl in rule_links:
        live.add(rl.src_state)
        live.add(rl.dst_state)

    remap = {old: i for i, old in enumerate(sorted(live))}
    for e in clean:
        e.src = remap[e.src]
        e.dst = remap[e.dst]
    for rl in rule_links:
        rl.src_state = remap[rl.src_state]
        rl.dst_state = remap[rl.dst_state]

    return NFA(
        start=remap[start],
        accepts={remap[a] for a in accepts},
        edges=clean,
        rule_links=rule_links,
    )


# ============================================================
# Public API
# ============================================================

def grammar_rule_to_nfa_dot(
    grammar: Grammar,
    rule_name: str,
    *,
    compact_literals: bool = True,
    allow_unknown_nodes_as_special: bool = True,
    expand_rules: Optional[Set[str]] = None,
    expand_charclasses: bool = True,
    expand_depth: Optional[int] = None,
) -> str:
    """Convert a single grammar rule to a DOT string (NFA graph).

    Parameters
    ----------
    grammar : Grammar
        The grammar containing the rule.
    rule_name : str
        Name of the rule to visualize.
    compact_literals : bool
        If True, show literal strings as single edges.
    expand_rules : set[str] | None
        Rules to expand inline instead of showing as references.
    expand_charclasses : bool
        If True, expand character classes into individual edges.
    expand_depth : int | None
        Maximum depth for recursive rule expansion.  ``None`` means
        unlimited (fully expand).  ``1`` expands one level, etc.

    Returns
    -------
    str
        A Graphviz DOT string.
    """
    rules = grammar.rules()
    builder = ThompsonBuilder(
        rules,
        compact_literals=compact_literals,
        allow_unknown_nodes_as_special=allow_unknown_nodes_as_special,
        expand_rules=expand_rules,
        expand_charclasses=expand_charclasses,
        max_expand_depth=expand_depth,
    )
    nfa = builder.build_rule(rule_name)
    return nfa.to_dot(name=rule_name)


def grammar_to_nfa_dot(
    grammar: Grammar,
    rule_names: Optional[List[str]] = None,
    *,
    name: str = "Grammar",
    compact_literals: bool = True,
    allow_unknown_nodes_as_special: bool = True,
    show_inter_rule_links: bool = False,
    expand_rules: Optional[Set[str]] = None,
    expand_charclasses: bool = True,
    expand_depth: Optional[int] = 1,
) -> str:
    """Convert multiple grammar rules to a single DOT string with subgraphs.

    Parameters
    ----------
    grammar : Grammar
        The grammar to visualize.
    rule_names : list[str] | None
        Rules to include.  ``None`` means user-defined rules only
        (via :func:`get_user_rules`).
    name : str
        Title for the DOT digraph.
    show_inter_rule_links : bool
        If True, draw dashed edges between rule references.
    expand_rules : set[str] | None
        Rules to expand inline.  Defaults to all *rule_names*.
    expand_depth : int | None
        Maximum depth for recursive rule expansion.  Defaults to ``1``
        (one level) to prevent combinatorial blowup in deeply nested
        grammars.  Use ``None`` for unlimited expansion.

    Returns
    -------
    str
        A Graphviz DOT string.
    """
    rules = grammar.rules()

    if rule_names is None:
        rule_names = get_user_rules(grammar)

    if expand_rules is None:
        expand_rules = set(rule_names)

    rendered: Dict[str, NFA] = {}
    safe_names: Dict[str, str] = {}

    for rule_name in rule_names:
        if rule_name not in rules:
            raise KeyError(f"Unknown rule: {rule_name!r}")
        builder = ThompsonBuilder(
            rules,
            compact_literals=compact_literals,
            allow_unknown_nodes_as_special=allow_unknown_nodes_as_special,
            expand_rules=expand_rules,
            expand_charclasses=expand_charclasses,
            max_expand_depth=expand_depth,
        )
        rendered[rule_name] = builder.build_rule(rule_name)
        safe_names[rule_name] = _safe_id(rule_name)

    lines = [
        f'digraph "{_dot_escape(name)}" {{',
        "  rankdir=LR;",
        "  compound=true;",
        '  graph [fontname="Helvetica"];',
        '  node [shape=circle, fontsize=10, fontname="Helvetica"];',
        '  edge [fontsize=9, fontname="Helvetica"];',
    ]

    for rule_name in rule_names:
        safe = safe_names[rule_name]
        nfa = rendered[rule_name]

        lines.append(f'  subgraph "cluster_{safe}" {{')
        lines.append(f'    label="{_dot_escape(rule_name)}";')
        lines.append("    style=rounded;")
        lines.append("    color=gray55;")

        for state in sorted(nfa.all_states()):
            if state == nfa.start and state in nfa.accepts:
                attrs = 'shape=doublecircle, penwidth=3.0'
            elif state == nfa.start:
                attrs = 'shape=circle, penwidth=3.0'
            elif state in nfa.accepts:
                attrs = 'shape=doublecircle'
            else:
                attrs = 'shape=circle'
            lines.append(f'    {safe}_s{state} [{attrs}, label="{state}"];')

        for e in nfa.edges:
            attrs = _edge_attrs(e)
            label = "ε" if e.label is None else _dot_escape(e.label)
            lines.append(
                f'    {safe}_s{e.src} -> {safe}_s{e.dst} [label="{label}"{attrs}];'
            )

        lines.append("  }")

    if show_inter_rule_links:
        for src_rule in rule_names:
            src_safe = safe_names[src_rule]
            nfa = rendered[src_rule]

            for rl in nfa.rule_links:
                if rl.target_rule not in safe_names:
                    continue

                dst_safe = safe_names[rl.target_rule]
                dst_nfa = rendered[rl.target_rule]

                lines.append(
                    f'  {src_safe}_s{rl.src_state} -> {dst_safe}_s{dst_nfa.start} '
                    f'[style=dashed, color=gray35, penwidth=1.2, '
                    f'ltail="cluster_{src_safe}", lhead="cluster_{dst_safe}", '
                    f'label="call"];'
                )

    lines.append("}")
    return "\n".join(lines)


def write_rule_dot(
    grammar: Grammar,
    rule_name: str,
    output_path: str | Path,
    *,
    compact_literals: bool = True,
    allow_unknown_nodes_as_special: bool = True,
    expand_rules: Optional[Set[str]] = None,
    expand_charclasses: bool = True,
    expand_depth: Optional[int] = None,
) -> Path:
    """Write the NFA for a single rule as a ``.dot`` file."""
    dot = grammar_rule_to_nfa_dot(
        grammar,
        rule_name,
        compact_literals=compact_literals,
        allow_unknown_nodes_as_special=allow_unknown_nodes_as_special,
        expand_rules=expand_rules,
        expand_charclasses=expand_charclasses,
        expand_depth=expand_depth,
    )
    output_path = Path(output_path)
    output_path.write_text(dot, encoding="utf-8")
    return output_path


def write_grammar_dot(
    grammar: Grammar,
    output_path: str | Path = "grammar_nfa.dot",
    *,
    rule_names: Optional[List[str]] = None,
    name: str = "Grammar",
    compact_literals: bool = True,
    allow_unknown_nodes_as_special: bool = True,
    show_inter_rule_links: bool = False,
    expand_rules: Optional[Set[str]] = None,
    expand_charclasses: bool = True,
    expand_depth: Optional[int] = 1,
) -> Path:
    """Write the NFA for a grammar as a ``.dot`` file."""
    dot = grammar_to_nfa_dot(
        grammar,
        rule_names=rule_names,
        name=name,
        compact_literals=compact_literals,
        allow_unknown_nodes_as_special=allow_unknown_nodes_as_special,
        show_inter_rule_links=show_inter_rule_links,
        expand_rules=expand_rules,
        expand_charclasses=expand_charclasses,
        expand_depth=expand_depth,
    )
    output_path = Path(output_path)
    output_path.write_text(dot, encoding="utf-8")
    return output_path


def render_dot_to_svg(
    dot_path: str | Path,
    svg_path: Optional[str | Path] = None,
) -> Path:
    """Render a ``.dot`` file to SVG using Graphviz.

    Raises
    ------
    RuntimeError
        If the ``dot`` command is not found in ``$PATH``.
    """
    dot_path = Path(dot_path)
    if svg_path is None:
        svg_path = dot_path.with_suffix(".svg")
    svg_path = Path(svg_path)

    if shutil.which("dot") is None:
        raise RuntimeError(
            "Graphviz is not installed or 'dot' is not in $PATH. "
            "Install it with: brew install graphviz (macOS) / "
            "apt install graphviz (Linux)"
        )

    subprocess.run(
        ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
        check=True,
    )
    return svg_path


def write_grammar_svg(
    grammar: Grammar,
    output_svg_path: str | Path = "grammar_nfa.svg",
    *,
    rule_names: Optional[List[str]] = None,
    name: str = "Grammar",
    compact_literals: bool = True,
    allow_unknown_nodes_as_special: bool = True,
    show_inter_rule_links: bool = False,
    keep_dot: bool = True,
    expand_rules: Optional[Set[str]] = None,
    expand_charclasses: bool = True,
    expand_depth: Optional[int] = 1,
) -> Path:
    """Write the NFA for a grammar directly as an SVG file.

    This is a convenience wrapper that generates a ``.dot`` file and
    then renders it to SVG via Graphviz.

    Parameters
    ----------
    grammar : Grammar
        The grammar to visualize.
    output_svg_path : str | Path
        Output SVG file path.
    keep_dot : bool
        If True (default), keep the intermediate ``.dot`` file.
    expand_depth : int | None
        Maximum depth for recursive rule expansion.  Defaults to ``1``.
        Use ``None`` for unlimited expansion.

    Returns
    -------
    Path
        The path to the generated SVG file.
    """
    output_svg_path = Path(output_svg_path)
    dot_path = output_svg_path.with_suffix(".dot")

    write_grammar_dot(
        grammar,
        output_path=dot_path,
        rule_names=rule_names,
        name=name,
        compact_literals=compact_literals,
        allow_unknown_nodes_as_special=allow_unknown_nodes_as_special,
        show_inter_rule_links=show_inter_rule_links,
        expand_rules=expand_rules,
        expand_charclasses=expand_charclasses,
        expand_depth=expand_depth,
    )
    render_dot_to_svg(dot_path, output_svg_path)

    if not keep_dot and dot_path.exists():
        dot_path.unlink()

    return output_svg_path


# ============================================================
# Auto-detection of user rules
# ============================================================

_PYGBNF_INFRA = frozenset({
    "ws", "json-string", "json-int", "json-float", "json-bool", "json-null",
})


def get_user_rules(grammar: Grammar) -> List[str]:
    """Return the names of user-defined rules in the grammar.

    Excludes infrastructure rules auto-generated by pygbnf
    (ws, json-*, *-args) and those created by ``from_tool_call`` /
    ``from_type``.
    """
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


# ============================================================
# Internal helpers
# ============================================================

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


def _edge_attrs(edge: Edge) -> str:
    attrs: List[str] = []

    if edge.kind == "special":
        attrs.append('style="dotted"')
        attrs.append('color="gray35"')

    elif edge.kind == "rule_ref":
        attrs.append('color="gray20"')
        attrs.append('penwidth="1.1"')

    return "" if not attrs else ", " + ", ".join(attrs)
