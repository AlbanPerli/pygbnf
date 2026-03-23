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

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set
import shutil
import subprocess

from ._visualization_builder import RegularSubsetError, StateFactory, ThompsonBuilder
from ._visualization_model import Edge, Fragment, NFA, RuleLink
from ._visualization_utils import _dot_escape, _edge_attrs, _safe_id, get_user_rules

if TYPE_CHECKING:
    from .grammar import Grammar


def grammar_rule_to_nfa_dot(
    grammar: "Grammar",
    rule_name: str,
    *,
    compact_literals: bool = True,
    allow_unknown_nodes_as_special: bool = True,
    expand_rules: Optional[Set[str]] = None,
    expand_charclasses: bool = True,
    expand_depth: Optional[int] = None,
) -> str:
    """Convert a single grammar rule to a DOT string (NFA graph)."""
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
    grammar: "Grammar",
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
    """Convert multiple grammar rules to a single DOT string with subgraphs."""
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
    grammar: "Grammar",
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
    grammar: "Grammar",
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
    """Render a ``.dot`` file to SVG using Graphviz."""
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
    grammar: "Grammar",
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
    """Write the NFA for a grammar directly as an SVG file."""
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
