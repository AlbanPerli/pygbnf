#!/usr/bin/env python3
"""Example: Visualize a grammar as an NFA diagram (DOT / SVG).

Builds a small arithmetic expression grammar, then exports its NFA
as a Graphviz DOT file (and SVG if Graphviz is installed).

Usage::

    python examples/demo_visualization.py
    # → arithmetic_nfa.dot  (always)
    # → arithmetic_nfa.svg  (if `dot` is in $PATH)
"""

import pygbnf as cfg
from pygbnf import select, one_or_more, zero_or_more, optional
from pygbnf.visualization import write_grammar_svg

g = cfg.Grammar()


@g.rule
def number():
    sign = optional("-")
    digits = one_or_more(select("0123456789"))
    return sign + digits


@g.rule
def operator():
    return select(["+", "-", "*", "/"])


@g.rule
def expression():
    atom = select([number(), "(" + expression() + ")"])
    return atom + zero_or_more(
        cfg.group(" " + operator() + " " + expression())
    )


g.start("expression")

# Print the GBNF grammar for reference
print("GBNF grammar:")
print("-" * 40)
g.pretty_print()
print()

# Export the NFA diagram — rule_names defaults to user-defined rules
try:
    svg = write_grammar_svg(
        g,
        output_svg_path="arithmetic_nfa.svg",
        name="Arithmetic expressions",
        keep_dot=True,
    )
    print(f"Wrote: {svg} (+ .dot)")
except RuntimeError as e:
    print(f"SVG skipped ({e})")
    from pygbnf.visualization import write_grammar_dot

    dot = write_grammar_dot(g, output_path="arithmetic_nfa.dot",
                            name="Arithmetic expressions")
    print(f"Wrote: {dot}")
