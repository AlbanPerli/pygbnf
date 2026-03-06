#!/usr/bin/env python3
"""Example 1 — Arithmetic expressions.

Demonstrates recursion, alternation, repetition, and operator precedence.
"""

import pygbnf as cfg
from pygbnf import select, one_or_more, zero_or_more, optional

g = cfg.Grammar()


@g.rule
def ws():
    return zero_or_more(select(" \t"))


@g.rule
def number():
    sign = optional("-")
    digits = one_or_more(select("0123456789"))
    return sign + digits


@g.rule
def factor():
    return select([
        number(),
        "(" + ws() + expression() + ws() + ")",
    ])


@g.rule
def term():
    return factor() + zero_or_more(
        cfg.group(ws() + select(["*", "/"]) + ws() + factor())
    )


@g.rule
def expression():
    return term() + zero_or_more(
        cfg.group(ws() + select(["+", "-"]) + ws() + term())
    )


g.start("expression")

print("=" * 60)
print("Arithmetic Expression Grammar")
print("=" * 60)
g.pretty_print()

print()
print("Left-recursion analysis:")
cycles = g.detect_left_recursion()
if not cycles:
    print("  No left recursion detected ✓")
else:
    for c in cycles:
        print(f"  Cycle: {' -> '.join(c)}")
