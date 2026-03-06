#!/usr/bin/env python3
"""Example from the specification — quick-start demo."""

import pygbnf as cfg
from pygbnf import select, one_or_more, zero_or_more

g = cfg.Grammar()


@g.rule
def number():
    n = one_or_more(select("0123456789"))
    return select(['-' + n, n])


@g.rule
def operator():
    return select(['+', '*', '**', '/', '-'])


@g.rule
def expression():
    return select([
        number(),
        expression() + zero_or_more(" ") + operator() + zero_or_more(" ") + expression(),
        "(" + expression() + ")"
    ])


g.start("expression")

print(g.to_gbnf())
