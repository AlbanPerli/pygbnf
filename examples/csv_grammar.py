#!/usr/bin/env python3
"""Example 2 — CSV grammar.

Demonstrates repetition, alternation, character classes, and escaping.
"""

import pygbnf as cfg
from pygbnf import select, one_or_more, zero_or_more, optional, CharacterClass

g = cfg.Grammar()


@g.rule
def ws():
    return zero_or_more(select(" \t"))


@g.rule
def newline():
    return select(["\r\n", "\n"])


@g.rule
def unquoted_field():
    return zero_or_more(CharacterClass(pattern='^,\\n\\r"', negated=False))


@g.rule
def escaped_char():
    return '""'  # doubled quote inside quoted string


@g.rule
def quoted_content():
    return select([
        CharacterClass(pattern='^"', negated=False),
        escaped_char(),
    ])


@g.rule
def quoted_field():
    return '"' + zero_or_more(quoted_content()) + '"'


@g.rule
def field():
    return ws() + select([quoted_field(), unquoted_field()]) + ws()


@g.rule
def record():
    return field() + zero_or_more(cfg.group("," + field()))


@g.rule
def csv():
    return record() + zero_or_more(cfg.group(newline() + record())) + optional(newline())


g.start("csv")

print("=" * 60)
print("CSV Grammar")
print("=" * 60)
g.pretty_print()
