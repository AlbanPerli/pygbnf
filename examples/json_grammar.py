#!/usr/bin/env python3
"""Example 3 — JSON-like grammar.

Demonstrates deep recursion, multiple rule references, string escaping,
character classes, and alternation.
"""

import pygbnf as cfg
from pygbnf import (
    CharacterClass,
    select,
    one_or_more,
    zero_or_more,
    optional,
)

g = cfg.Grammar()


@g.rule
def ws():
    return zero_or_more(select(" \t\n\r"))


@g.rule
def digit():
    return CharacterClass(pattern="0-9")


@g.rule
def digits():
    return one_or_more(digit())


@g.rule
def integer():
    return optional("-") + select([
        "0",
        CharacterClass(pattern="1-9") + zero_or_more(digit()),
    ])


@g.rule
def fraction():
    return optional(cfg.group("." + digits()))


@g.rule
def exponent():
    return optional(cfg.group(
        select("eE") + optional(select("+-")) + digits()
    ))


@g.rule
def json_number():
    return integer() + fraction() + exponent()


@g.rule
def hex_digit():
    return CharacterClass(pattern="0-9a-fA-F")


@g.rule
def escape():
    return "\\" + select([
        select('"\\\\/bfnrt'),
        "u" + hex_digit() + hex_digit() + hex_digit() + hex_digit(),
    ])


@g.rule
def safe_char():
    """Any char except " and \\ (simplified to printable ASCII range)."""
    return CharacterClass(pattern='^"\\\\', negated=False)


@g.rule
def json_string():
    return '"' + zero_or_more(select([safe_char(), escape()])) + '"'


@g.rule
def json_value():
    return ws() + select([
        json_string(),
        json_number(),
        json_object(),
        json_array(),
        "true",
        "false",
        "null",
    ]) + ws()


@g.rule
def pair():
    return ws() + json_string() + ws() + ":" + json_value()


@g.rule
def json_object():
    return "{" + select([
        ws() + "}",
        pair() + zero_or_more(cfg.group("," + pair())) + ws() + "}",
    ])


@g.rule
def json_array():
    return "[" + select([
        ws() + "]",
        json_value() + zero_or_more(cfg.group("," + json_value())) + ws() + "]",
    ])


g.start("json_value")

print("=" * 60)
print("JSON Grammar")
print("=" * 60)
g.pretty_print()
