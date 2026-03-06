#!/usr/bin/env python3
"""Example 5 — Token constraints for llama.cpp thinking blocks."""

import pygbnf as cfg
from pygbnf import token, not_token, token_id, not_token_id, zero_or_more

g = cfg.Grammar()


@g.rule
def thinking():
    return zero_or_more(not_token("think"))


@g.rule
def root():
    return token("think") + thinking() + token("think")


g.start("root")

print("=" * 60)
print("Token constraints (thinking block)")
print("=" * 60)
g.pretty_print()
