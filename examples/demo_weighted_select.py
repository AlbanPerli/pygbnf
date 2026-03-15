#!/usr/bin/env python3
"""Example: weighted_select — biased dish suggestion.

A restaurant chatbot suggests a dish.  The grammar allows 5 dishes,
but ``weighted_select`` steers the model towards today's special
("risotto") and away from an out-of-stock item ("burger").

Run 1 — plain ``select``:  the model picks by its own preference.
Run 2 — ``weighted_select``:  "risotto" dominates, "burger" nearly vanishes.

Requires a running llama-server::

    llama-server -m your-model.gguf
"""

from __future__ import annotations

import sys
from collections import Counter

from pygbnf import Grammar, GrammarLLM, select, weighted_select

BASE_URL = "http://localhost:8080/v1"
N = 20  # samples per condition

DISHES = ["pizza", "burger", "salad", "Meat"]

# Today's special is risotto (×25), burger is out of stock (×0.02).
WEIGHTS = {
    "pizza":   1.0,
    "burger":  0.2,   # almost out of stock
    "salad":   5.0,
    "Meat":   1.0,
}

MESSAGES = [
    {"role": "system", "content": "You are a restaurant waiter in a (salad) vegan restaurant. Suggest ONE dish."},
    {"role": "user", "content": "I love vegetables, what should I order today?"},
]


# ── Grammars ─────────────────────────────────────────────────────────

def make_grammar(weighted: bool) -> Grammar:
    g = Grammar()

    @g.rule
    def dish():
        return weighted_select(WEIGHTS) if weighted else select(DISHES)

    @g.rule
    def root():
        return "I recommend the " + dish() + "."

    return g


# ── Run ──────────────────────────────────────────────────────────────

llm = GrammarLLM(BASE_URL)


text, _ = llm.complete(messages=MESSAGES, grammar= make_grammar(weighted=False), temperature=0.5)
print(text)
text, _ = llm.complete(messages=MESSAGES, grammar= make_grammar(weighted=True), temperature=0.5)
print(text)
