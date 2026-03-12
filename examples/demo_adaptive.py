#!/usr/bin/env python3
"""Example: Adaptive streaming — switch grammars based on function output.

Demonstrates ``GrammarLLM.adaptive_stream()``:

1. The LLM picks a tool call (e.g. ``list_cities`` or ``get_weather``).
2. The tool is dispatched and returns real data.
3. A new grammar is built dynamically from that data, constraining the
   LLM to summarise using only the actual values.

This shows how function return values can shape the next grammar.

Requires a running llama-server::

    llama-server -m your-model.gguf
"""

from __future__ import annotations

import json
from typing import List

import pygbnf as cfg
from pygbnf import (
    Grammar,
    GrammarLLM,
    RuleEvent,
    Toolkit,
    select,
)


# ── Fake backend data ───────────────────────────────────────────────

CITIES_DB = {
    "france": ["Paris", "Lyon", "Marseille", "Toulouse"],
    "japan": ["Tokyo", "Osaka", "Kyoto", "Sapporo"],
    "brazil": ["São Paulo", "Rio de Janeiro", "Brasília"],
}

WEATHER_DB = {
    "Paris": {"temp": "12°C", "condition": "cloudy"},
    "Tokyo": {"temp": "18°C", "condition": "sunny"},
    "Lyon": {"temp": "10°C", "condition": "rainy"},
}


# ── Tools ────────────────────────────────────────────────────────────

toolkit = Toolkit()


@toolkit.tool
def list_cities(country: str) -> str:
    """List known cities for a country."""
    cities = CITIES_DB.get(country.lower(), [])
    return json.dumps(cities)


@toolkit.tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    info = WEATHER_DB.get(city, {"temp": "unknown", "condition": "unknown"})
    return json.dumps(info)


# ── Grammar builder from function output ─────────────────────────────

def grammar_from_tool_result(fn_name: str, result: str) -> Grammar | None:
    """Build a follow-up grammar dynamically from the tool's return value.

    The LLM will be constrained to only use values actually returned
    by the tool — no hallucination possible.
    """
    g = Grammar()

    if fn_name == "list_cities":
        cities: List[str] = json.loads(result)
        if not cities:
            return None

        @g.rule
        def answer():
            # Build a fixed-length enumeration: "Paris, Lyon, Marseille and Toulouse."
            if len(cities) == 1:
                return select(cities) + "."
            items = select(cities)
            # first (n-1) items joined by ", " then " and " before the last
            parts = items
            for _ in range(len(cities) - 2):
                parts = parts + ", " + items
            parts = parts + " and " + items + "."
            return select(["The cities are ", "Known cities: ", ""]) + parts

        g.start("answer")
        return g

    if fn_name == "get_weather":
        info = json.loads(result)
        temp = info.get("temp", "unknown")
        condition = info.get("condition", "unknown")

        @g.rule
        def answer():
            # Constrained to the actual temperature and condition
            return (
                select(["The weather is ", "Currently ", "It is "])
                + condition
                + select([" at ", " with ", ", "])
                + temp
                + "."
            )

        g.start("answer")
        return g

    return None


# ── Adaptive callback ───────────────────────────────────────────────

def on_tool_call(ev: RuleEvent) -> Grammar | None:
    """Intercept the completed tool_call rule, dispatch it, and build
    a follow-up grammar from the result."""
    try:
        call = json.loads(ev.text)
    except json.JSONDecodeError:
        return None

    fn_name = call["function"]
    result = toolkit.dispatch(ev.text)

    print(f"\n  → {fn_name}() returned: {result}")

    next_grammar = grammar_from_tool_result(fn_name, result)
    if next_grammar:
        print(f"  → Switching to result-constrained grammar")
    return next_grammar


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    llm = GrammarLLM("http://localhost:8080/v1")

    prompt = "What cities are in japan? List them."
    print(f"Prompt: {prompt}")
    print(f"System: {toolkit.system_prompt()}")
    print()
    print("Output: ", end="")

    for token, events in llm.adaptive_stream(
        messages=[
            {"role": "system", "content": toolkit.system_prompt()},
            {"role": "user", "content": prompt},
        ],
        grammar=toolkit.grammar,
        on_switch={"tool_call": on_tool_call},
        n_predict=256,
    ):
        print(token, end="", flush=True)

    print()

