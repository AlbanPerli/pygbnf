#!/usr/bin/env python3
"""Demo — Schema-based constrained generation with GrammarLLM.

Part 1: Generate structured JSON from a dataclass schema.
Part 2: Tool calling — the LLM picks a function and its arguments,
        then the demo actually calls it.

Usage:
    PYTHONPATH=. python examples/demo_schema.py
"""

import enum
import json
import sys
from dataclasses import dataclass
from typing import Optional

from pygbnf import Grammar, GrammarLLM, grammar_from_type, select, describe_tools

# =====================================================================
# Part 1 — Dataclass → JSON grammar
# =====================================================================


class Sentiment(enum.Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class MovieReview:
    """Structured movie review."""
    title: str
    year: int
    rating: float
    sentiment: Sentiment
    summary: str


# =====================================================================
# Part 2 — Tool calling: multiple functions
# =====================================================================


class Units(enum.Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


def get_weather(city: str, units: Units = Units.CELSIUS) -> str:
    """Get current weather for a city."""
    u = units.value if isinstance(units, Units) else units
    return f"☀ 22°{u[0].upper()} and sunny in {city}"


def send_email(to: str, subject: str, body: str, urgent: bool = False) -> str:
    """Send an email to someone."""
    tag = " [URGENT]" if urgent else ""
    return f"✉ Email sent to {to}{tag}: {subject!r}"


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web and return results."""
    return f"🔍 Found {max_results} results for {query!r}"


TOOLS = {
    "get_weather": get_weather,
    "send_email": send_email,
    "search_web": search_web,
}

g_tools = Grammar()

@g_tools.rule
def root():
    return select([g_tools.from_tool_call(fn) for fn in TOOLS.values()])

g_tools.start("root")


# =====================================================================
# Main
# =====================================================================

llm = GrammarLLM("http://localhost:8080/v1")

# -- Part 1: dataclass ------------------------------------------------

g_review = grammar_from_type(MovieReview)

print("=" * 60)
print("Part 1 — Dataclass → JSON (MovieReview)")
print("=" * 60)
print()
print(g_review.to_gbnf())

print("Streaming LLM output:\n")

result = ""
for token, _ in llm.stream(
    messages=[
        {"role": "system", "content": "You generate structured JSON. No commentary."},
        {"role": "user", "content": "Write a review of the movie Inception in JSON."},
    ],
    grammar=g_review,
    n_predict=512,
):
    sys.stdout.write(token)
    sys.stdout.flush()
    result += token

print("\n")
parsed = json.loads(result)
print(f"  ✓ Valid JSON: {list(parsed.keys())}")
review = MovieReview(**parsed)
print(f"    → MovieReview(title={review.title!r}, year={review.year}, rating={review.rating})")
print()

# -- Part 2: tool calling ---------------------------------------------

TOOL_LIST = describe_tools(*TOOLS.values())

print("=" * 60)
print("Part 2 — Tool calling (single function)")
print("=" * 60)
print()
print("Available tools:\n")
print(TOOL_LIST)
print()

QUERIES = [
    "What's the weather like in Tokyo (in farhenient)?",
    "Send an urgent email to bob@acme.com about the Q3 report being ready.",
    "Search for 'python GBNF grammar' on the web.",
]

print("=" * 60)
print("Part 2 — Tool calling (multiple functions)")
print("=" * 60)
print()
print(g_tools.to_gbnf())

for query in QUERIES:
    print(f"User: {query}")
    print("LLM:  ", end="")

    result = ""
    for token, _ in llm.stream(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a tool-calling assistant. Given a user request, "
                    "reply with a JSON object choosing the right function and arguments.\n"
                    f"Available tools:\n{TOOL_LIST}"
                ),
            },
            {"role": "user", "content": query},
        ],
        grammar=g_tools,
        n_predict=256,
    ):
        sys.stdout.write(token)
        sys.stdout.flush()
        result += token

    print()

    call = json.loads(result)
    fn_name = call["function"]
    fn_args = call["arguments"]
    fn = TOOLS[fn_name]

    print(f"  → calling {fn_name}({fn_args})")
    output = fn(**fn_args)
    print(f"  → {output}")
    print()
