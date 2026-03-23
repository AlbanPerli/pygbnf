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

from pygbnf import GrammarLLM, Toolkit, grammar_from_type


def parse_json_output(raw: str, *, context: str):
    """Parse streamed JSON output and fail with a useful diagnostic."""
    text = raw.strip()
    if not text:
        raise RuntimeError(
            f"{context}: the LLM returned no content.\n"
            "The backend may be running, but it did not emit any text.\n"
            "Check that your server actually supports the `grammar` field and that "
            "the selected model is producing content."
        )
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{context}: invalid JSON returned by the LLM.\n"
            f"Raw output: {text!r}"
        ) from exc


def stream_text(llm: GrammarLLM, **kwargs) -> str:
    """Collect streamed text and raise a clearer error if the backend fails."""
    text = ""
    try:
        for token, _ in llm.stream(**kwargs):
            sys.stdout.write(token)
            sys.stdout.flush()
            text += token
    except Exception as exc:
        raise RuntimeError(
            "Streaming request failed.\n"
            "Make sure your OpenAI-compatible server is reachable at "
            "`http://localhost:8080/v1` and supports the `grammar` field."
        ) from exc
    return text

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
# Part 2 — Tool calling with Toolkit decorator
# =====================================================================

toolkit = Toolkit()


class Units(enum.Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


@toolkit.tool
def get_weather(city: str, units: Units = Units.CELSIUS) -> str:
    """Get current weather for a city."""
    u = units.value if isinstance(units, Units) else units
    return f"☀ 22°{u[0].upper()} and sunny in {city}"


@toolkit.tool
def send_email(to: str, subject: str, body: str, urgent: bool = False) -> str:
    """Send an email to someone."""
    tag = " [URGENT]" if urgent else ""
    return f"✉ Email sent to {to}{tag}: {subject!r}"


@toolkit.tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web and return results."""
    return f"🔍 Found {max_results} results for {query!r}"


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    llm = GrammarLLM("http://localhost:8080/v1")

    # -- Part 1: dataclass ------------------------------------------------

    g_review = grammar_from_type(MovieReview)

    print("=" * 60)
    print("Part 1 — Dataclass → JSON (MovieReview)")
    print("=" * 60)
    print()
    print(g_review.to_gbnf())

    print("Streaming LLM output:\n")

    result = stream_text(
        llm,
        messages=[
            {"role": "system", "content": "You generate structured JSON. No commentary."},
            {"role": "user", "content": "Write a review of the movie Inception in JSON."},
        ],
        grammar=g_review,
        match=True,
        n_predict=512,
    )

    print("\n")
    parsed = parse_json_output(result, context="Part 1")
    print(f"  ✓ Valid JSON: {list(parsed.keys())}")
    parsed["sentiment"] = Sentiment(parsed["sentiment"])
    review = MovieReview(**parsed)
    print(
        f"    → MovieReview(title={review.title!r}, year={review.year}, "
        f"rating={review.rating}, sentiment={review.sentiment})"
    )
    print()

    # -- Part 2: tool calling with Toolkit ------------------------------------

    print("=" * 60)
    print("Part 2 — Tool calling with @toolkit.tool")
    print("=" * 60)
    print()
    print("Available tools:\n")
    print(toolkit.describe())
    print()
    print("Grammar:\n")
    print(toolkit.grammar.to_gbnf())

    queries = [
        "What's the weather like in NY (in farenheit)?",
        "Send an urgent email to bob@acme.com about the Q3 report being ready.",
        "Search for 'python GBNF grammar' on the web.",
    ]

    for query in queries:
        print(f"User: {query}")
        print("LLM:  ", end="")

        result = stream_text(
            llm,
            messages=[{"role": "user", "content": query}],
            toolkit=toolkit,
            n_predict=256,
        )

        call = parse_json_output(result, context=f"Tool call for query {query!r}")
        print(f"\n  → calling {call['function']}({call['arguments']})")
        output = toolkit.dispatch(result)
        print(f"  → {output}")
        print()


if __name__ == "__main__":
    main()
