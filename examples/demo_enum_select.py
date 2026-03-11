#!/usr/bin/env python3
"""Demo — LLM selects multiple values from an Enum.

The grammar constrains the LLM to output a JSON object whose
``tags`` field is a list of valid ``Tag`` enum values.

Usage:
    PYTHONPATH=. python examples/demo_enum_select.py
"""

import enum
import json
import sys
from dataclasses import dataclass
from typing import List

from pygbnf import GrammarLLM, grammar_from_type


class Tag(enum.Enum):
    URGENT = "urgent"
    BACKEND = "backend"
    FRONTEND = "frontend"
    BUGFIX = "bugfix"
    FEATURE = "feature"
    DOCS = "docs"
    SECURITY = "security"
    PERFORMANCE = "performance"


@dataclass
class TicketTags:
    tags: List[Tag]


g = grammar_from_type(TicketTags)

print("Grammar:\n")
print(g.to_gbnf())
print()

llm = GrammarLLM("http://localhost:8080/v1")

TICKETS = [
    "The login page crashes on Safari when using SSO.",
    "Add dark mode toggle to the settings panel.",
    "SQL queries on the dashboard take 12 seconds to load.",
    "Update the README with the new API endpoints.",
]

for ticket in TICKETS:
    print(f"Ticket: {ticket}")
    print("Tags:   ", end="")

    result = ""
    for token, _ in llm.stream(
        messages=[
            {"role": "system", "content": (
                "You are a ticket tagger. Given a ticket description, "
                "select all relevant tags. Reply with JSON only."
            )},
            {"role": "user", "content": ticket},
        ],
        grammar=g,
        n_predict=128,
    ):
        sys.stdout.write(token)
        sys.stdout.flush()
        result += token

    parsed = json.loads(result)
    tags = [Tag(t) for t in parsed["tags"]]
    print(f"\n  → {[t.value for t in tags]}")
    print()
