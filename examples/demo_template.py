#!/usr/bin/env python3
"""T() — constrain LLM output with f-string templates."""

import sys

import pygbnf as cfg
from pygbnf import T, GrammarLLM, one_or_more, select, number, line

g = cfg.Grammar()


@g.rule
def code_review():
    return T(f"""Severity: {select(["critical", "major", "minor", "info"])}
Confidence: {number()}/10
Summary: {line("")}
Affected files:
{line("-"):+}
Suggested fix:
{line("-"):+}
""")


g.start("code_review")

# ── Grammar ──────────────────────────────────────────────────────────

print(g.to_gbnf())
print()

# ── LLM generation ───────────────────────────────────────────────────

BASE_URL = "http://localhost:8080/v1"

llm = GrammarLLM(BASE_URL)

for token, _ in llm.stream(
    messages=[
        {"role": "system", "content": "You are a senior code reviewer. "
         "Respond using ONLY the format provided by the grammar."},
        {"role": "user", "content": "Review this code:\n\n"
         "def login(user, pwd):\n"
         "    query = f\"SELECT * FROM users WHERE name='{user}' AND pass='{pwd}'\"\n"
         "    return db.execute(query)"},
    ],
    grammar=g,
    n_predict=512,
):
    sys.stdout.write(token)
    sys.stdout.flush()

print()
