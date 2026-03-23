#!/usr/bin/env python3
"""T() — constrain LLM output with f-string templates."""

import sys

import pygbnf as cfg
from pygbnf import T, GrammarLLM, int_range, line, select

g = cfg.Grammar()

@g.rule
def code_review():
    return  T(f"""Severity: {select(["critical", "major", "minor", "info"])}
Confidence: {int_range(0,10)}/10
Summary:
From architect point of view: 
{line()}
From dev point of view:
{line()}
From security point of view:
{line()}
Suggested fix:
{line():3}
""")

g.start("code_review")

# ── Grammar ──────────────────────────────────────────────────────────

print(g.to_gbnf())
print()

# ── LLM generation ───────────────────────────────────────────────────

BASE_URL = "http://localhost:8080/v1"

llm = GrammarLLM(BASE_URL)

for token, events in llm.stream(
    messages=[
        {"role": "system", "content": "You are a senior code reviewer. "},
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
