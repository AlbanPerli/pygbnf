#!/usr/bin/env python3
"""
demo_react.py — ReAct loop avec outils via GrammarSpace.

Signal de routage structuré : Act.next = Literal["observe", "conclude"]
Le contenu (reasoning) est libre, seul le signal de navigation est contraint.

Automate :
  0:think → 1:toolkit → 2:observe → 3:Act ──next="observe"──→ 1
                                           └──next="conclude"──→ 4:conclude [final]
"""

from dataclasses import dataclass
from typing import Literal

import pygbnf as cfg
from pygbnf import GrammarSpace, GrammarLLM, Toolkit, T, line, write_space_svg

BASE_URL = "http://localhost:8080/v1"

QUESTION = "How should we handle authentication in a Python web API?"

# ── Signal de routage ─────────────────────────────────────────────────

@dataclass
class Act:
    reasoning: str                           # libre
    next: Literal["observe", "conclude"]     # signal de routage contraint

# ── Outils ────────────────────────────────────────────────────────────

toolkit = Toolkit()

@toolkit.tool
def search_docs(query: str) -> str:
    """Search the Python security documentation."""
    results = {
        "jwt":        "JWT (JSON Web Tokens) — stateless, signed tokens. Use PyJWT or python-jose.",
        "session":    "Session-based auth — server-side state. Use Flask-Login or Django sessions.",
        "oauth2":     "OAuth2 — delegated auth. Use authlib or python-social-auth.",
        "password":   "Password hashing — use bcrypt or argon2-cffi. Never store plaintext.",
        "middleware": "Auth middleware — FastAPI dependencies, Flask before_request hooks.",
    }
    key = next((k for k in results if k in query.lower()), None)
    return results[key] if key else f"No results for '{query}'."

@toolkit.tool
def check_vulnerability(code_snippet: str) -> str:
    """Check a code snippet for common authentication vulnerabilities."""
    issues = []
    if "md5" in code_snippet.lower():
        issues.append("MD5 is cryptographically broken — use bcrypt or argon2.")
    if "sha1" in code_snippet.lower():
        issues.append("SHA1 is weak for passwords — use bcrypt or argon2.")
    if "secret" in code_snippet.lower():
        issues.append("Possible hardcoded secret — use environment variables.")
    return " | ".join(issues) if issues else "No obvious vulnerabilities detected."

# ── Grammaires ────────────────────────────────────────────────────────

def make_think_grammar():
    g = cfg.Grammar()
    @g.rule
    def think():
        return T(f"""<think>
{line():1,3}
</think>""")
    g.start("think")
    return g

def make_observe_grammar():
    g = cfg.Grammar()
    @g.rule
    def observe():
        return T(f"""Observation:
{line():1,3}""")
    g.start("observe")
    return g

def make_conclude_grammar():
    g = cfg.Grammar()
    @g.rule
    def conclude():
        return T(f"""Conclusion:
{line():2,5}""")
    g.start("conclude")
    return g

think_g    = make_think_grammar()
observe_g  = make_observe_grammar()
conclude_g = make_conclude_grammar()

# ── Automate ──────────────────────────────────────────────────────────

MAX_ITERATIONS = 3

def _act_count(ctx):
    return sum(1 for e in ctx if e.arc[0] == 3)

space = GrammarSpace(
    q0 = 0,
    F  = {4},
    δ  = [
        (0, 1, think_g),
        (1, 2, toolkit),
        (2, 3, observe_g),
        (3, 1, Act, lambda v, ctx: v.next == "observe" and _act_count(ctx) < MAX_ITERATIONS),
        (3, 4, Act, lambda v, ctx: v.next == "conclude" or  _act_count(ctx) >= MAX_ITERATIONS),
        (4, 5, conclude_g),
    ]
)

STATE_NAMES = {
    0: "think",
    1: "toolkit",
    2: "observe",
    3: "act",
    4: "act→conclude",
    5: "conclude",
}

# ── Run ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    llm = GrammarLLM(BASE_URL)

    runner = space.run(llm, stream=True)
    runner |= (
        f"You are a careful security expert. Reason step by step.\\n"
        f"At each Act step, set next=observe to keep researching, "
        f"or next=conclude when you have enough to answer.\\n"
        f"Question: {QUESTION}"
    )
    ctx = runner.execute()

    print("\n" + "=" * 60)
    print("TRACE")
    print("=" * 60)
    for e in ctx:
        name = STATE_NAMES.get(e.arc[0], str(e.arc[0]))
        preview = repr(e.value)[:80]
        print(f"  [{name:12s}] {preview}")

    write_space_svg(space, "react_space.svg", state_names=STATE_NAMES)
    print("\nDiagram → react_space.svg")
