#!/usr/bin/env python3
"""
demo_chain.py — GrammarChain et GrammarSpace.

Deux modes :
  1. Impératif  — chain |= contexte, chain >> type
  2. Déclaratif — GrammarSpace(q0, F, δ) avec états entiers
"""

from dataclasses import dataclass
from typing import Literal

import pygbnf as cfg
from pygbnf import GrammarChain, GrammarSpace, GrammarLLM

BASE_URL = "http://localhost:8080/v1"
CODE = '''
def login(user, pwd):
    query = f"SELECT * FROM users WHERE name=\'{user}\' AND pass=\'{pwd}\'"
    return db.execute(query).fetchone()
'''

# ── Types ─────────────────────────────────────────────────────────────

@dataclass
class Category:
    value: Literal["security", "performance", "style"]

@dataclass
class Severity:
    value: Literal["critical", "high", "medium", "low"]

@dataclass
class Strategy:
    value: Literal["immediate-patch", "hotfix", "refactor", "document"]

@dataclass
class Diagnostic:
    text: str

@dataclass
class SecurityStrategy:
    value: Literal["immediate-patch", "hotfix", "rollback"]

@dataclass
class StyleStrategy:
    value: Literal["refactor", "document"]


# ── Mode 1 : impératif ────────────────────────────────────────────────

def run_imperative(llm):
    print("=" * 60)
    print("MODE IMPÉRATIF")
    print("=" * 60)

    chain = GrammarChain(llm, stream=True)
    chain |= f"You are a code reviewer.\nReview:\n{CODE}"

    category = chain >> Category
    print(f"\n  category  = {category}")

    severity = chain >> Severity
    print(f"\n  severity  = {severity}")

    strategy = chain >> Strategy
    print(f"\n  strategy  = {strategy}")

    diag = chain >> str
    print(f"\n  diag      = {diag}")

    print("\nTrace:")
    for e in chain.ctx:
        preview = repr(e.value)[:60]
        print(f"  state={e.arc[0]}  → {preview}")


# ── Mode 2 : déclaratif linéaire ──────────────────────────────────────

def run_declarative(llm):
    print("\n" + "=" * 60)
    print("MODE DÉCLARATIF (linéaire)")
    print("=" * 60)

    # États : 0=Category 1=Severity 2=Strategy 3=Diagnostic [final]
    space = GrammarSpace(
        q0 = 0,
        F  = {3},
        δ  = [
            (0, 1, Category),
            (1, 2, Severity),
            (2, 3, Strategy),
            (3, None, Diagnostic),
        ]
    )

    runner = space.run(llm, stream=True)
    runner |= f"You are a code reviewer.\nReview:\n{CODE}"
    ctx = runner.execute()

    STATE_NAMES = {0: "Category", 1: "Severity", 2: "Strategy", 3: "Diagnostic"}
    print("\nTrace:")
    for e in ctx:
        name = STATE_NAMES.get(e.arc[0], str(e.arc[0]))
        preview = repr(e.value)[:60]
        print(f"  [{name:12s}] → {preview}")


# ── Mode 3 : déclaratif avec branchement conditionnel ─────────────────

def run_conditional(llm):
    print("\n" + "=" * 60)
    print("MODE DÉCLARATIF (branchement conditionnel)")
    print("=" * 60)

    # États : 0=Category 1=Severity 2=SecurityStrategy 3=StyleStrategy
    # 4=Diagnostic [final]
    space = GrammarSpace(
        q0 = 0,
        F  = {4},
        δ  = [
            (0, 1, Category),
            (1, 2, Severity, lambda v, ctx: any(
                e.arc[2] is Category and e.value.value == "security"
                for e in ctx
            )),
            (1, 3, Severity),
            (2, 4, SecurityStrategy),
            (3, 4, StyleStrategy),
            (4, None, Diagnostic),
        ]
    )

    runner = space.run(llm, stream=True)
    runner |= f"You are a code reviewer.\nReview:\n{CODE}"
    ctx = runner.execute()

    STATE_NAMES = {
        0: "Category", 1: "Severity", 2: "SecStrategy",
        3: "StyleStrategy", 4: "Diagnostic",
    }
    print("\nTrace:")
    for e in ctx:
        name = STATE_NAMES.get(e.arc[0], str(e.arc[0]))
        preview = repr(e.value)[:60]
        print(f"  [{name:12s}] → {preview}")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    llm = GrammarLLM(BASE_URL)
    run_imperative(llm)
    run_declarative(llm)
    run_conditional(llm)
