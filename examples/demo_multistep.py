#!/usr/bin/env python3
"""
demo_multistep.py — Grammaire conditionnelle tour par tour.

Chaque tour utilise une petite grammaire conditionnée par les choix précédents.
Le KV cache du serveur absorbe le coût du contexte croissant.

Scénario : arbre de décision pour une revue de code.
  Tour 1 : catégorie  (security / performance / style)
  Tour 2 : sévérité   (conditionné par catégorie)
  Tour 3 : stratégie  (conditionné par catégorie + sévérité)
  Tour 4 : diagnostic (une ligne libre, conditionné par tout le reste)
"""

import sys
import time

import pygbnf as cfg
from pygbnf import GrammarLLM, line, select

BASE_URL = "http://localhost:8080/v1"

SYSTEM = "You are a senior code reviewer. Answer each question concisely."
CODE = '''
def login(user, pwd):
    query = f"SELECT * FROM users WHERE name=\'{user}\' AND pass=\'{pwd}\'"
    return db.execute(query).fetchone()

def register(user, pwd):
    db.execute(f"INSERT INTO users VALUES (\'{user}\', \'{pwd}\')")
    db.commit()
'''

# ── Grammaires ────────────────────────────────────────────────────────

SEVERITY_BY_CATEGORY = {
    "security":    ["critical", "high", "medium"],
    "performance": ["high", "medium", "low"],
    "style":       ["major", "minor"],
}

STRATEGY_BY_SEVERITY = {
    "critical": ["immediate-patch", "rollback", "hotfix"],
    "high":     ["patch-next-release", "workaround", "hotfix"],
    "medium":   ["patch-next-release", "refactor", "document"],
    "low":      ["refactor", "document", "ignore"],
    "major":    ["refactor", "document"],
    "minor":    ["document", "ignore"],
}

def g_category():
    g = cfg.Grammar()
    @g.rule
    def category():
        return select(["security", "performance", "style"])
    g.start("category")
    return g

def g_severity(category):
    g = cfg.Grammar()
    options = SEVERITY_BY_CATEGORY[category]
    @g.rule
    def severity():
        return select(options)
    g.start("severity")
    return g

def g_strategy(severity):
    g = cfg.Grammar()
    options = STRATEGY_BY_SEVERITY[severity]
    @g.rule
    def strategy():
        return select(options)
    g.start("strategy")
    return g

def g_diagnostic():
    g = cfg.Grammar()
    @g.rule
    def diagnostic():
        return line()
    g.start("diagnostic")
    return g


# ── Helpers ───────────────────────────────────────────────────────────

def step(llm, messages, grammar, label):
    t0 = time.perf_counter()
    ttft = None
    result = ""
    for token, _ in llm.stream(messages=messages, grammar=grammar):
        if ttft is None:
            ttft = time.perf_counter() - t0
        sys.stdout.write(token)
        sys.stdout.flush()
        result += token
    total = time.perf_counter() - t0
    print(f"  ← [{label}] TTFT={ttft:.3f}s total={total:.3f}s")
    return result.strip(), ttft, total


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    llm = GrammarLLM(BASE_URL)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f"Review this code:\n{CODE}\nWhat is the problem category?"},
    ]

    print("=" * 60)

    # Tour 1 — catégorie
    category, ttft1, t1 = step(llm, messages, g_category(), "tour 1")

    messages += [
        {"role": "assistant", "content": category},
        {"role": "user",      "content": f"Given category '{category}', what is the severity?"},
    ]

    # Tour 2 — sévérité
    severity, ttft2, t2 = step(llm, messages, g_severity(category), "tour 2")

    messages += [
        {"role": "assistant", "content": severity},
        {"role": "user",      "content": f"Given severity '{severity}', what is the recommended strategy?"},
    ]

    # Tour 3 — stratégie
    strategy, ttft3, t3 = step(llm, messages, g_strategy(severity), "tour 3")

    messages += [
        {"role": "assistant", "content": strategy},
        {"role": "user",      "content": "Write a one-line diagnostic summarizing the issue."},
    ]

    # Tour 4 — diagnostic libre
    diagnostic, ttft4, t4 = step(llm, messages, g_diagnostic(), "tour 4")

    print("=" * 60)
    print(f"  Catégorie  : {category}")
    print(f"  Sévérité   : {severity}")
    print(f"  Stratégie  : {strategy}")
    print(f"  Diagnostic : {diagnostic}")
    print()
    print(f"  TTFT par tour : {ttft1:.3f}s / {ttft2:.3f}s / {ttft3:.3f}s / {ttft4:.3f}s")
    print(f"  Total         : {t1+t2+t3+t4:.3f}s")
