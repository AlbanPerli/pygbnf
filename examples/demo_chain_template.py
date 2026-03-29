#!/usr/bin/env python3
"""
demo_chain_template.py — GrammarChain + templates T().

Chaque étape utilise une grammaire T() comme contrainte.
Les étapes sont conditionnées par les choix précédents via le GrammarSpace.

Scénario : analyse de code structurée en deux phases.
  Phase 1 — triage  : catégorie + sévérité en une seule grammaire T()
  Phase 2 — rapport : template conditionné par la phase 1
"""

import sys

import pygbnf as cfg
from pygbnf import GrammarChain, GrammarSpace, GrammarLLM, T, select, line, int_range

BASE_URL = "http://localhost:8080/v1"

CODE = '''
def login(user, pwd):
    query = f"SELECT * FROM users WHERE name=\'{user}\' AND pass=\'{pwd}\'"
    return db.execute(query).fetchone()

def register(user, pwd):
    db.execute(f"INSERT INTO users VALUES (\'{user}\', \'{pwd}\')")
    db.commit()
'''

# ── Grammaires T() ────────────────────────────────────────────────────

def make_triage_grammar():
    g = cfg.Grammar()

    @g.rule
    def triage():
        return T(f"""Category: {select(["security", "performance", "style"])}
Severity: {select(["critical", "high", "medium", "low"])}
Confidence: {int_range(0, 10)}/10""")

    g.start("triage")
    return g


def make_security_report_grammar():
    g = cfg.Grammar()

    @g.rule
    def report():
        return T(f"""Vulnerability: {line()}
Attack vector: {select(["network", "local", "physical"])}
Fix: {line():1,3}
Recommendation: {line()}""")

    g.start("report")
    return g


def make_generic_report_grammar():
    g = cfg.Grammar()

    @g.rule
    def report():
        return T(f"""Issue: {line()}
Impact: {line()}
Fix: {line():1,3}""")

    g.start("report")
    return g


# ── Mode impératif avec T() ───────────────────────────────────────────

def run_imperative(llm):
    print("=" * 60)
    print("IMPÉRATIF + T()")
    print("=" * 60)

    chain = GrammarChain(llm, stream=True)
    chain |= f"You are a senior code reviewer.\nReview:\n{CODE}"

    print("\n--- Triage ---")
    triage = chain >> make_triage_grammar()
    print(f"\n  → {triage!r}")

    # Choix conditionnel basé sur le triage
    is_security = "security" in triage.lower()
    grammar = make_security_report_grammar() if is_security else make_generic_report_grammar()

    print(f"\n--- Rapport ({'security' if is_security else 'generic'}) ---")
    report = chain >> grammar
    print(f"\n  → {report!r}")


# ── Mode déclaratif avec T() ──────────────────────────────────────────

def run_declarative(llm):
    print("\n" + "=" * 60)
    print("DÉCLARATIF + T()")
    print("=" * 60)

    triage_g    = make_triage_grammar()
    security_g  = make_security_report_grammar()
    generic_g   = make_generic_report_grammar()

    def next_report(ctx):
        triage_text = next(
            (e.value for e in ctx if e.state is triage_g), ""
        )
        return security_g if "security" in triage_text.lower() else generic_g

    space = GrammarSpace(
        Q  = {triage_g, security_g, generic_g},
        q0 = triage_g,
        F  = {security_g, generic_g},
        δ  = {
            triage_g: next_report,
        }
    )

    runner = space.run(llm, stream=True)
    runner |= f"You are a senior code reviewer.\nReview:\n{CODE}"

    print("\n--- Triage ---")
    ctx = runner.execute()

    print("\nTrace:")
    for e in ctx:
        label = getattr(e.state, '_start', None) or repr(e.state)[:30]
        print(f"  {label:30s} → {e.value!r}")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    llm = GrammarLLM(BASE_URL)
    run_imperative(llm)
    run_declarative(llm)
