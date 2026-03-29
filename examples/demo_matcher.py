#!/usr/bin/env python3
"""Demo — GrammarLLM + RegularMatcher + GrammarFST on streamed output.

Shows three complementary layers of the same grammar:

1. GBNF text          — grammar sent to the LLM for constrained generation
2. RegularMatcher     — live Python-side NFA matching as tokens arrive
                        (uses the same explicit-char NFA as the FST export)
3. GrammarFST export  — TensorAutomata AT&T transducer files + Julia script
                        for offline composition / grammar identification

Usage:
    python examples/demo_matcher.py [--no-llm]

    --no-llm   Skip the LLM streaming demo and only show the FST export.
"""

import os
import sys
import tempfile

# Prefer the local development tree over any installed package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pygbnf import (
    CharacterClass,
    Grammar,
    GrammarFST,
    GrammarLLM,
    RegularMatcher,
    group,
    one_or_more,
    optional,
    select,
    zero_or_more,
)


BASE_URL = "http://localhost:8080/v1"


# ── Grammar ───────────────────────────────────────────────────────────


def build_calculation_grammar() -> Grammar:
    g = Grammar()

    @g.rule
    def ws():
        return zero_or_more(select(" \t"))

    @g.rule
    def digit():
        return CharacterClass(pattern="0-9")

    @g.rule
    def integer():
        return optional("-") + one_or_more(digit())

    @g.rule
    def operator():
        return select(["+", "-", "*", "/"])

    @g.rule
    def expression():
        return integer() + zero_or_more(group(ws() + operator() + ws() + integer()))

    @g.rule
    def result():
        return expression() + ws() + "=" + ws() + integer()

    @g.rule
    def calculation():
        return "calc: " + result()

    g.start("calculation")
    return g


# ── FST export demo ───────────────────────────────────────────────────


def demo_fst_export(grammar: Grammar) -> None:
    """Export the grammar as TensorAutomata AT&T transducer files."""
    print("=" * 60)
    print("GrammarFST — TensorAutomata export")
    print("=" * 60)

    fst = GrammarFST(grammar, only={"integer", "operator", "expression", "calculation"})

    # Show the integer transducer (2-tape: tape1=chars, tape2=rule label)
    print("\n--- integer transducer (AT&T, first 10 lines) ---")
    att_lines = fst.to_att("integer").splitlines()
    for line in att_lines[:10]:
        print(" ", line)
    if len(att_lines) > 10:
        print(f"  ... ({len(att_lines)} lines total)")

    # Show the linear acceptor for a sample string
    sample = "3 + 14"
    print(f"\n--- linear acceptor for {sample!r} ---")
    print(fst.linear_acceptor(sample).to_att_string())

    # Write all files to a temp directory
    with tempfile.TemporaryDirectory(prefix="pygbnf_fst_") as d:
        fst.write_all(d)
        files = sorted(os.listdir(d))
        print(f"\n--- files written to {d} ---")
        for f in files:
            path = os.path.join(d, f)
            size = os.path.getsize(path)
            print(f"  {f:20s}  ({size} bytes)")

        # Preview the Julia script
        print("\n--- compose.jl (first 25 lines) ---")
        with open(os.path.join(d, "compose.jl")) as jl:
            for i, line in enumerate(jl):
                if i >= 25:
                    print("  ...")
                    break
                print(" ", line, end="")
    print()


# ── RegularMatcher demo (offline, no LLM) ────────────────────────────


def demo_offline_matching(grammar: Grammar) -> None:
    """Show the RegularMatcher working on a pre-recorded string."""
    print("=" * 60)
    print("RegularMatcher — offline simulation")
    print("=" * 60)

    sample_output = "calc: 12 + 7 = 19"
    print(f"\nSimulated LLM output (char by char): {sample_output!r}\n")

    matcher = RegularMatcher(
        grammar,
        only={"integer", "operator", "expression", "result", "calculation"},
    )

    matched: list = []

    def _on_event(ev):
        matched.append(ev)
        print(
            f"  [{ev.start:2d}:{ev.end:2d}]  rule={ev.rule:<12s}  text={ev.text!r}"
        )

    matcher.on("*", _on_event)

    print("  pos   rule         text")
    print("  " + "-" * 45)
    for ch in sample_output:
        matcher.feed(ch)

    print(f"\n  {len(matched)} rule event(s) fired.")

    # Cross-check with FST linear acceptor
    print("\n--- FST consistency check ---")
    fst = GrammarFST(grammar, only={"integer", "operator", "expression", "calculation"})
    for rule in ("integer", "operator", "expression"):
        att = fst.rule_transducer(rule)
        print(f"  T_{rule}: {att.num_states()} states, "
              f"{len(att.transitions)} transitions, "
              f"{len(att.finals)} final(s)")
    print()


# ── LLM streaming demo ────────────────────────────────────────────────


def demo_llm_streaming(grammar: Grammar) -> None:
    """Stream from a local LLM and match in real time."""
    print("=" * 60)
    print("GrammarLLM + RegularMatcher — live streaming")
    print("=" * 60)

    llm = GrammarLLM(BASE_URL)
    matcher = RegularMatcher(grammar, only={"calculation", "operator", "integer"})

    matcher.on(
        "calculation",
        lambda ev: print(
            f"\n  [matched] {ev.rule}: {ev.text!r} ({ev.start}:{ev.end})"
        ),
    )
    matcher.on(
        "operator",
        lambda ev: print(f"\n  [matched] operator: {ev.text!r}"),
    )
    matcher.on(
        "integer",
        lambda ev: print(f"\n  [matched] integer: {ev.text!r}"),
    )

    print("\nStreaming:\n")
    text = ""
    try:
        for token, _ in llm.stream(
            messages=[
                {
                    "role": "system",
                    "content": "You output one arithmetic calculation only. No commentary.",
                },
                {
                    "role": "user",
                    "content": "Produce a short calculation in the exact form `calc: A + B = C`.",
                },
            ],
            grammar=grammar,
            n_predict=128,
        ):
            sys.stdout.write(token)
            sys.stdout.flush()
            text += token
            matcher.feed(token)
    except Exception as exc:
        raise RuntimeError(
            "Streaming request failed.\n"
            "Make sure your OpenAI-compatible server is reachable at "
            f"`{BASE_URL}` and supports the `grammar` field.\n"
            "Run with --no-llm to skip this step."
        ) from exc

    print(f"\n\nFinal output: {text!r}\n")


# ── Entry point ───────────────────────────────────────────────────────


def main() -> None:
    no_llm = "--no-llm" in sys.argv

    grammar = build_calculation_grammar()

    print("\n--- GBNF grammar ---")
    print(grammar.to_gbnf())

    demo_fst_export(grammar)
    demo_offline_matching(grammar)

    if no_llm:
        print("(LLM streaming skipped — remove --no-llm to enable)")
    else:
        demo_llm_streaming(grammar)


if __name__ == "__main__":
    main()
