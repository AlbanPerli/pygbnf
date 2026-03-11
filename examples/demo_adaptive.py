#!/usr/bin/env python3
"""Example: Adaptive streaming — switch grammars mid-generation.

Demonstrates ``GrammarLLM.adaptive_stream()``: the LLM first picks a
response format (json or text), then the grammar switches dynamically
to constrain the rest of the output accordingly.

Requires a running llama-server::

    llama-server -m your-model.gguf
"""

import pygbnf as cfg
from pygbnf import Grammar, GrammarLLM, RuleEvent, select, one_or_more

# ── Phase 1 grammar: choose a format ────────────────────────────────

g1 = Grammar()


@g1.rule
def format_choice():
    return select(["json", "text"]) + ": "


g1.start("format_choice")


# ── Phase 2 grammars: constrained output per format ─────────────────

g_json = Grammar()


@g_json.rule
def json_output():
    key = '"' + one_or_more(select("abcdefghijklmnopqrstuvwxyz_")) + '"'
    value = '"' + one_or_more(select(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    )) + '"'
    pair = key + ": " + value
    return "{" + pair + cfg.zero_or_more(", " + pair) + "}"


g_json.start("json_output")


g_text = Grammar()


@g_text.rule
def text_output():
    return one_or_more(select(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'-"
    ))


g_text.start("text_output")


# ── Adaptive callback ───────────────────────────────────────────────

def on_format(ev: RuleEvent) -> Grammar | None:
    """Switch grammar based on the format the LLM chose."""
    chosen = ev.text.strip().rstrip(": ")
    print(f"\n  [format={chosen!r} → switching grammar]")
    if chosen == "json":
        return g_json
    elif chosen == "text":
        return g_text
    return None


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    llm = GrammarLLM("http://localhost:8080/v1")

    print("Prompt: Describe Paris in a few words.")
    print("Output: ", end="")

    for token, events in llm.adaptive_stream(
        messages=[{"role": "user", "content": "Describe Paris in a few words.(as text)"}],
        grammar=g1,
        on_switch={"format_choice": on_format},
        n_predict=128,
    ):
        print(token, end="", flush=True)

    print()
