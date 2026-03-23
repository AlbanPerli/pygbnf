#!/usr/bin/env python3
"""Tests for pure GrammarLLM behaviors that do not require a live backend."""

from pygbnf import Grammar, GrammarLLM, Literal, weighted_select


class _DummyToolkit:
    def __init__(self, grammar: Grammar) -> None:
        self.grammar = grammar

    def system_prompt(self) -> str:
        return "System prompt"


def _build_weighted_grammar() -> Grammar:
    g = Grammar()

    @g.rule
    def root():
        return weighted_select([" red", " blue"], weights=[2.0, 0.5])

    g.start("root")
    return g


def test_resolve_toolkit_prepends_system_message_when_missing():
    g = Grammar()

    @g.rule
    def root():
        return Literal("x")

    g.start("root")
    toolkit = _DummyToolkit(g)

    messages, grammar = GrammarLLM._resolve_toolkit(
        [{"role": "user", "content": "hello"}],
        None,
        toolkit,
    )

    assert messages[0] == {"role": "system", "content": "System prompt"}
    assert grammar is g


def test_resolve_toolkit_preserves_existing_system_message():
    g = Grammar()

    @g.rule
    def root():
        return Literal("x")

    g.start("root")
    toolkit = _DummyToolkit(g)

    messages, grammar = GrammarLLM._resolve_toolkit(
        [
            {"role": "system", "content": "Existing"},
            {"role": "user", "content": "hello"},
        ],
        None,
        toolkit,
    )

    assert messages[0]["content"] == "Existing"
    assert grammar is g


def test_compute_logit_bias_uses_first_diverging_token():
    llm = GrammarLLM.__new__(GrammarLLM)
    grammar = _build_weighted_grammar()

    def fake_tokenize(text: str) -> list[int]:
        table = {
            "": [],
            " red": [10],
            " blue": [20],
        }
        return table[text]

    bias = llm.compute_logit_bias(grammar, tokenize_fn=fake_tokenize, bias_scale=10.0)

    assert "10" in bias
    assert "20" in bias
    assert bias["10"] > 0
    assert bias["20"] < 0


def test_compute_weight_bias_returns_none_when_grammar_is_absent():
    llm = GrammarLLM.__new__(GrammarLLM)
    llm.tokenize = lambda text: [1]  # type: ignore[method-assign]
    assert llm._compute_weight_bias(None) is None
