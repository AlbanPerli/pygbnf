#!/usr/bin/env python3
"""Tests for visualization helpers and public API stability."""

from dataclasses import dataclass
from pathlib import Path

from pygbnf import Grammar, Literal, group, one_or_more, optional, select
from pygbnf.visualization import (
    NFA,
    ThompsonBuilder,
    get_user_rules,
    grammar_rule_to_nfa_dot,
    grammar_to_nfa_dot,
    write_grammar_dot,
    write_rule_dot,
)


def _build_arithmetic_grammar() -> Grammar:
    g = Grammar()

    @g.rule
    def digit():
        return select("0123456789")

    @g.rule
    def number():
        return optional("-") + one_or_more(digit())

    @g.rule
    def operator():
        return select(["+", "-", "*", "/"])

    @g.rule
    def expression():
        atom = select([number(), "(" + expression() + ")"])
        return atom + optional(group(" " + operator() + " " + atom))

    g.start("expression")
    return g


def test_grammar_rule_to_nfa_dot_returns_dot_graph():
    g = _build_arithmetic_grammar()
    dot = grammar_rule_to_nfa_dot(g, "expression")

    assert 'digraph "expression"' in dot
    assert "rankdir=LR" in dot


def test_grammar_to_nfa_dot_renders_clusters_and_inter_rule_links():
    g = _build_arithmetic_grammar()
    dot = grammar_to_nfa_dot(
        g,
        rule_names=["expression", "number"],
        show_inter_rule_links=True,
    )

    assert 'subgraph "cluster_expression"' in dot
    assert 'subgraph "cluster_number"' in dot
    assert 'label="call"' in dot


def test_get_user_rules_excludes_schema_infrastructure_rules():
    g = Grammar()

    @dataclass
    class Movie:
        title: str

    g.from_type(Movie)
    user_rules = get_user_rules(g)

    assert "ws" not in user_rules
    assert "json-string" not in user_rules


def test_write_rule_dot_and_write_grammar_dot_write_files(tmp_path: Path):
    g = _build_arithmetic_grammar()

    rule_path = write_rule_dot(g, "expression", tmp_path / "rule.dot")
    grammar_path = write_grammar_dot(g, tmp_path / "grammar.dot")

    assert rule_path.exists()
    assert grammar_path.exists()
    assert rule_path.read_text(encoding="utf-8").startswith('digraph "expression"')
    assert grammar_path.read_text(encoding="utf-8").startswith('digraph "Grammar"')


def test_visualization_exports_builder_and_model_types():
    g = _build_arithmetic_grammar()
    builder = ThompsonBuilder(g.rules())
    nfa = builder.build_rule("expression")

    assert isinstance(nfa, NFA)


def test_get_user_rules_prefers_user_defined_rules_when_no_schema_is_present():
    g = Grammar()

    @g.rule
    def root():
        return Literal("x")

    g.start("root")
    assert get_user_rules(g) == ["root"]
