#!/usr/bin/env python3
"""Pytest suite for core pygbnf behavior."""

from dataclasses import dataclass, field as dc_field
from typing import List, Optional
import warnings

import pytest

import pygbnf as cfg
from pygbnf import (
    Alternative,
    CharacterClass,
    Grammar,
    Group,
    Literal,
    Repeat,
    RuleReference,
    Sequence,
    group,
    one_or_more,
    optional,
    repeat,
    select,
    zero_or_more,
    token,
    token_id,
    not_token,
    not_token_id,
    WS,
    T,
    between,
    comma_list,
    float_number,
    identifier,
    keyword,
    number,
    string_literal,
)
from pygbnf.gbnf_codegen import _emit
from pygbnf.nodes import _tl
from pygbnf.optimizations import optimize_rules
from pygbnf.schema import grammar_from_type


@dataclass
class _AllRequired:
    name: str
    age: int


@dataclass
class _WithDefaults:
    name: str
    age: int
    active: bool = True
    nickname: Optional[str] = None


@dataclass
class _AllDefaults:
    x: int = 0
    y: int = 0


@pytest.fixture
def basic_number_grammar() -> Grammar:
    g = Grammar()

    @g.rule
    def digit():
        return select("0123456789")

    @g.rule
    def num():
        return optional("-") + one_or_more(digit())

    g.start("num")
    return g


@pytest.mark.parametrize(
    ("expr", "expected_type"),
    [
        (Literal("a") + Literal("b"), Sequence),
        ("x" + Literal("y"), Sequence),
        (Literal("x") + "y", Sequence),
        (Literal("a") | Literal("b"), Alternative),
        ("x" | Literal("y"), Alternative),
        (Literal("x") | "y", Alternative),
    ],
)
def test_node_operators_return_expected_types(expr, expected_type):
    assert isinstance(expr, expected_type)


def test_literal_and_character_class_store_values():
    assert Literal("hello").value == "hello"
    assert CharacterClass(pattern="0-9").pattern == "0-9"


def test_sequence_and_alternative_flatten_when_chained():
    seq = Literal("a") + Literal("b") + Literal("c")
    alt = Literal("a") | Literal("b") | Literal("c")

    assert len(seq.children) == 3
    assert len(alt.alternatives) == 3


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (select("abc"), CharacterClass),
        (select(["a", "b"]), Alternative),
        (select(["a"]), Literal),
        (group("x"), Group),
    ],
)
def test_combinators_return_expected_node_types(value, expected):
    assert isinstance(value, expected)


def test_repeat_combinators_store_bounds():
    assert (one_or_more("x").min, one_or_more("x").max) == (1, None)
    assert (zero_or_more("x").min, zero_or_more("x").max) == (0, None)
    assert (optional("x").min, optional("x").max) == (0, 1)
    assert (repeat("x", 2, 5).min, repeat("x", 2, 5).max) == (2, 5)


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        (token("think"), "<think>"),
        (token_id(1000), "<[1000]>"),
        (not_token("think"), "!<think>"),
        (not_token_id(1001), "!<[1001]>"),
        (Literal("hello"), '"hello"'),
        (Literal('say "hi"'), '"say \\"hi\\""'),
        (Literal("a\nb"), '"a\\nb"'),
        (CharacterClass(pattern="0-9"), "[0-9]"),
        (CharacterClass(pattern="abc", negated=True), "[^abc]"),
        (CharacterClass(pattern="^abc"), "[^abc]"),
        (RuleReference(name="my-rule"), "my-rule"),
        (Sequence(children=[Literal("a"), Literal("b")]), '"a" "b"'),
        (Alternative(alternatives=[Literal("a"), Literal("b")]), '"a" | "b"'),
        (Repeat(child=CharacterClass(pattern="0-9"), min=1, max=None), "[0-9]+"),
        (Repeat(child=Literal("x"), min=0, max=None), '"x"*'),
        (Repeat(child=Literal("x"), min=0, max=1), '"x"?'),
        (Repeat(child=Literal("x"), min=2, max=5), '"x"{2,5}'),
        (Repeat(child=Literal("x"), min=3, max=3), '"x"{3}'),
        (Repeat(child=Literal("x"), min=2, max=None), '"x"{2,}'),
        (Group(child=Alternative(alternatives=[Literal("a"), Literal("b")])), '("a" | "b")'),
    ],
)
def test_emit_renders_expected_gbnf(node, expected):
    assert _emit(node) == expected


def test_alternative_is_wrapped_in_sequence_context():
    rendered = _emit(
        Sequence(
            children=[
                Literal("x"),
                Alternative(alternatives=[Literal("a"), Literal("b")]),
            ]
        )
    )
    assert rendered == '"x" ("a" | "b")'


def test_basic_grammar_compiles_and_contains_expected_rules(basic_number_grammar: Grammar):
    gbnf = basic_number_grammar.to_gbnf()

    assert "root ::= num" in gbnf
    assert "digit ::=" in gbnf
    assert "num ::=" in gbnf
    assert "digit+" in gbnf


def test_dependency_graph_reports_rule_references(basic_number_grammar: Grammar):
    deps = basic_number_grammar.dependency_graph()

    assert "digit" in deps.get("num", set())
    assert deps.get("digit", set()) == set()


def test_left_recursion_detection_emits_warning():
    g = Grammar()

    @g.rule
    def expr():
        return select([expr() + "+" + expr(), "x"])

    g.start("expr")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cycles = g.detect_left_recursion()

    assert cycles
    assert caught


def test_optimizations_collapse_literals_and_singletons():
    literal_collapsed = optimize_rules(
        {"r": Sequence(children=[Literal("a"), Literal("b"), Literal("c")])}
    )
    singleton_alt = optimize_rules({"r": Alternative(alternatives=[Literal("x")])})
    singleton_seq = optimize_rules({"r": Sequence(children=[Literal("x")])})
    redundant_group = optimize_rules({"r": Group(child=Literal("x"))})

    assert literal_collapsed["r"] == Literal("abc")
    assert singleton_alt["r"] == Literal("x")
    assert singleton_seq["r"] == Literal("x")
    assert redundant_group["r"] == Literal("x")


def test_optimizations_merge_adjacent_repetitions():
    merged = optimize_rules(
        {
            "r": Sequence(
                children=[
                    Repeat(child=Literal("x"), min=0, max=1),
                    Repeat(child=Literal("x"), min=0, max=1),
                    Repeat(child=Literal("x"), min=0, max=1),
                    Repeat(child=Literal("x"), min=0, max=1),
                ]
            )
        }
    )["r"]

    assert isinstance(merged, Repeat)
    assert (merged.min, merged.max) == (0, 4)


def test_helper_nodes_emit_expected_patterns():
    assert _emit(WS()) == '[ \\t\\n]*'
    assert _emit(WS(required=True)) == '[ \\t\\n]+'
    assert _emit(keyword("return")) == '"return"'
    assert "a-zA-Z_" in _emit(identifier())
    assert "0-9" in _emit(number())
    assert '"."' in _emit(float_number())
    assert '[^"\\\\]' in _emit(string_literal())
    assert "," in _emit(comma_list(RuleReference(name="item")))
    assert _emit(between("(", RuleReference(name="expr"), ")")) == '"(" expr ")"'


def test_rule_names_are_converted_to_dashed_form():
    g = Grammar()

    @g.rule
    def my_rule():
        return Literal("x")

    g.start("my_rule")
    gbnf = g.to_gbnf()

    assert "my-rule ::=" in gbnf
    assert "root ::= my-rule" in gbnf


def test_schema_for_required_dataclass_has_no_optional_tail():
    gbnf = grammar_from_type(_AllRequired).to_gbnf()
    dc_line = [line for line in gbnf.splitlines() if "AllRequired" in line and "::=" in line][0]

    assert ")?" not in dc_line
    assert '\\"name\\"' in gbnf
    assert '\\"age\\"' in gbnf


def test_schema_for_dataclass_with_defaults_contains_optional_fields():
    gbnf = grammar_from_type(_WithDefaults).to_gbnf()

    assert ")?" in gbnf
    assert '\\"name\\"' in gbnf
    assert '\\"active\\"' in gbnf
    assert '\\"nickname\\"' in gbnf
    assert gbnf.count(")?") >= 2


def test_schema_for_all_default_dataclass_allows_empty_object():
    gbnf = grammar_from_type(_AllDefaults).to_gbnf()

    assert ")?" in gbnf
    assert '\\"x\\"' in gbnf
    assert '\\"y\\"' in gbnf


def test_template_builder_supports_literal_only_templates():
    rendered = _emit(T("Hello world\n"))
    assert '"Hello world" "\\n"' in rendered


def test_template_builder_supports_placeholders():
    rendered = _emit(T(f"Age: {number()}\n"))
    assert '"Age: "' in rendered
    assert '"\\n"' in rendered


@pytest.mark.parametrize(
    ("template", "needle"),
    [
        (lambda free: T(f"Items:\n- {free:+}\n"), "*"),
        (lambda free: T(f"Header:\n- {free:*}\n"), "*"),
    ],
)
def test_template_builder_supports_line_quantifiers(template, needle):
    free = one_or_more(CharacterClass(pattern="^\\n"))
    rendered = _emit(template(free))
    assert needle in rendered


def test_template_builder_supports_multiple_sections():
    free = one_or_more(CharacterClass(pattern="^\\n"))
    rendered = _emit(
        T(
            f"""# Section A:
- {free}+
# Section B:
- {free}+
Done:
"""
        )
    )

    assert '"Done:"' in rendered
    assert '"# Section A:"' in rendered


def test_template_builder_round_trips_inside_grammar():
    g = cfg.Grammar()

    @g.rule
    def template_test():
        f = one_or_more(CharacterClass(pattern="^\\n"))
        return T(
            f"""Nom: {identifier()}
Age: {number()}
"""
        )

    g.start("template_test")
    gbnf = g.to_gbnf()

    assert '"Nom: "' in gbnf
    assert "Age: " in gbnf


def test_template_builder_cleans_registry_after_use():
    T("Hello world\n")
    assert getattr(_tl, "counter", 0) == 0
