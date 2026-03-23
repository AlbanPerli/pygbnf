#!/usr/bin/env python3
"""Pytest coverage for core grammar, codegen, helper, and edge-case behavior."""

import contextlib
import io
import re
import warnings

import pytest

import pygbnf as cfg
from pygbnf import (
    Alternative,
    CharacterClass,
    Grammar,
    Group,
    Literal,
    Optional_,
    Repeat,
    RuleReference,
    Sequence,
    TokenReference,
    WS,
    any_char,
    between,
    comma_list,
    decimal_range,
    float_number,
    group,
    identifier,
    int_range,
    keyword,
    not_token,
    not_token_id,
    number,
    one_or_more,
    optional,
    repeat,
    select,
    separated_by,
    spaced_comma_list,
    string_literal,
    token,
    token_id,
    ws,
    ws_required,
)
from pygbnf.gbnf_codegen import _emit
from pygbnf.nodes import _coerce
from pygbnf.optimizations import optimize_rules


def test_node_attributes_cover_basic_leaf_types():
    assert Literal().value == ""
    assert CharacterClass(pattern="0-9").negated is False
    assert CharacterClass(pattern="abc", negated=True).negated is True
    assert RuleReference(name="my-rule").name == "my-rule"
    assert TokenReference(value="think").value == "think"
    assert TokenReference(value=42).value == 42
    assert TokenReference(value="x", negated=True).negated is True


def test_coerce_supports_str_and_node_and_rejects_other_types():
    assert isinstance(_coerce("hello"), Literal)
    assert _coerce("hi").value == "hi"
    literal = Literal("x")
    assert _coerce(literal) == literal
    with pytest.raises(TypeError):
        _coerce(42)  # type: ignore[arg-type]


def test_frozen_nodes_are_immutable():
    with pytest.raises(AttributeError):
        setattr(Literal("x"), "value", "y")
    with pytest.raises(AttributeError):
        setattr(Sequence(), "children", [])


def test_structural_equality_of_nodes_works():
    assert Literal("x") == Literal("x")
    assert Literal("x") != Literal("y")
    assert CharacterClass(pattern="0-9") == CharacterClass(pattern="0-9")
    assert Repeat(child=Literal("x"), min=0, max=1) == Repeat(
        child=Literal("x"), min=0, max=1
    )


def test_select_preserves_pattern_length_and_mixed_alternatives():
    assert select("abc").pattern == "abc"  # type: ignore[union-attr]
    assert len(select(["a", "b", "c"]).alternatives) == 3  # type: ignore[union-attr]
    mixed = select([Literal("x"), "y", CharacterClass(pattern="0-9")])
    assert len(mixed.alternatives) == 3  # type: ignore[union-attr]


def test_token_factories_preserve_expected_value_and_negation():
    tok = token("think")
    tok_id = token_id(1000)
    neg_tok = not_token("x")
    neg_tok_id = not_token_id(42)

    assert isinstance(tok, TokenReference)
    assert tok.value == "think"
    assert tok.negated is False
    assert isinstance(tok_id, TokenReference)
    assert tok_id.value == 1000
    assert neg_tok.negated is True
    assert neg_tok_id.negated is True


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        (Literal(""), '""'),
        (Literal("a\tb"), '"a\\tb"'),
        (Literal("a\\b"), '"a\\\\b"'),
        (Literal("a\rb"), '"a\\rb"'),
        (CharacterClass(pattern="a-zA-Z_"), "[a-zA-Z_]"),
        (CharacterClass(pattern='^"\\\\', negated=False), '[^"\\\\]'),
        (Repeat(child=Literal("x"), min=0, max=10), '"x"{0,10}'),
        (Optional_(child=Literal("x")), '"x"?'),
        (Repeat(child=Literal("x"), min=0, max=0), '"x"{0}'),
        (Sequence(children=[]), ""),
        (Alternative(alternatives=[]), ""),
    ],
)
def test_emit_covers_remaining_codegen_edge_cases(node, expected):
    assert _emit(node) == expected


def test_emit_wraps_compound_children_for_repeat():
    repeated_seq = _emit(
        Repeat(child=Sequence(children=[Literal("a"), Literal("b")]), min=1, max=None)
    )
    repeated_alt = _emit(
        Repeat(
            child=Alternative(alternatives=[Literal("a"), Literal("b")]),
            min=0,
            max=None,
        )
    )

    assert repeated_seq == '("a" "b")+'
    assert repeated_alt == '("a" | "b")*'


def test_compile_formats_long_alternatives_on_multiple_lines():
    g = Grammar()

    @g.rule
    def multi():
        return select(["alpha", "beta", "gamma", "delta"])

    g.start("multi")
    assert " |\n" in g.to_gbnf()


def test_character_class_dash_is_not_backslash_escaped():
    rendered = _emit(CharacterClass(pattern="+-"))
    assert "\\-" not in rendered
    assert "-" in rendered


def test_rule_named_registers_explicit_rule_name():
    g = Grammar()

    @g.rule_named("custom-name")
    def impl():
        return Literal("hello")

    g.start("custom-name")
    assert "custom-name ::=" in g.to_gbnf()


def test_ref_allows_named_reference_in_rule_body():
    g = Grammar()

    @g.rule
    def base():
        return Literal("x")

    @g.rule
    def uses_ref():
        return g.ref("base") + Literal("!")

    g.start("uses_ref")
    assert "base" in g.to_gbnf()


def test_starting_at_root_does_not_emit_root_alias():
    g = Grammar()

    @g.rule
    def root():
        return Literal("x")

    g.start("root")
    gbnf = g.to_gbnf()
    assert "root ::= root" not in gbnf
    assert 'root ::= "x"' in gbnf


def test_non_left_recursive_grammar_does_not_warn():
    g = Grammar()

    @g.rule
    def term_a():
        return Literal("x")

    @g.rule
    def expr_no_lr():
        return term_a() + optional(group("+" + term_a()))

    g.start("expr_no_lr")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cycles = g.detect_left_recursion()

    assert cycles == []
    assert caught == []


def test_missing_start_rule_raises_value_error():
    g = Grammar()

    @g.rule
    def some_rule():
        return Literal("x")

    g._start = "nonexistent"
    with pytest.raises(ValueError):
        g.to_gbnf()


def test_optimize_flag_controls_literal_collapsing():
    g = Grammar()

    @g.rule
    def lit_chain():
        return Literal("a") + Literal("b") + Literal("c")

    g.start("lit_chain")

    assert '"a" "b" "c"' in g.to_gbnf(optimize=False)
    assert '"abc"' in g.to_gbnf(optimize=True)


def test_pretty_print_writes_compiled_grammar_to_stdout():
    g = Grammar()

    @g.rule
    def pp_rule():
        return Literal("test")

    g.start("pp_rule")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        g.pretty_print()
    assert "pp-rule ::=" in buffer.getvalue()


def test_optimizer_handles_group_and_repetition_corner_cases():
    group_charclass = optimize_rules({"r": Group(child=CharacterClass(pattern="0-9"))})["r"]
    group_ref = optimize_rules({"r": Group(child=RuleReference(name="x"))})["r"]
    group_alt = optimize_rules(
        {"r": Group(child=Alternative(alternatives=[Literal("a"), Literal("b")]))}
    )["r"]
    no_merge = optimize_rules(
        {
            "r": Sequence(
                children=[
                    Repeat(child=Literal("x"), min=0, max=1),
                    Repeat(child=Literal("y"), min=0, max=1),
                ]
            )
        }
    )["r"]
    deep = optimize_rules(
        {"r": Group(child=Sequence(children=[Literal("a"), Literal("b")]))}
    )["r"]

    assert group_charclass == CharacterClass(pattern="0-9")
    assert group_ref == RuleReference(name="x")
    assert isinstance(group_alt, Group)
    assert isinstance(no_merge, Sequence)
    assert deep == Literal("ab")


def test_helper_aliases_and_additional_helpers_match_expected_output():
    assert _emit(ws()) == _emit(WS())
    assert _emit(ws_required()) == _emit(WS(required=True))
    assert keyword("return") == Literal("return")
    assert "\\" in _emit(string_literal())
    assert "'" in _emit(string_literal(quote="'"))
    assert '";"' in _emit(separated_by(";", RuleReference(name="stmt")))
    spaced = _emit(spaced_comma_list(RuleReference(name="item")))
    assert "item" in spaced
    assert "," in spaced
    assert isinstance(any_char(), CharacterClass)


def test_int_range_emits_inclusive_integer_alternatives():
    rendered = _emit(int_range(1, 3))
    assert rendered == "[1-3]"


def test_int_range_supports_negative_values_and_singletons():
    rendered = _emit(int_range(-2, 1))
    assert '"-"' in rendered
    assert "[1-2]" in rendered
    assert '"0"' in rendered
    assert '"1"' in rendered
    assert _emit(int_range(7, 7)) == '"7"'
    assert _emit(int_range(100, 999)) == "[1-9] [0-9]{2}"


def test_int_range_validates_bounds_and_types():
    with pytest.raises(ValueError):
        int_range(3, 1)
    with pytest.raises(TypeError):
        int_range(True, 2)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        int_range(1.5, 2)  # type: ignore[arg-type]


def test_decimal_range_supports_explicit_scale():
    rendered = _emit(decimal_range(1.2, 1.5, scale=1))
    assert rendered == '"1.2" | "1.3" | "1.4" | "1.5"'


def test_decimal_range_supports_decimal_strings_and_custom_steps():
    rendered = _emit(decimal_range("0.00", "1.00", step="0.25"))
    assert rendered == '"0.00" | "0.25" | "0.50" | "0.75" | "1.00"'
    assert _emit(decimal_range(2, 4, scale=0)) == '"2" | "3" | "4"'


def test_decimal_range_validates_configuration_and_alignment():
    with pytest.raises(ValueError):
        decimal_range(2.0, 1.0, scale=1)
    with pytest.raises(ValueError):
        decimal_range(0.0, 1.0)
    with pytest.raises(ValueError):
        decimal_range(0.0, 1.0, step=0.1, scale=1)
    with pytest.raises(ValueError):
        decimal_range(0.0, 1.0, step=0.3)
    with pytest.raises(ValueError):
        decimal_range(0.0, 2000.0, scale=0)
    with pytest.raises(ValueError):
        decimal_range(0.0, 1.0, step=0)
    with pytest.raises(ValueError):
        decimal_range(float("inf"), 1.0, scale=1)


def test_empty_grammar_compiles_to_empty_string():
    g = Grammar()
    g._start = None
    assert g.to_gbnf().strip() == ""


def test_long_literals_and_large_character_classes_are_supported():
    long_str = "a" * 200
    assert _emit(Literal(long_str)) == f'"{long_str}"'
    assert isinstance(select("abcdefghijklmnop"), CharacterClass)


def test_public_version_looks_like_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", cfg.__version__)


def test_special_rule_names_with_dashes_are_supported():
    g = Grammar()

    @g.rule_named("a-b-c")
    def special_impl():
        return Literal("x")

    g.start("a-b-c")
    assert "a-b-c ::=" in g.to_gbnf()


def test_nested_group_combinator_optimizes_down_to_literal():
    nested = group(group(Literal("x")))
    assert isinstance(nested, Group)
    assert isinstance(nested.child, Group)
    assert optimize_rules({"r": nested})["r"] == Literal("x")
