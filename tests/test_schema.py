#!/usr/bin/env python3
"""Pytest coverage for schema compilation and higher-level integrations."""

from dataclasses import dataclass, field as dc_field
import enum
from typing import Dict, List, Literal as TypingLiteral, Optional, Union

import pygbnf as cfg
import pytest

from pygbnf import (
    CharacterClass,
    Grammar,
    Literal,
    group,
    one_or_more,
    repeat,
    select,
)
from pygbnf.gbnf_codegen import compile_grammar
from pygbnf.schema import grammar_from_args, grammar_from_function, grammar_from_type


class Color(enum.Enum):
    RED = "red"
    GREEN = "green"


class Severity(enum.Enum):
    INFO = "info"
    ERROR = "error"


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Inner:
    value: int


@dataclass
class Outer:
    name: str
    child: Inner


@dataclass
class Project:
    title: str
    tags: List[str]


@dataclass
class SearchResult:
    title: str
    score: float


@dataclass
class Movie:
    title: str
    year: int


@dataclass
class LogEntry:
    message: str
    severity: Severity


@dataclass
class WithFactory:
    name: str
    items: List[str] = dc_field(default_factory=list)


@pytest.mark.parametrize(
    ("tp", "needle"),
    [
        (str, "json-string"),
        (int, "json-int"),
        (float, "json-float"),
        (bool, "json-bool"),
    ],
)
def test_primitive_types_compile_to_expected_json_rules(tp, needle):
    gbnf = grammar_from_type(tp).to_gbnf()
    assert needle in gbnf


def test_optional_and_literal_types_are_supported():
    optional_gbnf = grammar_from_type(Optional[int]).to_gbnf()
    literal_gbnf = grammar_from_type(TypingLiteral["red", "green", "blue"]).to_gbnf()

    assert "json-int" in optional_gbnf
    assert "json-null" in optional_gbnf
    assert '"\\"red\\""' in literal_gbnf
    assert '"\\"green\\""' in literal_gbnf
    assert '"\\"blue\\""' in literal_gbnf


def test_enum_types_compile_to_choice_rule():
    gbnf = grammar_from_type(Color).to_gbnf()
    assert '"\\"red\\""' in gbnf
    assert '"\\"green\\""' in gbnf


def test_list_and_dict_types_compile_to_json_structures():
    list_gbnf = grammar_from_type(List[int]).to_gbnf()
    dict_gbnf = grammar_from_type(Dict[str, int]).to_gbnf()

    assert '"["' in list_gbnf
    assert '"]"' in list_gbnf
    assert "json-int" in list_gbnf

    assert '"{"' in dict_gbnf
    assert '"}"' in dict_gbnf
    assert "json-string" in dict_gbnf


def test_simple_dataclass_compiles_to_fixed_object_rule():
    gbnf = grammar_from_type(Point).to_gbnf()
    assert '\\"x\\"' in gbnf
    assert '\\"y\\"' in gbnf
    assert '"{"' in gbnf


def test_nested_dataclass_compiles_nested_rules():
    gbnf = grammar_from_type(Outer).to_gbnf()

    assert '\\"name\\"' in gbnf
    assert '\\"child\\"' in gbnf
    assert "Inner ::=" in gbnf
    assert '\\"value\\"' in gbnf


def test_dataclass_with_list_field_compiles_array_rule():
    gbnf = grammar_from_type(Project).to_gbnf()

    assert '\\"title\\"' in gbnf
    assert '\\"tags\\"' in gbnf
    assert '"["' in gbnf


def test_grammar_from_function_uses_return_annotation():
    def search(query: str) -> SearchResult:
        raise NotImplementedError

    gbnf = grammar_from_function(search).to_gbnf()
    assert '\\"title\\"' in gbnf
    assert '\\"score\\"' in gbnf


def test_grammar_from_function_requires_return_annotation():
    def no_return(x: int):
        raise NotImplementedError

    with pytest.raises(TypeError):
        grammar_from_function(no_return)


def test_grammar_from_args_compiles_parameter_object():
    def send_email(to: str, subject: str, body: str, priority: int = 0):
        raise NotImplementedError

    gbnf = grammar_from_args(send_email).to_gbnf()

    assert '\\"to\\"' in gbnf
    assert '\\"subject\\"' in gbnf
    assert '\\"body\\"' in gbnf
    assert '\\"priority\\"' in gbnf


def test_grammar_from_type_method_supports_composition():
    g = Grammar()
    movie_node = g.from_type(Movie)

    @g.rule
    def review():
        return Literal('"review":') + movie_node

    g.start("review")
    gbnf = g.to_gbnf()

    assert "Movie ::=" in gbnf
    assert '\\"title\\"' in gbnf


def test_from_function_return_method_injects_return_schema():
    def get_movie() -> Movie:
        raise NotImplementedError

    g = Grammar()
    ret_node = g.from_function_return(get_movie)

    @g.rule
    def root():
        return ret_node

    g.start("root")
    gbnf = g.to_gbnf()
    assert "Movie ::=" in gbnf


def test_from_function_args_method_injects_argument_schema():
    def api_call(query: str, limit: int = 10):
        raise NotImplementedError

    g = Grammar()
    args_node = g.from_function_args(api_call)

    @g.rule
    def root():
        return args_node

    g.start("root")
    gbnf = g.to_gbnf()

    assert '\\"query\\"' in gbnf
    assert '\\"limit\\"' in gbnf


def test_non_optional_union_includes_all_member_types():
    gbnf = grammar_from_type(Union[str, int]).to_gbnf()
    assert "json-string" in gbnf
    assert "json-int" in gbnf


def test_dataclass_with_enum_field_includes_enum_values():
    gbnf = grammar_from_type(LogEntry).to_gbnf()

    assert '\\"message\\"' in gbnf
    assert '\\"severity\\"' in gbnf
    assert '"\\"info\\""' in gbnf
    assert '"\\"error\\""' in gbnf


@pytest.mark.parametrize("tp", [set, tuple])
def test_unsupported_types_raise_type_error(tp):
    with pytest.raises(TypeError):
        grammar_from_type(tp)


def test_default_factory_fields_are_optional():
    gbnf = grammar_from_type(WithFactory).to_gbnf()
    assert ")?" in gbnf
    assert '\\"name\\"' in gbnf
    assert '\\"items\\"' in gbnf


def test_integration_minilang_compiles_expected_rules():
    g = Grammar()

    @g.rule
    def lang_ws():
        return repeat(select(" \t"), 0, 8)

    @g.rule
    def lang_ident():
        return CharacterClass(pattern="a-zA-Z_") + repeat(
            CharacterClass(pattern="a-zA-Z0-9_"), 0, 20
        )

    @g.rule
    def lang_number():
        return one_or_more(CharacterClass(pattern="0-9"))

    @g.rule
    def lang_expr():
        return select([lang_number(), lang_ident()])

    @g.rule
    def lang_let():
        return Literal("let ") + lang_ident() + Literal(" = ") + lang_expr() + Literal(";")

    @g.rule
    def lang_print():
        return Literal("print(") + lang_expr() + Literal(");")

    @g.rule
    def lang_stmt():
        return select([lang_let(), lang_print()])

    @g.rule
    def lang_program():
        return lang_stmt() + repeat(group("\n" + lang_stmt()), 0, 10) + Literal("\n")

    g.start("lang_program")
    gbnf = g.to_gbnf()

    assert "root ::=" in gbnf
    assert '"let "' in gbnf
    assert '"print("' in gbnf
    assert "lang-ident" in gbnf
    assert len(gbnf) > 50
    assert len(g.rules()) == 8


def test_compile_grammar_standalone_supports_optimized_and_raw_modes():
    g = Grammar()

    @g.rule
    def cg_item():
        return select(["yes", "no"])

    g.start("cg_item")

    assert "root ::=" in compile_grammar(g, optimize=True)
    assert "root ::=" in compile_grammar(g, optimize=False)


def test_public_exports_expose_expected_names():
    assert cfg.Grammar is Grammar
    assert cfg.select is select
    assert cfg.grammar_from_type is grammar_from_type
    for name in cfg.__all__:
        assert name in dir(cfg)
