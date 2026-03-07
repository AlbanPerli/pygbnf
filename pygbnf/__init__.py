"""
pygbnf — A composable Python DSL for building GBNF grammars.

Quick start
-----------
::

    import pygbnf as cfg
    from pygbnf import select, one_or_more, zero_or_more

    g = cfg.Grammar()

    @g.rule
    def number():
        n = one_or_more(select("0123456789"))
        return select(['-' + n, n])

    @g.rule
    def operator():
        return select(['+', '*', '**', '/', '-'])

    @g.rule
    def expression():
        return select([
            number(),
            expression() + zero_or_more(" ") + operator()
                + zero_or_more(" ") + expression(),
            "(" + expression() + ")"
        ])

    g.start("expression")
    print(g.to_gbnf())
"""

from __future__ import annotations

# Core container
from .grammar import Grammar

# AST nodes (for advanced usage / type hints)
from .nodes import (
    Alternative,
    CharacterClass,
    Group,
    Literal,
    Node,
    Optional_,
    Repeat,
    RuleReference,
    Sequence,
    TokenReference,
)

# DSL combinators
from .combinators import (
    group,
    one_or_more,
    optional,
    repeat,
    select,
    zero_or_more,
)

# Token primitives
from .tokens import (
    not_token,
    not_token_id,
    token,
    token_id,
)

# Grammar helpers
from .helpers import (
    WS,
    any_char,
    between,
    comma_list,
    float_number,
    identifier,
    keyword,
    number,
    separated_by,
    spaced_comma_list,
    string_literal,
    ws,
    ws_required,
)

# Schema → Grammar generation
from .schema import (
    SchemaCompiler,
    describe_tools,
    grammar_from_args,
    grammar_from_function,
    grammar_from_tool_call,
    grammar_from_type,
)

# Stream matcher
from .matcher import GrammarMatcher, RuleEvent, MatchToken

# Unified LLM client
from .llm import GrammarLLM

__all__ = [
    # Container
    "Grammar",
    # Nodes
    "Alternative",
    "CharacterClass",
    "Group",
    "Literal",
    "Node",
    "Optional_",
    "Repeat",
    "RuleReference",
    "Sequence",
    "TokenReference",
    # Combinators
    "group",
    "one_or_more",
    "optional",
    "repeat",
    "select",
    "zero_or_more",
    # Tokens
    "not_token",
    "not_token_id",
    "token",
    "token_id",
    # Helpers
    "WS",
    "any_char",
    "between",
    "comma_list",
    "float_number",
    "identifier",
    "keyword",
    "number",
    "separated_by",
    "spaced_comma_list",
    "string_literal",
    "ws",
    "ws_required",
    # Schema
    "SchemaCompiler",
    "describe_tools",
    "grammar_from_args",
    "grammar_from_function",
    "grammar_from_tool_call",
    "grammar_from_type",
    # Matcher
    "GrammarMatcher",
    "RuleEvent",
    "MatchToken",
    # LLM client
    "GrammarLLM",
]

__version__ = "0.2.0"
