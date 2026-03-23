"""Automatic grammar generation from Python types.

Transforms dataclasses, function signatures, and type annotations into
GBNF grammars via pygbnf.

Examples
--------
>>> from dataclasses import dataclass
>>> from pygbnf.schema import grammar_from_type
>>>
>>> @dataclass
... class Movie:
...     title: str
...     year: int
...     rating: float
...
>>> g = grammar_from_type(Movie)
>>> print(g.to_gbnf())
"""

from __future__ import annotations

from ._schema_compiler import SchemaCompiler
from ._schema_functions import (
    describe_tools,
    grammar_from_args,
    grammar_from_function,
    grammar_from_tool_call,
    grammar_from_type,
)

__all__ = [
    "SchemaCompiler",
    "describe_tools",
    "grammar_from_args",
    "grammar_from_function",
    "grammar_from_tool_call",
    "grammar_from_type",
]
