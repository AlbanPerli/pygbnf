"""
pygbnf.helpers — Convenience grammar fragments for common patterns.

These helpers return :class:`~pygbnf.nodes.Node` sub-trees and can be used
directly inside ``@g.rule`` definitions or composed with other combinators.
"""

from __future__ import annotations

from typing import Union

from .nodes import (
    Alternative,
    CharacterClass,
    Group,
    Literal,
    Node,
    Repeat,
    RuleReference,
    Sequence,
    _coerce,
)
from .combinators import one_or_more, zero_or_more, optional, select, group


# ---------------------------------------------------------------------------
# Whitespace
# ---------------------------------------------------------------------------

def WS(*, required: bool = False) -> Node:
    """Whitespace (spaces, tabs, newlines).

    ``WS()``          → ``[ \\t\\n]*``
    ``WS(required=True)`` → ``[ \\t\\n]+``
    """
    ws_class = CharacterClass(pattern=" \\t\\n")
    if required:
        return one_or_more(ws_class)
    return zero_or_more(ws_class)


def ws() -> Node:
    """Alias for ``WS()`` — optional whitespace."""
    return WS()


def ws_required() -> Node:
    """Alias for ``WS(required=True)`` — at least one whitespace character."""
    return WS(required=True)


# ---------------------------------------------------------------------------
# Common token patterns
# ---------------------------------------------------------------------------

def keyword(word: str) -> Literal:
    """A literal keyword.

    >>> keyword("return")   # → "return"
    """
    return Literal(word)


def identifier() -> Node:
    """A typical identifier: ``[a-zA-Z_][a-zA-Z0-9_]*``.

    Compiles to::

        [a-zA-Z_] [a-zA-Z0-9_]*
    """
    head = CharacterClass(pattern="a-zA-Z_")
    tail = zero_or_more(CharacterClass(pattern="a-zA-Z0-9_"))
    return Sequence(children=[head, tail])


def number() -> Node:
    """An optionally-negative integer: ``"-"? [0-9]+``.

    Compiles to::

        "-"? [0-9]+
    """
    sign = optional(Literal("-"))
    digits = one_or_more(CharacterClass(pattern="0-9"))
    return Sequence(children=[sign, digits])


def float_number() -> Node:
    """A floating-point number: ``"-"? [0-9]+ ("." [0-9]+)?``."""
    sign = optional(Literal("-"))
    integer_part = one_or_more(CharacterClass(pattern="0-9"))
    frac = optional(
        Group(child=Sequence(children=[
            Literal("."),
            one_or_more(CharacterClass(pattern="0-9")),
        ]))
    )
    return Sequence(children=[sign, integer_part, frac])


def string_literal(*, quote: str = '"') -> Node:
    """A double-quoted (or custom-quoted) string with escape support.

    Compiles to::

        "\"" ([^"\\\\] | "\\\\" ["\\\\nrt/bfu])* "\""
    """
    body_char = Alternative(alternatives=[
        CharacterClass(pattern=f'^{quote}\\\\'),  # chars except quote and backslash
        Sequence(children=[
            Literal("\\"),
            CharacterClass(pattern=f'{quote}\\\\/bfnrtu'),  # known escape targets
        ]),
    ])
    return Sequence(children=[
        Literal(quote),
        zero_or_more(Group(child=body_char)),
        Literal(quote),
    ])


# ---------------------------------------------------------------------------
# Combinators for structure
# ---------------------------------------------------------------------------

def comma_list(item: Union[str, Node]) -> Node:
    """A comma-separated list of *item*: ``item ("," item)*``.

    >>> comma_list(identifier())
    """
    item = _coerce(item)
    return Sequence(children=[
        item,
        zero_or_more(Group(child=Sequence(children=[
            Literal(","),
            zero_or_more(Literal(" ")),
            item,
        ]))),
    ])


def spaced_comma_list(item: Union[str, Node]) -> Node:
    """Like :func:`comma_list` but with optional whitespace around commas."""
    item = _coerce(item)
    return Sequence(children=[
        item,
        zero_or_more(Group(child=Sequence(children=[
            WS(),
            Literal(","),
            WS(),
            item,
        ]))),
    ])


def between(left: Union[str, Node], expr: Union[str, Node], right: Union[str, Node]) -> Node:
    """Wrap *expr* between *left* and *right* delimiters.

    >>> between("(", expression(), ")")
    """
    return Sequence(children=[_coerce(left), _coerce(expr), _coerce(right)])


def separated_by(sep: Union[str, Node], item: Union[str, Node]) -> Node:
    """Alias for ``item (sep item)*``."""
    item = _coerce(item)
    sep = _coerce(sep)
    return Sequence(children=[
        item,
        zero_or_more(Group(child=Sequence(children=[sep, item]))),
    ])


def any_char() -> CharacterClass:
    """Match any single character (approximation via wide range).

    Compiles to ``.`` — but GBNF doesn't have ``.``; we use a broad class.
    In practice this is ``[^]`` which GBNF doesn't support either, so we
    use ``[\\x00-\\x7F]`` as a reasonable ASCII-range stand-in, or users
    should use a specific character class.
    """
    # GBNF uses [^\n] or similar; we'll provide a helper that covers
    # printable ASCII at minimum.  Users can replace with their own class.
    return CharacterClass(pattern="\\x00-\\x7F")
