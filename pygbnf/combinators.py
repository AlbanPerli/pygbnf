"""
pygbnf.combinators — DSL functions for building grammar nodes.

Provides ``select``, ``one_or_more``, ``zero_or_more``, ``optional``,
``repeat``, and ``group`` combinators that return AST :mod:`nodes`.
"""

from __future__ import annotations

from typing import List, Optional, Union

from .nodes import (
    Alternative,
    CharacterClass,
    Group,
    Literal,
    Node,
    Optional_,
    Repeat,
    Sequence,
    _coerce,
)


# ---------------------------------------------------------------------------
# Selection / alternation
# ---------------------------------------------------------------------------

def select(items: Union[str, List[Union[str, Node]]]) -> Node:
    """Create an alternative or character-class node.

    * If *items* is a **string**, return a :class:`CharacterClass` where each
      character is a member.  E.g. ``select("0123456789")`` → ``[0-9]``
      shorthand stored as ``[0123456789]``.
    * If *items* is a **list**, return an :class:`Alternative` over the
      elements (strings are auto-coerced to :class:`Literal`).

    Examples
    --------
    >>> select("abc")         # CharacterClass
    >>> select(["+", "-"])    # Alternative
    """
    if isinstance(items, str):
        return CharacterClass(pattern=items)
    alts = [_coerce(i) for i in items]
    if len(alts) == 1:
        return alts[0]
    return Alternative(alternatives=alts)


# ---------------------------------------------------------------------------
# Repetition helpers
# ---------------------------------------------------------------------------

def one_or_more(item: Union[str, Node]) -> Repeat:
    """``item+`` — match *item* one or more times."""
    return Repeat(child=_coerce(item), min=1, max=None)


def zero_or_more(item: Union[str, Node]) -> Repeat:
    """``item*`` — match *item* zero or more times."""
    return Repeat(child=_coerce(item), min=0, max=None)


def optional(item: Union[str, Node]) -> Repeat:
    """``item?`` — match *item* zero or one time."""
    return Repeat(child=_coerce(item), min=0, max=1)


def repeat(item: Union[str, Node], min: int = 0, max: Optional[int] = None) -> Repeat:
    """``item{min,max}`` — match *item* between *min* and *max* times.

    Parameters
    ----------
    item : str or Node
        The element to repeat.
    min : int
        Minimum number of repetitions (default 0).
    max : int or None
        Maximum number of repetitions.  ``None`` means unbounded.
    """
    return Repeat(child=_coerce(item), min=min, max=max)


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def group(item: Union[str, Node]) -> Group:
    """Wrap *item* in an explicit parenthesised group."""
    return Group(child=_coerce(item))
