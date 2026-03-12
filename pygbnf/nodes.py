"""
pygbnf.nodes — Abstract Syntax Tree node definitions for context-free grammars.

Every grammar construct is represented as a dataclass node. Nodes are composable
via the ``+`` (sequence) and ``|`` (alternative) operators, enabling a natural
Python DSL for grammar authoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Node:
    """Base class for all AST nodes.

    Provides ``__add__`` (sequence) and ``__or__`` (alternative) so that any
    two nodes can be combined with ``a + b`` or ``a | b``.
    """

    def __add__(self, other: Union["Node", str]) -> "Sequence":
        """Concatenate two nodes into a Sequence."""
        other = _coerce(other)
        left = self.children if isinstance(self, Sequence) else [self]
        right = other.children if isinstance(other, Sequence) else [other]
        return Sequence(children=list(left) + list(right))

    def __radd__(self, other: Union["Node", str]) -> "Sequence":
        other = _coerce(other)
        left = other.children if isinstance(other, Sequence) else [other]
        right = self.children if isinstance(self, Sequence) else [self]
        return Sequence(children=list(left) + list(right))

    def __or__(self, other: Union["Node", str]) -> "Alternative":
        """Create an Alternative between two nodes."""
        other = _coerce(other)
        left = self.alternatives if isinstance(self, Alternative) else [self]
        right = other.alternatives if isinstance(other, Alternative) else [other]
        return Alternative(alternatives=list(left) + list(right))

    def __ror__(self, other: Union["Node", str]) -> "Alternative":
        other = _coerce(other)
        left = other.alternatives if isinstance(other, Alternative) else [other]
        right = self.alternatives if isinstance(self, Alternative) else [self]
        return Alternative(alternatives=list(left) + list(right))


# ---------------------------------------------------------------------------
# Concrete node types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Literal(Node):
    """A literal string terminal, e.g. ``"hello"``."""
    value: str = ""


@dataclass(frozen=True)
class CharacterClass(Node):
    """A character class such as ``[0-9]`` or ``[^\\n]``.

    Parameters
    ----------
    pattern : str
        The inner content of the brackets, e.g. ``"0-9"`` or ``"^\\n"``.
    negated : bool
        Whether the class is negated (``[^...]``).
    """
    pattern: str = ""
    negated: bool = False


@dataclass(frozen=True)
class Sequence(Node):
    """An ordered sequence of nodes (e.g. ``a b c``)."""
    children: List[Node] = field(default_factory=list)


@dataclass(frozen=True)
class Alternative(Node):
    """A set of alternative productions (e.g. ``a | b | c``)."""
    alternatives: List[Node] = field(default_factory=list)


@dataclass(frozen=True)
class WeightedAlternative(Alternative):
    """Alternative with logit-bias weights for preferring certain branches.

    Each weight is a positive float interpreted as a probability multiplier.
    ``1.0`` is neutral, ``2.0`` doubles relative likelihood, ``0.5`` halves it.
    Weights are converted to logit biases via ``ln(weight)`` by
    :class:`~pygbnf.GrammarLLM`.

    In GBNF output this behaves identically to :class:`Alternative` — the
    grammar remains a binary filter.  Weights are applied as ``logit_bias``
    on the first distinguishing token of each branch.
    """
    weights: tuple = ()  # tuple[float, ...] parallel to alternatives


@dataclass(frozen=True)
class Optional_(Node):
    """An optional node (``child?``)."""
    child: Node = field(default_factory=lambda: Literal(""))


@dataclass(frozen=True)
class Repeat(Node):
    """A repeated node with quantifier bounds.

    If *min* = 0 and *max* = ``None`` → ``child*``
    If *min* = 1 and *max* = ``None`` → ``child+``
    If *min* = 0 and *max* = 1        → ``child?``
    Otherwise                           → ``child{min,max}``
    """
    child: Node = field(default_factory=lambda: Literal(""))
    min: int = 0
    max: Optional[int] = None  # None means unbounded


@dataclass(frozen=True)
class RuleReference(Node):
    """A reference to another grammar rule by name."""
    name: str = ""


@dataclass(frozen=True)
class TokenReference(Node):
    """A token-level constraint for llama.cpp.

    Supports ``<token>``, ``<[token_id]>``, ``!<token>``, ``!<[token_id]>``.
    """
    value: Union[str, int] = ""
    negated: bool = False


@dataclass(frozen=True)
class Group(Node):
    """Explicit parenthesised group ``( ... )``."""
    child: Node = field(default_factory=lambda: Literal(""))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _coerce(obj: Union[Node, str]) -> Node:
    """Convert a plain string into a :class:`Literal` node."""
    if isinstance(obj, str):
        return Literal(obj)
    if isinstance(obj, Node):
        return obj
    raise TypeError(f"Cannot coerce {type(obj)!r} to a grammar Node")
