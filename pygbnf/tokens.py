"""
pygbnf.tokens — Token-level constraint helpers for llama.cpp grammars.

llama.cpp GBNF supports matching specific tokeniser tokens rather than
character sequences.  This module exposes helpers that produce
:class:`~pygbnf.nodes.TokenReference` nodes.
"""

from __future__ import annotations

from .nodes import TokenReference


def token(text: str) -> TokenReference:
    """Match a token whose text is exactly *text*.

    Compiles to ``<text>`` in GBNF.

    Example
    -------
    >>> token("think")       # → <think>
    """
    return TokenReference(value=text, negated=False)


def token_id(tid: int) -> TokenReference:
    """Match a token by its vocabulary ID.

    Compiles to ``<[tid]>`` in GBNF.

    Example
    -------
    >>> token_id(1000)       # → <[1000]>
    """
    return TokenReference(value=tid, negated=False)


def not_token(text: str) -> TokenReference:
    """Match any token *except* the one whose text is *text*.

    Compiles to ``!<text>`` in GBNF.

    Example
    -------
    >>> not_token("think")   # → !<think>
    """
    return TokenReference(value=text, negated=True)


def not_token_id(tid: int) -> TokenReference:
    """Match any token *except* the one with vocabulary ID *tid*.

    Compiles to ``!<[tid]>`` in GBNF.

    Example
    -------
    >>> not_token_id(1001)   # → !<[1001]>
    """
    return TokenReference(value=tid, negated=True)
