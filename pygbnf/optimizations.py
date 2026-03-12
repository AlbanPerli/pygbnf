"""
pygbnf.optimizations — AST rewrite passes for grammar simplification.

These passes operate on the rule dictionary **before** code generation and
produce an equivalent (but simpler / faster-to-evaluate) grammar.

Implemented passes
------------------
1. **Sequence flattening** — nested ``Sequence`` nodes are merged.
2. **Literal collapsing** — adjacent ``Literal`` children in a ``Sequence``
   are merged into a single ``Literal``.
3. **Redundant group removal** — ``Group`` nodes wrapping a single leaf are
   unwrapped.
4. **Repetition merging** — consecutive optional or repeated occurrences of
   the same child are merged (e.g. ``x? x? x?`` → ``x{0,3}``).
5. **Single-child Alternative / Sequence collapse** — containers with a
   single element are replaced by that element.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

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
    WeightedAlternative,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def optimize_rules(rules: Dict[str, Node]) -> Dict[str, Node]:
    """Apply all optimisation passes to every rule and return a new dict."""
    return {name: _optimize_node(node) for name, node in rules.items()}


# ---------------------------------------------------------------------------
# Recursive optimiser dispatcher
# ---------------------------------------------------------------------------

def _optimize_node(node: Node) -> Node:
    """Apply all passes to *node* recursively (bottom-up)."""
    # First recurse into children
    node = _recurse(node)
    # Then apply local rewrites
    node = _flatten_sequences(node)
    node = _collapse_literals(node)
    node = _remove_redundant_groups(node)
    node = _merge_repetitions(node)
    node = _collapse_singletons(node)
    return node


def _recurse(node: Node) -> Node:
    """Recursively optimise child nodes."""
    if isinstance(node, Sequence):
        return Sequence(children=[_optimize_node(c) for c in node.children])
    if isinstance(node, WeightedAlternative):
        return WeightedAlternative(
            alternatives=[_optimize_node(a) for a in node.alternatives],
            weights=node.weights,
        )
    if isinstance(node, Alternative):
        return Alternative(alternatives=[_optimize_node(a) for a in node.alternatives])
    if isinstance(node, Repeat):
        return Repeat(child=_optimize_node(node.child), min=node.min, max=node.max)
    if isinstance(node, Group):
        return Group(child=_optimize_node(node.child))
    if isinstance(node, Optional_):
        return Optional_(child=_optimize_node(node.child))
    return node


# ---------------------------------------------------------------------------
# Pass 1 — Flatten nested Sequences
# ---------------------------------------------------------------------------

def _flatten_sequences(node: Node) -> Node:
    if not isinstance(node, Sequence):
        return node
    flat: List[Node] = []
    for child in node.children:
        if isinstance(child, Sequence):
            flat.extend(child.children)
        else:
            flat.append(child)
    return Sequence(children=flat)


# ---------------------------------------------------------------------------
# Pass 2 — Collapse adjacent Literals in a Sequence
# ---------------------------------------------------------------------------

def _collapse_literals(node: Node) -> Node:
    if not isinstance(node, Sequence):
        return node
    merged: List[Node] = []
    for child in node.children:
        if isinstance(child, Literal) and merged and isinstance(merged[-1], Literal):
            merged[-1] = Literal(value=merged[-1].value + child.value)
        else:
            merged.append(child)
    if len(merged) == len(node.children):
        return node  # nothing changed
    return Sequence(children=merged)


# ---------------------------------------------------------------------------
# Pass 3 — Remove redundant Groups
# ---------------------------------------------------------------------------

def _remove_redundant_groups(node: Node) -> Node:
    if isinstance(node, Group):
        inner = node.child
        if isinstance(inner, (Literal, CharacterClass, RuleReference, TokenReference)):
            return inner
    return node


# ---------------------------------------------------------------------------
# Pass 4 — Merge adjacent identical Repeats / Optionals
# ---------------------------------------------------------------------------

def _merge_repetitions(node: Node) -> Node:
    """Merge runs of ``x? x? x?`` into ``x{0,3}``, etc."""
    if not isinstance(node, Sequence):
        return node

    merged: List[Node] = []
    i = 0
    children = node.children
    while i < len(children):
        current = children[i]

        # Normalise Optional_ into Repeat{0,1} for comparison
        cur_repeat = _as_repeat(current)

        if cur_repeat is not None:
            # Look ahead for identical repeats
            j = i + 1
            total_min = cur_repeat.min
            total_max = cur_repeat.max
            while j < len(children):
                nxt_repeat = _as_repeat(children[j])
                if nxt_repeat is None or not _nodes_equal(cur_repeat.child, nxt_repeat.child):
                    break
                total_min += nxt_repeat.min
                if total_max is not None and nxt_repeat.max is not None:
                    total_max += nxt_repeat.max
                elif nxt_repeat.max is None:
                    total_max = None
                j += 1
            count = j - i
            if count > 1:
                merged.append(Repeat(child=cur_repeat.child, min=total_min, max=total_max))
                i = j
                continue

        merged.append(current)
        i += 1

    if merged == children:
        return node
    return Sequence(children=merged)


def _as_repeat(node: Node) -> Optional[Repeat]:
    """Return *node* as a :class:`Repeat` if applicable, else ``None``."""
    if isinstance(node, Repeat):
        return node
    if isinstance(node, Optional_):
        return Repeat(child=node.child, min=0, max=1)
    return None


def _nodes_equal(a: Node, b: Node) -> bool:
    """Structural equality check (relies on frozen dataclasses)."""
    return a == b


# ---------------------------------------------------------------------------
# Pass 5 — Collapse singleton containers
# ---------------------------------------------------------------------------

def _collapse_singletons(node: Node) -> Node:
    if isinstance(node, Sequence) and len(node.children) == 1:
        return node.children[0]
    if isinstance(node, Alternative) and len(node.alternatives) == 1:
        return node.alternatives[0]
    return node
