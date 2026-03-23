from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Set, Tuple

from .grammar import Grammar
from .nodes import (
    Alternative,
    Group,
    Literal,
    Node,
    Optional_,
    Repeat,
    RuleReference,
    Sequence,
    WeightedAlternative,
)


def compute_logit_bias(
    grammar: Grammar,
    *,
    tokenize_fn: Callable[[str], List[int]],
    bias_scale: float = 10.0,
) -> Dict[str, float]:
    """Extract weighted alternatives and convert them to an OpenAI logit_bias map."""
    weighted = collect_weighted(grammar)
    if not weighted:
        return {}

    bias: Dict[str, float] = {}
    for weighted_alt, ctx in weighted:
        ctx_tokens = tokenize_fn(ctx) if ctx else []
        for alt, weight in zip(weighted_alt.alternatives, weighted_alt.weights):
            prefix = first_literal_prefix(alt)
            if prefix is None or weight == 1.0:
                continue
            full_tokens = tokenize_fn(ctx + prefix)
            diff_idx = 0
            for i in range(min(len(ctx_tokens), len(full_tokens))):
                if ctx_tokens[i] != full_tokens[i]:
                    diff_idx = i
                    break
            else:
                diff_idx = len(ctx_tokens)

            if diff_idx < len(full_tokens):
                tid = str(full_tokens[diff_idx])
                adjustment = bias_scale * math.log(weight)
                bias[tid] = bias.get(tid, 0.0) + adjustment

    return bias


def safe_compute_logit_bias(
    grammar: Optional[Grammar],
    *,
    tokenize_fn: Callable[[str], List[int]],
    bias_scale: float = 10.0,
) -> Optional[Dict[str, float]]:
    """Compute logit bias if possible, otherwise return None."""
    if grammar is None:
        return None
    try:
        bias = compute_logit_bias(grammar, tokenize_fn=tokenize_fn, bias_scale=bias_scale)
        return bias if bias else None
    except RuntimeError:
        return None


def collect_weighted(grammar: Grammar) -> List[Tuple[WeightedAlternative, str]]:
    """Walk a grammar and collect weighted alternatives with preceding literal context."""
    rules = grammar.rules()
    results: List[Tuple[WeightedAlternative, str]] = []
    visited: Set[str] = set()

    start_name = getattr(grammar, "_start", None) or "root"
    entry = rules.get(start_name)
    if entry is not None:
        visited.add(start_name)
        walk_weighted(entry, "", results, rules, visited)

    for name, node in rules.items():
        if name not in visited:
            visited.add(name)
            walk_weighted(node, "", results, rules, visited)

    return results


def walk_weighted(
    node: Node,
    ctx: str,
    acc: List[Tuple[WeightedAlternative, str]],
    rules: Dict[str, Node],
    visited: Set[str],
) -> None:
    if isinstance(node, WeightedAlternative):
        acc.append((node, ctx))
    elif isinstance(node, Alternative):
        for alt in node.alternatives:
            walk_weighted(alt, ctx, acc, rules, visited)
    elif isinstance(node, Sequence):
        running_ctx = ctx
        for child in node.children:
            walk_weighted(child, running_ctx, acc, rules, visited)
            literal = full_literal_text(child)
            if literal is not None:
                running_ctx += literal
            else:
                running_ctx = ""
    elif isinstance(node, RuleReference):
        if node.name not in visited:
            visited.add(node.name)
            target = rules.get(node.name)
            if target is not None:
                walk_weighted(target, ctx, acc, rules, visited)
    elif isinstance(node, (Repeat, Optional_, Group)):
        child = getattr(node, "child", None)
        if child is not None:
            walk_weighted(child, ctx, acc, rules, visited)


def first_literal_prefix(node: Node) -> Optional[str]:
    """Extract the literal text at the start of a node, or None."""
    if isinstance(node, Literal):
        return node.value if node.value else None
    if isinstance(node, Sequence) and node.children:
        return first_literal_prefix(node.children[0])
    if isinstance(node, Group):
        return first_literal_prefix(node.child)
    if isinstance(node, (Alternative, WeightedAlternative)):
        return None
    return None


def full_literal_text(node: Node) -> Optional[str]:
    """Return the full literal text a node produces, or None if it is dynamic."""
    if isinstance(node, Literal):
        return node.value
    if isinstance(node, Sequence):
        parts: List[str] = []
        for child in node.children:
            text = full_literal_text(child)
            if text is None:
                return None
            parts.append(text)
        return "".join(parts)
    return None
