from __future__ import annotations

from typing import Dict, List, Optional, Set

from ._visualization_model import Edge, Fragment, NFA, RuleLink
from ._visualization_utils import (
    _display_literal,
    _node_display_name,
    _parse_charclass_pattern,
    _printable_char_label,
)
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


class RegularSubsetError(ValueError):
    pass


class StateFactory:
    def __init__(self) -> None:
        self._next = 0

    def new(self) -> int:
        s = self._next
        self._next += 1
        return s


class ThompsonBuilder:
    def __init__(
        self,
        rules: Dict[str, Node],
        *,
        compact_literals: bool = True,
        allow_unknown_nodes_as_special: bool = True,
        expand_rules: Optional[Set[str]] = None,
        expand_charclasses: bool = True,
        max_expand_depth: Optional[int] = None,
    ) -> None:
        self.rules = rules
        self.compact_literals = compact_literals
        self.allow_unknown_nodes_as_special = allow_unknown_nodes_as_special
        self.expand_rules = expand_rules
        self.expand_charclasses = expand_charclasses
        self.max_expand_depth = max_expand_depth
        self.sf = StateFactory()
        self.edges: List[Edge] = []
        self.rule_links: List[RuleLink] = []
        self._expand_stack: List[str] = []

    def build_rule(self, rule_name: str) -> NFA:
        if rule_name not in self.rules:
            raise KeyError(f"Unknown rule: {rule_name!r}")
        frag = self._build(self.rules[rule_name])
        nfa = NFA(
            start=frag.start,
            accepts=set(frag.accepts),
            edges=list(self.edges),
            rule_links=list(self.rule_links),
        )
        return _simplify_nfa(nfa)

    def _build(self, node: Node) -> Fragment:
        if isinstance(node, RuleReference):
            return self._build_rule_ref(node.name)

        if isinstance(node, Literal):
            return self._build_literal(node.value)

        if isinstance(node, CharacterClass):
            return self._build_charclass(
                pattern=getattr(node, "pattern", ""),
                negated=getattr(node, "negated", False),
            )

        if isinstance(node, Group):
            child = getattr(node, "child", None)
            if child is None:
                raise RegularSubsetError("Group sans child")
            return self._build(child)

        if isinstance(node, Sequence):
            children = list(getattr(node, "children", []))
            if not children:
                return self._build_epsilon()

            frag = self._build(children[0])
            for child in children[1:]:
                right = self._build(child)
                for a in frag.accepts:
                    self.edges.append(Edge(a, right.start, None))
                frag = Fragment(start=frag.start, accepts=set(right.accepts))
            return frag

        if isinstance(node, Alternative):
            alts = list(getattr(node, "alternatives", []))
            start = self.sf.new()
            end = self.sf.new()
            for alt in alts:
                f = self._build(alt)
                self.edges.append(Edge(start, f.start, None))
                for a in f.accepts:
                    self.edges.append(Edge(a, end, None))
            return Fragment(start=start, accepts={end})

        if isinstance(node, Optional_):
            child = getattr(node, "child", None)
            if child is None:
                raise RegularSubsetError("Optional_ sans child")
            inner = self._build(child)
            start = self.sf.new()
            end = self.sf.new()
            self.edges.append(Edge(start, inner.start, None))
            self.edges.append(Edge(start, end, None))
            for a in inner.accepts:
                self.edges.append(Edge(a, end, None))
            return Fragment(start=start, accepts={end})

        if isinstance(node, Repeat):
            return self._build_repeat(node)

        if isinstance(node, TokenReference):
            return self._build_special(f"<token:{_node_display_name(node)}>")

        if self.allow_unknown_nodes_as_special:
            return self._build_special(f"<{_node_display_name(node)}>")

        raise RegularSubsetError(f"Unsupported node type: {type(node).__name__}")

    def _build_rule_ref(self, name: str) -> Fragment:
        if name not in self.rules:
            raise RegularSubsetError(f"Unknown rule reference: {name}")

        depth_ok = (
            self.max_expand_depth is None
            or len(self._expand_stack) < self.max_expand_depth
        )
        if (
            self.expand_rules is not None
            and name in self.expand_rules
            and name not in self._expand_stack
            and depth_ok
        ):
            self._expand_stack.append(name)
            try:
                return self._build(self.rules[name])
            finally:
                self._expand_stack.pop()

        s = self.sf.new()
        t = self.sf.new()

        self.edges.append(Edge(s, t, f"<{name}>", kind="rule_ref"))
        self.rule_links.append(RuleLink(src_state=s, dst_state=t, target_rule=name))
        return Fragment(start=s, accepts={t})

    def _build_special(self, label: str) -> Fragment:
        s = self.sf.new()
        t = self.sf.new()
        self.edges.append(Edge(s, t, label, kind="special"))
        return Fragment(start=s, accepts={t})

    def _build_epsilon(self) -> Fragment:
        s = self.sf.new()
        t = self.sf.new()
        self.edges.append(Edge(s, t, None))
        return Fragment(start=s, accepts={t})

    def _build_literal(self, text: str) -> Fragment:
        if text == "":
            return self._build_epsilon()

        if self.compact_literals:
            s = self.sf.new()
            t = self.sf.new()
            self.edges.append(Edge(s, t, _display_literal(text)))
            return Fragment(start=s, accepts={t})

        start = self.sf.new()
        current = start
        for ch in text:
            nxt = self.sf.new()
            self.edges.append(Edge(current, nxt, _printable_char_label(ch)))
            current = nxt
        return Fragment(start=start, accepts={current})

    def _build_charclass(self, pattern: str, negated: bool) -> Fragment:
        if self.expand_charclasses and not negated:
            chars = _parse_charclass_pattern(pattern)
            if len(chars) > 1:
                start = self.sf.new()
                end = self.sf.new()
                for ch in chars:
                    self.edges.append(Edge(start, end, _printable_char_label(ch)))
                return Fragment(start=start, accepts={end})

        s = self.sf.new()
        t = self.sf.new()
        label = f"[^{pattern}]" if negated else f"[{pattern}]"
        self.edges.append(Edge(s, t, label))
        return Fragment(start=s, accepts={t})

    def _build_repeat(self, node: Repeat) -> Fragment:
        child = getattr(node, "child", None)
        if child is None:
            raise RegularSubsetError("Repeat sans child")

        m = getattr(node, "min", None)
        n = getattr(node, "max", None)

        if m is None:
            m = 0

        if m == 0 and n == 1:
            return self._build(Optional_(child=child))

        if m == 0 and n is None:
            inner = self._build(child)
            start = self.sf.new()
            end = self.sf.new()
            self.edges.append(Edge(start, end, None))
            self.edges.append(Edge(start, inner.start, None))
            for a in inner.accepts:
                self.edges.append(Edge(a, inner.start, None))
                self.edges.append(Edge(a, end, None))
            return Fragment(start=start, accepts={end})

        if m == 1 and n is None:
            first = self._build(child)
            end = self.sf.new()
            for a in first.accepts:
                self.edges.append(Edge(a, first.start, None))
                self.edges.append(Edge(a, end, None))
            return Fragment(start=first.start, accepts={end})

        if n is not None and n < m:
            raise RegularSubsetError(f"Invalid Repeat bounds: min={m}, max={n}")

        if m == 0:
            frag = self._build_epsilon()
        else:
            frag: Optional[Fragment] = None
            for _ in range(m):
                part = self._build(child)
                if frag is None:
                    frag = part
                else:
                    for a in frag.accepts:
                        self.edges.append(Edge(a, part.start, None))
                    frag = Fragment(start=frag.start, accepts=set(part.accepts))
            assert frag is not None

        if n is None:
            raise RegularSubsetError("Unexpected Repeat case")

        optional_count = n - m
        result = frag
        for _ in range(optional_count):
            part = self._build(Optional_(child=child))
            for a in result.accepts:
                self.edges.append(Edge(a, part.start, None))
            result = Fragment(start=result.start, accepts=set(part.accepts))

        return result


def _simplify_nfa(nfa: NFA) -> NFA:
    edges = list(nfa.edges)
    start = nfa.start
    accepts = set(nfa.accepts)
    rule_links = list(nfa.rule_links)

    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: int, b: int, keep: int) -> None:
        a, b = find(a), find(b)
        if a == b:
            return
        discard = b if keep == a else a
        parent[discard] = keep

    protected = {start} | accepts

    changed = True
    while changed:
        changed = False

        for e in edges:
            e.src = find(e.src)
            e.dst = find(e.dst)
        for rl in rule_links:
            rl.src_state = find(rl.src_state)
            rl.dst_state = find(rl.dst_state)

        edges = [e for e in edges if not (e.label is None and e.src == e.dst)]

        all_states: Set[int] = {start} | accepts
        for e in edges:
            all_states.add(e.src)
            all_states.add(e.dst)

        out_eps: Dict[int, List[Edge]] = {s: [] for s in all_states}
        out_lab: Dict[int, List[Edge]] = {s: [] for s in all_states}
        in_eps: Dict[int, List[Edge]] = {s: [] for s in all_states}
        in_lab: Dict[int, List[Edge]] = {s: [] for s in all_states}

        for e in edges:
            if e.label is None:
                out_eps[e.src].append(e)
                in_eps[e.dst].append(e)
            else:
                out_lab[e.src].append(e)
                in_lab[e.dst].append(e)

        for s in all_states:
            if s in protected:
                continue

            if len(out_eps[s]) == 1 and not out_lab[s]:
                target = out_eps[s][0].dst
                if target != s and find(target) not in protected | {find(s)}:
                    union(s, target, keep=find(target))
                    changed = True
                    break
                elif target != s:
                    union(s, target, keep=find(target))
                    changed = True
                    break

            if len(in_eps[s]) == 1 and not in_lab[s]:
                source = in_eps[s][0].src
                if source != s:
                    union(s, source, keep=find(source))
                    changed = True
                    break

    for e in edges:
        e.src = find(e.src)
        e.dst = find(e.dst)
    for rl in rule_links:
        rl.src_state = find(rl.src_state)
        rl.dst_state = find(rl.dst_state)
    start = find(start)
    accepts = {find(a) for a in accepts}

    seen: Set[tuple] = set()
    clean: List[Edge] = []
    for e in edges:
        if e.label is None and e.src == e.dst:
            continue
        key = (e.src, e.dst, e.label, e.kind)
        if key not in seen:
            seen.add(key)
            clean.append(e)

    live: Set[int] = {start} | accepts
    for e in clean:
        live.add(e.src)
        live.add(e.dst)
    for rl in rule_links:
        live.add(rl.src_state)
        live.add(rl.dst_state)

    remap = {old: i for i, old in enumerate(sorted(live))}
    for e in clean:
        e.src = remap[e.src]
        e.dst = remap[e.dst]
    for rl in rule_links:
        rl.src_state = remap[rl.src_state]
        rl.dst_state = remap[rl.dst_state]

    return NFA(
        start=remap[start],
        accepts={remap[a] for a in accepts},
        edges=clean,
        rule_links=rule_links,
    )
