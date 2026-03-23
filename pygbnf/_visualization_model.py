from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set

from ._visualization_utils import _dot_escape, _edge_attrs


@dataclass
class Edge:
    src: int
    dst: int
    label: Optional[str] = None
    kind: str = "normal"


@dataclass
class RuleLink:
    src_state: int
    dst_state: int
    target_rule: str


@dataclass
class Fragment:
    start: int
    accepts: Set[int]


@dataclass
class NFA:
    start: int
    accepts: Set[int]
    edges: List[Edge] = field(default_factory=list)
    rule_links: List[RuleLink] = field(default_factory=list)

    def all_states(self) -> Set[int]:
        states = {self.start} | set(self.accepts)
        for e in self.edges:
            states.add(e.src)
            states.add(e.dst)
        for rl in self.rule_links:
            states.add(rl.src_state)
            states.add(rl.dst_state)
        return states

    def to_dot(self, name: str = "NFA") -> str:
        lines = [
            f'digraph "{_dot_escape(name)}" {{',
            "  rankdir=LR;",
            '  node [shape=circle, fontsize=10];',
            '  edge [fontsize=9];',
        ]

        for state in sorted(self.all_states()):
            if state == self.start and state in self.accepts:
                attrs = 'shape=doublecircle, penwidth=3.0'
            elif state == self.start:
                attrs = 'shape=circle, penwidth=3.0'
            elif state in self.accepts:
                attrs = 'shape=doublecircle'
            else:
                attrs = 'shape=circle'
            lines.append(f'  s{state} [{attrs}, label="{state}"];')

        for e in self.edges:
            attrs = _edge_attrs(e)
            label = "ε" if e.label is None else _dot_escape(e.label)
            lines.append(f'  s{e.src} -> s{e.dst} [label="{label}"{attrs}];')

        lines.append("}")
        return "\n".join(lines)
