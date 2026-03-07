"""
pygbnf.matcher — Stream matcher for grammar-constrained LLM output.

Converts each named rule to a regex, then matches incrementally as tokens
arrive.  Fires callbacks when a named rule is fully recognised.

Usage
-----
::

    from pygbnf import Grammar, GrammarMatcher

    g = Grammar()

    @g.rule
    def number(): ...

    @g.rule
    def expression(): ...

    matcher = GrammarMatcher(g)

    # Register callbacks — use the **Python function name** (not the GBNF name)
    matcher.on("expression", lambda ev: print(f"matched: {ev.text}"))
    matcher.on("*", lambda ev: ...)           # wildcard — every rule

    # Feed tokens as they stream in
    for chunk in stream:
        events = matcher.feed(chunk.choices[0].delta.content or "")
"""

from __future__ import annotations

import re
import signal
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from .grammar import Grammar, _RuleProxy
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


# ── Public types ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class RuleEvent:
    """Emitted when a named rule is fully matched in the stream."""

    rule: str
    """Python function name of the matched rule (e.g. ``"expression"``)."""

    text: str
    """The matched substring."""

    start: int
    """Start offset in the accumulated buffer."""

    end: int
    """End offset (exclusive) in the accumulated buffer."""

    fn: object = None
    """The original decorated Python function, or ``None``."""

    @property
    def doc(self) -> str:
        """Docstring of the rule function (empty string if none)."""
        if self.fn is not None and getattr(self.fn, "__doc__", None):
            return self.fn.__doc__
        return ""


@dataclass(frozen=True)
class MatchToken:
    """Yielded by :meth:`GrammarMatcher.stream` for each token.

    Attributes
    ----------
    token : str
        The raw token text received from the LLM.
    events : list[RuleEvent]
        Grammar rules that completed after this token was appended.
        Empty if no new rule matched.
    """

    token: str
    """The raw token text."""

    events: List[RuleEvent] = field(default_factory=list)
    """Rules that completed on this token (may be empty)."""

    @property
    def matched(self) -> bool:
        """``True`` if at least one rule was matched."""
        return len(self.events) > 0

    @property
    def rules(self) -> List[str]:
        """Shortcut: list of matched rule names."""
        return [e.rule for e in self.events]


RuleCallback = Callable[["RuleEvent"], None]

# Max recursion depth when expanding recursive rule references into regex
_MAX_DEPTH = 8

# Max regex source length — patterns exceeding this are skipped to avoid
# catastrophic backtracking on complex grammars.
_MAX_REGEX_LEN = 4_000

# Per-rule match timeout in seconds (Unix only, via SIGALRM).
_MATCH_TIMEOUT = 0.5


# ── GrammarMatcher ───────────────────────────────────────────────────


class GrammarMatcher:
    """Incremental matcher that replays accumulated text against the grammar
    and fires :class:`RuleEvent` callbacks when a named rule boundary is
    recognised.

    Rule names exposed through events and ``on()`` always use the **Python
    function name** (``snake_case``), not the GBNF dashed form.

    Parameters
    ----------
    grammar : Grammar
        A fully defined pygbnf grammar (``to_gbnf()`` should succeed).
    only : set[str] | None
        If provided, only compile and track these rule names (Python names).
        Other rules are ignored entirely — saves CPU on large grammars.
    exclude : set[str] | None
        Rule names to skip (Python names).  Applied after *only*.
    """

    def __init__(
        self,
        grammar: Grammar,
        *,
        only: Optional[Set[str]] = None,
        exclude: Optional[Set[str]] = None,
    ) -> None:
        self._grammar = grammar
        self._buffer: str = ""
        self._callbacks: Dict[str, List[RuleCallback]] = {}
        self._emitted: Set[tuple[str, int, int]] = set()
        self._rule_fns: Dict[str, object] = grammar.rule_functions

        # Build rules dict  (GBNF name → node)
        self._rules = grammar.rules()

        # Build a mapping GBNF name ↔ Python name
        self._gbnf_to_py: Dict[str, str] = {}
        self._py_to_gbnf: Dict[str, str] = {}
        for gbnf_name in self._rules:
            py_name = gbnf_name.replace("-", "_")
            self._gbnf_to_py[gbnf_name] = py_name
            self._py_to_gbnf[py_name] = gbnf_name

        # Determine which rules to track
        _only = only
        _exclude = exclude or set()
        def _should_track(gbnf_name: str) -> bool:
            py_name = self._gbnf_to_py.get(gbnf_name, gbnf_name)
            if _only is not None and py_name not in _only:
                return False
            if py_name in _exclude:
                return False
            return True

        # Pre-compile a regex per rule (skip overly complex ones)
        self._patterns: Dict[str, re.Pattern[str]] = {}
        self._skipped: Set[str] = set()
        # Cache and cycle-detection for regex building
        self._regex_cache: Dict[str, str] = {}
        self._expanding: Set[str] = set()
        for gbnf_name, node in self._rules.items():
            if not _should_track(gbnf_name):
                continue
            try:
                regex = self._node_to_regex(node, rule_name=gbnf_name)
                if len(regex) > _MAX_REGEX_LEN:
                    self._skipped.add(gbnf_name)
                    continue
                self._patterns[gbnf_name] = re.compile(regex)
            except _Unsupported:
                self._skipped.add(gbnf_name)

    # ── Public API ───────────────────────────────────────────────────

    def on(self, rule_name: str, callback: RuleCallback) -> None:
        """Register *callback* for *rule_name* (Python function name).

        Use ``"*"`` as a wildcard to receive every rule event.
        """
        self._callbacks.setdefault(rule_name, []).append(callback)

    def feed(self, text: str) -> List[RuleEvent]:
        """Append *text* to the buffer, scan for new matches, and return
        a list of new :class:`RuleEvent` instances (also fires callbacks)."""
        if not text:
            return []
        self._buffer += text
        return self._scan()

    def reset(self) -> None:
        """Clear the buffer and the set of already-emitted events."""
        self._buffer = ""
        self._emitted.clear()

    @property
    def buffer(self) -> str:
        """The full accumulated text fed so far."""
        return self._buffer

    @property
    def rule_functions(self) -> Dict[str, object]:
        """Return ``{python_name: original_function}`` for every decorated rule.

        Convenience shortcut for ``grammar.rule_functions``.
        """
        return self._grammar.rule_functions

    def stream(
        self,
        tokens,
        *,
        exclude: Optional[Set[str]] = None,
    ):
        """Iterate over *tokens* and yield a ``(token, events)`` tuple.

        Parameters
        ----------
        tokens : iterable of str
            An iterable of token strings (e.g. from an LLM streaming response).
        exclude : set[str] | None
            Optional set of rule names to filter out from events.

        Yields
        ------
        tuple[str, list[RuleEvent] | None]
            ``(token, events)`` where *events* is a list of
            :class:`RuleEvent` when at least one rule matched, or
            ``None`` otherwise.

        Example
        -------
        ::

            for token, events in matcher.stream(token_iter):
                print(token, end="")
                if events:
                    for ev in events:
                        print(f"  ← [{ev.rule}] {ev.text}")
        """
        _exclude = exclude or set()
        for tok in tokens:
            if not tok:
                continue
            events = self.feed(tok)
            if _exclude:
                events = [e for e in events if e.rule not in _exclude]
            yield (tok, events if events else None)

    # ── Internals ────────────────────────────────────────────────────

    def _scan(self) -> List[RuleEvent]:
        events: List[RuleEvent] = []
        for gbnf_name, pat in self._patterns.items():
            py_name = self._gbnf_to_py.get(gbnf_name, gbnf_name)
            try:
                matches = list(_finditer_with_timeout(
                    pat, self._buffer, timeout=_MATCH_TIMEOUT
                ))
            except _MatchTimeout:
                # This pattern is too slow — disable it for future scans
                self._skipped.add(gbnf_name)
                continue
            for m in matches:
                key = (gbnf_name, m.start(), m.end())
                if key in self._emitted:
                    continue
                self._emitted.add(key)
                ev = RuleEvent(
                    rule=py_name,
                    text=m.group(),
                    start=m.start(),
                    end=m.end(),
                    fn=self._rule_fns.get(py_name),
                )
                events.append(ev)
                self._fire(ev)
        # Remove patterns that timed out
        for name in self._skipped:
            self._patterns.pop(name, None)
        events.sort(key=lambda e: (e.start, e.end))
        return events

    def _fire(self, ev: RuleEvent) -> None:
        for cb in self._callbacks.get(ev.rule, []):
            cb(ev)
        for cb in self._callbacks.get("*", []):
            cb(ev)

    # ── AST → regex ──────────────────────────────────────────────────

    # Fallback pattern for recursive references — matches typical grammar
    # content (digits, letters, operators, spaces, parens) but stops at
    # structural delimiters like quotes and brackets.
    _FALLBACK = r"[^\[\]'\"]+"

    def _node_to_regex(self, node: Node, *, rule_name: Optional[str] = None) -> str:
        """Convert an AST *node* into an equivalent Python regex string.

        Uses a per-rule cache so that non-recursive rules are expanded
        exactly once.  When a cycle is detected (a rule references itself
        directly or indirectly), the recursive reference is replaced by a
        permissive fallback pattern.
        """
        if isinstance(node, Literal):
            return re.escape(node.value)

        if isinstance(node, CharacterClass):
            neg = "^" if node.negated else ""
            return f"[{neg}{node.pattern}]"

        if isinstance(node, Sequence):
            parts = [self._node_to_regex(c) for c in node.children]
            return "".join(parts)

        if isinstance(node, Alternative):
            parts = [self._node_to_regex(c) for c in node.alternatives]
            return "(?:" + "|".join(parts) + ")"

        if isinstance(node, Repeat):
            inner = self._node_to_regex(node.child)
            lo, hi = node.min, node.max
            if lo == 0 and hi is None:
                return f"(?:{inner})*"
            if lo == 1 and hi is None:
                return f"(?:{inner})+"
            if lo == 0 and hi == 1:
                return f"(?:{inner})?"
            if hi is None:
                return f"(?:{inner}){{{lo},}}"
            return f"(?:{inner}){{{lo},{hi}}}"

        if isinstance(node, Optional_):
            inner = self._node_to_regex(node.child)
            return f"(?:{inner})?"

        if isinstance(node, Group):
            inner = self._node_to_regex(node.child)
            return f"(?:{inner})"

        if isinstance(node, (RuleReference, _RuleProxy)):
            name = node.name
            # Already cached — reuse
            if name in self._regex_cache:
                return self._regex_cache[name]
            # Cycle detected — use fallback
            if name in self._expanding:
                return self._FALLBACK
            ref_node = self._rules.get(name)
            if ref_node is None:
                raise _Unsupported(f"unknown rule: {name}")
            # Mark as currently expanding, build, cache, unmark
            self._expanding.add(name)
            try:
                regex = self._node_to_regex(ref_node)
            finally:
                self._expanding.discard(name)
            self._regex_cache[name] = regex
            return regex

        if isinstance(node, TokenReference):
            raise _Unsupported(f"TokenReference not convertible to regex")

        raise _Unsupported(f"unsupported node type: {type(node).__name__}")


class _Unsupported(Exception):
    """Raised when a node cannot be converted to a regex."""
    pass


class _MatchTimeout(Exception):
    """Raised when regex matching exceeds the allowed time."""
    pass


def _finditer_with_timeout(
    pattern: re.Pattern[str],
    text: str,
    *,
    timeout: float = _MATCH_TIMEOUT,
) -> List[re.Match[str]]:
    """Run ``pattern.finditer(text)`` with a wall-clock timeout.

    On Unix, uses ``SIGALRM``.  On other platforms (or when signals are
    unavailable), falls back to plain ``finditer`` without a timeout.
    """
    if sys.platform == "win32" or not hasattr(signal, "SIGALRM"):
        return list(pattern.finditer(text))

    def _alarm_handler(signum, frame):
        raise _MatchTimeout()

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    # setitimer gives sub-second precision
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        results = list(pattern.finditer(text))
    except _MatchTimeout:
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
    return results
