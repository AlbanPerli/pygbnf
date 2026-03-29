"""
pygbnf.chain — GrammarChain and GrammarSpace.

Constrained dynamic generation via automaton navigation.

GrammarChain — imperative:

    chain = GrammarChain(llm, stream=True)
    chain |= f"Review this code:\\n{code}"
    category = chain >> Category          # dataclass → instance
    severity = chain >> Severity          # enum → member
    score    = chain >> int               # primitive → int
    diag     = chain >> str               # free line → str

GrammarSpace — declarative automaton (OpenFST style):

    space = GrammarSpace(
        q0 = 0,
        F  = {3},
        δ  = [
            (0, 1, think_g),
            (1, 2, observe_g),
            (2, 3, act_g, lambda v: "conclude" in v),
            (2, 1, act_g, lambda v: "observe"  in v),
        ]
    )

Each arc is (from, to, grammar) or (from, to, grammar, guard).
guard(v) or guard(v, ctx) — first matching arc wins.

Context is a flat ordered trace of CtxEntry(space, arc, value).
"""

from __future__ import annotations

import enum
import inspect
import json
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .grammar import Grammar
from .llm import GrammarLLM


# ── Context ───────────────────────────────────────────────────────────


@dataclass
class CtxEntry:
    """One recorded generation step in the navigation trace."""

    space: "GrammarSpace"
    """The GrammarSpace that produced this entry."""

    arc: Tuple
    """The arc (from, to, grammar[, guard]) that fired."""

    value: Any
    """The Python value produced, after coercion."""

    @property
    def state(self) -> Any:
        """The origin state of the arc (for backwards compat)."""
        return self.arc[0]

    @property
    def grammar(self) -> Any:
        """The grammar used on this arc."""
        return self.arc[2]


# ── Default grammars for primitive Python types ───────────────────────


def _default_grammar(t: type) -> Tuple[Grammar, type]:
    """Return (Grammar, return_type) for a Python primitive or structured type."""
    from .helpers import line, float_number
    from .combinators import select, optional, one_or_more
    from .nodes import CharacterClass
    from .schema import grammar_from_type

    g = Grammar()

    if t is str:
        @g.rule
        def root():
            return line()
        g.start("root")
        return g, str

    if t is int:
        @g.rule
        def root():
            return optional("-") + one_or_more(CharacterClass("0-9"))
        g.start("root")
        return g, int

    if t is float:
        @g.rule
        def root():
            return float_number()
        g.start("root")
        return g, float

    if t is bool:
        @g.rule
        def root():
            return select(["true", "false"])
        g.start("root")
        return g, bool

    # dataclass / Enum / Literal / Union → grammar_from_type (JSON)
    return grammar_from_type(t), t


def _coerce(raw: str, t: type) -> Any:
    """Coerce a raw generated string to the target Python type."""
    raw = raw.strip()

    if t is str:
        return raw
    if t is int:
        return int(raw)
    if t is float:
        return float(raw)
    if t is bool:
        return raw.lower() == "true"

    if isinstance(t, type) and issubclass(t, enum.Enum):
        try:
            return t(raw)
        except ValueError:
            try:
                data = json.loads(raw)
                return t(data)
            except Exception:
                return t[raw]

    # dataclass or other structured type — JSON parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and hasattr(t, "__dataclass_fields__"):
            return t(**parsed)
        return parsed
    except Exception:
        return raw


# ── State resolution ──────────────────────────────────────────────────


def _resolve_grammar(grammar: Any) -> Tuple[Grammar, type]:
    """Return (grammar, return_type) for a grammar reference.

    Note: Toolkit is handled upstream in _generate — should not reach here.
    """
    from .toolkit import Toolkit
    if isinstance(grammar, Toolkit):
        raise ValueError(
            "Toolkit must be handled by _generate, not _resolve_grammar."
        )

    if isinstance(grammar, Grammar):
        return grammar, str

    if isinstance(grammar, type):
        return _default_grammar(grammar)

    # @g.rule reference — callable with _pygbnf_fn attribute
    if callable(grammar) and hasattr(grammar, "_pygbnf_fn"):
        fn = grammar._pygbnf_fn
        return_type = getattr(fn, "__annotations__", {}).get("return", str)
        g = Grammar()
        body = fn

        @g.rule
        def root():
            return body()

        g.start("root")
        return g, return_type

    raise ValueError(
        f"Cannot resolve {grammar!r} to a grammar. "
        "Expected a Grammar, type, or @g.rule reference."
    )


def _call_guard(guard: Callable, value: Any, ctx: List[CtxEntry]) -> bool:
    """Call a guard with (value) or (value, ctx) depending on its signature."""
    try:
        n = len(inspect.signature(guard).parameters)
    except (ValueError, TypeError):
        n = 1
    if n >= 2:
        return bool(guard(value, ctx))
    return bool(guard(value))


# ── GrammarChain — imperative ─────────────────────────────────────────


class GrammarChain:
    """Imperative constrained generation chain.

    Parameters
    ----------
    llm : GrammarLLM
        The LLM client.
    messages : list, optional
        Initial messages (system prompt, user context, etc.).
    ctx : list, optional
        Shared context trace (used when called from a GrammarSpace).
    stream : bool
        If True, stream tokens to stdout as they are generated.
    """

    def __init__(
        self,
        llm: GrammarLLM,
        *,
        messages: Optional[List[Dict[str, Any]]] = None,
        ctx: Optional[List[CtxEntry]] = None,
        stream: bool = False,
    ) -> None:
        self._llm = llm
        self._messages: List[Dict[str, Any]] = list(messages or [])
        self._ctx: List[CtxEntry] = ctx if ctx is not None else []
        self._stream = stream

    # ── Operators ─────────────────────────────────────────────────────

    def __ior__(self, text: str) -> "GrammarChain":
        """Append a user message (context injection)."""
        self._messages.append({"role": "user", "content": str(text)})
        return self

    def __rshift__(self, grammar: Any) -> Any:
        """Generate constrained by *grammar*, return coerced Python value."""
        return self._generate(grammar, arc=None, space=None)

    # ── Internal ──────────────────────────────────────────────────────

    def _generate(
        self,
        grammar: Any,
        *,
        arc: Optional[Tuple],
        space: Optional["GrammarSpace"],
    ) -> Any:
        from .toolkit import Toolkit

        # Ensure user/assistant alternation
        if self._messages and self._messages[-1]["role"] == "assistant":
            self._messages.append({"role": "user", "content": "Continue."})

        # ── Toolkit arc ──────────────────────────────────────────────
        if isinstance(grammar, Toolkit):
            toolkit = grammar
            # Inject tool descriptions into messages if no system prompt yet
            if not any(m["role"] == "system" for m in self._messages):
                self._messages.insert(0, {
                    "role": "system",
                    "content": toolkit.system_prompt(),
                })

            raw = ""
            if self._stream:
                for token, _ in self._llm.stream(
                    messages=self._messages, toolkit=toolkit
                ):
                    sys.stdout.write(token)
                    sys.stdout.flush()
                    raw += token
            else:
                raw, _ = self._llm.complete(
                    messages=self._messages, toolkit=toolkit
                )

            self._messages.append({"role": "assistant", "content": raw})

            # Dispatch and inject tool result as user message
            tool_result = toolkit.dispatch(raw)
            result_text = str(tool_result)
            self._messages.append({"role": "user", "content": result_text})

            value = tool_result
            if space is not None and arc is not None:
                self._ctx.append(CtxEntry(space=space, arc=arc, value=value))
            return value

        # ── Grammar arc ───────────────────────────────────────────────
        resolved_grammar, return_type = _resolve_grammar(grammar)

        raw = ""
        if self._stream:
            for token, _ in self._llm.stream(
                messages=self._messages, grammar=resolved_grammar
            ):
                sys.stdout.write(token)
                sys.stdout.flush()
                raw += token
        else:
            raw, _ = self._llm.complete(
                messages=self._messages, grammar=resolved_grammar
            )

        self._messages.append({"role": "assistant", "content": raw})
        value = _coerce(raw, return_type)

        if space is not None and arc is not None:
            self._ctx.append(CtxEntry(space=space, arc=arc, value=value))

        return value

    # ── Properties ────────────────────────────────────────────────────

    @property
    def ctx(self) -> List[CtxEntry]:
        """Ordered generation trace."""
        return self._ctx

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """Current message history."""
        return self._messages


# ── GrammarSpace — declarative automaton ─────────────────────────────


class GrammarSpace:
    """Declarative automaton over grammars (OpenFST style).

    Parameters
    ----------
    q0 : initial state (any hashable).
    F  : set of final (accepting) states.
    δ  : list of arcs. Each arc is a tuple:
         (from, to, grammar)             — unconditional
         (from, to, grammar, guard)      — conditional

         guard(value) → bool
         guard(value, ctx) → bool

         Multiple arcs from the same state are tried in order —
         the first whose guard returns True fires.
    llm : GrammarLLM, optional
        LLM bound to this space. Used when this space is nested inside
        another (e.g. a judge running on a different server).
        When running standalone via .run(), the llm passed to .run()
        takes precedence unless overridden here.
    """

    def __init__(
        self,
        *,
        q0: Any,
        F: Set[Any],
        δ: List[Tuple],
        llm: Optional[GrammarLLM] = None,
    ) -> None:
        self.q0 = q0
        self.F = F
        self.δ = δ
        self.llm = llm

        # Index arcs by origin state for fast lookup
        self._arcs_from: Dict[Any, List[Tuple]] = {}
        for arc in δ:
            src = arc[0]
            self._arcs_from.setdefault(src, []).append(arc)

    def run(
        self,
        llm: Optional[GrammarLLM] = None,
        *,
        messages: Optional[List[Dict[str, Any]]] = None,
        ctx: Optional[List[CtxEntry]] = None,
        stream: bool = False,
    ) -> "_SpaceRunner":
        """Return a runner ready to execute this space.

        The effective LLM is resolved in this order:
        1. ``llm`` argument passed to ``.run()``
        2. ``llm`` bound at construction time
        """
        effective_llm = llm or self.llm
        if effective_llm is None:
            raise ValueError(
                "No LLM provided — pass one to GrammarSpace(..., llm=...) "
                "or to .run(llm=...)."
            )
        return _SpaceRunner(
            space=self,
            llm=effective_llm,
            messages=list(messages or []),
            ctx=ctx,
            stream=stream,
        )


class _SpaceRunner:
    """Executes a GrammarSpace traversal."""

    def __init__(
        self,
        *,
        space: GrammarSpace,
        llm: GrammarLLM,
        messages: List[Dict[str, Any]],
        ctx: Optional[List[CtxEntry]] = None,
        stream: bool = False,
    ) -> None:
        self._space = space
        self._llm = llm
        self._messages = messages
        self._ctx: List[CtxEntry] = ctx if ctx is not None else []
        self._stream = stream

    def __ior__(self, text: str) -> "_SpaceRunner":
        """Inject context before execution."""
        self._messages.append({"role": "user", "content": str(text)})
        return self

    def execute(self) -> List[CtxEntry]:
        """Traverse the automaton and return the context trace."""
        chain = GrammarChain(
            self._llm,
            messages=self._messages,
            ctx=self._ctx,
            stream=self._stream,
        )
        _traverse(self._space, chain)
        self._messages = chain._messages
        return self._ctx

    @property
    def ctx(self) -> List[CtxEntry]:
        return self._ctx


# ── Traversal ─────────────────────────────────────────────────────────


def _traverse(space: GrammarSpace, chain: GrammarChain) -> None:
    """Traverse *space*, recording each step into *chain._ctx*.

    Semantics (OpenFST style):
    - Generate on the arc, then move to the destination.
    - Stop when arriving at a state with no matching outgoing arc.
    - A final state (F) is accepting — stop is valid there.
    - A final state with outgoing arcs continues normally.
    """
    current = space.q0

    while True:
        arcs = space._arcs_from.get(current, [])
        if not arcs:
            # No outgoing arcs — stop (accepted if in F, blocked otherwise)
            break

        # All arcs from this state share the same grammar — generate once
        grammar = arcs[0][2]

        if isinstance(grammar, GrammarSpace):
            # Use the sub-space's own LLM if it has one
            sub_chain = chain if grammar.llm is None else GrammarChain(
                grammar.llm,
                messages=chain._messages,
                ctx=chain._ctx,
                stream=chain._stream,
            )
            _traverse(grammar, sub_chain)
            if grammar.llm is not None:
                chain._messages = sub_chain._messages
            value = chain._ctx[-1].value if chain._ctx else None
        else:
            # Grammar or Toolkit — both handled by _generate
            value = chain._generate(grammar, arc=arcs[0], space=space)

        next_state = _match_arc(arcs, value, chain._ctx)
        if next_state is None:
            break

        current = next_state

        # Arrived at a final state with no outgoing arcs → stop
        next_arcs = space._arcs_from.get(current, [])
        if current in space.F and not next_arcs:
            break


def _match_arc(
    arcs: List[Tuple],
    value: Any,
    ctx: List[CtxEntry],
) -> Optional[Any]:
    """Return the destination of the first arc whose guard matches *value*."""
    for arc in arcs:
        to = arc[1]
        guard = arc[3] if len(arc) > 3 else None
        if guard is None or _call_guard(guard, value, ctx):
            return to
    return None


# ── Visualization ──────────────────────────────────────────────────────


def _arc_label(arc: Tuple, state_names: Optional[Dict[Any, str]] = None) -> str:
    """Return a short label for a DOT arc."""
    from .toolkit import Toolkit
    grammar = arc[2]
    has_guard = len(arc) > 3

    if isinstance(grammar, GrammarSpace):
        name = getattr(grammar, "name", None) or "subspace"
    elif isinstance(grammar, Toolkit):
        name = "toolkit"
    elif isinstance(grammar, Grammar):
        name = grammar._start or "grammar"
    elif isinstance(grammar, type):
        name = grammar.__name__
    else:
        name = repr(grammar)[:20]

    if has_guard:
        return f"{name} [?]"
    return name


def space_to_dot(
    space: "GrammarSpace",
    *,
    state_names: Optional[Dict[Any, str]] = None,
    name: str = "GrammarSpace",
) -> str:
    """Return a Graphviz DOT string representing *space*.

    Parameters
    ----------
    space : GrammarSpace
    state_names : dict, optional
        Mapping from state id to display name.
    name : str
        Graph name.
    """
    sn = state_names or {}

    def _sname(s: Any) -> str:
        if s in sn:
            return sn[s]
        if isinstance(s, type):
            return s.__name__
        return str(s)

    def _sid(s: Any) -> str:
        return f"s{abs(hash(s)) % 10_000_000}"

    # Collect all states
    states: Set[Any] = set()
    states.add(space.q0)
    for arc in space.δ:
        states.add(arc[0])
        states.add(arc[1])

    lines = [
        f'digraph "{name}" {{',
        '  rankdir=LR;',
        '  node [fontname="Helvetica" fontsize=11];',
        '  edge [fontname="Helvetica" fontsize=10];',
        '',
        '  // invisible entry arrow',
        f'  __start__ [shape=point width=0.2];',
        f'  __start__ -> {_sid(space.q0)};',
        '',
    ]

    for s in sorted(states, key=lambda x: str(x)):
        if s in space.F:
            lines.append(
                f'  {_sid(s)} [label="{s}" shape=doublecircle];'
            )
        else:
            lines.append(
                f'  {_sid(s)} [label="{s}" shape=circle];'
            )

    lines.append('')

    for arc in space.δ:
        src, dst = arc[0], arc[1]
        label = _arc_label(arc, sn)
        # Conditional arcs dashed
        style = 'dashed' if len(arc) > 3 else 'solid'
        lines.append(
            f'  {_sid(src)} -> {_sid(dst)} '
            f'[label="{label}" style={style}];'
        )

    lines.append('}')
    return '\n'.join(lines)


def write_space_dot(
    space: "GrammarSpace",
    path: str,
    **kwargs: Any,
) -> None:
    """Write a DOT file for *space*."""
    from pathlib import Path
    Path(path).write_text(space_to_dot(space, **kwargs))


def write_space_svg(
    space: "GrammarSpace",
    path: str,
    **kwargs: Any,
) -> None:
    """Render *space* to an SVG file (requires Graphviz dot)."""
    import shutil
    import subprocess
    from pathlib import Path

    dot_src = space_to_dot(space, **kwargs)
    dot_bin = shutil.which("dot")
    if dot_bin is None:
        raise RuntimeError("Graphviz 'dot' not found — install Graphviz.")
    result = subprocess.run(
        [dot_bin, "-Tsvg"],
        input=dot_src.encode(),
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode())
    Path(path).write_bytes(result.stdout)
