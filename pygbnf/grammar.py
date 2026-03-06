"""
pygbnf.grammar — Grammar container and rule registration.

The :class:`Grammar` class is the central hub for defining rules via the
``@g.rule`` decorator.  It stores the AST for each rule, resolves forward
references, tracks dependencies, and delegates to the code-generator for
GBNF output.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set

from .nodes import (
    Alternative,
    Group,
    Literal,
    Node,
    Optional_,
    Repeat,
    RuleReference,
    Sequence,
    TokenReference,
    CharacterClass,
)


# ---------------------------------------------------------------------------
# Lazy rule proxy — returned by calling a registered rule function inside
# another rule body.  It defers expansion to a plain RuleReference.
# ---------------------------------------------------------------------------

class _RuleProxy(Node):
    """Thin wrapper so that ``number()`` inside a rule body yields a
    :class:`RuleReference` rather than evaluating the rule function eagerly.
    """
    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        # bypass frozen=True from Node by using object.__setattr__
        object.__setattr__(self, "_name", name)

    # Act as a RuleReference in every practical sense
    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"_RuleProxy({self._name!r})"


# ---------------------------------------------------------------------------
# Grammar container
# ---------------------------------------------------------------------------

class Grammar:
    """Central grammar container.

    Example
    -------
    >>> g = Grammar()
    >>> @g.rule
    ... def number():
    ...     return select("0123456789")
    >>> g.start("number")
    >>> print(g.to_gbnf())
    """

    def __init__(self) -> None:
        self._rules: Dict[str, Node] = {}
        self._rule_order: List[str] = []
        self._start: Optional[str] = None
        self._building: bool = False  # guard against nested builds

    # ----- rule registration ------------------------------------------------

    def rule(self, fn: Callable[[], Node]) -> Callable[[], RuleReference]:
        """Decorator that registers *fn* as a grammar rule.

        When the decorated function is later *called* inside another rule body,
        it returns a :class:`RuleReference` (it does **not** re-evaluate the
        body).
        """
        name = _to_rule_name(fn.__name__)

        def _ref() -> _RuleProxy:
            return _RuleProxy(name)

        # Attach metadata so we can evaluate the body lazily
        _ref._pygbnf_name = name  # type: ignore[attr-defined]
        _ref._pygbnf_fn = fn      # type: ignore[attr-defined]
        _ref._pygbnf_registered = False  # type: ignore[attr-defined]

        # Register into the grammar immediately — but evaluate the body later
        # so that forward references work.
        self._rule_order.append(name)
        self._rules[name] = None  # placeholder  # type: ignore[assignment]

        # Store the callable so we can build later
        if not hasattr(self, "_deferred"):
            self._deferred: Dict[str, Callable] = {}
        self._deferred[name] = fn

        # Also store the wrapper so ``start()`` can look it up
        if not hasattr(self, "_wrappers"):
            self._wrappers: Dict[str, Callable] = {}
        self._wrappers[name] = _ref

        return _ref

    def rule_named(self, name: str) -> Callable:
        """Decorator that registers a rule with an explicit name.

        Useful when the GBNF rule name should differ from the Python function
        name (e.g. generated names like ``movie_title_items``).

        Parameters
        ----------
        name : str
            The rule name in the grammar.
        """
        rule_name = _to_rule_name(name)

        def decorator(fn: Callable[[], Node]) -> Callable[[], _RuleProxy]:
            def _ref() -> _RuleProxy:
                return _RuleProxy(rule_name)

            _ref._pygbnf_name = rule_name  # type: ignore[attr-defined]
            _ref._pygbnf_fn = fn           # type: ignore[attr-defined]

            if rule_name not in self._rule_order:
                self._rule_order.append(rule_name)
            self._rules[rule_name] = None  # type: ignore[assignment]

            if not hasattr(self, "_deferred"):
                self._deferred: Dict[str, Callable] = {}
            self._deferred[rule_name] = fn

            if not hasattr(self, "_wrappers"):
                self._wrappers: Dict[str, Callable] = {}
            self._wrappers[rule_name] = _ref

            return _ref
        return decorator

    def ref(self, name: str) -> RuleReference:
        """Create a :class:`RuleReference` to a rule by name.

        Parameters
        ----------
        name : str
            The target rule name (will be converted to GBNF naming).
        """
        return RuleReference(name=_to_rule_name(name))

    def from_type(self, tp: type) -> RuleReference:
        """Inject rules for a Python type and return a composable reference.

        Converts a dataclass, primitive type, ``Literal``, ``Enum``,
        ``Optional``, ``list[X]``, etc. into grammar rules (JSON format)
        and returns a :class:`RuleReference` that can be used inside any
        rule body.

        Parameters
        ----------
        tp : type
            The Python type to convert.

        Returns
        -------
        RuleReference
            A node that can be composed with ``+`` and ``|`` operators.

        Example
        -------
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Movie:
        ...     title: str
        ...     year: int
        >>> g = Grammar()
        >>> @g.rule
        ... def root():
        ...     return g.from_type(Movie)
        """
        from .schema import SchemaCompiler
        if not hasattr(self, '_schema_compiler'):
            self._schema_compiler = SchemaCompiler(grammar=self)
        return self._schema_compiler._type_to_node(tp, name_hint=tp.__name__.lower() if hasattr(tp, '__name__') else '')

    def from_function_return(self, func) -> RuleReference:
        """Inject rules for a function's return type.

        Parameters
        ----------
        func : callable
            A function with a ``-> ReturnType`` annotation.

        Returns
        -------
        RuleReference
        """
        from typing import get_type_hints
        hints = get_type_hints(func)
        ret = hints.get('return')
        if ret is None:
            raise TypeError(f"{func.__name__!r} has no return type annotation")
        return self.from_type(ret)

    def from_function_args(self, func) -> RuleReference:
        """Inject rules for a function's arguments as a JSON object.

        Parameters
        ----------
        func : callable
            A function with annotated parameters.

        Returns
        -------
        RuleReference
        """
        import dataclasses
        import inspect
        from typing import Any, get_type_hints
        hints = get_type_hints(func)
        sig = inspect.signature(func)
        dc_fields = []
        for name, _param in sig.parameters.items():
            if name in ('self', 'cls'):
                continue
            dc_fields.append((name, hints.get(name, Any)))
        ArgsClass = dataclasses.make_dataclass(
            f"{func.__name__}_args", dc_fields,
        )
        return self.from_type(ArgsClass)

    # ----- build (lazy evaluation) ------------------------------------------

    def _ensure_built(self) -> None:
        """Evaluate all deferred rule bodies (once)."""
        if self._building:
            return
        self._building = True
        try:
            for name in list(self._rule_order):
                if self._rules.get(name) is None and name in self._deferred:
                    body = self._deferred[name]()
                    self._rules[name] = _normalise(body)
        finally:
            self._building = False

    # ----- public API -------------------------------------------------------

    def start(self, rule_name: str) -> None:
        """Set the start (root) rule of the grammar."""
        name = _to_rule_name(rule_name)
        self._start = name

    def rules(self) -> Dict[str, Node]:
        """Return a mapping ``{rule_name: ast_node}``."""
        self._ensure_built()
        return dict(self._rules)

    def to_gbnf(self, *, optimize: bool = True) -> str:
        """Compile the grammar into a GBNF string.

        Parameters
        ----------
        optimize : bool
            If ``True`` (default), apply optimization passes before emitting.
        """
        self._ensure_built()
        from .gbnf_codegen import compile_grammar
        return compile_grammar(self, optimize=optimize)

    def pretty_print(self, *, optimize: bool = True) -> None:
        """Print the compiled GBNF grammar to stdout."""
        print(self.to_gbnf(optimize=optimize))

    def dependency_graph(self) -> Dict[str, Set[str]]:
        """Return a dict mapping each rule to the set of rules it references."""
        self._ensure_built()
        graph: Dict[str, Set[str]] = defaultdict(set)
        for name, node in self._rules.items():
            _collect_refs(node, graph[name])
        return dict(graph)

    def detect_left_recursion(self) -> List[List[str]]:
        """Return a list of left-recursive cycles (each cycle is a list of
        rule names).  Emits warnings for each cycle found.
        """
        self._ensure_built()
        cycles = _find_left_recursive_cycles(self._rules)
        for cycle in cycles:
            path = " -> ".join(cycle + [cycle[0]])
            warnings.warn(
                f"Left recursion detected: {path}. "
                f"Consider rewriting as iterative: base (op base)*",
                stacklevel=2,
            )
        return cycles


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_rule_name(py_name: str) -> str:
    """Convert a Python identifier (``snake_case``) into a valid GBNF rule
    name (``dashed-lowercase``)."""
    return py_name.replace("_", "-")


def _normalise(node: object) -> Node:
    """Ensure *node* is an AST node — convert ``_RuleProxy`` into
    :class:`RuleReference` recursively, and coerce plain strings."""
    if isinstance(node, _RuleProxy):
        return RuleReference(name=node.name)
    if isinstance(node, str):
        return Literal(node)
    if isinstance(node, Sequence):
        return Sequence(children=[_normalise(c) for c in node.children])
    if isinstance(node, Alternative):
        return Alternative(alternatives=[_normalise(a) for a in node.alternatives])
    if isinstance(node, Repeat):
        return Repeat(child=_normalise(node.child), min=node.min, max=node.max)
    if isinstance(node, Group):
        return Group(child=_normalise(node.child))
    if isinstance(node, Optional_):
        return Optional_(child=_normalise(node.child))
    if isinstance(node, Node):
        return node
    raise TypeError(f"Cannot normalise {type(node)!r} to a grammar Node")


def _collect_refs(node: Node, refs: Set[str]) -> None:
    """Walk *node* and collect all :class:`RuleReference` names into *refs*."""
    if isinstance(node, RuleReference):
        refs.add(node.name)
    elif isinstance(node, _RuleProxy):
        refs.add(node.name)
    elif isinstance(node, Sequence):
        for c in node.children:
            _collect_refs(c, refs)
    elif isinstance(node, Alternative):
        for a in node.alternatives:
            _collect_refs(a, refs)
    elif isinstance(node, (Repeat, Group, Optional_)):
        _collect_refs(node.child, refs)


def _first_refs(node: Node, rules: Dict[str, Node]) -> Set[str]:
    """Return rule names reachable as the *leftmost* symbol of *node*."""
    if isinstance(node, RuleReference):
        return {node.name}
    if isinstance(node, Sequence):
        if not node.children:
            return set()
        first = node.children[0]
        refs = _first_refs(first, rules)
        # If the first child can match ε, also look at the second, etc.
        # For simplicity, we only consider the strict first child here.
        return refs
    if isinstance(node, Alternative):
        out: Set[str] = set()
        for a in node.alternatives:
            out |= _first_refs(a, rules)
        return out
    if isinstance(node, (Repeat, Group, Optional_)):
        return _first_refs(node.child, rules)
    return set()


def _find_left_recursive_cycles(rules: Dict[str, Node]) -> List[List[str]]:
    """Detect left-recursive cycles via DFS on the *first-symbol* graph."""
    first_graph: Dict[str, Set[str]] = {}
    for name, node in rules.items():
        first_graph[name] = _first_refs(node, rules)

    visited: Set[str] = set()
    on_stack: Set[str] = set()
    stack: List[str] = []
    cycles: List[List[str]] = []

    def dfs(name: str) -> None:
        if name in on_stack:
            idx = stack.index(name)
            cycles.append(list(stack[idx:]))
            return
        if name in visited:
            return
        visited.add(name)
        on_stack.add(name)
        stack.append(name)
        for dep in first_graph.get(name, set()):
            dfs(dep)
        stack.pop()
        on_stack.discard(name)

    for name in rules:
        dfs(name)
    return cycles
