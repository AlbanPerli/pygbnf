"""
pygbnf.toolkit — Decorator-based tool registry.

Register functions with ``@toolkit.tool``, then pass the toolkit to
:class:`~pygbnf.GrammarLLM` methods.  The toolkit builds the GBNF
grammar, the system prompt, and dispatches parsed results.

Usage
-----
::

    from pygbnf import Toolkit, GrammarLLM

    toolkit = Toolkit()

    @toolkit.tool
    def get_weather(city: str) -> str:
        \"\"\"Get current weather for a city.\"\"\"
        return f"Sunny in {city}"

    llm = GrammarLLM("http://localhost:8080/v1")

    # One-liner: stream + dispatch
    result = llm.tool_call(toolkit, "What's the weather in Tokyo?")
    print(result)  # "Sunny in Tokyo"

    # Or lower-level — toolkit auto-injects grammar + system prompt
    for token, events in llm.stream(
        messages=[{"role": "user", "content": "Weather?"}],
        toolkit=toolkit,
    ):
        print(token, end="")
"""

from __future__ import annotations

import enum
import json
from typing import Any, Callable, Dict, List, Optional

from .grammar import Grammar
from .combinators import select
from .schema import describe_tools


class Toolkit:
    """Decorator-based tool registry with grammar-constrained calling.

    Parameters
    ----------
    system : str | None
        Custom system prompt prefix.  The tool list is always appended.
    rule_name : str
        Name of the root grammar rule (default ``"tool_call"``).

    Examples
    --------
    ::

        toolkit = Toolkit()

        @toolkit.tool
        def greet(name: str) -> str:
            \"\"\"Say hello.\"\"\"
            return f"Hello, {name}!"

        # toolkit.grammar   → Grammar with GBNF constraints
        # toolkit.describe() → "- greet(name: str) — Say hello."
        # toolkit.dispatch('{"function":"greet","arguments":{"name":"World"}}')
        #   → "Hello, World!"
    """

    def __init__(
        self,
        *,
        system: str | None = None,
        rule_name: str = "tool_call",
    ) -> None:
        self._tools: Dict[str, Callable] = {}
        self._system = system
        self._rule_name = rule_name
        self._grammar: Grammar | None = None  # lazily built

    # ── Registration ─────────────────────────────────────────────────

    def tool(self, fn: Callable) -> Callable:
        """Decorator that registers *fn* as a callable tool.

        The function must have type-annotated parameters.  Enum types
        are constrained to their values in the grammar.

        Returns the original function unchanged.
        """
        self._tools[fn.__name__] = fn
        self._grammar = None  # invalidate cache
        return fn

    # ── Properties ───────────────────────────────────────────────────

    @property
    def tools(self) -> Dict[str, Callable]:
        """Registered tools ``{name: function}``."""
        return dict(self._tools)

    @property
    def grammar(self) -> Grammar:
        """The GBNF grammar constraining tool-call generation.

        Built lazily and cached.  Invalidated when a new tool is registered.
        """
        if self._grammar is None:
            self._grammar = self._build_grammar()
        return self._grammar

    # ── Describe ─────────────────────────────────────────────────────

    def describe(self) -> str:
        """Human-readable tool descriptions for the system prompt.

        Returns
        -------
        str
            Multi-line list of tools with signatures and docstrings.
        """
        return describe_tools(*self._tools.values())

    # ── Grammar building ─────────────────────────────────────────────

    def _build_grammar(self) -> Grammar:
        g = Grammar()
        rule_name = self._rule_name

        @g.rule_named(rule_name)
        def _tool_call():
            return select([g.from_tool_call(fn) for fn in self._tools.values()])

        g.start(rule_name)
        return g

    # ── System prompt ────────────────────────────────────────────────

    def system_prompt(self, extra: str = "") -> str:
        """Build the full system prompt with tool descriptions.

        Parameters
        ----------
        extra : str
            Additional instructions appended after the tool list.
        """
        prefix = self._system or (
            "You are a tool-calling assistant. Given a user request, "
            "reply with a JSON object choosing the right function and arguments."
        )
        parts = [prefix, "", "Available tools:", self.describe()]
        if extra:
            parts.append("")
            parts.append(extra)
        return "\n".join(parts)

    # ── Dispatch ─────────────────────────────────────────────────────

    def dispatch(self, json_str: str, **extra_kwargs: Any) -> Any:
        """Parse a tool-call JSON string and call the matching function.

        Parameters
        ----------
        json_str : str
            JSON with ``{"function": "name", "arguments": {...}}``.
        **extra_kwargs
            Additional keyword arguments merged into the function call.

        Returns
        -------
        Any
            The return value of the called tool function.

        Raises
        ------
        KeyError
            If the function name is not a registered tool.
        """
        call = json.loads(json_str)
        fn_name = call["function"]
        fn_args = call["arguments"]
        fn = self._tools[fn_name]

        # Convert enum string values back to Enum instances
        import inspect
        from typing import get_type_hints
        hints = get_type_hints(fn)
        for param_name, param_type in hints.items():
            if param_name == "return":
                continue
            if isinstance(param_type, type) and issubclass(param_type, enum.Enum):
                if param_name in fn_args and isinstance(fn_args[param_name], str):
                    fn_args[param_name] = param_type(fn_args[param_name])

        fn_args.update(extra_kwargs)
        return fn(**fn_args)
