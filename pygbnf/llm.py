"""
pygbnf.llm — Unified LLM client with optional grammar-constrained streaming.

Wraps an OpenAI-compatible chat completion endpoint and optionally injects
a GBNF grammar + :class:`GrammarMatcher` for real-time rule detection.

Usage
-----
::

    from pygbnf import Grammar, GrammarLLM, Toolkit, select

    g = Grammar()

    @g.rule
    def root():
        return select(["yes", "no", "maybe"])

    llm = GrammarLLM("http://localhost:8080/v1")

    # Streaming with grammar only (no matching)
    for token, events in llm.stream(messages=[...], grammar=g):
        print(token, end="")

    # Streaming with grammar + matching
    for token, events in llm.stream(messages=[...], grammar=g, match=True):
        ...

    # With a Toolkit — grammar + system prompt auto-injected
    toolkit = Toolkit()

    @toolkit.tool
    def get_weather(city: str) -> str: ...

    for token, events in llm.stream(
        messages=[{"role": "user", "content": "Weather in Tokyo?"}],
        toolkit=toolkit,
    ):
        print(token, end="")

    # One-liner: stream + dispatch
    result = llm.tool_call(toolkit, "Weather in Tokyo?")
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from .grammar import Grammar
from .matcher import GrammarMatcher, RuleCallback, RuleEvent


class GrammarLLM:
    """Unified LLM client with optional GBNF grammar constraint.

    Wraps an OpenAI-compatible chat completion client.  The grammar is
    passed per-call to :meth:`stream` or :meth:`complete`, allowing you
    to switch grammars dynamically between requests.

    When a :class:`~pygbnf.Toolkit` is passed instead of a raw grammar,
    the toolkit's grammar and system prompt are injected automatically.

    Parameters
    ----------
    base_url : str
        Base URL of the OpenAI-compatible server.
    model : str
        Model name to pass to the API.
    api_key : str
        API key (default ``"sk-no-key-required"`` for local servers).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        model: str = "gpt-4-vision-preview",
        *,
        api_key: str = "sk-no-key-required",
    ) -> None:
        try:
            from openai import OpenAI  # noqa: F811
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for GrammarLLM.  "
                "Install it with:  pip install openai"
            ) from None

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._buffer: str = ""
        self._matcher: GrammarMatcher | None = None

    # ── Properties ───────────────────────────────────────────────────

    @property
    def matcher(self) -> GrammarMatcher | None:
        """The :class:`GrammarMatcher` from the last call, or ``None``."""
        return self._matcher

    @property
    def buffer(self) -> str:
        """Accumulated text from the last :meth:`stream` or :meth:`complete`."""
        if self._matcher is not None:
            return self._matcher.buffer
        return self._buffer

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _resolve_toolkit(
        messages: List[Dict[str, Any]],
        grammar: Grammar | None,
        toolkit: Any,
    ) -> Tuple[List[Dict[str, Any]], Grammar | None]:
        """Extract grammar from toolkit and prepend system prompt."""
        if toolkit is None:
            return messages, grammar

        if grammar is not None:
            raise ValueError("Cannot pass both 'grammar' and 'toolkit'.")

        grammar = toolkit.grammar

        # Prepend system prompt if no system message already present
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            messages = [
                {"role": "system", "content": toolkit.system_prompt()},
                *messages,
            ]

        return messages, grammar

    # ── Streaming ────────────────────────────────────────────────────

    def stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        grammar: Grammar | None = None,
        toolkit: Any = None,
        match: bool = False,
        only: Optional[Set[str]] = None,
        exclude: Optional[Set[str]] = None,
        on: Optional[Dict[str, RuleCallback]] = None,
        temperature: float = 0,
        n_predict: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[Tuple[str, Optional[List[RuleEvent]]]]:
        """Streaming chat completion yielding ``(token, events | None)``.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-format message list.
        grammar : Grammar | None
            GBNF grammar to constrain generation.  Mutually exclusive
            with *toolkit*.
        toolkit : Toolkit | None
            Tool registry — auto-injects ``toolkit.grammar`` and
            ``toolkit.system_prompt()``.  Mutually exclusive with
            *grammar*.
        match : bool
            Explicitly enable rule matching for all rules.
        only : set[str] | None
            Only track these rules — implies ``match=True``.
        exclude : set[str] | None
            Skip these rules — implies ``match=True``.
        on : dict[str, RuleCallback] | None
            Callbacks keyed by rule name — implies ``match=True``.
            Use ``"*"`` for a wildcard.
        temperature : float
            Sampling temperature.
        n_predict : int | None
            Maximum tokens to predict (llama-server specific).
        **kwargs
            Forwarded to ``client.chat.completions.create()``.

        Yields
        ------
        tuple[str, list[RuleEvent] | None]
        """
        messages, grammar = self._resolve_toolkit(messages, grammar, toolkit)

        # Build GBNF + optional matcher
        gbnf: str | None = None
        matcher: GrammarMatcher | None = None
        if grammar is not None:
            gbnf = grammar.to_gbnf()
            use_matcher = match or only is not None or exclude is not None or on is not None
            if use_matcher:
                matcher = GrammarMatcher(grammar, only=only, exclude=exclude)
                if on:
                    for rule_name, cb in on.items():
                        matcher.on(rule_name, cb)

        self._matcher = matcher
        self._buffer = ""

        extra_body = kwargs.pop("extra_body", {}) or {}
        if gbnf:
            extra_body["grammar"] = gbnf
        if n_predict is not None:
            extra_body["n_predict"] = n_predict

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            stream=True,
            extra_body=extra_body if extra_body else None,
            **kwargs,
        )

        for chunk in response:
            tok = chunk.choices[0].delta.content or ""
            if not tok:
                continue

            if matcher:
                events = matcher.feed(tok)
                yield (tok, events if events else None)
            else:
                self._buffer += tok
                yield (tok, None)

    # ── Non-streaming ────────────────────────────────────────────────

    def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        grammar: Grammar | None = None,
        toolkit: Any = None,
        match: bool = False,
        only: Optional[Set[str]] = None,
        exclude: Optional[Set[str]] = None,
        on: Optional[Dict[str, RuleCallback]] = None,
        temperature: float = 0,
        n_predict: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[str, List[RuleEvent]]:
        """Non-streaming chat completion.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-format message list.
        grammar : Grammar | None
            GBNF grammar.  Mutually exclusive with *toolkit*.
        toolkit : Toolkit | None
            Same as in :meth:`stream`.
        match : bool
            Enable rule matching.
        only / exclude / on
            Matcher configuration — see :meth:`stream`.
        temperature : float
            Sampling temperature.
        n_predict : int | None
            Maximum tokens to predict.
        **kwargs
            Forwarded to ``client.chat.completions.create()``.

        Returns
        -------
        tuple[str, list[RuleEvent]]
            ``(full_text, events)`` — *events* is empty if no matcher.
        """
        messages, grammar = self._resolve_toolkit(messages, grammar, toolkit)

        gbnf: str | None = None
        matcher: GrammarMatcher | None = None
        if grammar is not None:
            gbnf = grammar.to_gbnf()
            use_matcher = match or only is not None or exclude is not None or on is not None
            if use_matcher:
                matcher = GrammarMatcher(grammar, only=only, exclude=exclude)
                if on:
                    for rule_name, cb in on.items():
                        matcher.on(rule_name, cb)

        self._matcher = matcher
        self._buffer = ""

        extra_body = kwargs.pop("extra_body", {}) or {}
        if gbnf:
            extra_body["grammar"] = gbnf
        if n_predict is not None:
            extra_body["n_predict"] = n_predict

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            stream=False,
            extra_body=extra_body if extra_body else None,
            **kwargs,
        )

        text = response.choices[0].message.content or ""
        self._buffer = text

        events: List[RuleEvent] = []
        if matcher:
            events = matcher.feed(text)

        return (text, events)

    # ── Tool calling convenience ─────────────────────────────────────

    def tool_call(
        self,
        toolkit: Any,
        user_message: str,
        *,
        stream: bool = True,
        print_tokens: bool = True,
        on: Optional[Dict[str, RuleCallback]] = None,
        on_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        n_predict: int = 256,
        **kwargs: Any,
    ) -> Any:
        """Ask the LLM to pick a tool, then dispatch it automatically.

        Highest-level convenience method.  Sends the user message with
        the toolkit's grammar and system prompt, collects the constrained
        response, and calls the matching tool function.

        Parameters
        ----------
        toolkit : Toolkit
            Tool registry (grammar + system prompt + dispatch).
        user_message : str
            The user's natural-language request.
        stream : bool
            If ``True`` (default), stream tokens; otherwise use
            ``complete()``.
        print_tokens : bool
            If ``True`` and *stream* is ``True``, print tokens to
            stdout as they arrive.
        on : dict[str, RuleCallback] | None
            Per-rule matcher callbacks.
        on_call : callable | None
            Callback ``(fn_name, fn_args)`` called just before dispatch.
        n_predict : int
            Maximum tokens.
        **kwargs
            Forwarded to the underlying stream/complete call.

        Returns
        -------
        Any
            The return value of the dispatched tool function.
        """
        messages = [{"role": "user", "content": user_message}]

        if stream:
            result = ""
            for token, _ in self.stream(
                messages=messages,
                toolkit=toolkit,
                on=on,
                n_predict=n_predict,
                **kwargs,
            ):
                if print_tokens:
                    sys.stdout.write(token)
                    sys.stdout.flush()
                result += token
        else:
            result, _ = self.complete(
                messages=messages,
                toolkit=toolkit,
                n_predict=n_predict,
                **kwargs,
            )

        call = json.loads(result)
        if on_call:
            on_call(call["function"], call["arguments"])
        return toolkit.dispatch(result)
