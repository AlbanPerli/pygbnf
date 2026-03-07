"""
pygbnf.llm — Unified LLM client with optional grammar-constrained streaming.

Wraps an OpenAI-compatible chat completion endpoint and optionally injects
a GBNF grammar + :class:`GrammarMatcher` for real-time rule detection.

Usage
-----
::

    from pygbnf import Grammar, GrammarLLM, select

    g = Grammar()

    @g.rule
    def root():
        return select(["yes", "no", "maybe"])

    llm = GrammarLLM("http://localhost:8080/v1")

    # Streaming with grammar only (no matching) — events always None
    for token, events in llm.stream(messages=[...], grammar=g):
        print(token, end="")

    # Streaming with grammar + matching — yields (token, events | None)
    for token, events in llm.stream(messages=[...], grammar=g, match=True):
        print(token, end="")
        if events:
            for ev in events:
                print(f"  ← [{ev.rule}] {ev.text}")

    # Match only specific rules (implies match=True)
    for token, events in llm.stream(messages=[...], grammar=g, only={"root"}):
        ...

    # Non-streaming
    text, events = llm.complete(messages=[...], grammar=g, match=True)
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from .grammar import Grammar
from .matcher import GrammarMatcher, RuleCallback, RuleEvent


class GrammarLLM:
    """Unified LLM client with optional GBNF grammar constraint.

    Wraps an OpenAI-compatible chat completion client.  The grammar is
    passed per-call to :meth:`stream` or :meth:`complete`, allowing you
    to switch grammars dynamically between requests.

    The streaming interface always yields ``(token, events | None)``
    tuples, whether or not a grammar is provided.

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

    # ── Streaming ────────────────────────────────────────────────────

    def stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        grammar: Grammar | None = None,
        match: bool = False,
        only: Optional[Set[str]] = None,
        exclude: Optional[Set[str]] = None,
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
            If provided, the GBNF string is injected into the request
            to constrain generation.  A :class:`GrammarMatcher` is
            **not** created unless *match*, *only* or *exclude* is set.
        match : bool
            Explicitly enable rule matching for all rules.
        only : set[str] | None
            Only track these rules — implies ``match=True``.
        exclude : set[str] | None
            Skip these rules — implies ``match=True``.
        temperature : float
            Sampling temperature.
        n_predict : int | None
            Maximum number of tokens to predict (llama-server specific).
        **kwargs
            Extra keyword arguments forwarded to
            ``client.chat.completions.create()``.  ``extra_body`` is
            merged (grammar is added automatically).

        Yields
        ------
        tuple[str, list[RuleEvent] | None]
            ``(token, events)`` — *events* is a list of
            :class:`RuleEvent` when at least one rule matched, or
            ``None`` otherwise.  Always ``None`` when no matcher.
        """
        # Build GBNF + optional matcher
        gbnf: str | None = None
        matcher: GrammarMatcher | None = None
        if grammar is not None:
            gbnf = grammar.to_gbnf()
            use_matcher = match or only is not None or exclude is not None
            if use_matcher:
                matcher = GrammarMatcher(grammar, only=only, exclude=exclude)

        self._matcher = matcher
        self._buffer = ""

        # Prepare extra_body with grammar injection
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
        match: bool = False,
        only: Optional[Set[str]] = None,
        exclude: Optional[Set[str]] = None,
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
            If provided, the GBNF string is injected.  A matcher is
            only created when *match*, *only* or *exclude* is set.
        match : bool
            Explicitly enable rule matching for all rules.
        only : set[str] | None
            Only track these rules — implies ``match=True``.
        exclude : set[str] | None
            Skip these rules — implies ``match=True``.
        temperature : float
            Sampling temperature.
        n_predict : int | None
            Maximum number of tokens to predict (llama-server specific).
        **kwargs
            Extra keyword arguments forwarded to
            ``client.chat.completions.create()``.

        Returns
        -------
        tuple[str, list[RuleEvent]]
            ``(full_text, events)`` — *events* is empty if no matcher.
        """
        gbnf: str | None = None
        matcher: GrammarMatcher | None = None
        if grammar is not None:
            gbnf = grammar.to_gbnf()
            use_matcher = match or only is not None or exclude is not None
            if use_matcher:
                matcher = GrammarMatcher(grammar, only=only, exclude=exclude)

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
