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
import math
import sys
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from .grammar import Grammar
from .matcher import GrammarMatcher, RuleCallback, RuleEvent
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

        logit_bias = self._compute_weight_bias(grammar)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            stream=True,
            extra_body=extra_body if extra_body else None,
            logit_bias=logit_bias,
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

        logit_bias = self._compute_weight_bias(grammar)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            stream=False,
            extra_body=extra_body if extra_body else None,
            logit_bias=logit_bias,
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

    # ── Tokenization ─────────────────────────────────────────────────

    def tokenize(self, text: str) -> List[int]:
        """Tokenize *text* using the server's ``/tokenize`` endpoint.

        Works with llama-server which exposes ``/tokenize`` at the base URL.

        Raises
        ------
        RuntimeError
            If the server does not support ``/tokenize``.
        """
        base = str(self._client.base_url).rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]

        url = f"{base}/tokenize"
        payload = json.dumps({"content": text}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(
                f"Cannot reach {url} — is llama-server running? ({exc})"
            ) from exc

        return data["tokens"]

    # ── Logit bias from grammar weights ──────────────────────────────

    def compute_logit_bias(
        self,
        grammar: Grammar,
        *,
        tokenize_fn: Optional[Callable[[str], List[int]]] = None,
        bias_scale: float = 10.0,
    ) -> Dict[str, float]:
        """Extract :class:`WeightedAlternative` nodes and return ``logit_bias``.

        Walks the grammar AST, finds all weighted alternatives, tokenizes
        the first token of each branch **in context** (with the preceding
        literal text), and converts weights to logit adjustments via
        ``bias_scale * ln(weight)``.

        Parameters
        ----------
        grammar : Grammar
            A grammar potentially containing weighted alternatives.
        tokenize_fn : callable | None
            ``text → [token_id, ...]``.  Defaults to :meth:`tokenize`.
        bias_scale : float
            Multiplier for ``ln(weight)`` adjustments.  With the default
            of ``10.0``, a weight of ``50`` maps to a bias of ≈ +39
            and a weight of ``0.05`` maps to ≈ −30.

        Returns
        -------
        dict[str, float]
            ``{token_id_str: bias_value}`` ready for the OpenAI API.
        """
        if tokenize_fn is None:
            tokenize_fn = self.tokenize

        weighted = _collect_weighted(grammar)
        if not weighted:
            return {}

        bias: Dict[str, float] = {}
        for wa, ctx in weighted:
            ctx_tokens = tokenize_fn(ctx) if ctx else []
            for alt, weight in zip(wa.alternatives, wa.weights):
                prefix = _first_literal_prefix(alt)
                if prefix is None or weight == 1.0:
                    continue
                full_tokens = tokenize_fn(ctx + prefix)
                # Find the first diverging token after the shared context.
                diff_idx = 0
                for i in range(min(len(ctx_tokens), len(full_tokens))):
                    if ctx_tokens[i] != full_tokens[i]:
                        diff_idx = i
                        break
                else:
                    diff_idx = len(ctx_tokens)

                if diff_idx < len(full_tokens):
                    tid = str(full_tokens[diff_idx])
                    b = bias_scale * math.log(weight)
                    bias[tid] = bias.get(tid, 0.0) + b

        return bias

    # ── Internal: inject weight bias ─────────────────────────────────

    def _compute_weight_bias(
        self, grammar: Optional[Grammar],
    ) -> Optional[Dict[str, float]]:
        """If *grammar* has weighted alternatives, compute logit_bias dict."""
        if grammar is None:
            return None
        try:
            wb = self.compute_logit_bias(grammar)
            return wb if wb else None
        except RuntimeError:
            return None  # tokenize endpoint unavailable — skip biasing


# ── Weighted alternative helpers ─────────────────────────────────────

def _collect_weighted(grammar: Grammar) -> List[Tuple[WeightedAlternative, str]]:
    """Walk the grammar AST and collect ``(WeightedAlternative, context)`` pairs.

    *context* is the literal text that precedes the weighted node,
    resolved through :class:`RuleReference` boundaries.  This context is
    essential for correct BPE tokenization (space merging etc.).
    """
    rules = grammar.rules()
    results: List[Tuple[WeightedAlternative, str]] = []
    visited: Set[str] = set()

    # Determine the entry-point rule and walk from there so that
    # RuleReferences carry accumulated literal context.
    start_name = getattr(grammar, "_start", None) or "root"
    entry = rules.get(start_name)
    if entry is not None:
        visited.add(start_name)
        _walk_weighted(entry, "", results, rules, visited)

    # Pick up any weighted alternatives in unreachable rules.
    for name, node in rules.items():
        if name not in visited:
            visited.add(name)
            _walk_weighted(node, "", results, rules, visited)

    return results


def _walk_weighted(
    node: Node,
    ctx: str,
    acc: List[Tuple[WeightedAlternative, str]],
    rules: Dict[str, Node],
    visited: Set[str],
) -> None:
    if isinstance(node, WeightedAlternative):
        acc.append((node, ctx))
        # Don't recurse into alternatives — they're individual branches
    elif isinstance(node, Alternative):
        for a in node.alternatives:
            _walk_weighted(a, ctx, acc, rules, visited)
    elif isinstance(node, Sequence):
        running_ctx = ctx
        for child in node.children:
            _walk_weighted(child, running_ctx, acc, rules, visited)
            lit = _full_literal_text(child)
            if lit is not None:
                running_ctx += lit
            else:
                running_ctx = ""  # non-literal breaks context
    elif isinstance(node, RuleReference):
        if node.name not in visited:
            visited.add(node.name)
            target = rules.get(node.name)
            if target is not None:
                _walk_weighted(target, ctx, acc, rules, visited)
    elif isinstance(node, (Repeat, Optional_, Group)):
        child = getattr(node, "child", None)
        if child is not None:
            _walk_weighted(child, ctx, acc, rules, visited)


def _first_literal_prefix(node: Node) -> Optional[str]:
    """Extract the literal text at the start of a node, or None."""
    if isinstance(node, Literal):
        return node.value if node.value else None
    if isinstance(node, Sequence) and node.children:
        return _first_literal_prefix(node.children[0])
    if isinstance(node, Group):
        return _first_literal_prefix(node.child)
    if isinstance(node, (Alternative, WeightedAlternative)):
        return None
    return None


def _full_literal_text(node: Node) -> Optional[str]:
    """Return the full literal text a node produces, or None if it's dynamic."""
    if isinstance(node, Literal):
        return node.value
    if isinstance(node, Sequence):
        parts: List[str] = []
        for child in node.children:
            t = _full_literal_text(child)
            if t is None:
                return None
            parts.append(t)
        return "".join(parts)
    return None
