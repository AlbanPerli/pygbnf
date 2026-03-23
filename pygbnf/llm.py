"""
pygbnf.llm — Unified LLM client with optional grammar-constrained streaming.

Wraps an OpenAI-compatible chat completion endpoint and optionally injects
a GBNF grammar + :class:`GrammarMatcher` for real-time rule detection.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from .grammar import Grammar
from .matcher import GrammarMatcher, RuleCallback, RuleEvent
from ._llm_utils import build_completion_options, resolve_toolkit, tokenize_with_server
from ._llm_weights import compute_logit_bias as _compute_logit_bias
from ._llm_weights import safe_compute_logit_bias


class GrammarLLM:
    """Unified LLM client with optional GBNF grammar constraint."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        model: str = "gpt-4-vision-preview",
        *,
        api_key: str = "sk-no-key-required",
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for GrammarLLM.  "
                "Install it with:  pip install openai"
            ) from None

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._buffer: str = ""
        self._matcher: GrammarMatcher | None = None

    @property
    def matcher(self) -> GrammarMatcher | None:
        return self._matcher

    @property
    def buffer(self) -> str:
        if self._matcher is not None:
            return self._matcher.buffer
        return self._buffer

    @staticmethod
    def _resolve_toolkit(
        messages: List[Dict[str, Any]],
        grammar: Grammar | None,
        toolkit: Any,
    ) -> Tuple[List[Dict[str, Any]], Grammar | None]:
        return resolve_toolkit(messages, grammar, toolkit)

    @staticmethod
    def _build_matcher(
        grammar: Optional[Grammar],
        *,
        match: bool,
        only: Optional[Set[str]],
        exclude: Optional[Set[str]],
        on: Optional[Dict[str, RuleCallback]],
    ) -> Tuple[Optional[str], Optional[GrammarMatcher]]:
        if grammar is None:
            return None, None

        gbnf = grammar.to_gbnf()
        use_matcher = match or only is not None or exclude is not None or on is not None
        if not use_matcher:
            return gbnf, None

        matcher = GrammarMatcher(grammar, only=only, exclude=exclude)
        if on:
            for rule_name, callback in on.items():
                matcher.on(rule_name, callback)
        return gbnf, matcher

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
        messages, grammar = self._resolve_toolkit(messages, grammar, toolkit)
        gbnf, matcher = self._build_matcher(
            grammar,
            match=match,
            only=only,
            exclude=exclude,
            on=on,
        )

        self._matcher = matcher
        self._buffer = ""

        extra_body = build_completion_options(
            gbnf=gbnf,
            n_predict=n_predict,
            extra_body=kwargs.pop("extra_body", None),
        )
        logit_bias = self._compute_weight_bias(grammar)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            stream=True,
            extra_body=extra_body,
            logit_bias=logit_bias,
            **kwargs,
        )

        for chunk in response:
            token = chunk.choices[0].delta.content or ""
            if not token:
                continue

            if matcher:
                events = matcher.feed(token)
                yield (token, events if events else None)
            else:
                self._buffer += token
                yield (token, None)

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
        messages, grammar = self._resolve_toolkit(messages, grammar, toolkit)
        gbnf, matcher = self._build_matcher(
            grammar,
            match=match,
            only=only,
            exclude=exclude,
            on=on,
        )

        self._matcher = matcher
        self._buffer = ""

        extra_body = build_completion_options(
            gbnf=gbnf,
            n_predict=n_predict,
            extra_body=kwargs.pop("extra_body", None),
        )
        logit_bias = self._compute_weight_bias(grammar)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            stream=False,
            extra_body=extra_body,
            logit_bias=logit_bias,
            **kwargs,
        )

        text = response.choices[0].message.content or ""
        self._buffer = text

        events: List[RuleEvent] = []
        if matcher:
            events = matcher.feed(text)

        return (text, events)

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

    def tokenize(self, text: str) -> List[int]:
        return tokenize_with_server(self._client, text)

    def compute_logit_bias(
        self,
        grammar: Grammar,
        *,
        tokenize_fn: Optional[Callable[[str], List[int]]] = None,
        bias_scale: float = 10.0,
    ) -> Dict[str, float]:
        if tokenize_fn is None:
            tokenize_fn = self.tokenize
        return _compute_logit_bias(
            grammar,
            tokenize_fn=tokenize_fn,
            bias_scale=bias_scale,
        )

    def _compute_weight_bias(
        self,
        grammar: Optional[Grammar],
    ) -> Optional[Dict[str, float]]:
        return safe_compute_logit_bias(
            grammar,
            tokenize_fn=self.tokenize,
        )
