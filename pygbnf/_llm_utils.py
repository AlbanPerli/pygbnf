from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .grammar import Grammar


def resolve_toolkit(
    messages: List[Dict[str, Any]],
    grammar: Optional[Grammar],
    toolkit: Any,
) -> Tuple[List[Dict[str, Any]], Optional[Grammar]]:
    """Extract grammar from toolkit and prepend a system prompt when needed."""
    if toolkit is None:
        return messages, grammar

    if grammar is not None:
        raise ValueError("Cannot pass both 'grammar' and 'toolkit'.")

    grammar = toolkit.grammar
    has_system = any(message.get("role") == "system" for message in messages)
    if not has_system:
        messages = [{"role": "system", "content": toolkit.system_prompt()}, *messages]

    return messages, grammar


def build_completion_options(
    *,
    gbnf: Optional[str],
    n_predict: Optional[int],
    extra_body: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Build the extra_body payload for compatible completion backends."""
    body = extra_body or {}
    if gbnf:
        body["grammar"] = gbnf
    if n_predict is not None:
        body["n_predict"] = n_predict
    return body or None


def tokenize_with_server(client: Any, text: str) -> List[int]:
    """Tokenize text using a llama.cpp-compatible /tokenize endpoint."""
    base = str(client.base_url).rstrip("/")
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
