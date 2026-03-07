#!/usr/bin/env python3
"""Example — Vision + Grammar constrained generation.

Send an image (e.g. a math problem on a whiteboard) to a multimodal
model served by llama-server, and constrain the response with a GBNF
grammar so the output is always a valid arithmetic expression with its
result.

Requirements:
    pip install openai

Usage:
    # Start llama-server with a vision model, e.g.:
    # llama-server -m model.gguf --mmproj mmproj.gguf
    python examples/demo_vision.py path/to/image.png
"""

import base64
import sys
import time
from pathlib import Path

import pygbnf as cfg
from pygbnf import select, one_or_more, zero_or_more, optional, CharacterClass
from pygbnf import GrammarLLM

# ── Grammar: arithmetic expression = integer result ──────────────────

g = cfg.Grammar()


@g.rule
def ws():
    return zero_or_more(select(" \t"))


@g.rule
def digit():
    return CharacterClass(pattern="0-9")


@g.rule
def integer():
    return optional("-") + one_or_more(digit())


@g.rule
def operator():
    return select(["+", "-", "*", "/"])


@g.rule
def expression():
    """e.g. 215 * 74 or (2 + 3) * 4"""
    atom = select([
        integer(),
        "(" + ws() + expression() + ws() + ")",
    ])
    return atom + zero_or_more(
        cfg.group(ws() + operator() + ws() + atom)
    )

@g.rule
def full_expression():
    """e.g. 215 * 74 = 15910"""
    return "'" + expression() + ws() + "=" + ws() + integer() + "'"

@g.rule
def root():
    """e.g. 215 * 74 = 15910"""
    return ws() + "[" + ws() + one_or_more(select([
        full_expression(),
        ws() + "," + ws()
    ])) + ws() + "]"


# ── Helpers ──────────────────────────────────────────────────────────


def image_to_data_url(path: str) -> str:
    """Encode a local image as a base64 data-URL."""
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode()
    suffix = Path(path).suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    return f"data:{mime};base64,{b64}"


# ── Main ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
    else:
        # Default: examples/img/test.png (relative to repo root or script dir)
        script_dir = Path(__file__).resolve().parent
        image_path = str(script_dir / "img" / "test.png")

    if not Path(image_path).exists():
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    # ── Set up GrammarLLM ────────────────────────────────────────────

    llm = GrammarLLM("http://localhost:8080/v1")

    grammar = g.to_gbnf()
    print("=== GBNF Grammar ===")
    print(grammar)
    print("====================\n")

    t1 = time.perf_counter()
    data_url = image_to_data_url(image_path)
    t_encode = time.perf_counter() - t1

    # ── Streaming request ────────────────────────────────────────────

    messages = [
        {
            "role": "system",
            "content": (
                "You are a calculator. Look at the image and solve "
                "all the arithmetics problem. Reply ONLY with the expressions "
                "and theirs result. Example of expected output format: "
                "['2 + 3 = 5', '215 * 74 = 15910']"
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
                {
                    "type": "text",
                    "text": "Solve the math problem shown in this image.",
                },
            ],
        },
    ]

    t2 = time.perf_counter()

    print("Streaming response:")
    for token, events in llm.stream(messages, grammar=g,only={"full_expression"}):
        sys.stdout.write(token)
        sys.stdout.flush()
        if events:
            for ev in events:
                doc = f"  ({ev.doc})" if ev.doc else ""
                print(f"\n  ✅ [{ev.rule}] matched: {ev.text}{doc}", end="")

    t_inference = time.perf_counter() - t2

    print(f"\n\nFull output:\n{llm.buffer}")

    print(f"\n=== Timings ===")
    print(f"Image encoding      : {t_encode*1000:7.1f} ms")
    print(f"LLM inference       : {t_inference:7.2f} s")
    print(f"Total               : {(t_encode + t_inference):7.2f} s")


if __name__ == "__main__":
    main()
