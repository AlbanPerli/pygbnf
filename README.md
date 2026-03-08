# pygbnf

A composable Python DSL for building **GBNF grammars** compatible with [llama.cpp](https://github.com/ggml-org/llama.cpp).

1) Define [context-free grammars](https://en.wikipedia.org/wiki/Context-free_grammar) using expressive Python functions, 

2) Compile them into valid `G`[BNF](https://en.wikipedia.org/wiki/Backus–Naur_form) strings for constrained LLM generation.

## Quick Start

```python
import pygbnf as cfg
from pygbnf import select, one_or_more, zero_or_more

g = cfg.Grammar()

@g.rule
def number():
    n = one_or_more(select("0123456789"))
    return select(['-' + n, n])

@g.rule
def operator():
    return select(['+', '*', '**', '/', '-'])

@g.rule
def expression():
    return select([
        number(),
        expression() + zero_or_more(" ") + operator()
            + zero_or_more(" ") + expression(),
        "(" + expression() + ")"
    ])

g.start("expression")
print(g.to_gbnf())
```

Output:

```
root ::= expression

number ::= "-" [0123456789]+ | [0123456789]+
operator ::= "+" | "*" | "**" | "/" | "-"
expression ::=
    number
  | expression " "* operator " "* expression
  | "(" expression ")"
```

## Installation

```bash
pip install pygbnf
```

Or install from source:

```bash
git clone https://github.com/al/pygbnf.git
cd pygbnf
pip install -e .
```

## LLM Usage

pygbnf includes `GrammarLLM`, a thin wrapper around any OpenAI-compatible endpoint (llama.cpp, vLLM, Ollama…) that injects the GBNF grammar automatically.

### Basic constrained generation

```python
from pygbnf import Grammar, GrammarLLM, select

g = Grammar()

@g.rule
def answer():
    return select(["yes", "no", "maybe"])

g.start("answer")

llm = GrammarLLM("http://localhost:8080/v1")

for token, _ in llm.stream(
    messages=[{"role": "user", "content": "Is the sky blue?"}],
    grammar=g,
):
    print(token, end="", flush=True)
print()
```

The grammar constrains the LLM output — it can only produce `yes`, `no`, or `maybe`.

### Streaming with rule matching

Enable `match=True` (or pass `only`/`exclude`) to get real-time `RuleEvent`s as the LLM generates tokens:

```python
from pygbnf import Grammar, GrammarLLM, select, one_or_more

g = Grammar()

@g.rule
def name():
    """A person's name."""
    return one_or_more(select("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "))

@g.rule
def greeting():
    """A greeting message."""
    return select(["hello", "hi", "hey"]) + " " + name()

g.start("greeting")

llm = GrammarLLM("http://localhost:8080/v1")

for token, events in llm.stream(
    messages=[{"role": "user", "content": "Greet Alice."}],
    grammar=g,
    match=True,
):
    print(token, end="", flush=True)
    if events:
        for ev in events:
            print(f"\n  ← [{ev.rule}] {ev.text!r} (doc: {ev.doc})")
print()
```

Each `RuleEvent` carries:
- `rule` — the matched rule name
- `text` — the matched text
- `fn` — the original Python function
- `doc` — the function's docstring

### Non-streaming completion

```python
text, events = llm.complete(
    messages=[{"role": "user", "content": "Is the sky blue?"}],
    grammar=g,
    match=True,
)
print(text)
for ev in events:
    print(f"  [{ev.rule}] {ev.text!r}")
```

### Schema-based grammar with LLM

Combine `grammar_from_type` with `GrammarLLM` to constrain output to a JSON schema:

```python
from dataclasses import dataclass
from pygbnf import grammar_from_type, GrammarLLM

@dataclass
class City:
    name: str
    country: str
    population: int

g = grammar_from_type(City)
llm = GrammarLLM("http://localhost:8080/v1")

text, _ = llm.complete(
    messages=[{"role": "user", "content": "Describe Tokyo in JSON."}],
    grammar=g,
)
print(text)
# → {"name": "Tokyo", "country": "Japan", "population": 13960000}
```

### Tool calling with Toolkit

`Toolkit` is a decorator-based tool registry. Register functions with `@toolkit.tool`, then pass the toolkit to `llm.stream()` or `llm.complete()` — the grammar and system prompt are injected automatically.

```python
import enum
from pygbnf import GrammarLLM, Toolkit

toolkit = Toolkit()

class Units(enum.Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"

@toolkit.tool
def get_weather(city: str, units: Units = Units.CELSIUS) -> str:
    """Get current weather for a city."""
    return f"22° {units.value} in {city}"

@toolkit.tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web."""
    return f"Found {max_results} results for {query!r}"

llm = GrammarLLM("http://localhost:8080/v1")

# Stream with toolkit — grammar + system prompt auto-injected
result = ""
for token, _ in llm.stream(
    messages=[{"role": "user", "content": "Weather in Tokyo?"}],
    toolkit=toolkit,
):
    print(token, end="", flush=True)
    result += token

# Dispatch the JSON result to the matching function
output = toolkit.dispatch(result)
print(output)  # → "22° celsius in Tokyo"
```

The toolkit:
- **Builds a GBNF grammar** constraining the LLM to produce `{"function": "...", "arguments": {...}}` with only registered tool names and typed arguments
- **Generates a system prompt** listing available tools with signatures and docstrings
- **Dispatches** the parsed JSON to the right function, converting enum strings back to Python `Enum` instances automatically

You can also use `llm.tool_call()` as a one-liner that streams + dispatches:

```python
output = llm.tool_call(toolkit, "Weather in Tokyo?")
print(output)  # → "22° celsius in Tokyo"
```

> **Note:** `GrammarLLM` requires the `openai` package: `pip install openai`.
> The LLM server must support the `grammar` field in its API (llama.cpp does natively).

## Architecture

### AST Nodes

Every grammar construct is a frozen dataclass node. Nodes compose via `+` (sequence) and `|` (alternative):

| Node | Description | GBNF |
|------|------------|------|
| `Literal` | Double-quoted string | `"hello"` |
| `CharacterClass` | Character class | `[0-9]` |
| `Sequence` | Ordered concatenation | `a b c` |
| `Alternative` | Choice between options | `a \| b \| c` |
| `Repeat` | Quantified repetition | `x+`, `x*`, `x?`, `x{2,5}` |
| `RuleReference` | Reference to named rule | `expression` |
| `TokenReference` | Token-level constraint | `<think>`, `<[1000]>` |
| `Group` | Parenthesised group | `(a b)` |
| `Optional_` | Optional element | `x?` |

### DSL Combinators

```python
from pygbnf import select, one_or_more, zero_or_more, optional, repeat, group

# Character class from string
select("0123456789")          # → [0123456789]

# Alternative from list
select(["+", "-", "*"])       # → "+" | "-" | "*"

# Repetition
one_or_more(x)                # → x+
zero_or_more(x)               # → x*
optional(x)                   # → x?
repeat(x, 2, 5)              # → x{2,5}

# Grouping
group(a + b)                  # → (a b)

# Operators
a + b                         # → a b   (sequence)
a | b                         # → a | b (alternative)
```

### Rule Definition

Rules are defined with the `@g.rule` decorator. Calling a rule function inside another rule creates a **rule reference** (not an inline expansion):

```python
g = cfg.Grammar()

@g.rule
def digit():
    return select("0123456789")

@g.rule
def number():
    return one_or_more(digit())  # → digit+  (reference, not inlined)
```

Forward references work naturally — rules can reference rules defined later.

### Token Constraints

llama.cpp supports token-level matching:

```python
from pygbnf import token, token_id, not_token, not_token_id

token("think")        # → <think>
token_id(1000)        # → <[1000]>
not_token("think")    # → !<think>
not_token_id(1001)    # → !<[1001]>
```

### Grammar Helpers

Common patterns prebuilt:

```python
from pygbnf import (
    WS, ws, ws_required,           # whitespace
    keyword, identifier, number,    # basic tokens
    float_number, string_literal,   # complex tokens
    comma_list, between,           # structural patterns
    separated_by, spaced_comma_list,
)

comma_list(identifier())   # → ident ("," " "* ident)*
between("(", expr, ")")    # → "(" expr ")"
```

## Recursion Analysis

Detect left recursion in your grammar:

```python
cycles = g.detect_left_recursion()
# Warns: "Left recursion detected: expression -> expression"
# Suggests: rewrite as base (op base)*
```

## Examples

See the `examples/` directory:

| File | Description |
|------|------------|
| `quickstart.py` | The quick-start example from this README |
| `arithmetic.py` | Arithmetic expressions with operator precedence |
| `csv_grammar.py` | CSV file format |
| `json_grammar.py` | Full JSON grammar |
| `simple_lang.py` | A small programming language |
| `token_demo.py` | Token-level constraints |
| `demo_llm.py` | LLM constrained generation (requires llama-server) |
| `demo_schema.py` | Schema → grammar examples |
| `demo_hybrid.py` | DSL + Python types mixed |
| `demo_simple_lang.py` | Mini-language generation with LLM |
| `demo_vision.py` | Vision + grammar: solve math from an image |

Run any example:

```bash
python examples/arithmetic.py
```

## Schema Generation

Auto-generate grammars from Python types and dataclasses:

```python
from dataclasses import dataclass
from pygbnf import grammar_from_type

@dataclass
class Movie:
    title: str
    year: int
    rating: float

g = grammar_from_type(Movie)
print(g.to_gbnf())
```

Also supports function signatures:

```python
from pygbnf import grammar_from_args

def search(query: str, limit: int = 10):
    ...

g = grammar_from_args(search)
print(g.to_gbnf())
```

## Requirements

- Python 3.11+
- No external dependencies

## Acknowledgements

- [guidance-ai](https://github.com/guidance-ai/guidance) — pygbnf's composable API is inspired by their approach to constrained generation
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — for the GBNF format and the underlying inference engine
