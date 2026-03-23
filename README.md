# pygbnf

A composable Python DSL for building **GBNF grammars** compatible with [llama.cpp](https://github.com/ggml-org/llama.cpp).

1) Define [context-free grammars](https://en.wikipedia.org/wiki/Context-free_grammar) using expressive Python functions, 

2) Compile them into valid `G`[BNF](https://en.wikipedia.org/wiki/Backus–Naur_form) strings for constrained LLM generation.

3) Real-time rule matching during inference.

pygbnf supports both:
- classic node-by-node grammar composition with `select()`, `repeat()`, `group()`, etc.
- template-first grammar authoring with `T()`, which lets you write constrained text formats as natural Python f-strings

## Installation

```bash
pip install pygbnf          # core DSL only
pip install pygbnf[llm]     # + openai (for GrammarLLM)
pip install pygbnf[all]     # everything
```

For grammar visualization (DOT / SVG export), install [Graphviz](https://graphviz.org/):

```bash
brew install graphviz   # macOS
apt install graphviz    # Debian / Ubuntu
```

## Quick Start

Start llama-server with your favorite GGUF model.

```cli
$ llama-server -m LFM2-8B-A1B-Q4_K_M.gguf
```

Build grammar and constraint the model.

```python
from pygbnf import Grammar, GrammarLLM, select

g = Grammar()

@g.rule
def answer():
    return select(["yes", "no", "maybe"])

g.start("answer")

llm = GrammarLLM("http://localhost:8080/v1")

text, _ = llm.complete(
    messages=[{"role": "user", "content": "Is the sky blue?"}],
    grammar=g
)

print(text)
```

The grammar constrains the LLM output — it can only produce `yes`, `no`, or `maybe`.

## Template-First Grammars with `T()`

`T()` is one of the most expressive parts of the library. It lets you write
structured text grammars as readable templates instead of building everything
node-by-node.

```python
from pygbnf import T, line, identifier, number, Grammar

g = Grammar()

@g.rule
def person_card():
    return T(f"""Name: {identifier()}
Age: {number()}
Tags:
{line("- "):+}
""")

g.start("person_card")
print(g.to_gbnf())
```

This is especially useful for:
- prompt-shaped outputs
- markdown-like or bullet-list formats
- semi-structured text with repeated sections
- grammars that should stay readable in source form

`T()` works with normal grammar nodes embedded in f-strings, and supports
line-level quantifiers such as `{node:+}`, `{node:*}`, `{node:?}`, `{node:3}`,
and `{node:2,5}`.

## Guidance-Style GBNF

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

## LLM Usage

pygbnf includes `GrammarLLM`, a thin wrapper around any OpenAI-compatible endpoint (llama.cpp, vLLM, Ollama…) that injects the GBNF grammar automatically.

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
from pygbnf import T, line, select, one_or_more, zero_or_more, optional, repeat, group

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

# Template-first authoring
T(f"Name: {identifier()}\n")
line("- ")                    # → "- " [^\n]+

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
    T, line,
)

comma_list(identifier())   # → ident ("," " "* ident)*
between("(", expr, ")")    # → "(" expr ")"
line("- ")                 # → bullet-point line
T(f"Title: {identifier()}\n")
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
| `demo_schema.py` | Schema → grammar examples |
| `demo_enum_select.py` | Enum-based selection |
| `demo_template.py` | Template-first grammar authoring with `T()` |
| `demo_simple_lang.py` | Mini-language generation with LLM |
| `demo_vision.py` | Vision + grammar: solve math from an image |
| `demo_visualization.py` | Export grammar NFA as DOT / SVG |

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

## Visualization

Export any grammar as an NFA diagram in DOT or SVG format:

```python
import pygbnf as cfg
from pygbnf import select, one_or_more, optional
from pygbnf.visualization import write_grammar_svg

g = cfg.Grammar()

@g.rule
def number():
    return optional("-") + one_or_more(select("0123456789"))

@g.rule
def operator():
    return select(["+", "-", "*", "/"])

@g.rule
def expression():
    atom = select([number(), "(" + expression() + ")"])
    return atom + cfg.zero_or_more(cfg.group(" " + operator() + " " + expression()))

g.start("expression")

# Generates .dot + .svg (requires Graphviz)
write_grammar_svg(g, "arithmetic.svg")
```

When `rule_names` is omitted, only user-defined rules are included (auto-generated infrastructure rules like `ws`, `json-string`, etc. are filtered out).

## Requirements

- Python 3.8+
- **Optional:** `openai>=1.0` for `GrammarLLM` (`pip install pygbnf[llm]`)
- **Optional:** [Graphviz](https://graphviz.org/) CLI for SVG rendering

## Testing

Install the package in editable mode with `pytest`:

```bash
python -m pip install -e . pytest
```

Run the full test suite:

```bash
pytest -q
```

The suite is organized as native `pytest` modules:

- `tests/test_core.py` — core grammar, code generation, helpers, and edge cases
- `tests/test_pygbnf.py` — DSL composition, optimizations, and template builder coverage
- `tests/test_schema.py` — schema compilation and higher-level integrations
- `tests/test_matcher.py` — incremental matcher behavior

A minimal GitHub Actions workflow runs the suite on Python 3.11, 3.12, and 3.13.

## Acknowledgements

- [guidance-ai](https://github.com/guidance-ai/guidance) — pygbnf's composable API is inspired by their approach to constrained generation
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — for the GBNF format and the underlying inference engine
