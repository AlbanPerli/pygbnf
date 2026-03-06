# pygbnf

A composable Python DSL for building **GBNF grammars** compatible with [llama.cpp](https://github.com/ggml-org/llama.cpp).

Define context-free grammars using expressive Python functions, then compile them into valid GBNF strings for constrained LLM generation.

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

No external dependencies — pure Python 3.11+.

```bash
pip install pygbnf
```

Or install from source:

```bash
git clone https://github.com/al/pygbnf.git
cd pygbnf
pip install -e .
```

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
