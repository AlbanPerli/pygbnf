#!/usr/bin/env python3
"""Demo — Have an LLM write code constrained by the simple_lang grammar.

Uses pygbnf to generate the GBNF grammar of a mini programming language,
then asks the LLM to produce valid code in that language.
"""

import sys
import time
from openai import OpenAI
import pygbnf as cfg
from pygbnf import (
    CharacterClass,
    select,
    one_or_more,
    zero_or_more,
    optional,
    repeat,
    GrammarLLM,
)

# =====================================================================
# Grammar definition
# =====================================================================

g = cfg.Grammar()

# -- Whitespace & tokens ---------------------------------------------------

@g.rule
def sp():
    """Optional horizontal spaces (no newline)."""
    return repeat(select(" \t"), 0, 8)

@g.rule
def nl():
    """A single newline."""
    return select(["\r\n", "\n"])

@g.rule
def ws():
    """Bounded whitespace: spaces + newlines, max 20 chars."""
    return repeat(select(" \t\n\r"), 0, 20)

@g.rule
def ws1():
    """At least one whitespace char, bounded."""
    return select(" \t\n\r") + repeat(select(" \t\n\r"), 0, 10)

@g.rule
def ident():
    return CharacterClass(pattern="a-zA-Z_") + repeat(CharacterClass(pattern="a-zA-Z0-9_"), 0, 30)

@g.rule
def number():
    return one_or_more(CharacterClass(pattern="0-9"))

@g.rule
def string_lit():
    return '"' + repeat(CharacterClass(pattern='^"\\\\'), 0, 60) + '"'

# -- Expressions ------------------------------------------------------------

@g.rule
def atom():
    return select([
        number(),
        string_lit(),
        func_call(),
        ident(),
        "(" + ws() + expression() + ws() + ")",
    ])

@g.rule
def func_call():
    return ident() + "(" + ws() + optional(arg_list()) + ws() + ")"

@g.rule
def arg_list():
    return expression() + repeat(cfg.group(ws() + "," + ws() + expression()), 0, 8)

@g.rule
def mul_op():
    return select(["*", "/", "%"])

@g.rule
def add_op():
    return select(["+", "-"])

@g.rule
def cmp_op():
    return select(["==", "!=", "<=", ">=", "<", ">"])

@g.rule
def term():
    return atom() + repeat(cfg.group(ws() + mul_op() + ws() + atom()), 0, 4)

@g.rule
def arith_expr():
    return term() + repeat(cfg.group(ws() + add_op() + ws() + term()), 0, 4)

@g.rule
def expression():
    return arith_expr() + optional(cfg.group(ws() + cmp_op() + ws() + arith_expr()))

# -- Statements -------------------------------------------------------------

@g.rule
def block():
    return "{" + ws() + repeat(cfg.group(statement() + ws()), 0, 30) + "}"

@g.rule
def let_stmt():
    return "let" + ws1() + ident() + ws() + "=" + ws() + expression() + ws() + ";"

@g.rule
def assign_stmt():
    return ident() + ws() + "=" + ws() + expression() + ws() + ";"

@g.rule
def return_stmt():
    return "return" + ws1() + expression() + ws() + ";"

@g.rule
def if_stmt():
    return ("if" + ws() + "(" + ws() + expression() + ws() + ")"
            + ws() + block()
            + optional(cfg.group(ws() + "else" + ws() + select([block(), if_stmt()]))))

@g.rule
def while_stmt():
    return "while" + ws() + "(" + ws() + expression() + ws() + ")" + ws() + block()

@g.rule
def param_list():
    return ident() + repeat(cfg.group(ws() + "," + ws() + ident()), 0, 8)

@g.rule
def fn_def():
    return ("fn" + ws1() + ident()
            + ws() + "(" + ws() + optional(param_list()) + ws() + ")"
            + ws() + block())

@g.rule
def print_stmt():
    return "print" + ws() + "(" + ws() + arg_list() + ws() + ")" + ws() + ";"

@g.rule
def expr_stmt():
    return func_call() + ws() + ";"

@g.rule
def done_stmt():
    return  ws()+"done;\n"+ws()

@g.rule
def statement():
    return select([
        let_stmt(),
        assign_stmt(),
        return_stmt(),
        print_stmt(),
        if_stmt(),
        while_stmt(),
        fn_def(),
        expr_stmt(),
        done_stmt(),
    ])

@g.rule
def program():
    return ws() + fn_def() + ws()

g.start("program")

# =====================================================================
# Call the LLM with grammar constraints
# =====================================================================

LLAMA_BASE_URL = "http://localhost:8080/v1"

SYSTEM_PROMPT = """\
You are an expert programmer in a small custom language with the following syntax:

  fn name(args) { body }     — function definition
  let x = expr;              — variable declaration
  x = expr;                  — assignment
  if (cond) { } else { }     — conditional
  while (cond) { }           — loop
  return expr;               — return value
  print(expr, ...);          — print output
  name(args)                 — function call
  operators: + - * / % == != < > <= >=

Example of a valid program:
fn factorial(n) {
  if (n <= 1) {
    return 1;
  } else {
    return n * factorial(n - 1);
  }
}

fn main() {
  let result = factorial(5);
  print(result);
  let i = 0;
  while (i < result) {
    print(i);
    i = i + 1;
  }
  return i;
}

You always reply with ONLY the program code, nothing else."""

USER_PROMPT = """\
Write a program that defines a main() function which prints the numbers from 1 to 10."""

llm = GrammarLLM(LLAMA_BASE_URL)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT},
]

for token, events in llm.stream(
    messages,
    grammar=g,
    temperature=0,
    n_predict=96,
    stop=["\n\n\n"],
):
    sys.stdout.write(token)
    sys.stdout.flush()
    if events:
        for ev in events:
            text = ev.text if len(ev.text) <= 60 else ev.text[:57] + "..."
            doc = f"  ({ev.doc.strip()})" if ev.doc else ""
            print(f"\n  ✅ [{ev.rule}] {text}{doc}", end="")
