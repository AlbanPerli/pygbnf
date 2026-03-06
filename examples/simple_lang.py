#!/usr/bin/env python3
"""Example 4 — Simple programming language grammar.

Demonstrates keywords, identifiers, recursion, repetition, alternation,
and structured blocks.  The language supports:
- variable assignment (let x = expr;)
- if / else
- while loops
- function definitions (fn name(args) { body })
- return statements
- expressions with arithmetic and function calls
"""

import pygbnf as cfg
from pygbnf import (
    CharacterClass,
    select,
    one_or_more,
    zero_or_more,
    optional,
)

g = cfg.Grammar()


# -- Whitespace & low-level tokens ------------------------------------------

@g.rule
def ws():
    return zero_or_more(select(" \t\n\r"))


@g.rule
def ws1():
    """At least one whitespace character."""
    return one_or_more(select(" \t\n\r"))


@g.rule
def ident():
    head = CharacterClass(pattern="a-zA-Z_")
    tail = zero_or_more(CharacterClass(pattern="a-zA-Z0-9_"))
    return head + tail


@g.rule
def number():
    return optional("-") + one_or_more(CharacterClass(pattern="0-9"))


@g.rule
def string_lit():
    return '"' + zero_or_more(select([
        CharacterClass(pattern='^"\\\\', negated=False),
        "\\" + CharacterClass(pattern='.'),
    ])) + '"'


# -- Expressions -------------------------------------------------------------

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
    return expression() + zero_or_more(cfg.group(ws() + "," + ws() + expression()))


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
    return atom() + zero_or_more(cfg.group(ws() + mul_op() + ws() + atom()))


@g.rule
def arith_expr():
    return term() + zero_or_more(cfg.group(ws() + add_op() + ws() + term()))


@g.rule
def expression():
    return arith_expr() + optional(cfg.group(ws() + cmp_op() + ws() + arith_expr()))


# -- Statements --------------------------------------------------------------

@g.rule
def block():
    return "{" + ws() + zero_or_more(cfg.group(statement() + ws())) + "}"


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
    return ident() + zero_or_more(cfg.group(ws() + "," + ws() + ident()))


@g.rule
def fn_def():
    return ("fn" + ws1() + ident()
            + ws() + "(" + ws() + optional(param_list()) + ws() + ")"
            + ws() + block())


@g.rule
def expr_stmt():
    return expression() + ws() + ";"


@g.rule
def statement():
    return select([
        let_stmt(),
        assign_stmt(),
        return_stmt(),
        if_stmt(),
        while_stmt(),
        fn_def(),
        expr_stmt(),
    ])


@g.rule
def program():
    return ws() + zero_or_more(cfg.group(statement() + ws()))


g.start("program")

print("=" * 60)
print("Simple Programming Language Grammar")
print("=" * 60)
g.pretty_print()

print()
print("Dependency graph:")
for rule, deps in sorted(g.dependency_graph().items()):
    if deps:
        print(f"  {rule} → {', '.join(sorted(deps))}")

print()
print("Left-recursion analysis:")
cycles = g.detect_left_recursion()
if not cycles:
    print("  No left recursion detected ✓")
