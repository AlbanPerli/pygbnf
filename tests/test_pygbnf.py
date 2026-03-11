#!/usr/bin/env python3
"""Test suite for pygbnf — validates AST construction, codegen, and optimizations."""

import sys
import warnings

import pygbnf as cfg
from pygbnf import (
    Alternative,
    CharacterClass,
    Grammar,
    Group,
    Literal,
    Node,
    Repeat,
    RuleReference,
    Sequence,
    TokenReference,
    group,
    one_or_more,
    optional,
    repeat,
    select,
    zero_or_more,
    token,
    token_id,
    not_token,
    not_token_id,
    WS,
    identifier,
    number,
    keyword,
    comma_list,
    between,
    string_literal,
    float_number,
)
from pygbnf.gbnf_codegen import _emit
from pygbnf.optimizations import optimize_rules

passed = 0
failed = 0


def check(label: str, got, expected):
    global passed, failed
    if got == expected:
        passed += 1
        print(f"  ✓ {label}")
    else:
        failed += 1
        print(f"  ✗ {label}")
        print(f"    expected: {expected!r}")
        print(f"    got:      {got!r}")


# ── Node construction ──────────────────────────────────────────────────

print("Node construction:")

check("Literal", Literal("hello").value, "hello")
check("CharacterClass", CharacterClass(pattern="0-9").pattern, "0-9")
check("Sequence via +", isinstance(Literal("a") + Literal("b"), Sequence), True)
check("Alternative via |", isinstance(Literal("a") | Literal("b"), Alternative), True)
check("str + Node", isinstance("x" + Literal("y"), Sequence), True)
check("Node + str", isinstance(Literal("x") + "y", Sequence), True)
check("str | Node", isinstance("x" | Literal("y"), Alternative), True)

seq = Literal("a") + Literal("b") + Literal("c")
check("Sequence flattening via +", len(seq.children), 3)

alt = Literal("a") | Literal("b") | Literal("c")
check("Alternative flattening via |", len(alt.alternatives), 3)

# ── Combinators ────────────────────────────────────────────────────────

print("\nCombinators:")

check("select(str) → CharacterClass", isinstance(select("abc"), CharacterClass), True)
check("select(list) → Alternative", isinstance(select(["a", "b"]), Alternative), True)
check("select([single]) → Literal", isinstance(select(["a"]), Literal), True)
check("one_or_more min", one_or_more("x").min, 1)
check("one_or_more max", one_or_more("x").max, None)
check("zero_or_more min", zero_or_more("x").min, 0)
check("optional max", optional("x").max, 1)
check("repeat(x,2,5)", (repeat("x", 2, 5).min, repeat("x", 2, 5).max), (2, 5))
check("group", isinstance(group("x"), Group), True)

# ── Token references ──────────────────────────────────────────────────

print("\nToken references:")

check("token", _emit(token("think")), "<think>")
check("token_id", _emit(token_id(1000)), "<[1000]>")
check("not_token", _emit(not_token("think")), "!<think>")
check("not_token_id", _emit(not_token_id(1001)), "!<[1001]>")

# ── Code generation ───────────────────────────────────────────────────

print("\nCode generation:")

check("Literal emit", _emit(Literal("hello")), '"hello"')
check("Literal escape quotes", _emit(Literal('say "hi"')), '"say \\"hi\\""')
check("Literal escape newline", _emit(Literal("a\nb")), '"a\\nb"')
check("CharacterClass", _emit(CharacterClass(pattern="0-9")), "[0-9]")
check("CharacterClass negated", _emit(CharacterClass(pattern="abc", negated=True)), "[^abc]")
check("CharacterClass ^ in pattern", _emit(CharacterClass(pattern="^abc")), "[^abc]")
check("RuleReference", _emit(RuleReference(name="my-rule")), "my-rule")

seq_emit = _emit(Sequence(children=[Literal("a"), Literal("b")]))
check("Sequence emit", seq_emit, '"a" "b"')

alt_emit = _emit(Alternative(alternatives=[Literal("a"), Literal("b")]))
check("Alternative emit", alt_emit, '"a" | "b"')

rep_emit = _emit(Repeat(child=CharacterClass(pattern="0-9"), min=1, max=None))
check("Repeat + emit", rep_emit, "[0-9]+")

rep_star = _emit(Repeat(child=Literal("x"), min=0, max=None))
check("Repeat * emit", rep_star, '"x"*')

rep_opt = _emit(Repeat(child=Literal("x"), min=0, max=1))
check("Repeat ? emit", rep_opt, '"x"?')

rep_range = _emit(Repeat(child=Literal("x"), min=2, max=5))
check("Repeat {m,n} emit", rep_range, '"x"{2,5}')

rep_exact = _emit(Repeat(child=Literal("x"), min=3, max=3))
check("Repeat {n} emit", rep_exact, '"x"{3}')

rep_min = _emit(Repeat(child=Literal("x"), min=2, max=None))
check("Repeat {m,} emit", rep_min, '"x"{2,}')

grp = _emit(Group(child=Alternative(alternatives=[Literal("a"), Literal("b")])))
check("Group emit", grp, '("a" | "b")')

# Wrap alternatives in sequence context
alt_in_seq = _emit(
    Sequence(children=[
        Literal("x"),
        Alternative(alternatives=[Literal("a"), Literal("b")]),
    ])
)
check("Alt wrapped in seq", alt_in_seq, '"x" ("a" | "b")')

# ── Grammar container ────────────────────────────────────────────────

print("\nGrammar container:")

g = Grammar()


@g.rule
def digit():
    return select("0123456789")


@g.rule
def num():
    return optional("-") + one_or_more(digit())


g.start("num")
gbnf = g.to_gbnf()
check("Grammar basic output", "root ::= num" in gbnf, True)
check("Rule present: digit", "digit ::=" in gbnf, True)
check("Rule present: num", "num ::=" in gbnf, True)
check("Forward ref works", "digit+" in gbnf, True)

# ── Dependency graph ────────────────────────────────────────────────

print("\nDependency graph:")

deps = g.dependency_graph()
check("num depends on digit", "digit" in deps.get("num", set()), True)
check("digit has no deps", len(deps.get("digit", set())), 0)

# ── Left recursion detection ────────────────────────────────────────

print("\nLeft recursion detection:")

g2 = Grammar()


@g2.rule
def expr():
    return select([
        expr() + "+" + expr(),
        "x",
    ])


g2.start("expr")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    cycles = g2.detect_left_recursion()
    check("Detects left recursion", len(cycles) > 0, True)
    check("Warning emitted", len(w) > 0, True)

# ── Optimizations ────────────────────────────────────────────────────

print("\nOptimizations:")

# Literal collapsing
opt = optimize_rules({"r": Sequence(children=[Literal("a"), Literal("b"), Literal("c")])})
check("Literal collapse", opt["r"], Literal("abc"))

# Repetition merging: x? x? x? → x{0,3}
from pygbnf.nodes import Optional_

three_opt = Sequence(children=[
    Repeat(child=Literal("x"), min=0, max=1),
    Repeat(child=Literal("x"), min=0, max=1),
    Repeat(child=Literal("x"), min=0, max=1),
    Repeat(child=Literal("x"), min=0, max=1),
])
opt2 = optimize_rules({"r": three_opt})
node = opt2["r"]
check("Repetition merge x?x?x?x? → x{0,4}", isinstance(node, Repeat) and node.min == 0 and node.max == 4, True)

# Singleton collapse
opt3 = optimize_rules({"r": Alternative(alternatives=[Literal("x")])})
check("Singleton alt collapse", opt3["r"], Literal("x"))

opt4 = optimize_rules({"r": Sequence(children=[Literal("x")])})
check("Singleton seq collapse", opt4["r"], Literal("x"))

# Redundant group removal
opt5 = optimize_rules({"r": Group(child=Literal("x"))})
check("Redundant group removal", opt5["r"], Literal("x"))

# ── Helpers ──────────────────────────────────────────────────────────

print("\nHelpers:")

check("WS emit", _emit(WS()), '[ \\t\\n]*')
check("WS required", _emit(WS(required=True)), '[ \\t\\n]+')
check("keyword", _emit(keyword("return")), '"return"')
check("identifier emit", "a-zA-Z_" in _emit(identifier()), True)
check("number emit", "0-9" in _emit(number()), True)
check("float_number emit", '"."' in _emit(float_number()), True)
check("string_literal emit", '[^"\\\\]' in _emit(string_literal()), True)

cl = comma_list(RuleReference(name="item"))
cl_str = _emit(cl)
check("comma_list", "," in cl_str, True)

bt = between("(", RuleReference(name="expr"), ")")
bt_str = _emit(bt)
check("between", bt_str, '"(" expr ")"')

# ── Rule name conversion ─────────────────────────────────────────────

print("\nRule name conversion:")

g3 = Grammar()


@g3.rule
def my_rule():
    return Literal("x")


g3.start("my_rule")
gbnf3 = g3.to_gbnf()
check("snake_case → dashed", "my-rule ::=" in gbnf3, True)
check("start dashed", "root ::= my-rule" in gbnf3, True)

# ── Schema — optional dataclass fields ───────────────────────────────

print("\nSchema (optional fields):")

from dataclasses import dataclass, field as dc_field
from typing import Optional, List
from pygbnf.schema import grammar_from_type, SchemaCompiler

@dataclass
class _AllRequired:
    name: str
    age: int

@dataclass
class _WithDefaults:
    name: str
    age: int
    active: bool = True
    nickname: Optional[str] = None

@dataclass
class _AllDefaults:
    x: int = 0
    y: int = 0

# All-required: no optional groups in the dataclass rule itself
g_req = grammar_from_type(_AllRequired)
gbnf_req = g_req.to_gbnf()
# Find the dataclass rule line and check it has no )?
dc_line_req = [l for l in gbnf_req.splitlines() if "AllRequired" in l and "::=" in l][0]
check("all required — no ()?", ")?" not in dc_line_req, True)
check("all required — has name", '\\"name\\"' in gbnf_req, True)
check("all required — has age", '\\"age\\"' in gbnf_req, True)

# With defaults: optional trailing fields
g_def = grammar_from_type(_WithDefaults)
gbnf_def = g_def.to_gbnf()
check("with defaults — has ()?", ")?" in gbnf_def, True)
check("with defaults — name required", '\\"name\\"' in gbnf_def, True)
check("with defaults — active optional", '\\"active\\"' in gbnf_def, True)
check("with defaults — nickname optional", '\\"nickname\\"' in gbnf_def, True)
# nickname is nested inside active's optional
check("with defaults — nested optionals", gbnf_def.count(")?") >= 2, True)

# All defaults: entire body is optional
g_all = grammar_from_type(_AllDefaults)
gbnf_all = g_all.to_gbnf()
check("all defaults — has ()?", ")?" in gbnf_all, True)
# With no required fields, the JSON can be just {}
# Check the structure contains the field names
check("all defaults — has x", '\\"x\\"' in gbnf_all, True)
check("all defaults — has y", '\\"y\\"' in gbnf_all, True)

# ── Summary ──────────────────────────────────────────────────────────

print(f"\n{'=' * 40}")
print(f"Results: {passed} passed, {failed} failed")
if failed:
    sys.exit(1)
else:
    print("All tests passed ✓")
