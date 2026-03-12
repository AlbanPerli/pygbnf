#!/usr/bin/env python3
"""Suite de tests unitaires complète pour pygbnf.

Couvre :
- nodes.py        : construction AST, opérateurs +/|, coerce
- combinators.py  : select, one_or_more, zero_or_more, optional, repeat, group
- tokens.py       : token, token_id, not_token, not_token_id
- grammar.py      : @g.rule, rule_named, ref, start, to_gbnf, dependency_graph,
                    detect_left_recursion, from_type, from_function_return,
                    from_function_args, _to_rule_name, _normalise
- gbnf_codegen.py : _emit pour chaque type de nœud, _escape_literal,
                    _escape_char_class, compile_grammar, multi-line alternatives
- optimizations.py: 5 passes d'optimisation
- helpers.py      : WS, ws, keyword, identifier, number, float_number,
                    string_literal, comma_list, spaced_comma_list, between,
                    separated_by, any_char
- schema.py       : SchemaCompiler, grammar_from_type, grammar_from_function,
                    grammar_from_args, types optionnels, Literal, Enum,
                    dataclass, list, dict, Optional, Union, defaults

Usage :
    PYTHONPATH=. python tests/test_full.py
    PYTHONPATH=. python -m pytest tests/test_full.py -v
"""

import sys
import warnings
import enum
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Literal as TypingLiteral, Optional, Union

import pygbnf as cfg
from pygbnf import (
    Alternative,
    CharacterClass,
    Grammar,
    Group,
    Literal,
    Node,
    Optional_,
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
    ws,
    ws_required,
    identifier,
    number,
    keyword,
    comma_list,
    spaced_comma_list,
    between,
    separated_by,
    any_char,
    string_literal,
    float_number,
)
from pygbnf.gbnf_codegen import _emit, compile_grammar
from pygbnf.optimizations import optimize_rules
from pygbnf.nodes import _coerce
from pygbnf.schema import grammar_from_type, grammar_from_function, grammar_from_args, SchemaCompiler

# ══════════════════════════════════════════════════════════════════════
# Framework minimal
# ══════════════════════════════════════════════════════════════════════

passed = 0
failed = 0
_current_section = ""


def section(name: str):
    global _current_section
    _current_section = name
    print(f"\n{name}:")


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


def check_in(label: str, needle, haystack):
    """Vérifie que needle est dans haystack."""
    check(label, needle in haystack, True)


def check_not_in(label: str, needle, haystack):
    """Vérifie que needle n'est PAS dans haystack."""
    check(label, needle in haystack, False)


def check_raises(label: str, exc_type, fn):
    """Vérifie qu'une exception est levée."""
    global passed, failed
    try:
        fn()
        failed += 1
        print(f"  ✗ {label}")
        print(f"    expected {exc_type.__name__} but no exception raised")
    except exc_type:
        passed += 1
        print(f"  ✓ {label}")
    except Exception as e:
        failed += 1
        print(f"  ✗ {label}")
        print(f"    expected {exc_type.__name__} but got {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════
# 1. NODES — Construction AST
# ══════════════════════════════════════════════════════════════════════

section("Nodes — Construction de base")

check("Literal value", Literal("hello").value, "hello")
check("Literal vide", Literal().value, "")
check("CharacterClass pattern", CharacterClass(pattern="0-9").pattern, "0-9")
check("CharacterClass negated=False", CharacterClass(pattern="0-9").negated, False)
check("CharacterClass negated=True", CharacterClass(pattern="abc", negated=True).negated, True)
check("RuleReference name", RuleReference(name="my-rule").name, "my-rule")
check("TokenReference str", TokenReference(value="think").value, "think")
check("TokenReference int", TokenReference(value=42).value, 42)
check("TokenReference negated", TokenReference(value="x", negated=True).negated, True)

section("Nodes — Opérateurs + et |")

check("Node + Node → Sequence", isinstance(Literal("a") + Literal("b"), Sequence), True)
check("str + Node → Sequence", isinstance("x" + Literal("y"), Sequence), True)
check("Node + str → Sequence", isinstance(Literal("x") + "y", Sequence), True)
check("Node | Node → Alternative", isinstance(Literal("a") | Literal("b"), Alternative), True)
check("str | Node → Alternative", isinstance("x" | Literal("y"), Alternative), True)
check("Node | str → Alternative", isinstance(Literal("x") | "y", Alternative), True)

section("Nodes — Aplatissement automatique")

seq = Literal("a") + Literal("b") + Literal("c")
check("Sequence 3 enfants", len(seq.children), 3)

alt = Literal("a") | Literal("b") | Literal("c")
check("Alternative 3 branches", len(alt.alternatives), 3)

# Chaînage plus long
seq4 = Literal("a") + Literal("b") + Literal("c") + Literal("d")
check("Sequence 4 enfants", len(seq4.children), 4)

alt4 = Literal("a") | Literal("b") | Literal("c") | Literal("d")
check("Alternative 4 branches", len(alt4.alternatives), 4)

section("Nodes — _coerce")

check("coerce str → Literal", isinstance(_coerce("hello"), Literal), True)
check("coerce str valeur", _coerce("hi").value, "hi")
check("coerce Node → Node", _coerce(Literal("x")), Literal("x"))
check_raises("coerce int → TypeError", TypeError, lambda: _coerce(42))

section("Nodes — Frozen (immuable)")

check_raises("Literal frozen", AttributeError, lambda: setattr(Literal("x"), 'value', 'y'))
check_raises("Sequence frozen", AttributeError, lambda: setattr(Sequence(), 'children', []))

section("Nodes — Égalité structurelle")

check("Literal == Literal", Literal("x") == Literal("x"), True)
check("Literal != Literal", Literal("x") == Literal("y"), False)
check("CharClass == CharClass", CharacterClass(pattern="0-9") == CharacterClass(pattern="0-9"), True)
check("Repeat == Repeat",
      Repeat(child=Literal("x"), min=0, max=1) == Repeat(child=Literal("x"), min=0, max=1),
      True)

# ══════════════════════════════════════════════════════════════════════
# 2. COMBINATORS
# ══════════════════════════════════════════════════════════════════════

section("Combinators — select")

check("select(str) → CharacterClass", isinstance(select("abc"), CharacterClass), True)
check("select(str) pattern", select("abc").pattern, "abc")
check("select(list) → Alternative", isinstance(select(["a", "b"]), Alternative), True)
check("select([single]) → Literal", isinstance(select(["only"]), Literal), True)
check("select([single]) value", select(["only"]).value, "only")
check("select(list) len", len(select(["a", "b", "c"]).alternatives), 3)

# select mélange str et Node
mixed = select([Literal("x"), "y", CharacterClass(pattern="0-9")])
check("select mixed types", len(mixed.alternatives), 3)

section("Combinators — repeat/quantifiers")

check("one_or_more min", one_or_more("x").min, 1)
check("one_or_more max", one_or_more("x").max, None)
check("one_or_more child", one_or_more("x").child, Literal("x"))

check("zero_or_more min", zero_or_more("x").min, 0)
check("zero_or_more max", zero_or_more("x").max, None)

check("optional min", optional("x").min, 0)
check("optional max", optional("x").max, 1)

check("repeat(2,5) min", repeat("x", 2, 5).min, 2)
check("repeat(2,5) max", repeat("x", 2, 5).max, 5)
check("repeat(3,3) exact", repeat("x", 3, 3).min, 3)
check("repeat unbounded max", repeat("x", 2).max, None)

section("Combinators — group")

check("group → Group", isinstance(group("x"), Group), True)
check("group child", group("x").child, Literal("x"))

# ══════════════════════════════════════════════════════════════════════
# 3. TOKENS
# ══════════════════════════════════════════════════════════════════════

section("Tokens")

check("token type", isinstance(token("think"), TokenReference), True)
check("token value", token("think").value, "think")
check("token negated=False", token("think").negated, False)

check("token_id type", isinstance(token_id(1000), TokenReference), True)
check("token_id value", token_id(1000).value, 1000)

check("not_token negated=True", not_token("x").negated, True)
check("not_token_id negated=True", not_token_id(42).negated, True)

# ══════════════════════════════════════════════════════════════════════
# 4. CODEGEN — _emit
# ══════════════════════════════════════════════════════════════════════

section("Codegen — Literal")

check("emit Literal simple", _emit(Literal("hello")), '"hello"')
check("emit Literal vide", _emit(Literal("")), '""')
check("emit Literal guillemets", _emit(Literal('say "hi"')), '"say \\"hi\\""')
check("emit Literal newline", _emit(Literal("a\nb")), '"a\\nb"')
check("emit Literal tab", _emit(Literal("a\tb")), '"a\\tb"')
check("emit Literal backslash", _emit(Literal("a\\b")), '"a\\\\b"')
check("emit Literal CR", _emit(Literal("a\rb")), '"a\\rb"')

section("Codegen — CharacterClass")

check("emit CharClass simple", _emit(CharacterClass(pattern="0-9")), "[0-9]")
check("emit CharClass negated attr", _emit(CharacterClass(pattern="abc", negated=True)), "[^abc]")
check("emit CharClass ^ in pattern", _emit(CharacterClass(pattern="^abc")), "[^abc]")
check("emit CharClass a-zA-Z", _emit(CharacterClass(pattern="a-zA-Z_")), "[a-zA-Z_]")
check("emit CharClass with backslash", _emit(CharacterClass(pattern="^\"\\\\", negated=False)), '[^"\\\\]')

section("Codegen — Sequence, Alternative, Group")

check("emit Sequence", _emit(Sequence(children=[Literal("a"), Literal("b")])), '"a" "b"')
check("emit Alt", _emit(Alternative(alternatives=[Literal("a"), Literal("b")])), '"a" | "b"')
check("emit Group", _emit(Group(child=Alternative(alternatives=[Literal("a"), Literal("b")]))), '("a" | "b")')

# Alt imbriqué dans Sequence → wrapping automatique
alt_in_seq = _emit(Sequence(children=[
    Literal("x"),
    Alternative(alternatives=[Literal("a"), Literal("b")]),
]))
check("emit Alt dans Seq → parenthèses", alt_in_seq, '"x" ("a" | "b")')

section("Codegen — Repeat/quantifiers")

check("emit Repeat +", _emit(Repeat(child=CharacterClass(pattern="0-9"), min=1, max=None)), "[0-9]+")
check("emit Repeat *", _emit(Repeat(child=Literal("x"), min=0, max=None)), '"x"*')
check("emit Repeat ?", _emit(Repeat(child=Literal("x"), min=0, max=1)), '"x"?')
check("emit Repeat {2,5}", _emit(Repeat(child=Literal("x"), min=2, max=5)), '"x"{2,5}')
check("emit Repeat {3}", _emit(Repeat(child=Literal("x"), min=3, max=3)), '"x"{3}')
check("emit Repeat {2,}", _emit(Repeat(child=Literal("x"), min=2, max=None)), '"x"{2,}')
check("emit Repeat {0,10}", _emit(Repeat(child=Literal("x"), min=0, max=10)), '"x"{0,10}')

# Repeat sur Sequence → parenthèses
check("emit Repeat Seq → wrap",
      _emit(Repeat(child=Sequence(children=[Literal("a"), Literal("b")]), min=1, max=None)),
      '("a" "b")+')

# Repeat sur Alternative → parenthèses
check("emit Repeat Alt → wrap",
      _emit(Repeat(child=Alternative(alternatives=[Literal("a"), Literal("b")]), min=0, max=None)),
      '("a" | "b")*')

section("Codegen — RuleReference et TokenReference")

check("emit RuleRef", _emit(RuleReference(name="my-rule")), "my-rule")
check("emit token", _emit(token("think")), "<think>")
check("emit token_id", _emit(token_id(1000)), "<[1000]>")
check("emit not_token", _emit(not_token("x")), "!<x>")
check("emit not_token_id", _emit(not_token_id(42)), "!<[42]>")

section("Codegen — Optional_")

check("emit Optional_", _emit(Optional_(child=Literal("x"))), '"x"?')

section("Codegen — Multi-line alternatives")

g_ml = Grammar()
@g_ml.rule
def multi():
    return select(["alpha", "beta", "gamma", "delta"])
g_ml.start("multi")
gbnf_ml = g_ml.to_gbnf()
# Doit avoir des retours à la ligne avec " |" en fin de ligne
check_in("multi-line trailing |", " |\n", gbnf_ml)

section("Codegen — Dash dans char class")

# Le fameux bug : [-] doit être à la fin, pas \-
cc_dash = CharacterClass(pattern="+-")
emitted_dash = _emit(cc_dash)
check_not_in("pas de \\-", "\\-", emitted_dash)
# Le dash doit être présent quelque part dans le résultat
check_in("dash présent", "-", emitted_dash)

# ══════════════════════════════════════════════════════════════════════
# 5. GRAMMAR CONTAINER
# ══════════════════════════════════════════════════════════════════════

section("Grammar — Règles basiques")

g1 = Grammar()

@g1.rule
def digit():
    return select("0123456789")

@g1.rule
def num():
    return optional("-") + one_or_more(digit())

g1.start("num")
gbnf1 = g1.to_gbnf()

check_in("root ::= num", "root ::= num", gbnf1)
check_in("digit ::=", "digit ::=", gbnf1)
check_in("num ::=", "num ::=", gbnf1)
check_in("forward ref digit+", "digit+", gbnf1)

section("Grammar — rule_named")

g_named = Grammar()

@g_named.rule_named("custom-name")
def _impl():
    return Literal("hello")

g_named.start("custom-name")
gbnf_named = g_named.to_gbnf()
check_in("custom-name ::=", "custom-name ::=", gbnf_named)

section("Grammar — ref()")

g_ref = Grammar()

@g_ref.rule
def base():
    return Literal("x")

@g_ref.rule
def uses_ref():
    return g_ref.ref("base") + Literal("!")

g_ref.start("uses_ref")
gbnf_ref = g_ref.to_gbnf()
check_in("ref base", "base", gbnf_ref)

section("Grammar — start root direct")

g_root = Grammar()

@g_root.rule
def root():
    return Literal("x")

g_root.start("root")
gbnf_root = g_root.to_gbnf()
# Quand start == "root", pas de "root ::= root"
check_not_in("pas de root alias", "root ::= root", gbnf_root)
check_in("root ::= \"x\"", 'root ::= "x"', gbnf_root)

section("Grammar — snake_case → dashed")

g_sc = Grammar()

@g_sc.rule
def my_fancy_rule():
    return Literal("x")

g_sc.start("my_fancy_rule")
gbnf_sc = g_sc.to_gbnf()
check_in("my-fancy-rule ::=", "my-fancy-rule ::=", gbnf_sc)
check_in("root ::= my-fancy-rule", "root ::= my-fancy-rule", gbnf_sc)

section("Grammar — dependency_graph")

deps = g1.dependency_graph()
check_in("num dépend de digit", "digit", deps.get("num", set()))
check("digit sans deps", len(deps.get("digit", set())), 0)

section("Grammar — detect_left_recursion")

g_lr = Grammar()

@g_lr.rule
def expr_lr():
    return select([
        expr_lr() + "+" + expr_lr(),
        "x",
    ])

g_lr.start("expr_lr")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    cycles = g_lr.detect_left_recursion()
    check("détecte récursion gauche", len(cycles) > 0, True)
    check("warning émis", len(w) > 0, True)

section("Grammar — pas de récursion gauche si correcte")

g_no_lr = Grammar()

@g_no_lr.rule
def term_a():
    return Literal("x")

@g_no_lr.rule
def expr_no_lr():
    return term_a() + optional(group("+" + term_a()))

g_no_lr.start("expr_no_lr")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    cycles_no = g_no_lr.detect_left_recursion()
    check("pas de récursion gauche", len(cycles_no), 0)
    check("pas de warning", len(w), 0)

section("Grammar — start rule manquante")

g_missing = Grammar()

@g_missing.rule
def some_rule():
    return Literal("x")

g_missing._start = "nonexistent"
check_raises("start manquante → ValueError", ValueError, lambda: g_missing.to_gbnf())

section("Grammar — optimize=False")

g_noopt = Grammar()

@g_noopt.rule
def lit_chain():
    return Literal("a") + Literal("b") + Literal("c")

g_noopt.start("lit_chain")
gbnf_noopt = g_noopt.to_gbnf(optimize=False)
# Sans optimisation, les litéraux ne sont PAS fusionnés
check_in("pas fusionné sans opt", '"a" "b" "c"', gbnf_noopt)

gbnf_opt = g_noopt.to_gbnf(optimize=True)
# Avec optimisation, fusionnés
check_in("fusionné avec opt", '"abc"', gbnf_opt)

section("Grammar — pretty_print")

import io, contextlib
g_pp = Grammar()

@g_pp.rule
def pp_rule():
    return Literal("test")

g_pp.start("pp_rule")
f = io.StringIO()
with contextlib.redirect_stdout(f):
    g_pp.pretty_print()
check_in("pretty_print contient règle", "pp-rule ::=", f.getvalue())

# ══════════════════════════════════════════════════════════════════════
# 6. OPTIMIZATIONS
# ══════════════════════════════════════════════════════════════════════

section("Optimizations — Literal collapse")

opt1 = optimize_rules({"r": Sequence(children=[Literal("a"), Literal("b"), Literal("c")])})
check("literal collapse abc", opt1["r"], Literal("abc"))

section("Optimizations — Sequence flattening")

nested_seq = Sequence(children=[
    Literal("a"),
    Sequence(children=[Literal("b"), Literal("c")]),
])
opt_flat = optimize_rules({"r": nested_seq})
# Après flatten + literal collapse → "abc"
check("sequence flatten → abc", opt_flat["r"], Literal("abc"))

section("Optimizations — Singleton collapse")

opt_single_alt = optimize_rules({"r": Alternative(alternatives=[Literal("x")])})
check("singleton alt collapse", opt_single_alt["r"], Literal("x"))

opt_single_seq = optimize_rules({"r": Sequence(children=[Literal("x")])})
check("singleton seq collapse", opt_single_seq["r"], Literal("x"))

section("Optimizations — Redundant group removal")

opt_grp = optimize_rules({"r": Group(child=Literal("x"))})
check("group(literal) → literal", opt_grp["r"], Literal("x"))

opt_grp_cc = optimize_rules({"r": Group(child=CharacterClass(pattern="0-9"))})
check("group(charclass) → charclass", opt_grp_cc["r"], CharacterClass(pattern="0-9"))

opt_grp_ref = optimize_rules({"r": Group(child=RuleReference(name="x"))})
check("group(ref) → ref", opt_grp_ref["r"], RuleReference(name="x"))

# Group sur Alt ne doit PAS être supprimé (significatif)
opt_grp_alt = optimize_rules({"r": Group(child=Alternative(alternatives=[Literal("a"), Literal("b")]))})
check("group(alt) conservé", isinstance(opt_grp_alt["r"], Group), True)

section("Optimizations — Repetition merge")

three_opt = Sequence(children=[
    Repeat(child=Literal("x"), min=0, max=1),
    Repeat(child=Literal("x"), min=0, max=1),
    Repeat(child=Literal("x"), min=0, max=1),
])
opt_rep = optimize_rules({"r": three_opt})
node_rep = opt_rep["r"]
check("x? x? x? → x{0,3}", isinstance(node_rep, Repeat), True)
check("x? x? x? min=0", node_rep.min, 0)
check("x? x? x? max=3", node_rep.max, 3)

# Merge x+ x+ → x{2,}
two_plus = Sequence(children=[
    Repeat(child=Literal("y"), min=1, max=None),
    Repeat(child=Literal("y"), min=1, max=None),
])
opt_plus = optimize_rules({"r": two_plus})
node_plus = opt_plus["r"]
check("y+ y+ → y{2,}", isinstance(node_plus, Repeat), True)
check("y+ y+ min=2", node_plus.min, 2)
check("y+ y+ max=None", node_plus.max, None)

# Pas de merge si enfants différents
no_merge = Sequence(children=[
    Repeat(child=Literal("x"), min=0, max=1),
    Repeat(child=Literal("y"), min=0, max=1),
])
opt_no = optimize_rules({"r": no_merge})
check("x? y? pas de merge", isinstance(opt_no["r"], Sequence), True)

section("Optimizations — Combo deep")

# Test un cas profond : Group(Sequence(Literal("a"), Literal("b")))
# → après passes : "ab"
deep = Group(child=Sequence(children=[Literal("a"), Literal("b")]))
opt_deep = optimize_rules({"r": deep})
# Le Group n'est pas supprimé (Sequence est compound), mais les Literals sont fusionnés
# Résultat : Group(Literal("ab"))   → puis group(leaf) → Literal("ab")
check("deep combo → Literal ab", opt_deep["r"], Literal("ab"))

# ══════════════════════════════════════════════════════════════════════
# 7. HELPERS
# ══════════════════════════════════════════════════════════════════════

section("Helpers — Whitespace")

check_in("WS() → *", "*", _emit(WS()))
check_in("WS() → [ \\t\\n]", "[ \\t\\n]", _emit(WS()))
check_in("WS(required) → +", "+", _emit(WS(required=True)))
check_in("ws() same as WS()", _emit(ws()), _emit(WS()))
check_in("ws_required() same as WS(req)", _emit(ws_required()), _emit(WS(required=True)))

section("Helpers — keyword, identifier, number")

check("keyword → Literal", keyword("return"), Literal("return"))
check_in("identifier → a-zA-Z_", "a-zA-Z_", _emit(identifier()))
check_in("number → 0-9", "0-9", _emit(number()))
check_in("number → \"-\"", '"-"', _emit(number()))
check_in("float_number → \".\"", '"."', _emit(float_number()))

section("Helpers — string_literal")

sl = string_literal()
sl_emit = _emit(sl)
check_in("string_literal → quote", '"', sl_emit)
check_in("string_literal → escape", "\\\\", sl_emit)

# Custom quote
sl_sq = string_literal(quote="'")
sl_sq_emit = _emit(sl_sq)
check_in("string_literal quote='", "'", sl_sq_emit)

section("Helpers — comma_list, between, separated_by")

cl = comma_list(RuleReference(name="item"))
cl_str = _emit(cl)
check_in("comma_list → virgule", ",", cl_str)
check_in("comma_list → item", "item", cl_str)

bt = between("(", RuleReference(name="expr"), ")")
check("between", _emit(bt), '"(" expr ")"')

sep = separated_by(";", RuleReference(name="stmt"))
sep_str = _emit(sep)
check_in("separated_by → ;", '";"', sep_str)
check_in("separated_by → stmt", "stmt", sep_str)

section("Helpers — spaced_comma_list")

scl = spaced_comma_list(RuleReference(name="item"))
scl_str = _emit(scl)
check_in("spaced_comma_list → item", "item", scl_str)
check_in("spaced_comma_list → virgule", ",", scl_str)

section("Helpers — any_char")

ac = any_char()
check("any_char → CharacterClass", isinstance(ac, CharacterClass), True)

# ══════════════════════════════════════════════════════════════════════
# 8. SCHEMA — Types primitifs
# ══════════════════════════════════════════════════════════════════════

section("Schema — Types primitifs")

g_str = grammar_from_type(str)
gbnf_str = g_str.to_gbnf()
check_in("str → json-string", "json-string", gbnf_str)

g_int = grammar_from_type(int)
gbnf_int = g_int.to_gbnf()
check_in("int → json-int", "json-int", gbnf_int)

g_float = grammar_from_type(float)
gbnf_float = g_float.to_gbnf()
check_in("float → json-float", "json-float", gbnf_float)

g_bool = grammar_from_type(bool)
gbnf_bool = g_bool.to_gbnf()
check_in("bool → json-bool", "json-bool", gbnf_bool)

section("Schema — Optional, Literal, Enum")

g_opt = grammar_from_type(Optional[int])
gbnf_opt = g_opt.to_gbnf()
check_in("Optional[int] → json-int", "json-int", gbnf_opt)
check_in("Optional[int] → json-null", "json-null", gbnf_opt)

g_lit = grammar_from_type(TypingLiteral["red", "green", "blue"])
gbnf_lit = g_lit.to_gbnf()
check_in("Literal red", '"\\"red\\""', gbnf_lit)
check_in("Literal green", '"\\"green\\""', gbnf_lit)
check_in("Literal blue", '"\\"blue\\""', gbnf_lit)

class Color(enum.Enum):
    RED = "red"
    GREEN = "green"

g_enum = grammar_from_type(Color)
gbnf_enum = g_enum.to_gbnf()
check_in("Enum → red", '"\\"red\\""', gbnf_enum)
check_in("Enum → green", '"\\"green\\""', gbnf_enum)

section("Schema — list[X], dict[str, X]")

g_list = grammar_from_type(List[int])
gbnf_list = g_list.to_gbnf()
check_in("list[int] → [", '"["', gbnf_list)
check_in("list[int] → ]", '"]"', gbnf_list)
check_in("list[int] → json-int", "json-int", gbnf_list)

g_dict = grammar_from_type(Dict[str, int])
gbnf_dict = g_dict.to_gbnf()
check_in("dict → {", '"{"', gbnf_dict)
check_in("dict → }", '"}"', gbnf_dict)
check_in("dict → json-string", "json-string", gbnf_dict)

# ══════════════════════════════════════════════════════════════════════
# 9. SCHEMA — Dataclass simple
# ══════════════════════════════════════════════════════════════════════

section("Schema — Dataclass simple")

@dataclass
class Point:
    x: float
    y: float

g_point = grammar_from_type(Point)
gbnf_point = g_point.to_gbnf()
check_in("Point → x", '\\"x\\"', gbnf_point)
check_in("Point → y", '\\"y\\"', gbnf_point)
check_in("Point → {", '"{"', gbnf_point)

section("Schema — Dataclass imbriquée")

@dataclass
class Inner:
    value: int

@dataclass
class Outer:
    name: str
    child: Inner

g_outer = grammar_from_type(Outer)
gbnf_outer = g_outer.to_gbnf()
check_in("Outer → name", '\\"name\\"', gbnf_outer)
check_in("Outer → child", '\\"child\\"', gbnf_outer)
check_in("Inner rule", "Inner ::=", gbnf_outer)
check_in("Inner → value", '\\"value\\"', gbnf_outer)

section("Schema — Dataclass avec list")

@dataclass
class Project:
    title: str
    tags: List[str]

g_proj = grammar_from_type(Project)
gbnf_proj = g_proj.to_gbnf()
check_in("Project → title", '\\"title\\"', gbnf_proj)
check_in("Project → tags", '\\"tags\\"', gbnf_proj)
check_in("Project → array [", '"["', gbnf_proj)

# ══════════════════════════════════════════════════════════════════════
# 10. SCHEMA — Champs optionnels (defaults)
# ══════════════════════════════════════════════════════════════════════

section("Schema — Champs optionnels (defaults)")

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

# Tous obligatoires → pas de )?
g_req = grammar_from_type(_AllRequired)
gbnf_req = g_req.to_gbnf()
dc_lines_req = [l for l in gbnf_req.splitlines() if "AllRequired" in l and "::=" in l]
if dc_lines_req:
    check_not_in("all required — no )?", ")?", dc_lines_req[0])
else:
    check_in("AllRequired rule exists", "AllRequired", gbnf_req)
check_in("all required → name", '\\"name\\"', gbnf_req)
check_in("all required → age", '\\"age\\"', gbnf_req)

# Avec defaults → optional trailing
g_def = grammar_from_type(_WithDefaults)
gbnf_def = g_def.to_gbnf()
check_in("with defaults — has )?", ")?", gbnf_def)
check_in("with defaults → name", '\\"name\\"', gbnf_def)
check_in("with defaults → active", '\\"active\\"', gbnf_def)
check_in("with defaults → nickname", '\\"nickname\\"', gbnf_def)
check("with defaults — nested )?", gbnf_def.count(")?") >= 2, True)

# Tous defaults
g_all = grammar_from_type(_AllDefaults)
gbnf_all = g_all.to_gbnf()
check_in("all defaults — has )?", ")?", gbnf_all)
check_in("all defaults → x", '\\"x\\"', gbnf_all)
check_in("all defaults → y", '\\"y\\"', gbnf_all)

# ══════════════════════════════════════════════════════════════════════
# 11. SCHEMA — grammar_from_function / grammar_from_args
# ══════════════════════════════════════════════════════════════════════

section("Schema — grammar_from_function")

@dataclass
class SearchResult:
    title: str
    score: float

def search(query: str) -> SearchResult:
    ...

g_fn = grammar_from_function(search)
gbnf_fn = g_fn.to_gbnf()
check_in("fn return → title", '\\"title\\"', gbnf_fn)
check_in("fn return → score", '\\"score\\"', gbnf_fn)

# Fonction sans annotation retour
def no_return(x: int):
    ...

check_raises("fn sans retour → TypeError", TypeError, lambda: grammar_from_function(no_return))

section("Schema — grammar_from_args")

def send_email(to: str, subject: str, body: str, priority: int = 0):
    ...

g_args = grammar_from_args(send_email)
gbnf_args = g_args.to_gbnf()
check_in("args → to", '\\"to\\"', gbnf_args)
check_in("args → subject", '\\"subject\\"', gbnf_args)
check_in("args → body", '\\"body\\"', gbnf_args)
check_in("args → priority", '\\"priority\\"', gbnf_args)

# ══════════════════════════════════════════════════════════════════════
# 12. SCHEMA — from_type / from_function_return / from_function_args
#     (méthodes Grammar pour composition)
# ══════════════════════════════════════════════════════════════════════

section("Schema — Grammar.from_type() composition")

@dataclass
class Movie:
    title: str
    year: int

g_comp = Grammar()
movie_node = g_comp.from_type(Movie)

@g_comp.rule
def review():
    return Literal('"review":') + movie_node

g_comp.start("review")
gbnf_comp = g_comp.to_gbnf()
check_in("from_type → movie rule", "Movie ::=", gbnf_comp)
check_in("from_type → title", '\\"title\\"', gbnf_comp)

section("Schema — Grammar.from_function_return()")

def get_movie() -> Movie:
    ...

g_fr = Grammar()
ret_node = g_fr.from_function_return(get_movie)

@g_fr.rule
def fr_root():
    return ret_node

g_fr.start("fr_root")
gbnf_fr = g_fr.to_gbnf()
check_in("from_function_return → movie", "Movie ::=", gbnf_fr)

section("Schema — Grammar.from_function_args()")

def api_call(query: str, limit: int = 10):
    ...

g_fa = Grammar()
args_node = g_fa.from_function_args(api_call)

@g_fa.rule
def fa_root():
    return args_node

g_fa.start("fa_root")
gbnf_fa = g_fa.to_gbnf()
check_in("from_function_args → query", '\\"query\\"', gbnf_fa)
check_in("from_function_args → limit", '\\"limit\\"', gbnf_fa)

# ══════════════════════════════════════════════════════════════════════
# 13. SCHEMA — Union non-Optional
# ══════════════════════════════════════════════════════════════════════

section("Schema — Union[str, int]")

g_union = grammar_from_type(Union[str, int])
gbnf_union = g_union.to_gbnf()
check_in("Union → json-string", "json-string", gbnf_union)
check_in("Union → json-int", "json-int", gbnf_union)

# ══════════════════════════════════════════════════════════════════════
# 14. SCHEMA — Dataclass avec Enum
# ══════════════════════════════════════════════════════════════════════

section("Schema — Dataclass avec Enum")

class Severity(enum.Enum):
    INFO = "info"
    ERROR = "error"

@dataclass
class LogEntry:
    message: str
    severity: Severity

g_log = grammar_from_type(LogEntry)
gbnf_log = g_log.to_gbnf()
check_in("LogEntry → message", '\\"message\\"', gbnf_log)
check_in("LogEntry → severity", '\\"severity\\"', gbnf_log)
check_in("LogEntry → info", '"\\"info\\""', gbnf_log)
check_in("LogEntry → error", '"\\"error\\""', gbnf_log)

# ══════════════════════════════════════════════════════════════════════
# 15. SCHEMA — Type non supporté
# ══════════════════════════════════════════════════════════════════════

section("Schema — Type non supporté")

check_raises("set → TypeError", TypeError, lambda: grammar_from_type(set))
check_raises("tuple → TypeError", TypeError, lambda: grammar_from_type(tuple))

# ══════════════════════════════════════════════════════════════════════
# 16. Integration — Grammaire complète
# ══════════════════════════════════════════════════════════════════════

section("Integration — Grammaire mini-lang")

g_lang = Grammar()

@g_lang.rule
def lang_ws():
    return repeat(select(" \t"), 0, 8)

@g_lang.rule
def lang_ident():
    return CharacterClass(pattern="a-zA-Z_") + repeat(CharacterClass(pattern="a-zA-Z0-9_"), 0, 20)

@g_lang.rule
def lang_number():
    return one_or_more(CharacterClass(pattern="0-9"))

@g_lang.rule
def lang_expr():
    return select([
        lang_number(),
        lang_ident(),
    ])

@g_lang.rule
def lang_let():
    return Literal("let ") + lang_ident() + Literal(" = ") + lang_expr() + Literal(";")

@g_lang.rule
def lang_print():
    return Literal("print(") + lang_expr() + Literal(");")

@g_lang.rule
def lang_stmt():
    return select([lang_let(), lang_print()])

@g_lang.rule
def lang_program():
    return lang_stmt() + repeat(group("\n" + lang_stmt()), 0, 10) + Literal("\n")

g_lang.start("lang_program")
gbnf_lang = g_lang.to_gbnf()

check_in("integration → root", "root ::=", gbnf_lang)
check_in("integration → let", '"let "', gbnf_lang)
check_in("integration → print(", '"print("', gbnf_lang)
check_in("integration → lang-ident", "lang-ident", gbnf_lang)
check("integration → compile sans erreur", len(gbnf_lang) > 50, True)

# Nombre de règles
rules = g_lang.rules()
check("integration → 8 règles", len(rules), 8)

# ══════════════════════════════════════════════════════════════════════
# 17. Edge cases
# ══════════════════════════════════════════════════════════════════════

section("Edge cases")

# Grammaire vide (pas de start)
g_empty = Grammar()
g_empty._start = None
gbnf_empty = g_empty.to_gbnf()
check("grammaire sans start", gbnf_empty.strip(), "")

# Repeat {0,0} → vide
check("emit Repeat {0,0}", _emit(Repeat(child=Literal("x"), min=0, max=0)), '"x"{0}')

# Séquence vide
check("emit Sequence vide", _emit(Sequence(children=[])), "")

# Alternative vide
check("emit Alternative vide", _emit(Alternative(alternatives=[])), "")

# Chaîne très longue dans Literal
long_str = "a" * 200
check("literal longue", _emit(Literal(long_str)), f'"{long_str}"')

# CharacterClass avec beaucoup de chars
check("select long str", isinstance(select("abcdefghijklmnop"), CharacterClass), True)

# ══════════════════════════════════════════════════════════════════════
# 18. __init__ — exports publics
# ══════════════════════════════════════════════════════════════════════

section("Exports publics")

check("version", cfg.__version__, "0.3.1")
check("Grammar accessible", cfg.Grammar is Grammar, True)
check("select accessible", cfg.select is select, True)
check("grammar_from_type accessible", cfg.grammar_from_type is grammar_from_type, True)

# Vérifier que tous les __all__ sont importables
for name in cfg.__all__:
    check_in(f"__all__ : {name}", name, dir(cfg))

# ══════════════════════════════════════════════════════════════════════
# 19. Dataclass avec field(default_factory=...)
# ══════════════════════════════════════════════════════════════════════

section("Schema — default_factory")

@dataclass
class _WithFactory:
    name: str
    items: List[str] = dc_field(default_factory=list)

g_wf = grammar_from_type(_WithFactory)
gbnf_wf = g_wf.to_gbnf()
check_in("default_factory → optional )?", ")?", gbnf_wf)
check_in("default_factory → name present", '\\"name\\"', gbnf_wf)
check_in("default_factory → items present", '\\"items\\"', gbnf_wf)

# ══════════════════════════════════════════════════════════════════════
# 20. Multiples règles et ordre topologique
# ══════════════════════════════════════════════════════════════════════

section("Grammar — Tri topologique")

g_topo = Grammar()

@g_topo.rule
def c_rule():
    return Literal("c")

@g_topo.rule
def b_rule():
    return c_rule()

@g_topo.rule
def a_rule():
    return b_rule()

g_topo.start("a_rule")
gbnf_topo = g_topo.to_gbnf()

# Toutes les règles présentes
check_in("topo → a-rule", "a-rule ::=", gbnf_topo)
check_in("topo → b-rule", "b-rule ::=", gbnf_topo)
check_in("topo → c-rule", "c-rule ::=", gbnf_topo)

# ══════════════════════════════════════════════════════════════════════
# 21. compile_grammar standalone
# ══════════════════════════════════════════════════════════════════════

section("compile_grammar standalone")

g_cg = Grammar()

@g_cg.rule
def cg_item():
    return select(["yes", "no"])

g_cg.start("cg_item")

gbnf_cg_opt = compile_grammar(g_cg, optimize=True)
gbnf_cg_no = compile_grammar(g_cg, optimize=False)
check_in("compile_grammar opt → root", "root ::=", gbnf_cg_opt)
check_in("compile_grammar no opt → root", "root ::=", gbnf_cg_no)

# ══════════════════════════════════════════════════════════════════════
# 22. Caractères spéciaux dans les noms de règles
# ══════════════════════════════════════════════════════════════════════

section("Grammar — caractères spéciaux dans les noms")

g_special = Grammar()

@g_special.rule_named("a-b-c")
def _special_impl():
    return Literal("x")

g_special.start("a-b-c")
gbnf_special = g_special.to_gbnf()
check_in("nom avec tirets", "a-b-c ::=", gbnf_special)

# ══════════════════════════════════════════════════════════════════════
# 23. Combinateur group imbriqué
# ══════════════════════════════════════════════════════════════════════

section("Combinators — group imbriqué")

nested_group = group(group(Literal("x")))
check("group(group(x)) → Group", isinstance(nested_group, Group), True)
check("group(group(x)) inner → Group", isinstance(nested_group.child, Group), True)

# Après optimisation, les groups redondants devraient être supprimés
opt_ng = optimize_rules({"r": nested_group})
check("double group optimisé → Literal", opt_ng["r"], Literal("x"))

# ══════════════════════════════════════════════════════════════════════
# Résumé
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 50}")
print(f"Résultats : {passed} passés, {failed} échoués")
if failed:
    print(f"\n⚠ {failed} test(s) en échec !")
    sys.exit(1)
else:
    print("Tous les tests passent ✓")
