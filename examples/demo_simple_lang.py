#!/usr/bin/env python3
"""Démo — Faire écrire du code par un LLM, contraint par la grammaire simple_lang.

Utilise pygbnf pour générer la grammaire GBNF d'un mini-langage de
programmation, puis demande au LLM de produire du code valide dans ce langage.
"""

import time
import requests
from openai import OpenAI
import pygbnf as cfg
from pygbnf import (
    CharacterClass,
    select,
    one_or_more,
    zero_or_more,
    optional,
    repeat,
)

# =====================================================================
# Grammaire du mini-langage
# =====================================================================

g = cfg.Grammar()

# -- Whitespace & tokens ---------------------------------------------------
# Borner le whitespace pour éviter que le LLM boucle en émettant des espaces

@g.rule
def sp():
    """Espaces horizontaux optionnels (pas de newline)."""
    return repeat(select(" \t"), 0, 8)

@g.rule
def nl():
    """Un saut de ligne."""
    return select(["\r\n", "\n"])

@g.rule
def ws():
    """Whitespace borné : espaces + sauts de ligne, max 20 chars."""
    return repeat(select(" \t\n\r"), 0, 20)

@g.rule
def ws1():
    """Au moins un whitespace, borné."""
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

t0 = time.perf_counter()
grammar = g.to_gbnf()
t_compile = time.perf_counter() - t0

print("=" * 60)
print("Grammaire GBNF du mini-langage")
print("=" * 60)
print(grammar)
print("=" * 60)
print(f"Compilation de la grammaire : {t_compile*1000:.1f} ms")

# =====================================================================
# Appel au LLM contraint par la grammaire
# =====================================================================

LLAMA_BASE_URL = "http://localhost:8080"

# On utilise l'endpoint natif /completion de llama-server qui applique
# toujours la grammaire, contrairement à /v1/chat/completions qui
# l'ignore avec certains modèles.

PROMPT = """\
Écris un programme dans le mini-langage suivant.

Syntaxe du langage :
  fn name(args) { body }     — définition de fonction
  let x = expr;              — déclaration de variable
  x = expr;                  — assignation
  if (cond) { } else { }     — condition
  while (cond) { }           — boucle
  return expr;               — retour de valeur
  print(expr, ...);          — affichage
  name(args)                 — appel de fonction
  opérateurs : + - * / % == != < > <= >=

Exemple de programme valide :
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

done;

Maintenant, écris un programme qui :
- définit une fonction main() qui affiche les nombres de 1 à 10.
- tu finis le programme par l'instruction "done;" pour indiquer que c'est la fin.

Programme :
"""

print("\nEnvoi de la requête au LLM avec contrainte grammaticale...\n")

t1 = time.perf_counter()
response = requests.post(
    f"{LLAMA_BASE_URL}/completion",
    json={
        "prompt": PROMPT,
        "grammar": grammar,
        "n_predict": 96,
        "temperature": 0.0,
        "stop": ["\n\n\n"],  # arrêter après 2 lignes vides (fin du programme)
    },
)
t_gen = time.perf_counter() - t1

data = response.json()
code = data.get("content", "")
timings = data.get("timings", {})
tokens_predicted = data.get("tokens_predicted", 0)
tokens_evaluated = data.get("tokens_evaluated", 0)

print("=== Code généré par le LLM ===")
print(code)
print("==============================")
print(f"\n--- Mesures ---")
print(f"Compilation grammaire : {t_compile*1000:.1f} ms")
print(f"Génération LLM        : {t_gen:.2f} s")
print(f"Tokens prompt         : {tokens_evaluated}")
print(f"Tokens completion     : {tokens_predicted}")
if tokens_predicted and t_gen > 0:
    tps = tokens_predicted / t_gen
    print(f"Débit                 : {tps:.1f} tokens/s")
if timings:
    prompt_ms = timings.get("prompt_ms", 0)
    predicted_ms = timings.get("predicted_ms", 0)
    pred_per_s = timings.get("predicted_per_second", 0)
    print(f"Temps prompt (serveur): {prompt_ms:.0f} ms")
    print(f"Temps génération      : {predicted_ms:.0f} ms")
    print(f"Débit (serveur)       : {pred_per_s:.1f} tokens/s")
