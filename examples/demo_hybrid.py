#!/usr/bin/env python3
"""Démo — Mini-langage hybride : grammaire manuelle + types Python.

Combine :
- Des dataclasses Python pour décrire les structures de données (JSON)
- Des règles manuelles pygbnf pour la syntaxe du langage

Le LLM est contraint de produire un programme valide dans ce DSL
qui manipule des objets typés.

Usage :
    PYTHONPATH=. python demo_hybrid.py          # affiche la grammaire
    PYTHONPATH=. python demo_hybrid.py --llm    # envoie au LLM
"""

import enum
import time
import sys
from dataclasses import dataclass
from typing import Optional

import pygbnf as cfg
from pygbnf import (
    CharacterClass,
    select,
    one_or_more,
    zero_or_more,
    optional,
    repeat,
    group,
)

LLAMA_BASE_URL = "http://localhost:8080"


# =====================================================================
# 1. Types Python — décrivent les structures de données du langage
# =====================================================================

class Severity(enum.Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    message: str
    severity: Severity
    code: int


@dataclass
class Config:
    host: str
    port: int
    debug: bool = False
    timeout: Optional[int] = None


@dataclass
class HttpResponse:
    status: int
    body: str
    ok: bool


# =====================================================================
# 2. Grammaire du mini-langage — mélange règles manuelles et types
# =====================================================================

g = cfg.Grammar()

# -- Injecter les types Python dans la grammaire -----------------------
# Chaque appel crée les règles GBNF correspondantes et retourne
# un nœud composable qu'on peut utiliser dans les règles manuelles.

log_entry_node = g.from_type(LogEntry)
config_node = g.from_type(Config)
http_response_node = g.from_type(HttpResponse)


# -- Tokens de base ----------------------------------------------------

@g.rule
def ws():
    return repeat(select(" \t"), 0, 20)


@g.rule
def nl():
    return repeat(select(" \t"), 0, 20) + "\n" + repeat(select(" \t\n"), 0, 20)


@g.rule
def ident():
    return CharacterClass(pattern="a-zA-Z_") + repeat(
        CharacterClass(pattern="a-zA-Z0-9_"), 0, 30
    )


@g.rule
def string_lit():
    return '"' + repeat(
        select([CharacterClass(pattern='^"\\\\'), "\\" + select('"\\\\nrt')]),
        0, 60,
    ) + '"'


@g.rule
def int_lit():
    return optional("-") + one_or_more(CharacterClass(pattern="0-9"))


@g.rule
def bool_lit():
    return select(["true", "false"])


# -- Expressions -------------------------------------------------------

@g.rule
def atom():
    return select([
        int_lit(),
        string_lit(),
        bool_lit(),
        ident(),
        "(" + ws() + expr() + ws() + ")",
    ])


@g.rule
def call_expr():
    args = optional(group(
        expr() + repeat(group(ws() + "," + ws() + expr()), 0, 10)
    ))
    return ident() + "(" + ws() + args + ws() + ")"


@g.rule
def member_expr():
    return atom() + repeat(group("." + ident()), 0, 5)


@g.rule
def cmp_op():
    return select(["==", "!=", "<=", ">=", "<", ">"])


@g.rule
def arith_op():
    return select(["+", "-", "*", "/", "%"])


@g.rule
def expr():
    return select([
        call_expr(),
        member_expr() + optional(group(
            ws() + select([cmp_op(), arith_op()]) + ws() + expr()
        )),
    ])


# -- Typed literals : ici on branche les types Python ! ----------------
# Un "typed_value" est soit un littéral JSON généré depuis une dataclass,
# soit une expression du langage.

@g.rule
def typed_value():
    return select([
        "LogEntry" + ws() + "(" + ws() + log_entry_node + ws() + ")",
        "Config" + ws() + "(" + ws() + config_node + ws() + ")",
        "HttpResponse" + ws() + "(" + ws() + http_response_node + ws() + ")",
    ])


# -- Statements --------------------------------------------------------

@g.rule
def let_stmt():
    return "let" + ws() + ident() + ws() + "=" + ws() + select([
        typed_value(),
        expr(),
    ]) + ";"


@g.rule
def assign_stmt():
    return ident() + repeat(group("." + ident()), 0, 5) + ws() + "=" + ws() + expr() + ";"


@g.rule
def if_stmt():
    return (
        "if" + ws() + "(" + ws() + expr() + ws() + ")" + ws()
        + "{" + nl() + stmts() + nl() + "}"
        + optional(group(
            ws() + "else" + ws() + "{" + nl() + stmts() + nl() + "}"
        ))
    )


@g.rule
def while_stmt():
    return (
        "while" + ws() + "(" + ws() + expr() + ws() + ")" + ws()
        + "{" + nl() + stmts() + nl() + "}"
    )


@g.rule
def log_stmt():
    return "log" + ws() + "(" + ws() + expr() + ws() + ")" + ";"


@g.rule
def send_stmt():
    return (
        "send" + ws() + "(" + ws()
        + expr() + ws() + "," + ws() + expr()
        + ws() + ")" + ";"
    )


@g.rule
def return_stmt():
    return "return" + ws() + expr() + ";"


@g.rule
def expr_stmt():
    return call_expr() + ";"


@g.rule
def statement():
    return select([
        let_stmt(),
        assign_stmt(),
        if_stmt(),
        while_stmt(),
        log_stmt(),
        send_stmt(),
        return_stmt(),
        expr_stmt(),
    ])


@g.rule
def stmts():
    return statement() + repeat(group(nl() + statement()), 0, 8)


# -- Fonctions ---------------------------------------------------------

@g.rule
def param_list():
    return optional(group(
        ident() + repeat(group(ws() + "," + ws() + ident()), 0, 10)
    ))


@g.rule
def fn_decl():
    return (
        "fn" + ws() + ident() + "(" + ws() + param_list() + ws() + ")"
        + ws() + "{" + nl() + stmts() + nl() + "}"
    )


# -- Programme ---------------------------------------------------------

@g.rule
def top_level():
    return select([fn_decl(), statement()])


@g.rule
def program():
    return top_level() + repeat(group(nl() + nl() + top_level()), 2, 6) + nl()


g.start("program")


# =====================================================================
# 3. Compilation et requête LLM
# =====================================================================

def main():
    use_llm = "--llm" in sys.argv

    t0 = time.perf_counter()
    grammar = g.to_gbnf()
    t_compile = (time.perf_counter() - t0) * 1000

    print("=" * 60)
    print("Grammaire GBNF — Mini-langage hybride")
    print("=" * 60)
    print(grammar)
    print("=" * 60)
    print(f"Compilation : {t_compile:.1f} ms")
    print(f"Règles : {len(g.rules())}")
    print()

    if not use_llm:
        print("(ajoutez --llm pour envoyer au LLM)")
        return

    import requests

    # -- Exemple de programme attendu (few-shot) -----------------------
    example = """\
let cfg = Config({"host": "api.example.com", "port": 8080});

fn check(response) {
  if (response.ok == true) {
    log(response.body);
  } else {
    let err = LogEntry({"message": "request failed", "severity": "error", "code": response.status});
    log(err.message);
  }
}"""

    prompt = f"""\
Génère du code dans un mini-langage typé. Réponds UNIQUEMENT avec le code, rien d'autre.

Types disponibles (valeurs JSON) :
- Config {{ host: string, port: int, debug?: bool, timeout?: int|null }} (debug et timeout sont optionnels, on peut les omettre)
- LogEntry {{ message: string, severity: "info"|"warning"|"error"|"critical", code: int }}
- HttpResponse {{ status: int, body: string, ok: bool }}

Syntaxe : let x = expr; | let x = TypeName({{...}}); | if/else | while | fn name(args) {{}} | log(expr); | send(expr, expr);

Exemple de programme valide :
{example}

---

Écris un programme qui :
- Crée une Config pour un serveur www.google.com sur le port 80, en mode debug (omet timeout)
- Définit une fonction handle(resp) qui :
  - si resp.ok est true : log le body
  - sinon : crée un LogEntry d'erreur et log le message
- Appelle handle avec un HttpResponse

Code :
"""

    print("Envoi au LLM...\n")
    t0 = time.perf_counter()
    resp = requests.post(f"{LLAMA_BASE_URL}/completion", json={
        "prompt": prompt,
        "grammar": grammar,
        "n_predict": 1024,
        "temperature": 0.3,
    })
    elapsed = time.perf_counter() - t0

    data = resp.json()
    result = data.get("content", "").strip()
    tokens = data.get("tokens_predicted", "?")
    timings = data.get("timings", {})
    tps = timings.get("predicted_per_second", 0)

    print("=== Programme généré ===")
    print(result)
    print("========================")
    print()
    print(f"Tokens prompt  : {data.get('tokens_evaluated', '?')}")
    print(f"Tokens générés : {tokens}")
    print(f"Temps total    : {elapsed:.2f} s")
    if tps:
        print(f"Débit serveur  : {tps:.1f} tokens/s")


if __name__ == "__main__":
    main()
