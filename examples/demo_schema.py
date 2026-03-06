#!/usr/bin/env python3
"""Démo — Génération automatique de grammaires depuis des types Python.

Montre comment transformer des dataclasses, des fonctions annotées et des
enums en grammaires GBNF, puis les appliquer à un LLM via llama-server.

Usage :
    PYTHONPATH=. python demo_schema.py          # tous les exemples (offline)
    PYTHONPATH=. python demo_schema.py --llm    # avec appel au LLM
    PYTHONPATH=. python demo_schema.py 2 --llm  # exemple 2 uniquement
"""

import enum
import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Literal, Optional

import requests

from pygbnf import (
    Grammar, grammar_from_type, grammar_from_function, grammar_from_args,
    select, optional, group, repeat, CharacterClass,
)

LLAMA_BASE_URL = "http://localhost:8080"


def _strip_markdown(text: str) -> str:
    """Retire les balises ```json ... ``` si présentes."""
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*\n(.*?)\n?```\s*$", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Parfois juste ``` au début
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _parse_json(text: str, label: str = "") -> dict | None:
    """Parse du JSON depuis le texte LLM, avec nettoyage et gestion d'erreur."""
    cleaned = _strip_markdown(text)
    try:
        parsed = json.loads(cleaned)
        return parsed
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON invalide{' ('+label+')' if label else ''} : {e}")
        if cleaned[:80] != text[:80]:
            print(f"    (après nettoyage markdown : {cleaned[:120]}...)")
        return None


def call_llm(prompt: str, grammar: str, max_tokens: int = 512) -> str:
    """Appelle llama-server avec la grammaire GBNF. Retourne le texte généré."""
    t0 = time.perf_counter()
    resp = requests.post(f"{LLAMA_BASE_URL}/completion", json={
        "prompt": prompt,
        "grammar": grammar,
        "n_predict": max_tokens,
        "temperature": 0.0,
    })
    elapsed = time.perf_counter() - t0

    if resp.status_code != 200:
        print(f"  ✗ Erreur serveur HTTP {resp.status_code}: {resp.text[:200]}")
        return ""

    data = resp.json()
    content = data.get("content", "")
    tokens = data.get("tokens_predicted", 0)
    timings = data.get("timings", {})
    tps = timings.get("predicted_per_second", 0)

    print(f"  ({elapsed:.2f}s, {tokens} tokens, {tps:.0f} t/s)")

    # Diagnostic : si la grammaire n'a pas été appliquée, le serveur
    # peut renvoyer un champ d'erreur ou générer du texte libre.
    if content and not content.lstrip().startswith("{") and not content.lstrip().startswith("["):
        print("  ⚠ La sortie ne commence pas par { ou [ — la grammaire ")
        print("    n'est peut-être pas appliquée par le serveur.")
        print("    Vérifiez les logs de llama-server pour des erreurs de parsing GBNF.")

    return content


# =====================================================================
# Exemple 1 — Dataclass simple
# =====================================================================

@dataclass
class MovieReview:
    """Critique de film structurée."""
    title: str
    year: int
    rating: float
    sentiment: Literal["positive", "negative", "neutral"]
    summary: str


def demo_1(use_llm: bool = False):
    print("=" * 60)
    print("Exemple 1 — Dataclass simple → Grammaire JSON")
    print("=" * 60)
    print()

    g = grammar_from_type(MovieReview)
    gbnf = g.to_gbnf()
    print(gbnf)

    if use_llm:
        print("--- Résultat LLM ---")
        result = call_llm(
            "Génère une critique du film Inception en JSON.\nJSON :\n",
            gbnf,
        )
        print(result)
        print()
        parsed = _parse_json(result, "MovieReview")
        if parsed:
            print(f"  ✓ JSON valide : {list(parsed.keys())}")
    print()


# =====================================================================
# Exemple 2 — Dataclass imbriquée + Enum
# =====================================================================

class Priority(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Task:
    title: str
    priority: Priority
    done: bool


@dataclass
class Project:
    name: str
    description: str
    tasks: list[Task]


def demo_2(use_llm: bool = False):
    print("=" * 60)
    print("Exemple 2 — Dataclass imbriquée (Project → Task[])")
    print("=" * 60)
    print()

    g = grammar_from_type(Project)
    gbnf = g.to_gbnf()
    print(gbnf)

    if use_llm:
        print("--- Résultat LLM ---")
        result = call_llm(
            "Génère un projet de développement web avec 3 tâches en JSON.\nJSON :\n",
            gbnf, max_tokens=1024,
        )
        print(result)
        print()
        parsed = _parse_json(result, "Project")
        if parsed:
            tasks = parsed.get("tasks", [])
            print(f"  ✓ JSON valide, {len(tasks)} tâches")
    print()


# =====================================================================
# Exemple 3 — Type retour d'une fonction
# =====================================================================

@dataclass
class SearchResult:
    query: str
    results: list[str]
    total: int


def search(query: str, max_results: int = 10, lang: str = "fr") -> SearchResult:
    """Recherche dans la base de données."""
    ...


def demo_3(use_llm: bool = False):
    print("=" * 60)
    print("Exemple 3 — Type retour d'une fonction → Grammaire")
    print("=" * 60)
    print()

    g = grammar_from_function(search)
    gbnf = g.to_gbnf()
    print(gbnf)

    if use_llm:
        print("--- Résultat LLM ---")
        result = call_llm(
            'Génère un résultat de recherche pour "Python programming" en JSON.\nJSON :\n',
            gbnf,
        )
        print(result)
        print()
        parsed = _parse_json(result, "SearchResult")
        if parsed:
            print(f"  ✓ JSON valide : query={parsed.get('query', '?')!r}, {parsed.get('total', '?')} résultats")
    print()


# =====================================================================
# Exemple 4 — Arguments de fonction (tool calling)
# =====================================================================

def send_email(
    to: str,
    subject: str,
    body: str,
    cc: Optional[str] = None,
    urgent: bool = False,
) -> None:
    """Envoie un email."""
    ...


def demo_4(use_llm: bool = False):
    print("=" * 60)
    print("Exemple 4 — Arguments de fonction → Grammaire (tool calling)")
    print("=" * 60)
    print()

    g = grammar_from_args(send_email)
    gbnf = g.to_gbnf()
    print(gbnf)

    if use_llm:
        print("--- Résultat LLM ---")
        result = call_llm(
            "L'utilisateur dit : 'Envoie un email urgent à alice@example.com "
            "pour lui rappeler la réunion de demain.'\n"
            "Génère les arguments de la fonction send_email en JSON.\nJSON :\n",
            gbnf,
        )
        print(result)
        print()
        parsed = _parse_json(result, "send_email args")
        if parsed:
            print(f"  ✓ JSON valide : to={parsed.get('to', '?')!r}, urgent={parsed.get('urgent', '?')}")
    print()


# =====================================================================
# Exemple 5 — Optional fields
# =====================================================================

@dataclass
class UserProfile:
    name: str
    age: int
    email: Optional[str]
    bio: Optional[str]
    verified: bool


def demo_5(use_llm: bool = False):
    print("=" * 60)
    print("Exemple 5 — Optional fields (null autorisé)")
    print("=" * 60)
    print()

    g = grammar_from_type(UserProfile)
    gbnf = g.to_gbnf()
    print(gbnf)

    if use_llm:
        print("--- Résultat LLM ---")
        result = call_llm(
            "Génère un profil utilisateur en JSON. "
            "L'email et la bio peuvent être null.\nJSON :\n",
            gbnf,
        )
        print(result)
        print()
        parsed = _parse_json(result, "UserProfile")
        if parsed:
            print(f"  ✓ JSON valide : name={parsed.get('name', '?')!r}, email={parsed.get('email', '?')}")
    print()


# =====================================================================
# Exemple 6 — Composition : types + règles manuelles dans la même grammaire
# =====================================================================

class Sentiment(enum.Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class Review:
    """Critique structurée."""
    title: str
    rating: float
    sentiment: Sentiment
    summary: str


@dataclass
class UserInfo:
    name: str
    email: Optional[str]


def demo_6(use_llm: bool = False):
    print("=" * 60)
    print("Exemple 6 — Composition : types + règles manuelles")
    print("=" * 60)
    print()

    g = Grammar()

    # Injecter les types comme sous-grammaires composables
    review_node = g.from_type(Review)
    user_node = g.from_type(UserInfo)

    # Construire une structure personnalisée qui utilise les deux
    @g.rule
    def root():
        ws = g.ref("ws")
        return ("{" + ws
                + '"review"' + ws + ":" + ws + review_node + ws + "," + ws
                + '"author"' + ws + ":" + ws + user_node + ws + "," + ws
                + '"tags"' + ws + ":" + ws + tags() + ws + "," + ws
                + '"recommend"' + ws + ":" + ws + select(["true", "false"]) + ws
                + "}")

    @g.rule
    def tag():
        return '"' + repeat(CharacterClass(pattern="a-z0-9_"), 1, 20) + '"'

    @g.rule
    def tags():
        ws = g.ref("ws")
        return "[" + ws + optional(group(
            tag() + repeat(group(ws + "," + ws + tag()), 0, 10)
        )) + ws + "]"

    g.start("root")
    gbnf = g.to_gbnf()
    print(gbnf)

    if use_llm:
        print("--- Résultat LLM ---")
        result = call_llm(
            "Génère une critique du film Dune avec auteur et tags, en JSON.\nJSON :\n",
            gbnf, max_tokens=512,
        )
        print(result)
        print()
        parsed = _parse_json(result, "Composed")
        if parsed:
            print(f"  ✓ JSON valide")
            print(f"    review.title  = {parsed.get('review', {}).get('title', '?')!r}")
            print(f"    author.name   = {parsed.get('author', {}).get('name', '?')!r}")
            print(f"    tags          = {parsed.get('tags', '?')}")
            print(f"    recommend     = {parsed.get('recommend', '?')}")
    print()


# =====================================================================

DEMOS = [
    ("Dataclass simple (MovieReview)", demo_1),
    ("Dataclass imbriquée (Project → Task[])", demo_2),
    ("Type retour de fonction (SearchResult)", demo_3),
    ("Arguments de fonction — tool calling (send_email)", demo_4),
    ("Optional fields (UserProfile)", demo_5),
    ("Composition : types + règles manuelles", demo_6),
]

if __name__ == "__main__":
    use_llm = "--llm" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--llm"]

    if args and args[0].isdigit():
        choice = int(args[0])
        if 1 <= choice <= len(DEMOS):
            DEMOS[choice - 1][1](use_llm=use_llm)
        else:
            print(f"Numéro invalide : {choice} (1-{len(DEMOS)})")
            sys.exit(1)
    else:
        print("Démos schema → GBNF :")
        for i, (desc, _) in enumerate(DEMOS, 1):
            print(f"  {i}. {desc}")
        if not use_llm:
            print("\n  (ajoutez --llm pour envoyer au LLM)")
        print()

        for i, (desc, fn) in enumerate(DEMOS, 1):
            try:
                fn(use_llm=use_llm)
            except Exception as e:
                print(f"  ✗ Erreur dans l'exemple {i}: {e}")
                print()
