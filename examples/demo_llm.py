# pip install openai
from openai import OpenAI
import pygbnf as cfg
from pygbnf import select, one_or_more, zero_or_more, optional, CharacterClass

# --- Grammaire construite avec pygbnf ---
g = cfg.Grammar()

@g.rule
def root():
    return select(["yes", "no", "maybe"])

grammar = g.to_gbnf()
print("=== Grammaire GBNF générée ===")
print(grammar)
print("==============================\n")

# --- Appel llama.cpp server ---
LLAMA_BASE_URL = "http://localhost:8080/v1"
MODEL_NAME = "gpt-3.5-turbo"  # llama-server accepte souvent n'importe quel nom ici

client = OpenAI(
    base_url=LLAMA_BASE_URL,
    api_key="sk-no-key-required",
)

resp = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "Replique par 'yes', 'no' ou si tu ne sais pas 'maybe'."},
        {"role": "user", "content": "Est-ce que Paris est la capitale de la France ?"},
    ],
    temperature=0,
    # IMPORTANT: champ non-standard OpenAI, mais supporté par llama.cpp server
    extra_body={"grammar": grammar},
)

print(resp.choices[0].message.content)


# =====================================================================
# Requête 2 — Grammaire structurée : réponse JSON avec note et avis
# =====================================================================

g2 = cfg.Grammar()

@g2.rule
def ws():
    return zero_or_more(select(" \t\n"))

@g2.rule
def digit():
    return CharacterClass(pattern="0-9")

@g2.rule
def integer():
    return one_or_more(digit())

@g2.rule
def safe_char():
    return CharacterClass(pattern='^"\\\\')

@g2.rule
def string_val():
    return '"' + zero_or_more(safe_char()) + '"'

@g2.rule
def rating():
    """Note entre 1 et 5."""
    return select("12345")

@g2.rule
def sentiment():
    return select(["positive", "negative", "neutral"])

@g2.rule
def root():
    return ("{" + ws()
        + '"rating":' + ws() + rating() + "," + ws()
        + '"sentiment":' + ws() + '"' + sentiment() + '"' + "," + ws()
        + '"summary":' + ws() + string_val() + ws()
    + "}")

grammar2 = g2.to_gbnf()
print("\n=== Grammaire GBNF #2 (JSON structuré) ===")
print(grammar2)
print("==========================================\n")

resp2 = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": (
            "Tu es un critique de films. "
            "Réponds uniquement en JSON avec les champs: "
            'rating (1-5), sentiment (positive/negative/neutral), summary (texte court).'
        )},
        {"role": "user", "content": "Donne ton avis sur le film Inception."},
    ],
    temperature=0.7,
    extra_body={"grammar": grammar2},
)

print(resp2.choices[0].message.content)