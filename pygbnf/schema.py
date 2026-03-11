"""Génération automatique de grammaires à partir de types Python.

Permet de transformer des dataclasses, TypedDict, signatures de fonctions
et annotations de types en grammaires GBNF via pygbnf.

Exemples
--------
>>> from dataclasses import dataclass
>>> from pygbnf.schema import grammar_from_type
>>>
>>> @dataclass
... class Movie:
...     title: str
...     year: int
...     rating: float
...
>>> g = grammar_from_type(Movie)
>>> print(g.to_gbnf())
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import typing
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pygbnf import Grammar, CharacterClass, group, optional, select, one_or_more, repeat


# ── Helpers internes ─────────────────────────────────────────────────

def _is_optional(tp: Any) -> Tuple[bool, Any]:
    """Détecte Optional[X] (= Union[X, None]) et retourne (True, X)."""
    origin = get_origin(tp)
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1 and type(None) in get_args(tp):
            return True, args[0]
    return False, tp


def _is_list(tp: Any) -> Tuple[bool, Any]:
    """Détecte list[X] / List[X]."""
    origin = get_origin(tp)
    if origin is list:
        args = get_args(tp)
        return True, args[0] if args else Any
    return False, tp


def _is_dict(tp: Any) -> Tuple[bool, Any, Any]:
    """Détecte dict[K, V] / Dict[K, V]."""
    origin = get_origin(tp)
    if origin is dict:
        args = get_args(tp)
        k = args[0] if len(args) > 0 else str
        v = args[1] if len(args) > 1 else Any
        return True, k, v
    return False, None, None


def _is_literal(tp: Any) -> Tuple[bool, Tuple[Any, ...]]:
    """Détecte Literal['a', 'b', 'c']."""
    origin = get_origin(tp)
    if origin is Literal:
        return True, get_args(tp)
    return False, ()


def _is_enum(tp: Any) -> bool:
    """Détecte une sous-classe de enum.Enum."""
    return isinstance(tp, type) and issubclass(tp, enum.Enum)


def _is_dataclass(tp: Any) -> bool:
    return dataclasses.is_dataclass(tp) and isinstance(tp, type)


# ── Construction de grammaire ────────────────────────────────────────


class SchemaCompiler:
    """Compile des types Python en règles pygbnf.

    Parcourt récursivement les annotations de type et crée
    les règles GBNF correspondantes (format JSON).
    """

    def __init__(self, grammar: Grammar | None = None) -> None:
        self.g = grammar or Grammar()
        self._registered: set[str] = set()

    # ── Primitives JSON ──────────────────────────────────────────

    def _ensure_json_primitives(self) -> None:
        """Enregistre les règles de base (ws, string, number, bool, null)."""
        if "ws" in self._registered:
            return

        g = self.g

        @g.rule
        def ws():
            return repeat(select(" \t\n\r"), 0, 20)

        @g.rule
        def json_string():
            return '"' + repeat(
                select([
                    CharacterClass(pattern='^"\\\\'),
                    "\\" + select('"\\\\bfnrt/'),
                ]),
                0, 500,
            ) + '"'

        @g.rule
        def json_int():
            return optional("-") + select([
                "0",
                CharacterClass(pattern="1-9") + repeat(CharacterClass(pattern="0-9"), 0, 18),
            ])

        @g.rule
        def json_float():
            return json_int() + optional(group(
                "." + one_or_more(CharacterClass(pattern="0-9"))
            )) + optional(group(
                select("eE") + optional(select("+-")) + one_or_more(CharacterClass(pattern="0-9"))
            ))

        @g.rule
        def json_bool():
            return select(["true", "false"])

        @g.rule
        def json_null():
            return "null"

        self._registered.update(["ws", "json_string", "json_int", "json_float",
                                  "json_bool", "json_null"])

    # ── Dispatch par type ────────────────────────────────────────

    def _type_to_node(self, tp: Any, name_hint: str = "") -> Any:
        """Convertit un type Python en nœud AST pygbnf.

        Retourne soit une référence de règle, soit un nœud en ligne.
        """
        self._ensure_json_primitives()
        g = self.g

        # --- Optional[X] → X | null ---
        is_opt, inner = _is_optional(tp)
        if is_opt:
            inner_node = self._type_to_node(inner, name_hint)
            return select([inner_node, g.ref("json_null")])

        # --- Literal['a', 'b'] → "a" | "b" ---
        is_lit, values = _is_literal(tp)
        if is_lit:
            return select([f'"{v}"' if isinstance(v, str) else str(v) for v in values])

        # --- Enum → named sub-rule with alternatives ---
        if _is_enum(tp):
            rule_name = tp.__name__.lstrip("_")
            if rule_name not in self._registered:
                self._registered.add(rule_name)
                members = [f'"{m.value}"' for m in tp]

                @g.rule_named(rule_name)
                def _enum_rule(_m=members):
                    return select(_m)

            return g.ref(rule_name)

        # --- list[X] → "[" X ("," X)* "]" ---
        is_lst, item_tp = _is_list(tp)
        if is_lst:
            rule_name = f"{name_hint}_items" if name_hint else "list_items"
            item_node = self._type_to_node(item_tp, f"{name_hint}_item")

            # Cap list size to enum member count when items are Enum
            max_extra = 50
            if _is_enum(item_tp):
                max_extra = len(item_tp) - 1  # first item is mandatory inside optional()

            if rule_name not in self._registered:
                self._registered.add(rule_name)

                @g.rule_named(rule_name)
                def _list_rule(_it=item_node, _max=max_extra):
                    return "[" + g.ref("ws") + optional(group(
                        _it + repeat(group(
                            g.ref("ws") + "," + g.ref("ws") + _it
                        ), 0, _max)
                    )) + g.ref("ws") + "]"

            return g.ref(rule_name)

        # --- dict[str, V] → objet JSON libre ---
        is_dct, key_tp, val_tp = _is_dict(tp)
        if is_dct:
            rule_name = f"{name_hint}_dict" if name_hint else "json_dict"
            val_node = self._type_to_node(val_tp, f"{name_hint}_val")

            if rule_name not in self._registered:
                self._registered.add(rule_name)

                @g.rule_named(rule_name)
                def _dict_rule(_v=val_node):
                    pair = g.ref("json_string") + g.ref("ws") + ":" + g.ref("ws") + _v
                    return "{" + g.ref("ws") + optional(group(
                        pair + repeat(group(
                            g.ref("ws") + "," + g.ref("ws") + pair
                        ), 0, 50)
                    )) + g.ref("ws") + "}"

            return g.ref(rule_name)

        # --- Union[A, B, C] (non-Optional) ---
        origin = get_origin(tp)
        if origin is Union:
            args = get_args(tp)
            return select([self._type_to_node(a, name_hint) for a in args])

        # --- Dataclass → objet JSON avec champs fixes ---
        if _is_dataclass(tp):
            return self._dataclass_to_rule(tp)

        # --- Primitives ---
        if tp is str:
            return g.ref("json_string")
        if tp is int:
            return g.ref("json_int")
        if tp is float:
            return g.ref("json_float")
        if tp is bool:
            return g.ref("json_bool")
        if tp is type(None):
            return g.ref("json_null")
        if tp is Any:
            # Fallback : accepter string/number/bool/null
            return select([
                g.ref("json_string"),
                g.ref("json_float"),
                g.ref("json_bool"),
                g.ref("json_null"),
            ])

        msg = f"Type non supporté : {tp}"
        raise TypeError(msg)

    # ── Dataclass → règle JSON ───────────────────────────────────

    @staticmethod
    def _field_has_default(f: dataclasses.Field) -> bool:
        """Un champ a-t-il une valeur par défaut (default ou default_factory) ?"""
        return (f.default is not dataclasses.MISSING
                or f.default_factory is not dataclasses.MISSING)

    def _dataclass_to_rule(self, cls: type) -> Any:
        """Génère une règle pour un objet JSON à partir d'une dataclass.

        Les champs sans valeur par défaut sont obligatoires (toujours présents).
        Les champs avec ``default`` ou ``default_factory`` sont optionnels
        et peuvent être omis du JSON.  L'ordre est fixe : d'abord les champs
        obligatoires, puis les optionnels imbriqués avec ``optional()``.

        Exemples de sortie pour ``Config(host: str, port: int, debug: bool = False, timeout: Optional[int] = None)`` ::

            { "host": ..., "port": ... }                              # minimum
            { "host": ..., "port": ..., "debug": ... }                # +debug
            { "host": ..., "port": ..., "debug": ..., "timeout": ... }# +timeout
        """
        rule_name = cls.__name__.lstrip("_")

        if rule_name in self._registered:
            return self.g.ref(rule_name)

        self._registered.add(rule_name)
        hints = get_type_hints(cls)
        all_fields = dataclasses.fields(cls)
        ws = self.g.ref("ws")

        # Séparer champs obligatoires / optionnels
        required_fields = []
        optional_fields = []
        for f in all_fields:
            if self._field_has_default(f):
                optional_fields.append(f)
            else:
                required_fields.append(f)

        def _make_field_node(f):
            ft = hints[f.name]
            val_node = self._type_to_node(ft, name_hint=f"{rule_name}_{f.name}")
            return f'"{f.name}"' + ws + ":" + ws + val_node

        # -- Construire la partie obligatoire --------------------------
        req_nodes = [_make_field_node(f) for f in required_fields]

        if not req_nodes:
            body = ws
        elif len(req_nodes) == 1:
            body = ws + req_nodes[0]
        else:
            body = ws + req_nodes[0]
            for rn in req_nodes[1:]:
                body = body + ws + "," + ws + rn

        # -- Champs optionnels : imbrication inversée ------------------
        # On construit de droite à gauche :
        #   optional("," + fieldN)
        # puis :
        #   optional("," + fieldN-1 + ^ci-dessus)
        # etc.
        if optional_fields:
            opt_tail = None  # terme le plus à droite
            for f in reversed(optional_fields):
                fn = _make_field_node(f)
                if opt_tail is None:
                    opt_tail = optional(group(ws + "," + ws + fn))
                else:
                    opt_tail = optional(group(ws + "," + ws + fn + opt_tail))

            body = body + opt_tail

        body = body + ws

        @self.g.rule_named(rule_name)
        def _dc_rule(_b=body):
            return "{" + _b + "}"

        return self.g.ref(rule_name)

    # ── API publique ─────────────────────────────────────────────

    def compile(self, tp: type, *, start_name: str | None = None) -> Grammar:
        """Compile un type en grammaire complète.

        Parameters
        ----------
        tp : type
            Le type racine (dataclass, primitif, etc.)
        start_name : str, optional
            Nom de la règle de départ. Par défaut ``root``.
        """
        root_name = start_name or "root"
        root_node = self._type_to_node(tp, name_hint=root_name)

        @self.g.rule_named(root_name)
        def _root(_n=root_node):
            return _n

        self.g.start(root_name)
        return self.g


# ── Fonctions de convenance ──────────────────────────────────────────


def grammar_from_type(tp: type, **kwargs: Any) -> Grammar:
    """Génère une grammaire GBNF à partir d'un type Python.

    Parameters
    ----------
    tp : type
        Dataclass, primitif, ou type annoté.

    Returns
    -------
    Grammar
        Grammaire prête à compiler via ``to_gbnf()``.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Point:
    ...     x: float
    ...     y: float
    >>> g = grammar_from_type(Point)
    >>> print(g.to_gbnf())
    """
    compiler = SchemaCompiler()
    return compiler.compile(tp, **kwargs)


def grammar_from_function(func: Any, **kwargs: Any) -> Grammar:
    """Génère une grammaire GBNF à partir du type retour d'une fonction.

    La fonction doit avoir une annotation de retour.

    Parameters
    ----------
    func : callable
        Fonction avec annotation ``-> ReturnType``.

    Returns
    -------
    Grammar

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Result:
    ...     answer: str
    ...     confidence: float
    >>> def process(query: str) -> Result: ...
    >>> g = grammar_from_function(process)
    """
    hints = get_type_hints(func)
    ret = hints.get("return")
    if ret is None:
        msg = f"La fonction {func.__name__!r} n'a pas d'annotation de retour"
        raise TypeError(msg)
    return grammar_from_type(ret, **kwargs)


def grammar_from_args(func: Any, **kwargs: Any) -> Grammar:
    """Génère une grammaire GBNF correspondant aux arguments d'une fonction.

    Crée un objet JSON dont les clés sont les noms des paramètres
    et les valeurs correspondent aux types annotés.

    Parameters
    ----------
    func : callable
        Fonction avec des paramètres annotés.

    Returns
    -------
    Grammar

    Examples
    --------
    >>> def search(query: str, max_results: int = 10) -> None: ...
    >>> g = grammar_from_args(search)
    >>> print(g.to_gbnf())  # { "query": ..., "max_results": ... }
    """
    hints = get_type_hints(func)
    sig = inspect.signature(func)

    # Créer une dataclass dynamique à partir de la signature
    dc_fields: list[tuple[str, type]] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        tp = hints.get(name, Any)
        dc_fields.append((name, tp))

    # Créer un dataclass dynamique
    ArgsClass = dataclasses.make_dataclass(
        f"{func.__name__}_args",
        [(n, t) for n, t in dc_fields],
    )

    return grammar_from_type(ArgsClass, **kwargs)


def grammar_from_tool_call(func: Any, **kwargs: Any) -> Grammar:
    """Generate a GBNF grammar for a complete tool-call JSON object.

    Produces::

        {"function": "<func_name>", "arguments": {<args>}}

    Parameters
    ----------
    func : callable
        A function with annotated parameters.

    Returns
    -------
    Grammar

    Examples
    --------
    >>> def search(query: str, limit: int = 10) -> None: ...
    >>> g = grammar_from_tool_call(search)
    >>> print(g.to_gbnf())  # {"function": "search", "arguments": {"query": ..., "limit": ...}}
    """
    g = Grammar(**kwargs)

    @g.rule
    def root():
        return g.from_tool_call(func)

    g.start("root")
    return g


def describe_tools(*funcs: Any) -> str:
    """Return a human-readable description of tool functions.

    Useful for building system prompts that list available tools.

    Parameters
    ----------
    *funcs : callable
        Functions with type annotations and docstrings.

    Returns
    -------
    str
        A formatted multi-line description.

    Examples
    --------
    >>> def search(query: str, limit: int = 10) -> str:
    ...     \"\"\"Search the web.\"\"\"  # noqa
    ...     ...
    >>> print(describe_tools(search))
    - search(query: str, limit: int = 10) — Search the web.
    """
    lines = []
    for fn in funcs:
        sig = inspect.signature(fn)
        params = []
        hints = get_type_hints(fn)
        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue
            tp = hints.get(name)
            # For Enum types, show choices instead of opaque type name
            if tp and _is_enum(tp):
                choices = "|".join(repr(m.value) for m in tp)
                tp_name = choices
            else:
                tp_name = getattr(tp, "__name__", str(tp)) if tp else "Any"
            if param.default is not inspect.Parameter.empty:
                default = param.default
                if isinstance(default, enum.Enum):
                    default_str = repr(default.value)
                else:
                    default_str = repr(default)
                params.append(f"{name}: {tp_name} = {default_str}")
            else:
                params.append(f"{name}: {tp_name}")
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        line = f"- {fn.__name__}({', '.join(params)})"
        if doc:
            line += f" \u2014 {doc}"
        lines.append(line)
    return "\n".join(lines)