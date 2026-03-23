from __future__ import annotations

import dataclasses
from typing import Any, get_args, get_origin, get_type_hints, Union

from .combinators import group, one_or_more, optional, repeat, select
from .grammar import Grammar
from .nodes import CharacterClass
from ._schema_type_utils import (
    is_dataclass_type,
    is_dict_type,
    is_enum_type,
    is_list_type,
    is_literal_type,
    is_optional_type,
)


class SchemaCompiler:
    """Compile Python types into pygbnf rules."""

    def __init__(self, grammar: Grammar | None = None) -> None:
        self.g = grammar or Grammar()
        self._registered: set[str] = set()

    def _ensure_json_primitives(self) -> None:
        """Register the primitive JSON rules once."""
        if "ws" in self._registered:
            return

        g = self.g

        @g.rule
        def ws():
            return repeat(select(" \t\n\r"), 0, 20)

        @g.rule
        def json_string():
            return '"' + repeat(
                select(
                    [
                        CharacterClass(pattern='^"\\\\'),
                        "\\" + select('"\\\\bfnrt/'),
                    ]
                ),
                0,
                500,
            ) + '"'

        @g.rule
        def json_int():
            return optional("-") + select(
                [
                    "0",
                    CharacterClass(pattern="1-9")
                    + repeat(CharacterClass(pattern="0-9"), 0, 18),
                ]
            )

        @g.rule
        def json_float():
            return json_int() + optional(
                group("." + one_or_more(CharacterClass(pattern="0-9")))
            ) + optional(
                group(
                    select("eE")
                    + optional(select("+-"))
                    + one_or_more(CharacterClass(pattern="0-9"))
                )
            )

        @g.rule
        def json_bool():
            return select(["true", "false"])

        @g.rule
        def json_null():
            return "null"

        self._registered.update(
            ["ws", "json_string", "json_int", "json_float", "json_bool", "json_null"]
        )

    def _type_to_node(self, tp: Any, name_hint: str = "") -> Any:
        """Convert a Python type into a pygbnf AST node or rule reference."""
        self._ensure_json_primitives()
        g = self.g

        is_opt, inner = is_optional_type(tp)
        if is_opt:
            inner_node = self._type_to_node(inner, name_hint)
            return select([inner_node, g.ref("json_null")])

        is_lit, values = is_literal_type(tp)
        if is_lit:
            return select([f'"{v}"' if isinstance(v, str) else str(v) for v in values])

        if is_enum_type(tp):
            rule_name = tp.__name__.lstrip("_")
            if rule_name not in self._registered:
                self._registered.add(rule_name)
                members = [f'"{m.value}"' for m in tp]

                @g.rule_named(rule_name)
                def _enum_rule(_m=members):
                    return select(_m)

            return g.ref(rule_name)

        is_lst, item_tp = is_list_type(tp)
        if is_lst:
            rule_name = f"{name_hint}_items" if name_hint else "list_items"
            item_node = self._type_to_node(item_tp, f"{name_hint}_item")

            max_extra = 50
            if is_enum_type(item_tp):
                max_extra = len(item_tp) - 1

            if rule_name not in self._registered:
                self._registered.add(rule_name)

                @g.rule_named(rule_name)
                def _list_rule(_it=item_node, _max=max_extra):
                    return "[" + g.ref("ws") + optional(
                        group(
                            _it
                            + repeat(
                                group(g.ref("ws") + "," + g.ref("ws") + _it), 0, _max
                            )
                        )
                    ) + g.ref("ws") + "]"

            return g.ref(rule_name)

        is_dct, _key_tp, val_tp = is_dict_type(tp)
        if is_dct:
            rule_name = f"{name_hint}_dict" if name_hint else "json_dict"
            val_node = self._type_to_node(val_tp, f"{name_hint}_val")

            if rule_name not in self._registered:
                self._registered.add(rule_name)

                @g.rule_named(rule_name)
                def _dict_rule(_v=val_node):
                    pair = g.ref("json_string") + g.ref("ws") + ":" + g.ref("ws") + _v
                    return "{" + g.ref("ws") + optional(
                        group(
                            pair
                            + repeat(
                                group(g.ref("ws") + "," + g.ref("ws") + pair), 0, 50
                            )
                        )
                    ) + g.ref("ws") + "}"

            return g.ref(rule_name)

        origin = get_origin(tp)
        if origin is Union:
            args = get_args(tp)
            return select([self._type_to_node(a, name_hint) for a in args])

        if is_dataclass_type(tp):
            return self._dataclass_to_rule(tp)

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
            return select(
                [g.ref("json_string"), g.ref("json_float"), g.ref("json_bool"), g.ref("json_null")]
            )

        raise TypeError(f"Type non supporté : {tp}")

    @staticmethod
    def _field_has_default(field: dataclasses.Field) -> bool:
        return (
            field.default is not dataclasses.MISSING
            or field.default_factory is not dataclasses.MISSING
        )

    def _dataclass_to_rule(self, cls: type) -> Any:
        rule_name = cls.__name__.lstrip("_")

        if rule_name in self._registered:
            return self.g.ref(rule_name)

        self._registered.add(rule_name)
        hints = get_type_hints(cls)
        all_fields = dataclasses.fields(cls)
        ws = self.g.ref("ws")

        required_fields = []
        optional_fields = []
        for field in all_fields:
            if self._field_has_default(field):
                optional_fields.append(field)
            else:
                required_fields.append(field)

        def _make_field_node(field: dataclasses.Field):
            field_type = hints[field.name]
            value_node = self._type_to_node(field_type, name_hint=f"{rule_name}_{field.name}")
            return f'"{field.name}"' + ws + ":" + ws + value_node

        req_nodes = [_make_field_node(field) for field in required_fields]

        if not req_nodes:
            body = ws
        elif len(req_nodes) == 1:
            body = ws + req_nodes[0]
        else:
            body = ws + req_nodes[0]
            for required_node in req_nodes[1:]:
                body = body + ws + "," + ws + required_node

        if optional_fields:
            opt_tail = None
            for field in reversed(optional_fields):
                field_node = _make_field_node(field)
                if opt_tail is None:
                    opt_tail = optional(group(ws + "," + ws + field_node))
                else:
                    opt_tail = optional(group(ws + "," + ws + field_node + opt_tail))
            body = body + opt_tail

        body = body + ws

        @self.g.rule_named(rule_name)
        def _dc_rule(_b=body):
            return "{" + _b + "}"

        return self.g.ref(rule_name)

    def compile(self, tp: type, *, start_name: str | None = None) -> Grammar:
        """Compile a root type into a complete grammar."""
        root_name = start_name or "root"
        root_node = self._type_to_node(tp, name_hint=root_name)

        @self.g.rule_named(root_name)
        def _root(_n=root_node):
            return _n

        self.g.start(root_name)
        return self.g
