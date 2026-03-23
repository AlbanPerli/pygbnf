from __future__ import annotations

import dataclasses
import enum
import inspect
from typing import Any, get_type_hints

from .grammar import Grammar
from ._schema_compiler import SchemaCompiler
from ._schema_type_utils import is_enum_type


def grammar_from_type(tp: type, **kwargs: Any) -> Grammar:
    """Generate a GBNF grammar from a Python type."""
    compiler = SchemaCompiler()
    return compiler.compile(tp, **kwargs)


def grammar_from_function(func: Any, **kwargs: Any) -> Grammar:
    """Generate a GBNF grammar from a function return type."""
    hints = get_type_hints(func)
    ret = hints.get("return")
    if ret is None:
        raise TypeError(f"La fonction {func.__name__!r} n'a pas d'annotation de retour")
    return grammar_from_type(ret, **kwargs)


def grammar_from_args(func: Any, **kwargs: Any) -> Grammar:
    """Generate a GBNF grammar for a function's arguments as a JSON object."""
    hints = get_type_hints(func)
    sig = inspect.signature(func)

    dc_fields: list[tuple[str, type]] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        tp = hints.get(name, Any)
        dc_fields.append((name, tp))

    args_class = dataclasses.make_dataclass(
        f"{func.__name__}_args",
        [(name, tp) for name, tp in dc_fields],
    )

    return grammar_from_type(args_class, **kwargs)


def grammar_from_tool_call(func: Any, **kwargs: Any) -> Grammar:
    """Generate a GBNF grammar for a complete tool-call JSON object."""
    g = Grammar(**kwargs)

    @g.rule
    def root():
        return g.from_tool_call(func)

    g.start("root")
    return g


def describe_tools(*funcs: Any) -> str:
    """Return a human-readable description of tool functions."""
    lines = []
    for fn in funcs:
        sig = inspect.signature(fn)
        params = []
        hints = get_type_hints(fn)
        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue
            tp = hints.get(name)
            if tp and is_enum_type(tp):
                choices = "|".join(repr(member.value) for member in tp)
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
            line += f" — {doc}"
        lines.append(line)
    return "\n".join(lines)
