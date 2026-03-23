from __future__ import annotations

import dataclasses
import enum
from typing import Any, Literal, Tuple, Union, get_args, get_origin


def is_optional_type(tp: Any) -> Tuple[bool, Any]:
    """Detect Optional[X] (= Union[X, None]) and return (True, X)."""
    origin = get_origin(tp)
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1 and type(None) in get_args(tp):
            return True, args[0]
    return False, tp


def is_list_type(tp: Any) -> Tuple[bool, Any]:
    """Detect list[X] / List[X]."""
    origin = get_origin(tp)
    if origin is list:
        args = get_args(tp)
        return True, args[0] if args else Any
    return False, tp


def is_dict_type(tp: Any) -> Tuple[bool, Any, Any]:
    """Detect dict[K, V] / Dict[K, V]."""
    origin = get_origin(tp)
    if origin is dict:
        args = get_args(tp)
        key_type = args[0] if len(args) > 0 else str
        value_type = args[1] if len(args) > 1 else Any
        return True, key_type, value_type
    return False, None, None


def is_literal_type(tp: Any) -> Tuple[bool, Tuple[Any, ...]]:
    """Detect Literal['a', 'b', 'c']."""
    origin = get_origin(tp)
    if origin is Literal:
        return True, get_args(tp)
    return False, ()


def is_enum_type(tp: Any) -> bool:
    """Detect enum.Enum subclasses."""
    return isinstance(tp, type) and issubclass(tp, enum.Enum)


def is_dataclass_type(tp: Any) -> bool:
    """Detect dataclass types."""
    return dataclasses.is_dataclass(tp) and isinstance(tp, type)
