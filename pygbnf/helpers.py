"""
pygbnf.helpers — Convenience grammar fragments for common patterns.

These helpers return :class:`~pygbnf.nodes.Node` sub-trees and can be used
directly inside ``@g.rule`` definitions or composed with other combinators.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re
from typing import Union

from .nodes import (
    Alternative,
    CharacterClass,
    Group,
    Literal,
    Node,
    Repeat,
    RuleReference,
    Sequence,
    _coerce,
    _tl,
)
from .combinators import one_or_more, zero_or_more, optional, repeat, select, group


_MAX_ENUMERATED_RANGE = 1000


# ---------------------------------------------------------------------------
# Whitespace
# ---------------------------------------------------------------------------

def WS(*, required: bool = False) -> Node:
    """Whitespace (spaces, tabs, newlines).

    ``WS()``          → ``[ \\t\\n]*``
    ``WS(required=True)`` → ``[ \\t\\n]+``
    """
    ws_class = CharacterClass(pattern=" \\t\\n")
    if required:
        return one_or_more(ws_class)
    return zero_or_more(ws_class)


def ws() -> Node:
    """Alias for ``WS()`` — optional whitespace."""
    return WS()


def ws_required() -> Node:
    """Alias for ``WS(required=True)`` — at least one whitespace character."""
    return WS(required=True)


# ---------------------------------------------------------------------------
# Common token patterns
# ---------------------------------------------------------------------------

def _labeled(node: Node, label: str) -> Node:
    """Attach a human-readable *label* to *node* for template display."""
    object.__setattr__(node, "_label", label)
    return node


def keyword(word: str) -> Literal:
    """A literal keyword.

    >>> keyword("return")   # → "return"
    """
    return Literal(word)


def identifier() -> Node:
    """A typical identifier: ``[a-zA-Z_][a-zA-Z0-9_]*``."""
    head = CharacterClass(pattern="a-zA-Z_")
    tail = zero_or_more(CharacterClass(pattern="a-zA-Z0-9_"))
    return _labeled(Sequence(children=[head, tail]), "identifier")


def number() -> Node:
    """An optionally-negative integer: ``"-"? [0-9]+``."""
    sign = optional(Literal("-"))
    digits = one_or_more(CharacterClass(pattern="0-9"))
    return _labeled(Sequence(children=[sign, digits]), "number")


def float_number() -> Node:
    """A floating-point number: ``"-"? [0-9]+ ("." [0-9]+)?``."""
    sign = optional(Literal("-"))
    integer_part = one_or_more(CharacterClass(pattern="0-9"))
    frac = optional(
        Group(child=Sequence(children=[
            Literal("."),
            one_or_more(CharacterClass(pattern="0-9")),
        ]))
    )
    return _labeled(Sequence(children=[sign, integer_part, frac]), "float")


def _digit_node(min_digit: int, max_digit: int) -> Node:
    if min_digit == max_digit:
        return Literal(str(min_digit))
    return CharacterClass(pattern=f"{min_digit}-{max_digit}")


def _any_digits(count: int) -> Node:
    if count <= 0:
        return Literal("")
    if count == 1:
        return CharacterClass(pattern="0-9")
    return Repeat(child=CharacterClass(pattern="0-9"), min=count, max=count)


def _concat_nodes(left: Node, right: Node) -> Node:
    if isinstance(left, Literal) and left.value == "":
        return right
    if isinstance(right, Literal) and right.value == "":
        return left
    return left + right


def _range_same_length(lo: str, hi: str) -> Node:
    if len(lo) != len(hi):
        raise ValueError("_range_same_length() requires equal-length bounds")
    if lo == hi:
        return Literal(lo)
    if len(lo) == 1:
        return _digit_node(int(lo), int(hi))
    if lo == "0" * len(lo) and hi == "9" * len(hi):
        return _any_digits(len(lo))

    if lo[0] == hi[0]:
        return _concat_nodes(Literal(lo[0]), _range_same_length(lo[1:], hi[1:]))

    first_digit = int(lo[0])
    last_digit = int(hi[0])
    rest_len = len(lo) - 1
    alts = [
        _concat_nodes(Literal(lo[0]), _range_same_length(lo[1:], "9" * rest_len))
    ]
    if first_digit + 1 <= last_digit - 1:
        alts.append(
            _concat_nodes(_digit_node(first_digit + 1, last_digit - 1), _any_digits(rest_len))
        )
    alts.append(
        _concat_nodes(Literal(hi[0]), _range_same_length("0" * rest_len, hi[1:]))
    )
    return select(alts)


def _full_positive_length(length: int) -> Node:
    if length == 1:
        return CharacterClass(pattern="1-9")
    return CharacterClass(pattern="1-9") + _any_digits(length - 1)


def _positive_int_range(min_value: int, max_value: int) -> Node:
    if min_value == max_value:
        return Literal(str(min_value))
    if min_value < 1 or max_value < min_value:
        raise ValueError("_positive_int_range() requires 1 <= min_value <= max_value")

    lo = str(min_value)
    hi = str(max_value)
    if lo == "1" + "0" * (len(lo) - 1) and hi == "9" * len(hi):
        return _full_positive_length(len(lo))
    if len(lo) == len(hi):
        return _range_same_length(lo, hi)

    alts = [_range_same_length(lo, "9" * len(lo))]
    for length in range(len(lo) + 1, len(hi)):
        alts.append(_full_positive_length(length))
    alts.append(_range_same_length("1" + "0" * (len(hi) - 1), hi))
    return select(alts)


def int_range(min_value: int, max_value: int) -> Node:
    """An inclusive integer range encoded as a compact digit grammar.

    Examples
    --------
    >>> int_range(1, 3)
    >>> int_range(-20, 20)
    >>> int_range(100, 999)
    """
    if isinstance(min_value, bool) or isinstance(max_value, bool):
        raise TypeError("int_range() does not accept bool values")
    if not isinstance(min_value, int) or not isinstance(max_value, int):
        raise TypeError("int_range() expects integer bounds")
    if min_value > max_value:
        raise ValueError("int_range() requires min_value <= max_value")

    if min_value == max_value:
        return _labeled(Literal(str(min_value)), f"int_range({min_value}, {max_value})")

    if min_value >= 0:
        node = Literal("0") if max_value == 0 else (
            _digit_node(min_value, max_value) if 0 <= min_value <= max_value <= 9
            else (
                select([Literal("0"), _positive_int_range(1, max_value)])
                if min_value == 0
                else _positive_int_range(min_value, max_value)
            )
        )
        return _labeled(node, f"int_range({min_value}, {max_value})")

    if max_value < 0:
        node = Literal("-") + _positive_int_range(abs(max_value), abs(min_value))
        return _labeled(node, f"int_range({min_value}, {max_value})")

    parts = [Literal("-") + _positive_int_range(1, abs(min_value)), Literal("0")]
    if max_value >= 1:
        parts.append(_positive_int_range(1, max_value))
    return _labeled(select(parts), f"int_range({min_value}, {max_value})")


def decimal_range(
    min_value: Union[int, float, str],
    max_value: Union[int, float, str],
    *,
    step: Union[int, float, str, None] = None,
    scale: int | None = None,
) -> Node:
    """An inclusive decimal range over a discrete step.

    Exactly one of `step` or `scale` must be provided.

    Examples
    --------
    >>> decimal_range(1.2, 1.5, scale=1)
    >>> decimal_range("0.00", "1.00", step="0.25")
    """

    def _to_decimal(value: Union[int, float, str]) -> Decimal:
        try:
            dec = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise TypeError(
                "decimal_range() expects numeric or decimal-string bounds"
            ) from exc
        if not dec.is_finite():
            raise ValueError("decimal_range() requires finite bounds")
        return dec

    if (step is None) == (scale is None):
        raise ValueError("decimal_range() requires exactly one of: step or scale")
    if scale is not None and scale < 0:
        raise ValueError("decimal_range() requires scale >= 0")

    lo = _to_decimal(min_value)
    hi = _to_decimal(max_value)
    if lo > hi:
        raise ValueError("decimal_range() requires min_value <= max_value")

    if scale is not None:
        step_dec = Decimal(1).scaleb(-scale)
        display_scale = scale
    else:
        step_dec = _to_decimal(step)  # type: ignore[arg-type]
        if step_dec <= 0:
            raise ValueError("decimal_range() requires step > 0")
        display_scale = max(
            0,
            -lo.as_tuple().exponent,
            -hi.as_tuple().exponent,
            -step_dec.as_tuple().exponent,
        )

    quotient = (hi - lo) / step_dec
    if quotient != quotient.to_integral_value():
        raise ValueError("decimal_range() requires bounds aligned with the chosen step")

    count = int(quotient) + 1
    if count > _MAX_ENUMERATED_RANGE:
        raise ValueError(
            f"decimal_range() would generate {count} alternatives; "
            f"maximum supported is {_MAX_ENUMERATED_RANGE}"
        )

    quant = Decimal(1).scaleb(-display_scale)
    values = []
    current = lo
    for _ in range(count):
        if display_scale == 0:
            text = str(current.quantize(quant))
        else:
            text = format(current.quantize(quant), f".{display_scale}f")
        values.append(text)
        current += step_dec

    return _labeled(
        select(values),
        f"decimal_range({min_value}, {max_value})",
    )

def string_literal(*, quote: str = '"') -> Node:
    """A double-quoted (or custom-quoted) string with escape support.

    Compiles to::

        "\"" ([^"\\\\] | "\\\\" ["\\\\nrt/bfu])* "\""
    """
    body_char = Alternative(alternatives=[
        CharacterClass(pattern=f'^{quote}\\\\'),  # chars except quote and backslash
        Sequence(children=[
            Literal("\\"),
            CharacterClass(pattern=f'{quote}\\\\/bfnrtu'),  # known escape targets
        ]),
    ])
    return _labeled(Sequence(children=[
        Literal(quote),
        zero_or_more(Group(child=body_char)),
        Literal(quote),
    ]), "string")


# ---------------------------------------------------------------------------
# Combinators for structure
# ---------------------------------------------------------------------------

def comma_list(item: Union[str, Node]) -> Node:
    """A comma-separated list of *item*: ``item ("," item)*``.

    >>> comma_list(identifier())
    """
    item = _coerce(item)
    return Sequence(children=[
        item,
        zero_or_more(Group(child=Sequence(children=[
            Literal(","),
            zero_or_more(Literal(" ")),
            item,
        ]))),
    ])


def spaced_comma_list(item: Union[str, Node]) -> Node:
    """Like :func:`comma_list` but with optional whitespace around commas."""
    item = _coerce(item)
    return Sequence(children=[
        item,
        zero_or_more(Group(child=Sequence(children=[
            WS(),
            Literal(","),
            WS(),
            item,
        ]))),
    ])


def between(left: Union[str, Node], expr: Union[str, Node], right: Union[str, Node]) -> Node:
    """Wrap *expr* between *left* and *right* delimiters.

    >>> between("(", expression(), ")")
    """
    return Sequence(children=[_coerce(left), _coerce(expr), _coerce(right)])


def separated_by(sep: Union[str, Node], item: Union[str, Node]) -> Node:
    """Alias for ``item (sep item)*``."""
    item = _coerce(item)
    sep = _coerce(sep)
    return Sequence(children=[
        item,
        zero_or_more(Group(child=Sequence(children=[sep, item]))),
    ])


def any_char() -> CharacterClass:
    """Match any single character (approximation via wide range).

    Compiles to ``.`` — but GBNF doesn't have ``.``; we use a broad class.
    In practice this is ``[^]`` which GBNF doesn't support either, so we
    use ``[\\x00-\\x7F]`` as a reasonable ASCII-range stand-in, or users
    should use a specific character class.
    """
    # GBNF uses [^\n] or similar; we'll provide a helper that covers
    # printable ASCII at minimum.  Users can replace with their own class.
    return CharacterClass(pattern="\\x00-\\x7F")


def line(prefix: str = "- ") -> Node:
    """A bullet-point line: *prefix* followed by free text.

    Does **not** append ``\\n`` — use inside :func:`T` which adds line
    endings automatically.

    >>> line()        # → "- " [^\\n]+
    >>> line("* ")    # → "* " [^\\n]+
    """
    lbl = f"line({prefix!r})" if prefix != "- " else "line"
    return _labeled(Sequence(children=[
        Literal(prefix),
        one_or_more(CharacterClass(pattern="^\\n")),
    ]), lbl)

# ---------------------------------------------------------------------------
# f-string template builder
# ---------------------------------------------------------------------------

_MARKER_RE = re.compile(r"\x00\x01(\d+)\x00")


def T(template: str) -> Node:
    """Build a grammar node from an f-string template.

    Use Python f-strings to embed :class:`~pygbnf.nodes.Node` expressions
    directly inside a text template.  Each line in the template becomes a
    sequence terminated by ``\\n``.

    Use format specs for line-level quantifiers:

    - ``{node:+}`` → the line is matched **one or more** times
    - ``{node:*}`` → the line is matched **zero or more** times
    - ``{node:?}`` → the line is matched **zero or one** time

    Examples
    --------
    ::

        free = one_or_more(CharacterClass(pattern="^\\n"))

        return T(f\"\"\"# Reformulation:
        - {free:+}
        # Structure:
        - {free:+}
        Traduction:
        \"\"\")
    """
    registry = getattr(_tl, "nodes", {})
    children = []

    # Reconstruct human-readable template
    raw_lines = template.strip("\n").split("\n")
    readable_lines = []

    for line in raw_lines:
        # Build readable version of this line
        rparts = _MARKER_RE.split(line)
        readable = ""
        for ri, rp in enumerate(rparts):
            if ri % 2 == 1:
                marker = f"\x00\x01{rp}\x00"
                if marker in registry:
                    _, spec, label = registry[marker]
                    readable += "{" + label + (":" + spec if spec else "") + "}"
                else:
                    readable += "{?}"
            else:
                readable += rp
        readable_lines.append(readable)

        # Split on markers, interleave Literal and Node
        parts = _MARKER_RE.split(line)
        line_children = []
        line_repeat = None  # quantifier from format spec
        for i, part in enumerate(parts):
            if i % 2 == 1:  # captured group = counter id
                marker = f"\x00\x01{part}\x00"
                if marker not in registry:
                    raise ValueError(
                        f"Unknown node marker (id={part}). "
                        "Make sure to call T() with an f-string."
                    )
                node, spec, _label = registry[marker]
                line_children.append(node)
                if spec:
                    line_repeat = spec
            elif part:
                line_children.append(Literal(part))

        line_children.append(Literal("\n"))
        ln = Sequence(children=line_children) if len(line_children) > 1 else line_children[0]

        if line_repeat == "+":
            children.append(ln)
            children.append(zero_or_more(ln))
        elif line_repeat == "*":
            children.append(zero_or_more(ln))
        elif line_repeat == "?":
            children.append(optional(ln))
        elif line_repeat is not None:
            # Numeric: "N", "N,M", or "N,"
            if "," in line_repeat:
                parts_q = line_repeat.split(",", 1)
                rmin = int(parts_q[0])
                rmax = int(parts_q[1]) if parts_q[1] else None
            else:
                rmin = int(line_repeat)
                rmax = rmin
            children.append(repeat(ln, min=rmin, max=rmax))
        else:
            children.append(ln)

    # Clean up registry
    _tl.nodes = {}
    _tl.counter = 0

    readable = "\n".join(readable_lines)

    if len(children) == 1:
        result = children[0]
    else:
        result = Sequence(children=children)

    object.__setattr__(result, "_template", readable)
    return result
