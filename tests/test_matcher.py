#!/usr/bin/env python3
"""Tests for pygbnf.matcher — GrammarMatcher and RuleEvent."""

import pygbnf as cfg
from pygbnf import (
    Grammar,
    GrammarMatcher,
    RuleEvent,
    CharacterClass,
    select,
    one_or_more,
    zero_or_more,
    optional,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _build_arithmetic_grammar() -> Grammar:
    """Build a simple arithmetic grammar for testing."""
    g = Grammar()

    @g.rule
    def ws():
        return zero_or_more(select(" \t"))

    @g.rule
    def digit():
        return CharacterClass(pattern="0-9")

    @g.rule
    def integer():
        return optional("-") + one_or_more(digit())

    @g.rule
    def operator():
        return select(["+", "-", "*", "/"])

    @g.rule
    def expression():
        return integer() + zero_or_more(
            cfg.group(ws() + operator() + ws() + integer())
        )

    @g.rule
    def result():
        return expression() + ws() + "=" + ws() + integer()

    g.start("result")
    # Force build by compiling
    g.to_gbnf()
    return g


# ── Tests ────────────────────────────────────────────────────────────

class TestGrammarMatcherBasic:
    """Basic matcher construction and properties."""

    def test_create_matcher(self):
        g = _build_arithmetic_grammar()
        m = GrammarMatcher(g)
        assert m.buffer == ""

    def test_feed_accumulates_buffer(self):
        g = _build_arithmetic_grammar()
        m = GrammarMatcher(g)
        m.feed("hello ")
        m.feed("world")
        assert m.buffer == "hello world"

    def test_reset_clears_buffer(self):
        g = _build_arithmetic_grammar()
        m = GrammarMatcher(g)
        m.feed("something")
        m.reset()
        assert m.buffer == ""

    def test_feed_empty_string(self):
        g = _build_arithmetic_grammar()
        m = GrammarMatcher(g)
        events = m.feed("")
        assert events == []
        assert m.buffer == ""


class TestRuleEvents:
    """Verify that rule events are emitted with correct names and text."""

    def test_integer_matched(self):
        g = _build_arithmetic_grammar()
        m = GrammarMatcher(g)
        events = m.feed("42")
        rule_names = {e.rule for e in events}
        assert "integer" in rule_names

    def test_expression_matched(self):
        g = _build_arithmetic_grammar()
        m = GrammarMatcher(g)
        events = m.feed("11 + 12")
        rule_names = {e.rule for e in events}
        assert "expression" in rule_names

    def test_result_matched(self):
        g = _build_arithmetic_grammar()
        m = GrammarMatcher(g)
        events = m.feed("11 + 12 = 23")
        rule_names = {e.rule for e in events}
        assert "result" in rule_names

    def test_event_text_is_correct(self):
        g = _build_arithmetic_grammar()
        m = GrammarMatcher(g)
        events = m.feed("3 * 7 = 21")
        result_events = [e for e in events if e.rule == "result"]
        assert len(result_events) >= 1
        assert result_events[0].text == "3 * 7 = 21"

    def test_event_uses_python_name_not_gbnf(self):
        """Rule names in events should be snake_case Python names."""
        g = Grammar()

        @g.rule
        def my_number():
            return one_or_more(CharacterClass(pattern="0-9"))

        @g.rule
        def root():
            return my_number()

        g.start("root")
        g.to_gbnf()

        m = GrammarMatcher(g)
        events = m.feed("123")
        rule_names = {e.rule for e in events}
        # Should be "my_number" not "my-number"
        assert "my_number" in rule_names

    def test_incremental_feed(self):
        """Events should fire once text is complete, even if fed char by char."""
        g = _build_arithmetic_grammar()
        m = GrammarMatcher(g)
        all_events = []
        for ch in "5 + 3 = 8":
            all_events.extend(m.feed(ch))
        rule_names = {e.rule for e in all_events}
        assert "result" in rule_names


class TestCallbacks:
    """Test the on() callback mechanism."""

    def test_specific_rule_callback(self):
        g = _build_arithmetic_grammar()
        m = GrammarMatcher(g)
        received = []
        m.on("result", lambda ev: received.append(ev))
        m.feed("10 + 5 = 15")
        assert len(received) >= 1
        assert received[0].rule == "result"

    def test_wildcard_callback(self):
        g = _build_arithmetic_grammar()
        m = GrammarMatcher(g)
        received = []
        m.on("*", lambda ev: received.append(ev))
        m.feed("42 = 42")
        assert len(received) > 0

    def test_no_duplicate_events(self):
        """The same match should not fire twice."""
        g = _build_arithmetic_grammar()
        m = GrammarMatcher(g)
        received = []
        m.on("result", lambda ev: received.append(ev))
        m.feed("1 + 2 = 3")
        count1 = len(received)
        # Feed more — the old match should not re-fire
        m.feed(" done")
        count2 = len([e for e in received if e.text == "1 + 2 = 3"])
        assert count2 == count1

    def test_multiple_callbacks_same_rule(self):
        g = _build_arithmetic_grammar()
        m = GrammarMatcher(g)
        a, b = [], []
        m.on("integer", lambda ev: a.append(ev))
        m.on("integer", lambda ev: b.append(ev))
        m.feed("99")
        assert len(a) >= 1
        assert len(b) >= 1


class TestRuleEventDataclass:
    """Verify RuleEvent fields."""

    def test_fields(self):
        ev = RuleEvent(rule="foo", text="bar", start=0, end=3)
        assert ev.rule == "foo"
        assert ev.text == "bar"
        assert ev.start == 0
        assert ev.end == 3
        assert ev.fn is None
        assert ev.doc == ""

    def test_fn_and_doc(self):
        def my_rule():
            """A doc."""
            pass
        ev = RuleEvent(rule="my_rule", text="x", start=0, end=1, fn=my_rule)
        assert ev.fn is my_rule
        assert ev.doc == "A doc."

    def test_frozen(self):
        ev = RuleEvent(rule="foo", text="bar", start=0, end=3)
        try:
            ev.rule = "baz"  # type: ignore
            assert False, "Should raise"
        except AttributeError:
            pass
