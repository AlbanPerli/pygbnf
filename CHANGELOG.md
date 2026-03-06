# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] — 2026-03-06

### Added
- `schema` module — auto-generate grammars from Python types, dataclasses, and function signatures
  - `grammar_from_type()`, `grammar_from_function()`, `grammar_from_args()`
  - `Grammar.from_type()`, `Grammar.from_function_return()`, `Grammar.from_function_args()` for composition
  - Optional dataclass fields (fields with `default` / `default_factory`) are properly omissible
- `WS()`, `ws_required()` whitespace helpers
- `float_number()` helper
- `spaced_comma_list()` helper
- Bounded `repeat(item, min, max)` combinator
- `token()`, `token_id()`, `not_token()`, `not_token_id()` for llama.cpp token-level constraints
- Left-recursion detection with `Grammar.detect_left_recursion()`
- Dependency graph with `Grammar.dependency_graph()`
- Pretty-print with `Grammar.pretty_print()`

### Fixed
- Character class dash rendering (`[+-]` instead of `[+\-]`)
- Multi-line alternative formatting (trailing `|`)
- Root rule duplication when start name equals `"root"`

## [0.1.0] — 2026-03-01

### Added
- Initial release
- Core AST nodes: `Literal`, `CharacterClass`, `Sequence`, `Alternative`, `Repeat`, `RuleReference`, `TokenReference`, `Group`, `Optional_`
- DSL combinators: `select()`, `one_or_more()`, `zero_or_more()`, `optional()`, `group()`
- Grammar container with `@g.rule` decorator and named rules
- GBNF code generation with 5 optimization passes
- Prebuilt helpers: `identifier()`, `number()`, `string_literal()`, `comma_list()`, `between()`, `separated_by()`, `keyword()`
