# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] — 2026-03-13

### Added
- `weighted_select()` combinator — like `select()` but with probability weights that bias the LLM towards preferred alternatives
- `WeightedAlternative` AST node — stores per-branch weights, transparent in GBNF output (grammar stays a binary filter)
- `GrammarLLM.compute_logit_bias(grammar)` — walks the AST, tokenizes alternatives **in context** (BPE-aware), and returns a `logit_bias` dict
- `GrammarLLM.tokenize(text)` — calls the server's `/tokenize` endpoint
- `bias_scale` parameter on `compute_logit_bias()` (default `10.0`) to control bias magnitude
- `logit_bias` is automatically injected into `stream()` and `complete()` when the grammar contains weighted alternatives
- New example: `demo_weighted_select.py` — restaurant chatbot that steers dish suggestions with weights

### Fixed
- Context-aware tokenization: `compute_logit_bias` now resolves `RuleReference` nodes and accumulates preceding literal text before tokenizing, so BPE merges (e.g. `" red"` vs `"red"`) produce the correct token IDs

## [0.3.1] — 2026-03-12

### Added
- `expand_depth` parameter on all visualization functions (`grammar_to_nfa_dot`, `write_grammar_dot`, `write_grammar_svg`, `grammar_rule_to_nfa_dot`, `write_rule_dot`)
  - Limits recursive rule expansion depth to prevent combinatorial DOT blowup
  - Multi-rule functions default to `expand_depth=1` (one level); single-rule functions default to `None` (unlimited)

### Fixed
- Visualization: deeply nested grammars no longer produce enormous DOT files (e.g. 24 000 → 1 300 lines for a 10-rule equation grammar)

## [0.3.0] — 2026-03-11

### Added
- `visualization` module — export grammars as NFA diagrams in DOT / SVG format
  - `grammar_rule_to_nfa_dot()`, `grammar_to_nfa_dot()` — generate Graphviz DOT strings
  - `write_rule_dot()`, `write_grammar_dot()` — write `.dot` files
  - `render_dot_to_svg()`, `write_grammar_svg()` — render to SVG (requires Graphviz CLI)
  - `get_user_rules()` — auto-detect user-defined rules (excludes infrastructure rules)
  - When `rule_names=None` (default), visualization functions automatically select user rules
- `GrammarLLM` — unified OpenAI-compatible LLM client with grammar-constrained streaming
- `Toolkit` — decorator-based tool registry with grammar-constrained calling and auto-dispatch
- `Grammar.from_tool_call()` — inject rules for a complete tool-call JSON object
- `grammar_from_tool_call()`, `describe_tools()` in `schema` module
- New example: `demo_visualization.py`

### Changed
- Optional dependencies declared in `pyproject.toml` (`pip install pygbnf[llm]`, `pip install pygbnf[all]`)

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
