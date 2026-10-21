# AGENTS.md

Instructions for AI coding agents (and a quick reference for people) working in `qccompute`. CONTRIBUTING.md explains the same workflow in more detail.

## This package

`qccompute` only executes: it resolves a `qccodec.Codec`, runs the native invocation (`run_subprocess` or `run_python`), decodes, collects files, and builds the `ProgramOutput`. It must never branch on a program name or contain program-specific logic; that belongs in `qccodec`.

- Failures keep partial results: decoding always runs, exceptions carry the completed `ProgramOutput`, and the first failure stays primary, except that a codec's explanation of a plain nonzero exit (`qccodec.CalculationError`) leads, with the exit status kept as an additional failure.
- Tests use stand-in programs. Integration tests with real programs live in qccodec (`tests/integration/`), which uses qccompute as a test-only dependency.

## The qc* suite

Dependencies point one way: qcref -> qcdata -> {qcinf, qccodec} -> qccompute. Never import a package that sits later in that order, and keep changes that span packages in separate pull requests, released in that order.

## Commands

```sh
uv sync --all-extras --all-groups     # install everything
uv run pre-commit install             # once per clone: run the checks on every commit
uv run pytest                         # tests
uv run ruff check --fix . && uv run ruff format .
uv run mypy
uv run pre-commit run --all-files     # exactly what CI checks
```

Use `uv add` / `uv remove` for dependencies; never edit `uv.lock` by hand.

## Code conventions

- Use type annotations for all function and method parameters (except `self` and `cls`) and return values, including `-> None`. mypy enforces this for library code.
- Annotate class and instance attributes, including dataclass and pydantic fields. Annotate local variables only when inference is insufficient.
- Prefer precise types; avoid `Any` unless genuinely necessary.
- Write Google-style docstrings for all public modules, classes, functions, and methods (ruff enforces this). Start with a one-line summary ending in a period; add Args, Returns, Yields, and Raises sections where applicable; describe meaning and units without repeating types already in annotations.
- Use `X | None`, not `Optional[X]`. Target Python 3.11.
- Keep changes focused. Do not reformat or refactor unrelated code.

## Before you finish

- `uv run pre-commit run --all-files` and `uv run pytest` pass.
- New behavior has tests; bug fixes have a test that failed before the fix.
- User-visible changes are listed under `## [unreleased]` in CHANGELOG.md.
- `pyproject.toml` has no `[tool.uv.sources]` path or git entries for qc* packages (CI rejects them).
- Do not commit, push, tag, or release unless asked to.
