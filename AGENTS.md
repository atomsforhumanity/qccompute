# AGENTS.md

Instructions for AI coding agents (and a quick reference for people) working in `qccompute`. CONTRIBUTING.md explains the same workflow in more detail.

## This package

`qccompute` runs programs: the Program protocol (`qccompute.program`), the `Runner` (`qccompute.runner`: finding, planning, placing and running programs, resources, lists, sessions, the store hook), and worker processes (`qccompute.serve` and its client `qccompute.worker`). It depends on qcdata only and must never import qccodec, branch on a program name, or contain program-specific logic; external programs belong in `qccodec`, which registers them as programs. Design: `docs/design/program-interface.md`.

- Failures raise `qcdata.ProgramError` with the failed output as `output`; lists and session `compute_many` run everything, then raise one group holding every result in order.
- The runner never changes a number by itself; keep the in-process path cheap (it is measured: about 4 us over a direct call of a `FunctionProgram`, planning included).
- Tests use stand-in programs (`qccompute.testing.toy`, `tests/serve_programs.py`). Integration tests with real programs live in qccodec (`tests/integration/`), which uses qccompute as a test-only dependency.

## The qc* suite

Dependencies point one way: qcref -> qcdata -> qccompute -> qccodec. qccodec depends on qccompute and registers its programs with it through entry points; qccompute reaches qccodec only through the optional extra `qccompute[qccodec]` and never imports it. Never import a package that sits later in that order, and keep changes that span packages in separate pull requests, released in that order.

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
