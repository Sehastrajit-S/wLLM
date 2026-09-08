# Contributing to wLLM

Thanks for considering a contribution. This project is a from-scratch, single-GPU,
single-machine, Windows-native inference engine — that scope is deliberate (see
the README's "What's not (yet) here" section), so before starting significant
new work, open an issue to check it fits before investing the time.

## Setup

```bash
git clone https://github.com/Sehastrajit-S/wLLM.git
cd wLLM
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
python scripts/check_toolchain.py   # confirms MSVC + nvcc + CUDA extension building works
```

## Running tests

```bash
python scripts/run_tests.py
```

Use this instead of a single `pytest tests/` invocation. The suite loads dozens
of real models across 30+ files; chaining them all in one process can exhaust
a consumer GPU's VRAM regardless of which code is under test. `scripts/run_tests.py`
runs them in separate subprocess batches instead — see its docstring for the
full explanation. If you hit an out-of-memory error in a single test file
during development, that's a real bug worth reporting; if it only shows up
when running many files together, it's very likely this same known limitation.

Tests that need a GPU skip automatically (`pytest.skip("CUDA not available")`)
when none is present — CI runs on a GPU-less runner and validates everything
that doesn't need one (packaging, imports, CPU-only logic).

## Code style

- `ruff check src/ tests/ scripts/` must pass (pyflakes + import sorting; see
  `pyproject.toml`'s `[tool.ruff.lint]` for why the ruleset is scoped the way
  it is).
- No comments explaining *what* code does — names should do that. Comments
  are for *why*: a non-obvious constraint, a workaround, an invariant a
  reader could easily violate.
- Match the existing testing philosophy: exact-match correctness against a
  proven baseline wherever one exists (a pure-PyTorch reference
  implementation, a known-good HF baseline, a manually-computed expected
  value), not just "it doesn't crash."

## Pull requests

- Keep PRs scoped to one change. A bug fix doesn't need drive-by refactoring.
- Add or update tests for anything behavioral. If a change can't be tested
  (e.g. it needs specific hardware you don't have access to verify), say so
  explicitly in the PR description rather than leaving it implicit.
- Run `python scripts/run_tests.py` and `ruff check` before opening the PR.

## Reporting issues

Include: your GPU, CUDA Toolkit version, Visual Studio version, Python
version, and the output of `python scripts/check_toolchain.py` if the issue
is build-related.
