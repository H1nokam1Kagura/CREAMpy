# Contributing to CREAMpy

Thank you for considering a contribution. This document covers how to report
issues, propose changes, and submit pull requests.

## Reporting bugs

Open an issue at https://github.com/H1nokam1Kagura/CREAMpy/issues.

Include:
- Python version and OS
- Minimal reproducible example
- Expected vs actual output

## Proposing changes

Before writing code, open an issue describing what you want to change and why.
This avoids duplicate work and lets us discuss the design before implementation.

## Development setup

```bash
git clone https://github.com/H1nokam1Kagura/CREAMpy
cd CREAMpy
pip install -e ".[dev]"
pytest tests/ -v
```

## Adding a test

Every code change must include a test. The test suite is in `tests/` and uses
pytest. Analytical test cases (known closed-form solutions) are strongly
preferred over round-trip or regression tests.

## Mathematical changes

If you change any formula in `model.py` or `adoption/bass.py`:

1. State the source reference (book, paper, equation number).
2. Add or update the relevant analytical test case in `tests/`.
3. Update the module docstring formula block.

The project follows the ANP (1995) and Bass (1969) derivations exactly.
Deviations must be documented and justified.

## Code style

- Python ≥ 3.9, no runtime dependencies.
- Type annotations on all public functions and dataclass fields.
- No comments explaining *what* the code does — only *why* (hidden constraints,
  non-obvious invariants, workarounds).
- No new dependencies without discussion.

## Commit messages

One-sentence summary on the first line. Blank line, then detail if needed.
Reference issues with `Fixes #N` or `See #N`.

## Pull requests

- Target the `main` branch.
- All CI checks must pass.
- Include a description of what changed and why.
