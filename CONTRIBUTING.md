# Contributing

1. **Open an issue** describing the bug or feature (use the templates).
2. **Create a branch** from `main`: `feature/<short-name>`, `fix/<short-name>` or `docs/<short-name>`.
3. **Install dev tools**: `make install` (editable install + pre-commit hooks).
4. **Write tests first** for new behaviour. Any change touching features or splits must keep the
   leakage tests green: `tests/test_features.py` and `tests/test_split.py`.
5. **Commit** using [Conventional Commits](https://www.conventionalcommits.org/):
   `feat:`, `fix:`, `test:`, `docs:`, `ci:`, `refactor:`, `chore:`.
6. **Run locally** `make lint test` before pushing.
7. **Open a pull request** that references the issue (`Closes #12`). CI must pass before merging.

## Releasing
Update `version` in `pyproject.toml` and `src/oulad_risk/__init__.py`, add a `CHANGELOG.md` entry,
then `git tag v0.x.y && git push --tags`. The release workflow builds and publishes the GitHub Release.

## Code style
Ruff (line length 100) for linting and formatting; type hints and NumPy-style docstrings for
public functions; no notebooks inside `src/`.
