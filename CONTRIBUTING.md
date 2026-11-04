# Contributing to RAGPocket

Thanks for helping improve RAGPocket. The project is early development, so
small, focused changes with clear behavior are especially valuable.

## Development setup

RAGPocket supports Python 3.10 and newer. Create an isolated environment and
install the checkout in editable mode:

```console
$ python -m venv .venv
$ . .venv/bin/activate
$ python -m pip install --upgrade pip
$ python -m pip install --editable .
```

There are no required runtime or test dependencies beyond Python's standard
library and SQLite. Keep optional provider dependencies out of the core path.

## Run the tests

The repository uses the standard-library `unittest` runner:

```console
$ python -m unittest discover -s tests -v
```

Run the same command before opening a pull request. The GitHub Actions workflow
runs it across the supported Python versions.

## Making changes

- Read [`docs/SPEC.md`](docs/SPEC.md) before changing observable behavior.
- Preserve deterministic ordering, chunk IDs, scores, and citations.
- Keep core commands local-only and dependency-light.
- Do not add network access, model downloads, telemetry, or implicit plugin
  discovery.
- Add or update focused tests for behavior changes. Keep fixtures small and
  self-contained under `tests/fixtures/`.
- Update the README or specification when a user-visible contract changes.
- Do not commit generated build output, virtual environments, local indexes,
  credentials, or cache directories.

## Pull requests

Describe the user-visible change, the relevant design or compatibility
considerations, and the exact checks you ran. Keep each pull request focused
on one logical change when practical.

Embedding integrations should be proposed as explicit plugins. They must not
become a dependency of installation, indexing, BM25 search, or the default
test suite, and their network behavior must be documented clearly.
