# Contributing

## Setup
- Python 3.10+ (`pip install -e ".[dev,finetune]"`, torch from the index that matches your GPU) and Go (see `go.mod`).
- Checks that CI runs: `ruff check src tests`, `pytest -q`, `go vet ./...`, `go test ./...`.
- Sign off commits (`git commit -s`): by doing so you certify the Developer Certificate of Origin (https://developercertificate.org).
- Commit messages follow Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:` ...); they drive release notes.

## Definition of done
A change counts as done when it has: tests (including failure paths), docs saying what is *not* covered, metrics/log
events where relevant, a threat-model note for anything touching network/auth/files, a runbook entry for its failure
modes, passing CI, and an ADR (`docs/adr/`) if it changes a design decision. Benchmarks need the raw JSON committed.

## Dependencies
Edit `pyproject.toml`, then run `make lock` and commit `requirements/`. Never edit locks by hand.

## Security issues
See `SECURITY.md`. Do not open public issues for vulnerabilities.
