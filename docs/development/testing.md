# Testing

Run the complete local gate:

```bash
uv run pre-commit run --all-files
```

It checks the lockfile, Ruff, ty, and pytest. Pytest enforces at least 95 percent
branch-aware coverage and exercises the axis values, phantom constraints, `beartype`
boundaries, complete enumeration, configuration loading, serialization, OpenRouter request
contracts, cache behavior, conversation execution, independent judging, exact boolean parsing,
exact-text message QA and fail-closed inference, trace-sensitive judgment caching, pairwise
ordering, position balance, resumable data
collection, synthetic Bradley-Terry recovery, ties, clustered bootstrap, disconnected graphs,
axis and position effects, sparse-design stratification, degree balance, minimum connectivity
repair, and reproducibility, analysis artifacts,
and all nine utilities. Sampling and analysis tests run against the real 24-pair bank at its real
size: 81 conditions per instruction side, 3,645 eligible pairings per pair, and the 720-pairing
default.

Thread pinning is no longer a test fixture. `reasonese.analysis` scopes BLAS threads around
the fit itself, so the suite exercises the same path a real run takes and a regression there
shows up as a failing test rather than a slow one. See
[`../reference/architecture.md`](../reference/architecture.md).

All tests are offline. A separately authorized live smoke test is needed to validate current
provider availability and behavior.
