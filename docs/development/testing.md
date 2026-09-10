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
size: 90 conditions per instruction side, 4,500 eligible pairings per pair, and the 720-pairing
default.

`tests/conftest.py` pins the BLAS thread count to one before NumPy is imported. The
Bradley-Terry fit solves one small dense system per component, and on a many-core machine
OpenBLAS spends far longer synchronizing threads than on the arithmetic: a 180x180
`numpy.linalg.solve` measured 2089 ms with default threads and 1.99 ms pinned to one. The same
setting is worth exporting for real analysis runs.

All tests are offline. A separately authorized live smoke test is needed to validate current
provider availability and behavior.
