# Testing

Run the complete local gate:

```bash
uv run pre-commit run --all-files
```

It checks the lockfile, Ruff, ty, and pytest. Pytest enforces at least 95 percent
branch-aware coverage and exercises the axis values, phantom constraints, `beartype`
boundaries, complete enumeration, configuration loading, serialization, OpenRouter request
contracts, cache behavior, conversation execution, independent judging, exact boolean parsing,
trace-sensitive judgment caching, pairwise ordering, position balance, resumable data
collection, synthetic Bradley-Terry recovery, ties, clustered bootstrap, disconnected graphs,
axis and position effects, analysis artifacts, and all six utilities.

All tests are offline. A separately authorized live smoke test is needed to validate current
provider availability and behavior.
