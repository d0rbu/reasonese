# Testing

Run the complete local gate:

```bash
uv run pre-commit run --all-files
```

It checks the lockfile, Ruff, ty, and pytest. Pytest enforces at least 95 percent
branch-aware coverage and exercises the axis values, phantom constraints, `beartype`
boundaries, complete enumeration, TOML loading, JSONL output, and CLI.

All tests are offline. They do not validate future prompt transformations or model behavior.
