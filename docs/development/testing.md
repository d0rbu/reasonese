# Testing

Run the complete local gate:

```bash
uv run pre-commit run --all-files
```

Its local hooks run:

```bash
uv lock --check
uv run ruff check .
uv run ty check
uv run pytest
```

Pytest enforces at least 95 percent branch-aware coverage. The suite exercises axis
metadata, configuration rejection, complete design enumeration, content-addressed IDs,
JSONL round trips and failure cleanup, and CLI integration.

All current tests are offline. They do not validate the quality of future model-authored
framings or behavior of future executor models.
