"""Versioned JSONL input and output for prompt specifications."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from reasonese.planning import PromptSpec


def write_prompt_specs(path: Path, specs: Iterable[PromptSpec]) -> None:
    """Atomically write prompt specifications as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for spec in specs:
                handle.write(json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_prompt_specs(path: Path) -> tuple[PromptSpec, ...]:
    """Read strict prompt specifications from JSONL."""
    specs: list[PromptSpec] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank line at {path}:{line_number}")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error.msg}") from error
            if not isinstance(payload, dict):
                raise ValueError(f"prompt spec at {path}:{line_number} must be an object")
            specs.append(PromptSpec.from_dict(payload))
    return tuple(specs)
