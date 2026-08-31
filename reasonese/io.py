"""Write planned prompt specifications."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from beartype import beartype

from reasonese.planning import PromptSpec


@beartype
def write_prompt_specs(path: Path, specs: tuple[PromptSpec, ...]) -> None:
    """Write prompt specifications as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for spec in specs:
            handle.write(json.dumps(asdict(spec), sort_keys=True))
            handle.write("\n")
