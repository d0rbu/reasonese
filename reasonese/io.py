"""JSONL persistence with strict record validation and atomic replacement."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, TextIO, cast

from reasonese.schemas import ResponseRecord, ScoredOutcome, Trial


class _Serializable(Protocol):
    def to_dict(self) -> dict[str, Any]: ...


@contextmanager
def _atomic_text_writer(path: Path) -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            yield cast("TextIO", handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def write_jsonl(path: Path, records: Iterable[_Serializable]) -> None:
    """Atomically replace ``path`` with canonical UTF-8 JSONL records."""
    with _atomic_text_writer(path) as handle:
        for record in records:
            json.dump(record.to_dict(), handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically replace ``path`` with indented, deterministic JSON."""
    with _atomic_text_writer(path) as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_jsonl[RecordT](
    path: Path, parser: Callable[[dict[str, Any]], RecordT]
) -> list[RecordT]:
    records: list[RecordT] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise ValueError("record must be a JSON object")
                    records.append(parser(cast("dict[str, Any]", raw)))
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    raise ValueError(f"invalid record at {path}:{line_number}: {error}") from error
    except OSError as error:
        raise ValueError(f"could not read {path}: {error}") from error
    return records


def read_trials(path: Path) -> list[Trial]:
    return _read_jsonl(path, Trial.from_dict)


def read_responses(path: Path) -> list[ResponseRecord]:
    return _read_jsonl(path, ResponseRecord.from_dict)


def read_outcomes(path: Path) -> list[ScoredOutcome]:
    return _read_jsonl(path, ScoredOutcome.from_dict)
