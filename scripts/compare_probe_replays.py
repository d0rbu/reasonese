#!/usr/bin/env python3
"""Compare every replay output, allowing only declared output paths and numeric tolerance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparisons = 0
    max_absolute_error = 0.0
    exact = True

    def compare(a: Any, b: Any, path: str) -> None:
        nonlocal comparisons, max_absolute_error, exact
        require(type(a) is type(b), f"Type mismatch: {path}")
        if isinstance(a, dict):
            require(set(a) == set(b), f"Keys changed: {path}")
            for key in a:
                compare(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, list):
            require(len(a) == len(b), f"Length changed: {path}")
            for index, (left, right) in enumerate(zip(a, b, strict=True)):
                compare(left, right, f"{path}/{index}")
        elif isinstance(a, float):
            require(math.isfinite(a) and math.isfinite(b), f"Nonfinite value: {path}")
            require(abs(a - b) <= 1e-6 + 1e-6 * abs(a), f"Numeric mismatch: {path}: {a}, {b}")
            comparisons += 1
            max_absolute_error = max(max_absolute_error, abs(a - b))
            exact = exact and a == b
        else:
            require(a == b, f"Discrete value changed: {path}: {a}, {b}")

    for filename in ("probe_qa_cache.json", "probe_qa_report.json", "probe_posthoc_manifest.json"):
        a, b = (load(path / "replay" / filename) for path in (args.baseline, args.candidate))
        output_fields = {
            "probe_qa_report.json": ("source_manifest",),
            "probe_posthoc_manifest.json": ("probe_report", "probe_cache"),
        }.get(filename, ())
        for field in output_fields:
            for value, directory in ((a[field], args.baseline), (b[field], args.candidate)):
                require(
                    Path(value).parent.resolve() == (directory / "replay").resolve(),
                    f"Unexpected output path: {filename}/{field}",
                )
            require(Path(a[field]).name == Path(b[field]).name, f"Output file changed: {field}")
            a[field] = b[field] = Path(a[field]).name
        compare(a, b, filename)

    activation_files = [path / "capture-hashes.json" for path in (args.baseline, args.candidate)]
    present = [path.is_file() for path in activation_files]
    require(all(present) or not any(present), "Only one run has activation hashes")
    activations = None
    if all(present):
        a, b = (load(path) for path in activation_files)
        require(a.keys() == b.keys(), "Captured prefix identities changed")
        for key in a:
            require(len(a[key]["output_sha256"]) == 1, "Baseline duplicate captures were not stable")
            for field in ("shape", "dtype", "input_tokens", "output_sha256"):
                require(a[key][field] == b[key][field], f"Activation mismatch: {key}: {field}")
        activations = {
            "unique_captures": len(a),
            "bitwise_equal": True,
            "baseline_calls": sum(row["calls"] for row in a.values()),
            "candidate_calls": sum(row["calls"] for row in b.values()),
        }
    result = {
        "passed": True,
        "float_comparisons": comparisons,
        "floats_exact": exact,
        "max_absolute_error": max_absolute_error,
        "atol": 1e-6,
        "rtol": 1e-6,
        "discrete_records_exact": True,
        "activations": activations,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
