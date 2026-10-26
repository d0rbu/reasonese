#!/usr/bin/env python3
"""Measure a complete local probe replay over an existing saved collection.

Run separate baseline/candidate processes with the same interpreter and inputs.
Each output directory must be new. CPU profiling is a separate attribution run;
its duration is not benchmark evidence. No model provider is invoked.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import pstats
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from threadpoolctl import threadpool_info

import reasonese.local_probe_qa as local
from reasonese.probe_posthoc import _collection_studies, _SavedStudy, score_saved_collections
from reasonese.probe_qa import ProbeQaDiagnosticIssue, ProbeQaRequest, probe_qa_contexts
from reasonese.role_probe_extraction import NativeTemplateAdapter
from reasonese.study_cache import SqliteStudyCache


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def saved_requests(
    collection: Path,
) -> tuple[tuple[_SavedStudy, ...], tuple[ProbeQaRequest, ...], list[ProbeQaDiagnosticIssue]]:
    """Inventory all available contexts without changing source study databases."""
    studies = _collection_studies(collection)
    by_database: dict[Path, list[_SavedStudy]] = defaultdict(list)
    for saved in studies:
        by_database[saved.database].append(saved)
    requests: list[ProbeQaRequest] = []
    issues: list[ProbeQaDiagnosticIssue] = []
    for database, saved_studies in by_database.items():
        trials = tuple(trial for saved in saved_studies for trial in saved.trials)
        traces = SqliteStudyCache(database).load_traces_readonly(trials)
        for saved in saved_studies:
            contexts = tuple(
                (int(trial.permutation), traces[trial.trial_id].setup)
                for trial in saved.trials if trial.trial_id in traces
            )
            found, missing = probe_qa_contexts(saved.study, contexts, missing_reason="not saved")
            requests.extend(found)
            issues.extend(missing)
    return studies, tuple(requests), issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--role-probes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-profile", action="store_true")
    args = parser.parse_args()
    args.collection = args.collection.resolve()
    args.role_probes = args.role_probes.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    # Make the active run visible to other GPU admission checks during CPU setup.
    # Context initialization and the fixed warmup are outside the measured API.
    torch.empty(1, dtype=torch.uint8, device="cuda:0")
    torch.cuda.synchronize()

    root = Path(local.__file__).resolve().parent.parent
    studies, requests, issues = saved_requests(args.collection)
    if not requests:
        raise ValueError("Source collection has no scoreable contexts")
    sources = {str(saved.directory / "study.yaml") for saved in studies}
    sources.update(str(saved.database) for saved in studies if saved.database.is_file())
    source_hashes = {path: file_sha256(Path(path)) for path in sorted(sources)}
    bundles = local.load_probe_bundles(args.role_probes)
    if len(bundles) != 1:
        raise ValueError("Benchmark one qualified probe bundle at a time")
    bundle = bundles[0]
    identity = local.probe_bundle_identity(args.role_probes)
    manifest = json.loads((bundle.checkpoint / "prefix-checkpoint-manifest.json").read_text())
    write_json(args.output / "launch.json", {
        "pid": os.getpid(),
        "source_root": str(root),
        "commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": file_sha256(Path(local.__file__)),
        "script_sha256": file_sha256(Path(__file__)),
        "probe_sha256": file_sha256(bundle.probe_path),
        "weights_sha256": manifest["weights_sha256"],
        "source_hashes": source_hashes,
        "requests": len(requests),
        "missing": len(issues),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "threadpools": threadpool_info(),
        "compiler_cache": {
            name: os.environ.get(name)
            for name in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")
        },
        "profile": args.cpu_profile,
        "started_at": time.time(),
    })
    print(
        f"Warmup: {len(requests)} real requests; {len(issues)} unavailable contexts.", flush=True
    )
    # A separate scorer warms the first and last real requests. Its cache and
    # static-artifact preflight are not reused by the measured replay.
    warmup = local.LocalProbeQaScorer(args.role_probes, args.output / "warmup-cache.json")
    warmup.check((requests[0], requests[-1]))
    del warmup
    torch.cuda.synchronize()

    capture_hashes: dict[str, dict[str, Any]] = {}
    if args.cpu_profile:
        real_capture = local.capture_token_activations

        def audited_capture(
            model: object,
            adapter: NativeTemplateAdapter,
            *,
            input_ids: tuple[int, ...],
            token_positions: tuple[int, ...],
            layers: tuple[int, ...],
        ) -> np.ndarray:
            captured = real_capture(
                model, adapter, input_ids=input_ids, token_positions=token_positions, layers=layers
            )
            capture_kwargs = {
                "input_ids": input_ids, "token_positions": token_positions, "layers": layers,
            }
            key = hashlib.sha256(json.dumps(capture_kwargs, sort_keys=True).encode()).hexdigest()
            digest = hashlib.sha256(captured.tobytes(order="C")).hexdigest()
            row = capture_hashes.setdefault(key, {
                "calls": 0,
                "shape": list(captured.shape),
                "dtype": str(captured.dtype),
                "input_tokens": len(input_ids),
                "output_sha256": [],
            })
            row["calls"] += 1
            if digest not in row["output_sha256"]:
                row["output_sha256"].append(digest)
            return captured

        # Instrument only this separate profiling process.
        cast(Any, local).capture_token_activations = audited_capture

    output = args.output / "replay"
    scorer = local.LocalProbeQaScorer(args.role_probes, output / "probe_qa_cache.json")
    profile = cProfile.Profile() if args.cpu_profile else None
    if profile is not None:
        profile.enable()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    print("Measured replay started.", flush=True)
    summary = score_saved_collections(args.collection, output, scorer, identity)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if profile is not None:
        profile.disable()
        profile.dump_stats(str(args.output / "cpu-profile.pstats"))
        with (args.output / "cpu-profile.txt").open("w") as stream:
            pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(60)
        write_json(args.output / "capture-hashes.json", capture_hashes)
    if summary["error_requests"] or summary["scores"] != len(requests):
        raise RuntimeError(f"Incomplete replay: {summary}")
    if any(file_sha256(Path(path)) != expected for path, expected in source_hashes.items()):
        raise RuntimeError("Source collection changed during measurement")
    cache = json.loads((output / "probe_qa_cache.json").read_text())
    report = json.loads((output / "probe_qa_report.json").read_text())
    result = {
        "elapsed_seconds": elapsed,
        "summary": summary,
        "source_unchanged": True,
        "cache_sha256": file_sha256(output / "probe_qa_cache.json"),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "cache_records": len(cache["records"]),
        "counts": report["counts"],
    }
    write_json(args.output / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
