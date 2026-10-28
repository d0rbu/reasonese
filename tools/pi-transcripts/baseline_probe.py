"""Observe real pinned source behavior. Baselines are not post-implementation oracles.

Run inside the prepared scenario sandbox, from its repository root. All fixtures
are temporary. Imports intentionally resolve in that scenario's dependency setup.
"""

from __future__ import annotations

import copy
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def airflow() -> dict:
    cls = importlib.import_module("airflow_shared.secrets_masker.secrets_masker").SecretsMasker
    masker = cls()
    masker.sensitive_variables_fields = ["password"]
    old = {"password": "old-secret"}
    masked = masker.redact(old, replacement="<hidden>")
    custom = masker.merge(masked, old, replacement="<hidden>")
    legacy = masker.merge({"password": "***"}, old, replacement="<hidden>")
    assert masked == custom == {"password": "<hidden>"}
    assert legacy == old == {"password": "old-secret"}
    return {"custom_round_trip": custom, "legacy_marker": legacy}


def digests() -> dict:
    digest = importlib.import_module("praktika.digest").Digest
    job = importlib.import_module("praktika.job").Job
    docker = importlib.import_module("praktika.docker").Docker
    original = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        os.chdir(directory)
        try:
            Path("src").mkdir()
            config = job.Config(name="fixture", runs_on=["local"], command="true",
                                digest_config=job.CacheDigestConfig(include_paths=["src"], exclude_paths=["*.skip"]))
            image = docker.Config(name="fixture", path="src", depends_on=[], platforms=["linux/amd64"])

            def both() -> tuple[str, str]:
                return (digest().calc_job_digest(config, {}, {}), digest().calc_docker_digest(image, []))

            Path("src/a.txt").write_text("same")
            before = both()
            Path("src/a.txt").rename("src/b.txt")
            renamed = both()
            assert before == renamed
            Path("src/b.txt").write_text("changed")
            changed = both()
            assert all(a != b for a, b in zip(renamed, changed, strict=True))
            # The pinned traversal visits c before b (reverse lexical order).
            Path("src/c.txt").write_text("ab")
            Path("src/b.txt").write_text("c")
            boundary_one = both()
            Path("src/c.txt").write_text("a")
            Path("src/b.txt").write_text("bc")
            boundary_two = both()
            assert boundary_one == boundary_two
            return {"rename_equal": True, "content_change_detected": True,
                    "file_boundary_collision": True, "callers": ["job", "docker"]}
        finally:
            os.chdir(original)


def results() -> dict:
    result = importlib.import_module("praktika.result").Result
    utils = importlib.import_module("praktika.utils").Utils
    children = [result.create_new("a", "SKIPPED"), result.create_new("b", "SKIPPED")]
    parent = result.create_from(name="fixture", results=children, stopwatch=utils.Stopwatch())
    initial = parent.status
    parent._update_status()
    assert initial == parent.status == "OK"
    assert result.from_dict(result.to_dict(parent)).status == "OK"
    child = result.create_new("error", "ERROR")
    failed = result.create_from(name="fixture", results=[child], stopwatch=utils.Stopwatch())
    constructor_error = failed.status
    failed._update_status()
    assert constructor_error == "ERROR" and failed.status == "FAIL"
    return {"all_skipped_constructor": initial, "all_skipped_update": parent.status,
            "error_constructor": constructor_error, "error_update": failed.status}


def requirements() -> dict:
    parse = importlib.import_module("mlflow.utils.requirements_utils")._parse_requirements
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name, content in {"root.txt": "-r left.txt\n-r right.txt\n",
                              "left.txt": "-r common.txt\n", "right.txt": "-r common.txt\n",
                              "common.txt": "demo==1.0\n"}.items():
            (root / name).write_text(content)
        records = list(parse(str(root / "root.txt"), is_constraint=False))
        assert records == [("demo==1.0", False), ("demo==1.0", False)]
        (root / "cycle.txt").write_text("-r cycle.txt\n")
        previous_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(100)
        try:
            list(parse(str(root / "cycle.txt"), is_constraint=False))
        except RecursionError:
            cycle = "RecursionError"
        else:
            raise AssertionError("Expected the pinned parser's unhandled cycle")
        finally:
            sys.setrecursionlimit(previous_limit)
        return {"diamond_records": records, "cycle_error": cycle}


def checksums() -> dict:
    script = Path("scripts/patch-dist-manifest-checksums.py").resolve()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / "manifest.json"
        sidecar = root / "uv.tar.gz.sha256"
        initial = {"metadata": "keep", "artifacts": {"uv.tar.gz": {"checksums": {"sha256": "a" * 64}}}}
        manifest.write_text(json.dumps(initial))
        sidecar.write_text("b" * 64 + "  uv.tar.gz\n")
        command = [sys.executable, str(script), "--manifest", str(manifest), "--artifacts-dir", str(root)]
        first = subprocess.run(command, capture_output=True, text=True, check=False)
        assert first.returncode == 0
        assert json.loads(manifest.read_text())["artifacts"]["uv.tar.gz"]["checksums"]["sha256"] == "b" * 64
        manifest.write_text(json.dumps(initial))
        sidecar.write_text("z" * 64)
        invalid = subprocess.run(command, capture_output=True, text=True, check=False)
        assert invalid.returncode == 0
        assert json.loads(manifest.read_text())["artifacts"]["uv.tar.gz"]["checksums"]["sha256"] == "z" * 64
        return {"conflict_exit_code": first.returncode, "conflict_replaces_existing": True,
                "non_hex_accepted": True}


def fallbacks() -> dict:
    runnable = importlib.import_module("langchain_core.runnables").RunnableLambda
    calls = []

    def backup(value: str) -> str:
        calls.append(value)
        return "backup"

    chain = runnable(lambda _: None).with_fallbacks([runnable(backup)])
    value = chain.invoke("request")
    assert value is None and not calls
    batch = chain.batch(["first", "second"])
    assert batch == [None, None] and not calls
    return {"invoke": value, "batch": batch, "backup_calls": len(calls)}


def flavors() -> dict:
    targeted = importlib.import_module("defs.functional_test_selection").targeted_variants
    job = importlib.import_module("praktika.job").Job
    jobs = [job.Config(name=f"Functional ({parameter})", runs_on=[runner],
                       command=f"run '{parameter}'", parameter=parameter, requires=["build"], provides=["obsolete"])
            for parameter, runner in [("release, parallel, 1/2", "large"), ("release, sequential, 1/2", "small")]]
    original = copy.deepcopy(jobs)
    variants = targeted(jobs)
    assert jobs == original
    assert len(variants) == 1 and variants[0].runs_on == ["large"]
    assert not variants[0].provides and variants[0].allow_failure
    assert targeted(list(reversed(jobs))) == variants
    inconsistent = copy.deepcopy(jobs[1])
    inconsistent.runs_on = ["different-small"]
    # The discarded sequential flavor is not remembered for later consistency checks.
    assert len(targeted([*jobs, inconsistent])) == 1
    return {"variant_count": len(variants), "runner": variants[0].runs_on,
            "input_unchanged": True, "inconsistent_discarded_flavor_accepted": True}


PROBES = {"airflow-redaction-marker": airflow, "clickhouse-cache-file-identity": digests,
          "clickhouse-all-skipped-status": results, "mlflow-requirements-includes": requirements,
          "uv-release-checksum-conflict": checksums, "langchain-none-fallback": fallbacks,
          "clickhouse-targeted-job-flavors": flavors}

if __name__ == "__main__":
    print(json.dumps({"kind": "pinned_source_baseline", "scenario": sys.argv[1],
                      "observed": PROBES[sys.argv[1]]()}, indent=2))
