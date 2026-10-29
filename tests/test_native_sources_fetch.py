"""Pinned source downloads preserve evidence and reject unsafe outer archives."""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/pi-transcripts/fetch-sources.py"


def bundle(tmp_path: Path, name: str = "task/template.json", *, link: bool = False) -> Path:
    path = tmp_path / "source.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        item = tarfile.TarInfo(name)
        if link:
            item.type = tarfile.SYMTYPE
            item.linkname = "/etc/passwd"
            archive.addfile(item)
        else:
            data = b'{"kind":"fixture_metadata"}\n'
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    return path


def fetch(tmp_path: Path, archive: Path, digest: str | None = None, *extra: str):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"artifacts": [{"scenario": "task", "url": archive.as_uri(),
        "sha256": digest or hashlib.sha256(archive.read_bytes()).hexdigest()}]}))
    return subprocess.run([sys.executable, str(SCRIPT), "--catalog", str(catalog), "--output",
                           str(tmp_path / "download"), *extra], capture_output=True, text=True, check=False)


def test_pinned_download_and_no_overwrite(tmp_path: Path) -> None:
    archive = bundle(tmp_path)
    result = fetch(tmp_path, archive)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "download/task/template.json").read_bytes())["kind"] == "fixture_metadata"
    assert json.loads(result.stdout)["scenario"] == "task"
    again = fetch(tmp_path, archive)
    assert again.returncode != 0 and "refusing to overwrite" in again.stderr


@pytest.mark.parametrize("name,link", [("task/link", True), ("task/../../escape", False), ("other/file", False)])
def test_untrusted_outer_archive_cannot_escape(tmp_path: Path, name: str, link: bool) -> None:
    result = fetch(tmp_path, bundle(tmp_path, name, link=link))
    assert result.returncode != 0
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "download/task/link").exists()


def test_changed_download_and_unknown_case_fail_closed(tmp_path: Path) -> None:
    archive = bundle(tmp_path)
    result = fetch(tmp_path, archive, "0" * 64)
    assert result.returncode != 0 and "hash mismatch" in result.stderr
    assert not (tmp_path / "download/task").exists()
    assert (tmp_path / "download/task.source.tar.gz").exists()
    unknown = fetch(tmp_path, archive, None, "--case", "unknown")
    assert unknown.returncode != 0 and "unknown or empty" in unknown.stderr


@pytest.mark.parametrize("reverse", [False, True])
def test_multipart_source_assembly_pins_each_part_and_complete_bytes(tmp_path: Path, reverse: bool) -> None:
    archive = bundle(tmp_path)
    data = archive.read_bytes()
    pieces = (data[:len(data) // 2], data[len(data) // 2:])
    refs = []
    for index, piece in enumerate(pieces):
        path = tmp_path / f"part-{index}"
        path.write_bytes(piece)
        refs.append({"url": path.as_uri(), "sha256": hashlib.sha256(piece).hexdigest(), "bytes": len(piece)})
    catalog = tmp_path / "multipart.json"
    catalog.write_text(json.dumps({"artifacts": [{"scenario": "task", "parts": refs[::-1] if reverse else refs,
                        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}]}))
    result = subprocess.run([sys.executable, str(SCRIPT), "--catalog", str(catalog), "--output",
                             str(tmp_path / "download")], capture_output=True, text=True, check=False)
    assert (result.returncode != 0) is reverse, result.stderr
    assert (tmp_path / "download/task.source.tar.gz").read_bytes() == (b"".join(pieces[::-1]) if reverse else data)
    assert (tmp_path / "download/task/template.json").exists() is not reverse
    if reverse:
        assert "hash mismatch" in result.stderr


def test_context_only_uses_separate_pinned_lightweight_artifact(tmp_path: Path) -> None:
    archive = bundle(tmp_path)
    catalog = tmp_path / "context-catalog.json"
    catalog.write_text(json.dumps({"artifacts": [{"scenario": "task", "url": "file:///unavailable-full-source",
        "sha256": "0" * 64, "contextOnlyBundle": {"url": archive.as_uri(),
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "bytes": archive.stat().st_size}}]}))
    result = subprocess.run([sys.executable, str(SCRIPT), "--catalog", str(catalog), "--output",
                             str(tmp_path / "context"), "--context-only"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["contextOnly"] is True
    assert (tmp_path / "context/task.context.tar.gz").exists()
    assert not (tmp_path / "context/task.source.tar.gz").exists()
    assert (tmp_path / "context/task/template.json").exists()
