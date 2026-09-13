from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterator, Mapping
from pathlib import Path
from urllib.parse import unquote

import pytest

from scripts import download_probe_prefix as subject


def _safetensors(tensors: list[tuple[str, str, list[int], bytes]]) -> bytes:
    cursor = 0
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    for name, dtype, shape, data in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + len(data)],
        }
        cursor += len(data)
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_header += b" " * (-len(raw_header) % 8)
    return struct.pack("<Q", len(raw_header)) + raw_header + b"".join(item[3] for item in tensors)


class FakeResponse:
    def __init__(self, body: bytes, start: int, end: int, total: int, *, truncate: bool = False):
        self.status_code = 206
        self.headers = {
            "Content-Range": f"bytes {start}-{end}/{total}",
            "Content-Length": str(end - start + 1),
        }
        self._body = body[:-1] if truncate else body
        self.closed = False

    @property
    def content(self) -> bytes:
        return self._body

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(
        self,
        files: dict[str, bytes],
        *,
        truncate_start: int | None = None,
        status_code: int = 206,
    ):
        self.files = files
        self.truncate_start = truncate_start
        self.status_code = status_code
        self.requested: list[tuple[str, int, int]] = []

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        stream: bool,
        timeout: tuple[int, float],
    ) -> FakeResponse:
        del stream, timeout
        match = subject.re.fullmatch(r"bytes=(\d+)-(\d+)", str(headers["Range"]))
        assert match is not None
        start, end = (int(value) for value in match.groups())
        filename = unquote(url.rsplit("/", 1)[-1])
        source = self.files[filename]
        self.requested.append((filename, start, end))
        truncate = start == self.truncate_start
        if truncate:
            self.truncate_start = None
        response = FakeResponse(source[start : end + 1], start, end, len(source), truncate=truncate)
        response.status_code = self.status_code
        return response


def _metadata_dir(path: Path, *, model_id: str, revision: str, weight_map: dict[str, str]) -> None:
    path.mkdir()
    for filename in ("config.json", "tokenizer_config.json", "metadata.json"):
        value = {"id": model_id, "sha": revision} if filename == "metadata.json" else {}
        (path / filename).write_text(json.dumps(value))
    (path / "chat_template.jinja").write_text("{{ messages }}")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"source": "test"}, "weight_map": weight_map})
    )


def test_model_selectors_are_scoped_to_the_language_backbones() -> None:
    expected_tokenizer_files = (
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    assert subject.PRESETS["nemotron"].auxiliary_files == expected_tokenizer_files
    assert subject.PRESETS["gemma"].auxiliary_files == expected_tokenizer_files
    assert subject.keep_nemotron("backbone.embeddings.weight")
    assert subject.keep_nemotron("backbone.layers.25.mixer.in_proj.weight")
    assert subject.keep_nemotron("backbone.layers.26.norm.weight")
    assert not subject.keep_nemotron("other.layers.0.weight")
    assert not subject.keep_nemotron("backbone.layers.26.mixer.in_proj.weight")

    assert subject.keep_gemma("model.language_model.embed_tokens.weight")
    assert subject.keep_gemma("model.language_model.layers.29.mlp.down_proj.weight")
    assert subject.keep_gemma("model.language_model.layers.30.self_attn.q_proj.weight")
    assert not subject.keep_gemma("vision_tower.layers.0.self_attn.q_proj.weight")
    assert not subject.keep_gemma("model.language_model.layers.30.mlp.down_proj.weight")


def test_download_prefix_builds_exact_filtered_checkpoint(tmp_path: Path) -> None:
    model_id = "example/model"
    revision = "a" * 40
    shard = _safetensors(
        [
            ("drop.before", "U8", [3], b"abc"),
            ("keep.one", "U8", [4], b"defg"),
            ("keep.two", "U8", [2], b"hi"),
            ("drop.after", "U8", [3], b"jkl"),
        ]
    )
    tokenizer = b'{"version":"1.0"}'
    metadata_dir = tmp_path / "metadata"
    output_dir = tmp_path / "output"
    weight_map = dict.fromkeys(
        ("drop.before", "keep.one", "keep.two", "drop.after"),
        "model-00001-of-00001.safetensors",
    )
    _metadata_dir(metadata_dir, model_id=model_id, revision=revision, weight_map=weight_map)
    preset = subject.ModelPreset(
        repository=model_id,
        revision=revision,
        metadata_model_id=model_id,
        adapter="test-native-v1",
        max_layer=1,
        auxiliary_files=("tokenizer.json",),
        selection="test selection",
        keep=lambda name: name.startswith("keep."),
    )
    session = FakeSession({"model-00001-of-00001.safetensors": shard, "tokenizer.json": tokenizer})

    manifest = subject.download_prefix(
        preset, metadata_dir, output_dir, token=None, timeout=1, session=session
    )

    output_shard = output_dir / "model-00001-of-00001.safetensors"
    with output_shard.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
        data = handle.read()
    assert set(header) == {"__metadata__", "keep.one", "keep.two"}
    assert data == b"defghi"
    assert (output_dir / "tokenizer.json").read_bytes() == tokenizer
    filtered_index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    assert filtered_index["weight_map"] == {
        "keep.one": output_shard.name,
        "keep.two": output_shard.name,
    }
    assert manifest["adapter"] == "test-native-v1"
    assert manifest["weights_hash_kind"] == "sha256-filtered-index-and-shard-files-v1"
    assert len(manifest["weights_sha256"]) == 64
    assert json.loads((output_dir / "prefix-checkpoint-manifest.json").read_text()) == manifest
    requested_weight_bytes = sum(
        end - start + 1 for name, start, end in session.requested if name == output_shard.name
    )
    assert requested_weight_bytes < len(shard)


def test_filtered_shard_resumes_after_truncated_range(tmp_path: Path) -> None:
    source_bytes = _safetensors([("drop", "U8", [2], b"ab"), ("keep", "U8", [6], b"cdefgh")])
    inspect_session = FakeSession({"weights.safetensors": source_bytes})
    source = subject.inspect_source_shard(
        inspect_session,
        "weights.safetensors",
        "https://example/weights.safetensors",
        token=None,
        timeout=1,
    )
    selected = next(tensor for tensor in source.tensors if tensor.name == "keep")
    selected_start = source.data_start + selected.source_start
    output = tmp_path / "weights.safetensors"
    interrupted = FakeSession({"weights.safetensors": source_bytes}, truncate_start=selected_start)
    with pytest.raises(RuntimeError, match="body has 5 bytes, expected 6"):
        subject.download_filtered_shard(
            interrupted, source, {"keep"}, output, token=None, timeout=1, chunk_size=2
        )
    partial_size = (tmp_path / "weights.safetensors.partial").stat().st_size

    resumed = FakeSession({"weights.safetensors": source_bytes})
    subject.download_filtered_shard(
        resumed, source, {"keep"}, output, token=None, timeout=1, chunk_size=2
    )
    assert resumed.requested == [("weights.safetensors", selected_start + 5, selected_start + 5)]
    assert output.stat().st_size == partial_size + 1
    assert not (tmp_path / "weights.safetensors.partial").exists()


def test_inspection_rejects_server_that_ignores_range() -> None:
    source = _safetensors([("tensor", "U8", [1], b"x")])
    session = FakeSession({"weights.safetensors": source}, status_code=200)
    with pytest.raises(RuntimeError, match="HTTP 200, expected 206"):
        subject.inspect_source_shard(
            session,
            "weights.safetensors",
            "https://example/weights.safetensors",
            token=None,
            timeout=1,
        )


def test_completed_shard_must_match_selection(tmp_path: Path) -> None:
    source_bytes = _safetensors([("keep", "U8", [2], b"ok")])
    session = FakeSession({"weights.safetensors": source_bytes})
    source = subject.inspect_source_shard(
        session,
        "weights.safetensors",
        "https://example/weights.safetensors",
        token=None,
        timeout=1,
    )
    output = tmp_path / "weights.safetensors"
    output.write_bytes(hashlib.sha256(b"wrong").digest())
    with pytest.raises(ValueError, match="does not match the requested selection"):
        subject.download_filtered_shard(session, source, {"keep"}, output, token=None, timeout=1)
