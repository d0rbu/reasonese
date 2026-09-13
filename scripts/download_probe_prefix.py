#!/usr/bin/env python3
"""Download only the checkpoint tensors needed by the role-probe prefix runner.

The Hugging Face sharded index and model metadata must already be present locally.
Every weight request is pinned to the configured repository revision and uses an
HTTP byte range. The output is a valid, filtered safetensors checkpoint; source
shards are never downloaded in full.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Protocol, cast
from urllib.parse import quote

import requests

_MAX_HEADER_BYTES = 64 * 1024 * 1024
_MAX_AUXILIARY_BYTES = 512 * 1024 * 1024
_STREAM_CHUNK_BYTES = 8 * 1024 * 1024
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2": 1,
    "F8_E5M2FNUZ": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


JsonObject = dict[str, Any]


class HttpResponse(Protocol):
    status_code: int
    headers: Any
    content: bytes

    def iter_content(self, chunk_size: int) -> Iterator[bytes]: ...

    def close(self) -> None: ...


class HttpSession(Protocol):
    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        stream: bool,
        timeout: tuple[int, float],
    ) -> HttpResponse: ...


@dataclass(frozen=True, slots=True)
class ModelPreset:
    """One pinned source checkpoint and its exact prefix tensor selection."""

    repository: str
    revision: str
    metadata_model_id: str
    adapter: str
    max_layer: int
    auxiliary_files: tuple[str, ...]
    selection: str
    keep: Callable[[str], bool]


def _layer_number(key: str) -> int | None:
    match = _LAYER.search(key)
    return int(match.group(1)) if match is not None else None


def keep_nemotron(key: str) -> bool:
    """Keep the Nemotron embedding, complete layers 0..25, and layer-26 norm."""
    if key == "backbone.embeddings.weight":
        return True
    if not key.startswith("backbone.layers."):
        return False
    layer = _layer_number(key)
    return layer is not None and (
        layer < 26 or (layer == 26 and key == "backbone.layers.26.norm.weight")
    )


_GEMMA_LAYER_30 = (
    "model.language_model.layers.30.input_layernorm.weight",
    "model.language_model.layers.30.post_attention_layernorm.weight",
    "model.language_model.layers.30.pre_feedforward_layernorm.weight",
)


def keep_gemma(key: str) -> bool:
    """Keep the Gemma text embedding, layers 0..29, and layer-30 attention input."""
    if key == "model.language_model.embed_tokens.weight":
        return True
    if not key.startswith("model.language_model.layers."):
        return False
    layer = _layer_number(key)
    if layer is None:
        return False
    if layer < 30:
        return True
    if layer != 30:
        return False
    return key in _GEMMA_LAYER_30 or key.startswith("model.language_model.layers.30.self_attn.")


PRESETS = {
    "nemotron": ModelPreset(
        repository="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        revision="a9904d24bcc1d289a1950fa9d2b978c47cf903b9",
        metadata_model_id="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        adapter="nemotron-3.5-lightning-native-v1",
        max_layer=26,
        auxiliary_files=("chat_template.jinja", "tokenizer.json", "tokenizer_config.json"),
        selection="backbone.embeddings; layers<26; layer26.norm",
        keep=keep_nemotron,
    ),
    "gemma": ModelPreset(
        repository="google/gemma-4-31B-it",
        revision="842da3794eaa0b77d5f08bae87a17459d91ff475",
        metadata_model_id="google/gemma-4-31B-it",
        adapter="gemma-4-31b-native-v1",
        max_layer=30,
        auxiliary_files=("chat_template.jinja", "tokenizer.json", "tokenizer_config.json"),
        selection=(
            "language embed_tokens; layers<30; layer30 input/post-attention/pre-feedforward "
            "norms and self_attn"
        ),
        keep=keep_gemma,
    ),
}


@dataclass(frozen=True, slots=True)
class TensorRecord:
    name: str
    dtype: str
    shape: tuple[int, ...]
    source_start: int
    source_end: int

    @property
    def size(self) -> int:
        return self.source_end - self.source_start


@dataclass(frozen=True, slots=True)
class SourceShard:
    filename: str
    url: str
    total_size: int
    data_start: int
    header_sha256: str
    metadata: JsonObject | None
    tensors: tuple[TensorRecord, ...]


@dataclass(frozen=True, slots=True)
class SourceRange:
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_object(raw: bytes, description: str) -> JsonObject:
    try:
        value = json.loads(raw, object_pairs_hook=_object_without_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return cast(JsonObject, value)


def _tensor_record(name: str, raw: object, data_size: int) -> TensorRecord:
    if not isinstance(raw, dict) or set(raw) != {"dtype", "shape", "data_offsets"}:
        raise ValueError(f"tensor {name!r} has an invalid safetensors header entry")
    entry = cast(dict[str, object], raw)
    dtype = entry["dtype"]
    shape = entry["shape"]
    offsets = entry["data_offsets"]
    if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
        raise ValueError(f"tensor {name!r} has unsupported dtype {dtype!r}")
    if not isinstance(shape, list) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in shape
    ):
        raise ValueError(f"tensor {name!r} has an invalid shape")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
    ):
        raise ValueError(f"tensor {name!r} has invalid data offsets")
    checked_shape = cast(list[int], shape)
    start, end = cast(list[int], offsets)
    if start < 0 or end < start or end > data_size:
        raise ValueError(f"tensor {name!r} has out-of-bounds data offsets")
    expected = math.prod(checked_shape) * _DTYPE_BYTES[dtype]
    if end - start != expected:
        raise ValueError(
            f"tensor {name!r} byte size {end - start} does not match {dtype} shape {shape}"
        )
    return TensorRecord(name, dtype, tuple(checked_shape), start, end)


def _range_response(
    session: HttpSession,
    url: str,
    start: int,
    end: int,
    *,
    token: str | None,
    timeout: float,
    expected_total: int | None = None,
) -> HttpResponse:
    if start < 0 or end < start:
        raise ValueError(f"invalid requested byte range {start}-{end}")
    headers = {
        "Accept-Encoding": "identity",
        "Range": f"bytes={start}-{end}",
        "User-Agent": "reasonese-probe-prefix/1",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = session.get(url, headers=headers, stream=True, timeout=(30, timeout))
    if response.status_code != 206:
        response.close()
        raise RuntimeError(
            f"range request returned HTTP {response.status_code}, expected 206: {url}"
        )
    content_range = response.headers.get("Content-Range", "")
    match = _CONTENT_RANGE.fullmatch(content_range.strip())
    if match is None:
        response.close()
        raise RuntimeError(f"range response has invalid Content-Range {content_range!r}: {url}")
    actual_start, actual_end, total = (int(value) for value in match.groups())
    if (actual_start, actual_end) != (start, end):
        response.close()
        raise RuntimeError(
            f"range response returned bytes {actual_start}-{actual_end}, expected {start}-{end}: {url}"
        )
    if expected_total is not None and total != expected_total:
        response.close()
        raise RuntimeError(
            f"range response total size changed from {expected_total} to {total}: {url}"
        )
    content_length = response.headers.get("Content-Length")
    if content_length is not None and int(content_length) != end - start + 1:
        response.close()
        raise RuntimeError(f"range response has incorrect Content-Length {content_length}: {url}")
    return response


def _read_range(
    session: HttpSession,
    url: str,
    start: int,
    end: int,
    *,
    token: str | None,
    timeout: float,
    expected_total: int | None = None,
) -> tuple[bytes, int]:
    response = _range_response(
        session, url, start, end, token=token, timeout=timeout, expected_total=expected_total
    )
    try:
        body = response.content
        match = _CONTENT_RANGE.fullmatch(response.headers["Content-Range"].strip())
        if match is None:  # _range_response already validates this invariant.
            raise RuntimeError("validated Content-Range disappeared from response")
        total = int(match.group(3))
    finally:
        response.close()
    if len(body) != end - start + 1:
        raise RuntimeError(
            f"range response body has {len(body)} bytes, expected {end - start + 1}: {url}"
        )
    return body, total


def inspect_source_shard(
    session: HttpSession,
    filename: str,
    url: str,
    *,
    token: str | None,
    timeout: float,
) -> SourceShard:
    """Read and fully validate one remote safetensors header using two range requests."""
    prefix, total_size = _read_range(session, url, 0, 7, token=token, timeout=timeout)
    header_size = struct.unpack("<Q", prefix)[0]
    if header_size == 0 or header_size > _MAX_HEADER_BYTES:
        raise ValueError(f"{filename} declares invalid header size {header_size}")
    data_start = 8 + header_size
    if data_start > total_size:
        raise ValueError(f"{filename} header extends beyond the source file")
    raw_header, _ = _read_range(
        session,
        url,
        8,
        data_start - 1,
        token=token,
        timeout=timeout,
        expected_total=total_size,
    )
    header = _json_object(raw_header, f"{filename} safetensors header")
    metadata = header.pop("__metadata__", None)
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError(f"{filename} __metadata__ must be an object")
    data_size = total_size - data_start
    tensors = tuple(_tensor_record(name, raw, data_size) for name, raw in header.items())
    by_offset = sorted(tensors, key=lambda tensor: (tensor.source_start, tensor.source_end))
    cursor = 0
    for tensor in by_offset:
        if tensor.source_start != cursor:
            raise ValueError(
                f"{filename} tensor data has a gap or overlap before {tensor.name!r}: "
                f"expected offset {cursor}, found {tensor.source_start}"
            )
        cursor = tensor.source_end
    if cursor != data_size:
        raise ValueError(
            f"{filename} tensor data covers {cursor} bytes but source data has {data_size}"
        )
    return SourceShard(
        filename,
        url,
        total_size,
        data_start,
        hashlib.sha256(raw_header).hexdigest(),
        cast(JsonObject | None, metadata),
        tensors,
    )


def _filtered_header(
    source: SourceShard, selected_names: set[str]
) -> tuple[bytes, tuple[TensorRecord, ...]]:
    selected = tuple(
        sorted(
            (tensor for tensor in source.tensors if tensor.name in selected_names),
            key=lambda tensor: (tensor.source_start, tensor.source_end, tensor.name),
        )
    )
    if {tensor.name for tensor in selected} != selected_names:
        missing = sorted(selected_names - {tensor.name for tensor in selected})
        raise ValueError(f"{source.filename} header is missing indexed tensors: {missing}")
    header: JsonObject = {}
    if source.metadata is not None:
        header["__metadata__"] = source.metadata
    cursor = 0
    for tensor in selected:
        header[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [cursor, cursor + tensor.size],
        }
        cursor += tensor.size
    raw = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw, selected


def _coalesced_ranges(
    source: SourceShard, tensors: tuple[TensorRecord, ...]
) -> tuple[SourceRange, ...]:
    ranges: list[SourceRange] = []
    for tensor in tensors:
        if tensor.size == 0:
            continue
        start = source.data_start + tensor.source_start
        end = source.data_start + tensor.source_end - 1
        if ranges and ranges[-1].end + 1 == start:
            ranges[-1] = SourceRange(ranges[-1].start, end)
        else:
            ranges.append(SourceRange(start, end))
    return tuple(ranges)


def _stream_to(response: HttpResponse, output: BinaryIO, expected: int, *, chunk_size: int) -> str:
    digest = hashlib.sha256()
    written = 0
    try:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            written += len(chunk)
            if written > expected:
                raise RuntimeError("range response body is longer than Content-Range")
            digest.update(chunk)
            output.write(chunk)
    finally:
        response.close()
    if written != expected:
        raise RuntimeError(f"range response body has {written} bytes, expected {expected}")
    return digest.hexdigest()


def _remaining_ranges(ranges: tuple[SourceRange, ...], completed: int) -> Iterator[SourceRange]:
    cursor = 0
    for source_range in ranges:
        next_cursor = cursor + source_range.size
        if completed < next_cursor:
            consumed = max(0, completed - cursor)
            yield SourceRange(source_range.start + consumed, source_range.end)
        cursor = next_cursor
    if completed > cursor:
        raise ValueError(
            f"partial output contains {completed} data bytes, expected at most {cursor}"
        )


def _receipt_path(path: Path) -> Path:
    return path.with_name(path.name + ".receipt")


def _receipt_json(path: Path, kind: str) -> JsonObject:
    try:
        value = _json_object(path.read_bytes(), f"{kind} integrity receipt")
    except OSError as error:
        raise ValueError(f"{kind} integrity receipt cannot be read: {path}") from error
    if value.get("format_version") != 1 or value.get("kind") != kind:
        raise ValueError(f"unsupported {kind} integrity receipt: {path}")
    return value


def _chunk_record(source_range: SourceRange, digest: str) -> JsonObject:
    return {
        "source_start": source_range.start,
        "source_end": source_range.end,
        "sha256": digest,
    }


def _receipt_chunks(
    receipt: JsonObject, ranges: tuple[SourceRange, ...], *, prefix_bytes: int, path: Path
) -> int:
    raw_chunks = receipt.get("chunks")
    if not isinstance(raw_chunks, list) or len(raw_chunks) > len(ranges):
        raise ValueError(f"invalid integrity receipt chunks: {path}")
    completed = 0
    for index, raw_chunk in enumerate(raw_chunks):
        if not isinstance(raw_chunk, dict):
            raise ValueError(f"invalid integrity receipt chunk: {path}")
        chunk = cast(dict[str, Any], raw_chunk)
        expected = ranges[index]
        if (
            chunk.get("source_start") != expected.start
            or chunk.get("source_end") != expected.end
            or not isinstance(chunk.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", cast(str, chunk["sha256"]))
        ):
            raise ValueError(f"integrity receipt chunk does not match its source range: {path}")
        completed += expected.size

    if receipt.get("completed_data_bytes") != completed:
        raise ValueError(f"integrity receipt completed byte count is invalid: {path}")
    if receipt.get("prefix_bytes") != prefix_bytes:
        raise ValueError(f"integrity receipt prefix size is invalid: {path}")
    return completed


def _matches_expected_bytes(handle: BinaryIO, expected: bytes) -> bool:
    offset = 0
    while offset < len(expected):
        requested = min(_STREAM_CHUNK_BYTES, len(expected) - offset)
        chunk = handle.read(requested)
        if not chunk or len(chunk) > requested:
            return False
        if chunk != expected[offset : offset + len(chunk)]:
            return False
        offset += len(chunk)
    return True


def _sha256_region(handle: BinaryIO, expected_size: int) -> str:
    digest = hashlib.sha256()
    remaining = expected_size
    while remaining:
        chunk = handle.read(min(_STREAM_CHUNK_BYTES, remaining))
        if not chunk or len(chunk) > remaining:
            raise ValueError("integrity receipt covers bytes missing from file")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def _verify_receipt_payload(
    path: Path,
    receipt: JsonObject,
    ranges: tuple[SourceRange, ...],
    *,
    prefix: bytes,
    expected_size: int,
    complete: bool,
) -> int:
    """Verify the bytes covered by a receipt and return completed data bytes."""
    if receipt.get("expected_size") != expected_size:
        raise ValueError(f"integrity receipt expected size does not match source: {path}")
    legacy_manifest = complete and receipt.get("legacy_manifest") is True
    if legacy_manifest:
        if receipt.get("chunks") != []:
            raise ValueError(f"invalid legacy integrity receipt chunks: {path}")
        completed = sum(source_range.size for source_range in ranges)
        if receipt.get("completed_data_bytes") != completed:
            raise ValueError(f"legacy integrity receipt byte count is invalid: {path}")
        if receipt.get("prefix_bytes") != len(prefix):
            raise ValueError(f"legacy integrity receipt prefix size is invalid: {path}")
    else:
        completed = _receipt_chunks(receipt, ranges, prefix_bytes=len(prefix), path=path)
    minimum_size = len(prefix) + completed
    actual_size = path.stat().st_size
    if actual_size < minimum_size:
        raise ValueError(f"integrity receipt covers bytes missing from {path}")
    if complete:
        if completed != sum(source_range.size for source_range in ranges):
            raise ValueError(f"completed integrity receipt is incomplete: {path}")
        if actual_size != expected_size:
            raise ValueError(f"completed file size does not match its integrity receipt: {path}")
    with path.open("rb") as handle:
        if not _matches_expected_bytes(handle, prefix):
            raise ValueError(f"integrity receipt prefix does not match source selection: {path}")
        for raw_chunk, source_range in zip(
            cast(list[dict[str, Any]], receipt["chunks"]), ranges, strict=False
        ):
            try:
                digest = _sha256_region(handle, source_range.size)
            except ValueError as error:
                raise ValueError(f"integrity receipt covers bytes missing from {path}") from error
            if digest != raw_chunk["sha256"]:
                raise ValueError(f"integrity receipt payload checksum mismatch: {path}")
        if complete:
            file_sha256 = receipt.get("file_sha256")
            if not isinstance(file_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", file_sha256):
                raise ValueError(f"completed integrity receipt lacks a file checksum: {path}")
            if _sha256_file(path) != file_sha256:
                raise ValueError(f"completed file checksum does not match its receipt: {path}")
    return completed


def _new_shard_receipt(
    source: SourceShard,
    prefix: bytes,
    ranges: tuple[SourceRange, ...],
    expected_size: int,
    chunks: list[JsonObject],
) -> JsonObject:
    return {
        "format_version": 1,
        "kind": "filtered-shard",
        "source_filename": source.filename,
        "source_url": source.url,
        "source_total_bytes": source.total_size,
        "source_header_sha256": source.header_sha256,
        "selected_header_sha256": hashlib.sha256(prefix[8:]).hexdigest(),
        "prefix_bytes": len(prefix),
        "expected_size": expected_size,
        "source_ranges": [[item.start, item.end] for item in ranges],
        "chunks": chunks,
        "completed_data_bytes": sum(
            item["source_end"] - item["source_start"] + 1 for item in chunks
        ),
    }


def _validate_shard_receipt(
    path: Path,
    receipt: JsonObject,
    source: SourceShard,
    prefix: bytes,
    ranges: tuple[SourceRange, ...],
    expected_size: int,
    *,
    complete: bool,
) -> int:
    expected_ranges = [[item.start, item.end] for item in ranges]
    if (
        receipt.get("source_filename") != source.filename
        or receipt.get("source_url") != source.url
        or receipt.get("source_total_bytes") != source.total_size
        or receipt.get("source_header_sha256") != source.header_sha256
        or receipt.get("selected_header_sha256") != hashlib.sha256(prefix[8:]).hexdigest()
        or receipt.get("source_ranges") != expected_ranges
    ):
        raise ValueError(f"integrity receipt does not match source selection: {path}")
    return _verify_receipt_payload(
        path, receipt, ranges, prefix=prefix, expected_size=expected_size, complete=complete
    )


def _shard_receipt_from_manifest(
    record: Mapping[str, Any],
    source: SourceShard,
    prefix: bytes,
    ranges: tuple[SourceRange, ...],
    expected_size: int,
) -> JsonObject:
    receipt = _new_shard_receipt(source, prefix, ranges, expected_size, [])
    fields = (
        "source_url",
        "source_total_bytes",
        "source_header_sha256",
        "selected_header_sha256",
        "source_ranges",
    )
    if record.get("filename") != source.filename or any(
        record.get(field) != receipt[field] for field in fields
    ):
        raise ValueError("existing shard manifest record does not match source selection")
    digest = record.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("existing shard manifest record lacks a valid file checksum")
    receipt["completed_data_bytes"] = sum(item.size for item in ranges)
    receipt["file_sha256"] = digest
    receipt["legacy_manifest"] = True
    return receipt


def _new_auxiliary_receipt(
    url: str, total_size: int, ranges: tuple[SourceRange, ...], chunks: list[JsonObject]
) -> JsonObject:
    return {
        "format_version": 1,
        "kind": "auxiliary-file",
        "source_url": url,
        "source_total_bytes": total_size,
        "prefix_bytes": 0,
        "expected_size": total_size,
        "source_ranges": [[item.start, item.end] for item in ranges],
        "chunks": chunks,
        "completed_data_bytes": sum(
            item["source_end"] - item["source_start"] + 1 for item in chunks
        ),
    }


def _validate_auxiliary_receipt(
    path: Path,
    receipt: JsonObject,
    url: str,
    total_size: int,
    ranges: tuple[SourceRange, ...],
    *,
    complete: bool,
) -> int:
    if (
        receipt.get("source_url") != url
        or receipt.get("source_total_bytes") != total_size
        or receipt.get("source_ranges") != [[item.start, item.end] for item in ranges]
    ):
        raise ValueError(f"integrity receipt does not match auxiliary source: {path}")
    chunks = receipt.get("chunks")
    if not (complete and receipt.get("legacy_manifest") is True and chunks == []) and (
        not isinstance(chunks, list)
        or not chunks
        or not isinstance(chunks[0], dict)
        or chunks[0].get("source_start") != 0
    ):
        raise ValueError(f"auxiliary integrity receipt must begin at byte zero: {path}")
    return _verify_receipt_payload(
        path, receipt, ranges, prefix=b"", expected_size=total_size, complete=complete
    )


def _auxiliary_receipt_from_manifest(
    record: Mapping[str, Any],
    filename: str,
    url: str,
    total_size: int,
    ranges: tuple[SourceRange, ...],
) -> JsonObject:
    receipt = _new_auxiliary_receipt(url, total_size, ranges, [])
    if (
        record.get("filename") != filename
        or record.get("source_url") != url
        or record.get("bytes") != total_size
    ):
        raise ValueError("existing auxiliary manifest record does not match source")
    digest = record.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("existing auxiliary manifest record lacks a valid file checksum")
    receipt["completed_data_bytes"] = total_size
    receipt["file_sha256"] = digest
    receipt["legacy_manifest"] = True
    return receipt


def download_filtered_shard(
    session: HttpSession,
    source: SourceShard,
    selected_names: set[str],
    output_path: Path,
    *,
    token: str | None,
    timeout: float,
    chunk_size: int = _STREAM_CHUNK_BYTES,
    expected_record: Mapping[str, Any] | None = None,
) -> JsonObject:
    """Download selected tensor ranges into one resumable, atomically published shard."""
    prefix, tensors = _filtered_header(source, selected_names)
    ranges = _coalesced_ranges(source, tensors)
    data_size = sum(tensor.size for tensor in tensors)
    expected_size = len(prefix) + data_size
    partial = output_path.with_name(output_path.name + ".partial")
    output_receipt = _receipt_path(output_path)
    partial_receipt = _receipt_path(partial)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        if not output_receipt.exists():
            if partial_receipt.exists():
                # Recover the narrow crash window between publishing the shard and its receipt.
                receipt = _receipt_json(partial_receipt, "filtered-shard")
                _validate_shard_receipt(
                    output_path,
                    receipt,
                    source,
                    prefix,
                    ranges,
                    expected_size,
                    complete=True,
                )
                os.replace(partial_receipt, output_receipt)
            elif expected_record is not None:
                receipt = _shard_receipt_from_manifest(
                    expected_record, source, prefix, ranges, expected_size
                )
                _validate_shard_receipt(
                    output_path,
                    receipt,
                    source,
                    prefix,
                    ranges,
                    expected_size,
                    complete=True,
                )
                _atomic_json(output_receipt, receipt)
            else:
                raise ValueError(
                    f"completed shard has no integrity receipt; remove or verify it: {output_path}"
                )
        receipt = _receipt_json(output_receipt, "filtered-shard")
        _validate_shard_receipt(
            output_path, receipt, source, prefix, ranges, expected_size, complete=True
        )
        return _shard_manifest(source, tensors, ranges, prefix)

    if partial.exists():
        if not partial_receipt.exists():
            raise ValueError(
                f"partial shard has no integrity receipt; remove or verify it: {partial}"
            )
        receipt = _receipt_json(partial_receipt, "filtered-shard")
        completed = _validate_shard_receipt(
            partial, receipt, source, prefix, ranges, expected_size, complete=False
        )
        with partial.open("r+b") as handle:
            handle.truncate(len(prefix) + completed)
        chunks = cast(list[JsonObject], receipt["chunks"])
    else:
        with partial.open("xb") as handle:
            handle.write(prefix)
            handle.flush()
            os.fsync(handle.fileno())
        completed = 0
        chunks = []
        _atomic_json(
            partial_receipt,
            _new_shard_receipt(source, prefix, ranges, expected_size, chunks),
        )

    remaining = tuple(_remaining_ranges(ranges, completed))
    with partial.open("ab") as handle:
        for index, source_range in enumerate(remaining, start=1):
            print(
                f"{source.filename}: range {index}/{len(remaining)} "
                f"bytes={source_range.start}-{source_range.end}",
                file=sys.stderr,
                flush=True,
            )
            response = _range_response(
                session,
                source.url,
                source_range.start,
                source_range.end,
                token=token,
                timeout=timeout,
                expected_total=source.total_size,
            )
            digest = _stream_to(response, handle, source_range.size, chunk_size=chunk_size)
            handle.flush()
            os.fsync(handle.fileno())
            chunks.append(_chunk_record(source_range, digest))
            _atomic_json(
                partial_receipt,
                _new_shard_receipt(source, prefix, ranges, expected_size, chunks),
            )

    if partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"filtered shard has {partial.stat().st_size} bytes, expected {expected_size}: {partial}"
        )
    receipt = _new_shard_receipt(source, prefix, ranges, expected_size, chunks)
    receipt["file_sha256"] = _sha256_file(partial)
    _atomic_json(partial_receipt, receipt)
    os.replace(partial, output_path)
    os.replace(partial_receipt, output_receipt)
    return _shard_manifest(source, tensors, ranges, prefix)


def _shard_manifest(
    source: SourceShard,
    tensors: tuple[TensorRecord, ...],
    ranges: tuple[SourceRange, ...],
    prefix: bytes,
) -> JsonObject:
    return {
        "filename": source.filename,
        "source_url": source.url,
        "source_total_bytes": source.total_size,
        "source_header_sha256": source.header_sha256,
        "selected_header_sha256": hashlib.sha256(prefix[8:]).hexdigest(),
        "selected_tensors": len(tensors),
        "selected_parameters": sum(math.prod(tensor.shape) for tensor in tensors),
        "selected_data_bytes": sum(tensor.size for tensor in tensors),
        "source_ranges": [[item.start, item.end] for item in ranges],
    }


def _atomic_json(path: Path, value: object) -> None:
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_STREAM_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _weights_identity(index_path: Path, shard_paths: Sequence[Path]) -> tuple[str, dict[str, str]]:
    """Hash the exact filtered index and shards through a small canonical identity record."""
    shard_hashes = {path.name: _sha256_file(path) for path in sorted(shard_paths)}
    identity = {
        "index_sha256": _sha256_file(index_path),
        "shard_sha256": shard_hashes,
    }
    canonical = json.dumps(identity, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), shard_hashes


def _copy_metadata(metadata_dir: Path, output_dir: Path) -> None:
    for name in ("config.json", "tokenizer_config.json", "chat_template.jinja", "metadata.json"):
        source = metadata_dir / name
        if not source.is_file():
            raise ValueError(f"required metadata file does not exist: {source}")
        destination = output_dir / name
        data = source.read_bytes()
        if destination.exists() and destination.read_bytes() == data:
            continue
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, destination)


def _existing_manifest_records(
    output_dir: Path,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    """Read prior complete-file checksums for safe migration to receipt sidecars."""
    path = output_dir / "prefix-checkpoint-manifest.json"
    if not path.exists():
        return {}, {}
    manifest = _json_object(path.read_bytes(), str(path))

    def records(name: str) -> dict[str, Mapping[str, Any]]:
        value = manifest.get(name)
        if not isinstance(value, list):
            raise ValueError(f"existing prefix checkpoint manifest lacks {name}")
        result: dict[str, Mapping[str, Any]] = {}
        for record in value:
            if not isinstance(record, dict) or not isinstance(record.get("filename"), str):
                raise ValueError(
                    f"existing prefix checkpoint manifest has an invalid {name} record"
                )
            filename = cast(str, record["filename"])
            if filename in result:
                raise ValueError(f"existing prefix checkpoint manifest repeats {filename}")
            result[filename] = record
        return result

    return records("shards"), records("auxiliary_files")


def _download_auxiliary_file(
    session: HttpSession,
    url: str,
    output_path: Path,
    *,
    token: str | None,
    timeout: float,
    expected_record: Mapping[str, Any] | None = None,
) -> JsonObject:
    """Download a bounded non-weight artifact through a validated, resumable byte range."""
    first_byte, total_size = _read_range(session, url, 0, 0, token=token, timeout=timeout)
    if total_size <= 0 or total_size > _MAX_AUXILIARY_BYTES:
        raise ValueError(f"auxiliary file size {total_size} is outside the allowed range: {url}")
    partial = output_path.with_name(output_path.name + ".partial")
    output_receipt = _receipt_path(output_path)
    partial_receipt = _receipt_path(partial)
    ranges = (
        (SourceRange(0, 0),)
        if total_size == 1
        else (SourceRange(0, 0), SourceRange(1, total_size - 1))
    )
    if output_path.exists():
        if not output_receipt.exists():
            if partial_receipt.exists():
                receipt = _receipt_json(partial_receipt, "auxiliary-file")
                _validate_auxiliary_receipt(
                    output_path,
                    receipt,
                    url,
                    total_size,
                    ranges,
                    complete=True,
                )
                os.replace(partial_receipt, output_receipt)
            elif expected_record is not None:
                receipt = _auxiliary_receipt_from_manifest(
                    expected_record, output_path.name, url, total_size, ranges
                )
                _validate_auxiliary_receipt(
                    output_path, receipt, url, total_size, ranges, complete=True
                )
                _atomic_json(output_receipt, receipt)
            else:
                raise ValueError(
                    f"completed auxiliary file has no integrity receipt; "
                    f"remove or verify it: {output_path}"
                )
        receipt = _receipt_json(output_receipt, "auxiliary-file")
        _validate_auxiliary_receipt(output_path, receipt, url, total_size, ranges, complete=True)
        return {
            "filename": output_path.name,
            "bytes": total_size,
            "sha256": _sha256_file(output_path),
            "source_url": url,
        }
    if partial.exists():
        if not partial_receipt.exists():
            raise ValueError(
                f"partial auxiliary file has no integrity receipt; remove or verify it: {partial}"
            )
        receipt = _receipt_json(partial_receipt, "auxiliary-file")
        completed = _validate_auxiliary_receipt(
            partial, receipt, url, total_size, ranges, complete=False
        )
        with partial.open("r+b") as handle:
            handle.truncate(completed)
        chunks = cast(list[JsonObject], receipt["chunks"])
    else:
        with partial.open("xb") as handle:
            handle.write(first_byte)
            handle.flush()
            os.fsync(handle.fileno())
        completed = 1
        chunks = [_chunk_record(SourceRange(0, 0), hashlib.sha256(first_byte).hexdigest())]
        _atomic_json(
            partial_receipt,
            _new_auxiliary_receipt(url, total_size, ranges, chunks),
        )

    if completed < total_size:
        response = _range_response(
            session,
            url,
            completed,
            total_size - 1,
            token=token,
            timeout=timeout,
            expected_total=total_size,
        )
        with partial.open("ab") as handle:
            digest = _stream_to(
                response, handle, total_size - completed, chunk_size=_STREAM_CHUNK_BYTES
            )
            handle.flush()
            os.fsync(handle.fileno())
        chunks.append(_chunk_record(SourceRange(completed, total_size - 1), digest))
        _atomic_json(
            partial_receipt,
            _new_auxiliary_receipt(url, total_size, ranges, chunks),
        )

    if partial.stat().st_size != total_size:
        raise RuntimeError(
            f"auxiliary file has {partial.stat().st_size} bytes, expected {total_size}: {partial}"
        )
    receipt = _new_auxiliary_receipt(url, total_size, ranges, chunks)
    receipt["file_sha256"] = _sha256_file(partial)
    _atomic_json(partial_receipt, receipt)
    os.replace(partial, output_path)
    os.replace(partial_receipt, output_receipt)
    return {
        "filename": output_path.name,
        "bytes": total_size,
        "sha256": _sha256_file(output_path),
        "source_url": url,
    }


def _load_inputs(metadata_dir: Path, preset: ModelPreset) -> tuple[JsonObject, JsonObject, str]:
    index_path = metadata_dir / "model.safetensors.index.json"
    metadata_path = metadata_dir / "metadata.json"
    raw_index = index_path.read_bytes()
    index = _json_object(raw_index, str(index_path))
    metadata = _json_object(metadata_path.read_bytes(), str(metadata_path))
    if metadata.get("id") != preset.metadata_model_id or metadata.get("sha") != preset.revision:
        raise ValueError(
            "metadata model id or revision does not match the selected pinned checkpoint: "
            f"expected {preset.metadata_model_id}@{preset.revision}"
        )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model.safetensors.index.json has no non-empty weight_map")
    if any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in weight_map.items()
    ):
        raise ValueError("model.safetensors.index.json weight_map must map strings to strings")
    return index, metadata, hashlib.sha256(raw_index).hexdigest()


def _source_url(preset: ModelPreset, filename: str) -> str:
    encoded_repo = quote(preset.repository, safe="/")
    encoded_revision = quote(preset.revision, safe="")
    encoded_filename = quote(filename, safe="/")
    return f"https://huggingface.co/{encoded_repo}/resolve/{encoded_revision}/{encoded_filename}"


def download_prefix(
    preset: ModelPreset,
    metadata_dir: Path,
    output_dir: Path,
    *,
    token: str | None,
    timeout: float,
    max_workers: int = 6,
    session: HttpSession | None = None,
) -> JsonObject:
    """Download and publish a complete filtered prefix checkpoint."""
    if metadata_dir.resolve() == output_dir.resolve():
        raise ValueError("--output must differ from --metadata-dir")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    index, metadata, index_sha256 = _load_inputs(metadata_dir, preset)
    weight_map = cast(dict[str, str], index["weight_map"])
    selected_map = {key: filename for key, filename in weight_map.items() if preset.keep(key)}
    if not selected_map:
        raise ValueError("prefix selection did not match any tensors")
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_shard_records, prior_auxiliary_records = _existing_manifest_records(output_dir)
    _copy_metadata(metadata_dir, output_dir)

    auxiliary_manifests: list[JsonObject] = []
    if session is not None:
        auxiliary_manifests = [
            _download_auxiliary_file(
                session,
                _source_url(preset, filename),
                output_dir / filename,
                token=token,
                timeout=timeout,
                expected_record=prior_auxiliary_records.get(filename),
            )
            for filename in preset.auxiliary_files
        ]
    else:
        with requests.Session() as auxiliary_session:
            auxiliary_manifests = [
                _download_auxiliary_file(
                    auxiliary_session,
                    _source_url(preset, filename),
                    output_dir / filename,
                    token=token,
                    timeout=timeout,
                    expected_record=prior_auxiliary_records.get(filename),
                )
                for filename in preset.auxiliary_files
            ]

    filenames = sorted(set(selected_map.values()))

    def fetch_shard(filename: str, active_session: HttpSession) -> JsonObject:
        selected_names = {key for key, shard in selected_map.items() if shard == filename}
        url = _source_url(preset, filename)
        source = inspect_source_shard(active_session, filename, url, token=token, timeout=timeout)
        return download_filtered_shard(
            active_session,
            source,
            selected_names,
            output_dir / filename,
            token=token,
            timeout=timeout,
            expected_record=prior_shard_records.get(filename),
        )

    shard_manifests: list[JsonObject]
    if session is not None:
        # An injected session is intentionally serial so deterministic fakes need not be thread-safe.
        shard_manifests = [fetch_shard(filename, session) for filename in filenames]
    else:

        def fetch_with_owned_session(filename: str) -> JsonObject:
            with requests.Session() as active_session:
                return fetch_shard(filename, active_session)

        completed: dict[str, JsonObject] = {}
        with ThreadPoolExecutor(max_workers=min(max_workers, len(filenames))) as executor:
            futures = {
                executor.submit(fetch_with_owned_session, filename): filename
                for filename in filenames
            }
            for future in as_completed(futures):
                filename = futures[future]
                completed[filename] = future.result()
        shard_manifests = [completed[filename] for filename in filenames]

    original_metadata = index.get("metadata")
    filtered_metadata = dict(original_metadata) if isinstance(original_metadata, dict) else {}
    selected_data_sizes = [int(shard["selected_data_bytes"]) for shard in shard_manifests]
    filtered_metadata["total_size"] = sum(selected_data_sizes)
    filtered_metadata["total_parameters"] = sum(
        int(shard["selected_parameters"]) for shard in shard_manifests
    )
    filtered_index = {"metadata": filtered_metadata, "weight_map": selected_map}
    index_path = output_dir / "model.safetensors.index.json"
    _atomic_json(index_path, filtered_index)
    weights_sha256, shard_hashes = _weights_identity(
        index_path, [output_dir / filename for filename in filenames]
    )
    for shard in shard_manifests:
        shard["sha256"] = shard_hashes[cast(str, shard["filename"])]

    manifest: JsonObject = {
        "format_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "repository": preset.repository,
        "revision": preset.revision,
        "model_id": preset.metadata_model_id,
        "metadata_model_id": metadata["id"],
        "adapter": preset.adapter,
        "max_layer": preset.max_layer,
        "source_index_sha256": index_sha256,
        "weights_hash_kind": "sha256-filtered-index-and-shard-files-v1",
        "weights_sha256": weights_sha256,
        "selection": preset.selection,
        "selected_tensor_count": len(selected_map),
        "selected_data_bytes": sum(selected_data_sizes),
        "auxiliary_files": auxiliary_manifests,
        "shards": shard_manifests,
    }
    _atomic_json(output_dir / "prefix-checkpoint-manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(PRESETS), required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--read-timeout", type=float, default=300.0)
    parser.add_argument("--max-workers", type=int, default=6)
    args = parser.parse_args(argv)
    if args.read_timeout <= 0:
        parser.error("--read-timeout must be positive")
    if args.max_workers <= 0:
        parser.error("--max-workers must be positive")
    try:
        manifest = download_prefix(
            PRESETS[args.model],
            args.metadata_dir,
            args.output,
            token=os.environ.get("HF_TOKEN"),
            timeout=args.read_timeout,
            max_workers=args.max_workers,
        )
    except (OSError, requests.RequestException, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
