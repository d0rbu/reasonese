"""Untouched native-dialogue activation extraction and portable artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from beartype import beartype

from reasonese.axes import Assistant
from reasonese.openrouter import validate_completion
from reasonese.probe_rendering import render_native_dialogue_context
from reasonese.role_probe_extraction import (
    EXTRACTION_PROTOCOL,
    ExtractionIdentity,
    NativeTemplateAdapter,
    ProbeRole,
    capture_token_activations,
)
from reasonese.role_probes import (
    CONTENT_TOKENS_ONLY,
    UNTOUCHED_CONVERSATIONS,
    ActivationDataset,
    ActivationProvenance,
)

_FILES = frozenset(
    {
        "activations.npy",
        "document_index.npy",
        "role.npy",
        "content_token_index.npy",
        "content_token_id.npy",
        "sequence_token_index.npy",
        "documents.jsonl",
    }
)
_FREE_ROUTES = {
    Assistant.NEMOTRON_3_5_LIGHTNING: "nvidia/nemotron-3.5-lightning:free",
    Assistant.GEMMA_4_31B_IT: "google/gemma-4-31b-it:free",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error


@beartype
@dataclass(frozen=True, slots=True)
class NativeDialogue:
    """One pinned hosted response used only for zero-shot probe validation."""

    document_id: str
    split: str
    prompt: str
    reasoning: str
    final: str
    source_file: str
    source_sha256: str
    route: str
    provider: str
    response_model: str
    prompt_source: str
    prompt_revision: str
    request_sha256: str

    def __post_init__(self) -> None:
        if self.split not in {"calibration", "test"}:
            raise ValueError("native dialogue split must be calibration or test")
        for name in (
            "document_id",
            "source_file",
            "route",
            "provider",
            "response_model",
            "prompt_source",
            "prompt_revision",
        ):
            value = getattr(self, name)
            if not value or value.strip() != value:
                raise ValueError(f"native dialogue {name} must be non-empty and trimmed")
        for name in ("prompt", "reasoning", "final"):
            value = getattr(self, name)
            if not value or not value.strip():
                raise ValueError(f"native dialogue {name} must contain non-whitespace text")
        for name in ("source_sha256", "request_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"native dialogue {name} must be a SHA-256 digest")


@beartype
def load_native_dialogues(
    paths: tuple[Path, ...],
    *,
    assistant: Assistant,
    split: str,
    prompt_partitions: Path,
) -> tuple[NativeDialogue, ...]:
    """Load exactly one frozen partition without retaining raw text in artifacts."""
    raw_partitions = _json(prompt_partitions)
    if not isinstance(raw_partitions, list):
        raise ValueError("native prompt partitions must be a list")
    partitions: dict[str, dict[str, Any]] = {}
    for raw in raw_partitions:
        if not isinstance(raw, dict) or set(raw) != {
            "index",
            "source_id",
            "normalized_prompt_sha256",
            "split",
        }:
            raise ValueError("invalid native prompt partition record")
        record = cast(dict[str, Any], raw)
        source_id = record["source_id"]
        if (
            not isinstance(source_id, str)
            or source_id in partitions
            or not isinstance(record["index"], int)
            or isinstance(record["index"], bool)
            or record["split"] not in {"calibration", "test"}
            or not isinstance(record["normalized_prompt_sha256"], str)
        ):
            raise ValueError("native prompt partition IDs must be distinct strings")
        normalized_digest = record["normalized_prompt_sha256"]
        if len(normalized_digest) != 64 or any(
            character not in "0123456789abcdef" for character in normalized_digest
        ):
            raise ValueError("native prompt partition contains an invalid prompt digest")
        partitions[source_id] = record
    if (
        len(partitions) != 24
        or {record["index"] for record in partitions.values()} != set(range(24))
        or sum(record["split"] == "calibration" for record in partitions.values()) != 12
        or sum(record["split"] == "test" for record in partitions.values()) != 12
    ):
        raise ValueError("native prompt partition must contain the frozen 12/12 assignment")

    dialogues: list[NativeDialogue] = []
    for path in paths:
        raw = _json(path)
        if not isinstance(raw, dict) or set(raw) != {
            "assistant",
            "route",
            "prompt",
            "request",
            "response",
        }:
            raise ValueError(f"invalid native dialogue record: {path}")
        record = cast(dict[str, Any], raw)
        if record["assistant"] != str(assistant):
            raise ValueError(f"native dialogue assistant mismatch: {path}")
        prompt = record["prompt"]
        request = record["request"]
        response = record["response"]
        if (
            not isinstance(prompt, dict)
            or not isinstance(request, dict)
            or not isinstance(response, dict)
        ):
            raise ValueError(f"invalid native dialogue objects: {path}")
        prompt = cast(dict[str, Any], prompt)
        request = cast(dict[str, Any], request)
        response = cast(dict[str, Any], response)
        document_id = prompt.get("id")
        prompt_text = prompt.get("text")
        partition = partitions.get(document_id) if isinstance(document_id, str) else None
        if partition is None or prompt.get("split") != partition["split"]:
            raise ValueError(f"native dialogue partition metadata disagrees: {path}")
        if partition["split"] != split:
            continue
        if not isinstance(prompt_text, str):
            raise ValueError(f"native dialogue prompt text is invalid: {path}")
        normalized_prompt = " ".join(prompt_text.lower().split())
        if (
            hashlib.sha256(normalized_prompt.encode()).hexdigest()
            != partition["normalized_prompt_sha256"]
        ):
            raise ValueError(f"native dialogue prompt text differs from its partition: {path}")
        document_id = cast(str, document_id)
        expected_request = {
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0.7,
            "reasoning": {"enabled": True, "exclude": False},
        }
        if request != expected_request:
            raise ValueError(f"native dialogue request does not match frozen protocol: {path}")
        validate_completion(response)
        route = record["route"]
        prompt_source = prompt.get("source")
        prompt_revision = prompt.get("revision")
        provider = response.get("provider")
        response_model = response.get("model")
        if (
            route != _FREE_ROUTES[assistant]
            or response_model != route
            or any(
                not isinstance(value, str) or not value or value.strip() != value
                for value in (provider, prompt_source, prompt_revision)
            )
        ):
            raise ValueError(f"native dialogue source identity is invalid: {path}")
        try:
            choices = response["choices"]
            message = choices[0]["message"]
            reasoning = message["reasoning"]
            final = message["content"]
        except (IndexError, KeyError, TypeError) as error:
            raise ValueError(
                f"native dialogue response lacks reasoning or final text: {path}"
            ) from error
        if not isinstance(reasoning, str) or not isinstance(final, str):
            raise ValueError(f"native dialogue response spans must be text: {path}")
        dialogues.append(
            NativeDialogue(
                document_id,
                split,
                prompt_text,
                reasoning,
                final,
                path.name,
                _file_sha256(path),
                cast(str, route),
                cast(str, provider),
                cast(str, response_model),
                cast(str, prompt_source),
                cast(str, prompt_revision),
                _canonical_sha256(request),
            )
        )
    if len(dialogues) != 12 or len({row.document_id for row in dialogues}) != 12:
        raise ValueError("native extraction requires exactly 12 distinct dialogues per split")
    return tuple(sorted(dialogues, key=lambda row: row.document_id))


@beartype
def extract_native_activations(
    model: object,
    tokenizer: object,
    adapter: NativeTemplateAdapter,
    dialogues: tuple[NativeDialogue, ...],
    *,
    layers: tuple[int, ...],
    identity: ExtractionIdentity,
    activation_dtype: str = "float32",
) -> ActivationDataset:
    """Replay full native dialogues and retain reasoning/final content tokens."""
    if activation_dtype != "float32":
        raise ValueError("native qualification protocol requires float32 activation storage")
    if not dialogues or len({row.split for row in dialogues}) != 1:
        raise ValueError("native dialogues must contain one non-empty frozen split")
    activations: list[np.ndarray] = []
    document_ids: list[str] = []
    roles: list[str] = []
    content_indices: list[int] = []
    content_token_ids: list[int] = []
    sequence_indices: list[int] = []
    masked = 0
    for dialogue in dialogues:
        rendered = render_native_dialogue_context(
            tokenizer,
            adapter,
            prompt=dialogue.prompt,
            reasoning=dialogue.reasoning,
            final=dialogue.final,
        )
        positions = tuple(position for span in rendered.token_positions for position in span)
        captured = capture_token_activations(
            model,
            adapter,
            input_ids=rendered.input_ids,
            token_positions=positions,
            layers=layers,
        )
        if captured.shape[0] != len(positions):
            raise ValueError("native activation capture returned the wrong token count")
        activations.append(captured.astype(activation_dtype, copy=False))
        masked += len(rendered.input_ids) - len(positions)
        cursor = 0
        for role, span, token_ids in zip(
            ("reasoning", "assistant"),
            rendered.token_positions,
            rendered.content_token_ids,
            strict=True,
        ):
            count = len(span)
            document_ids.extend((dialogue.document_id,) * count)
            roles.extend((role,) * count)
            content_indices.extend(range(count))
            content_token_ids.extend(token_ids)
            sequence_indices.extend(span)
            cursor += count
        if cursor != captured.shape[0]:
            raise ValueError("native span rows do not cover every captured activation")
    source_records = [
        {
            "document_id": row.document_id,
            "source_file": row.source_file,
            "sha256": row.source_sha256,
            "route": row.route,
            "provider": row.provider,
            "response_model": row.response_model,
            "prompt_source": row.prompt_source,
            "prompt_revision": row.prompt_revision,
            "request_sha256": row.request_sha256,
        }
        for row in dialogues
    ]
    provenance = ActivationProvenance(
        dataset_kind=UNTOUCHED_CONVERSATIONS,
        model_id=adapter.model_id,
        model_revision=adapter.model_revision,
        weights_sha256=identity.weights_sha256,
        weights_hash_kind=identity.weights_hash_kind,
        tokenizer_id=identity.tokenizer_id,
        tokenizer_revision=identity.tokenizer_revision,
        chat_template_sha256=adapter.chat_template_sha256,
        native_template_adapter=adapter.name,
        activation_site=adapter.activation_site,
        runtime_sha256=identity.runtime_sha256,
        model_dtype=identity.model_dtype,
        activation_dtype=activation_dtype,
        layer_indices=layers,
        hidden_size=activations[0].shape[2],
        roles=tuple(str(role) for role in ProbeRole),
        source_name=f"hosted-native-dialogues-{dialogues[0].split}",
        source_sha256=_canonical_sha256(source_records),
        extraction_protocol=EXTRACTION_PROTOCOL,
        content_mask=CONTENT_TOKENS_ONLY,
        masked_control_tokens=masked,
        masked_filler_tokens=0,
        filler_pool_kind="none",
        filler_source_sha256=None,
        filler_documents=0,
    )
    return ActivationDataset(
        provenance,
        np.concatenate(activations),
        np.asarray(document_ids),
        np.asarray(roles),
        np.asarray(content_indices, dtype=np.int32),
        np.asarray(content_token_ids, dtype=np.int32),
        np.asarray(sequence_indices, dtype=np.int32),
        np.full(len(roles), "", dtype=np.str_),
    )


@beartype
def save_native_activation_dataset(dataset: ActivationDataset, output: Path) -> None:
    """Atomically save a checksummed untouched-conversation activation artifact."""
    if dataset.provenance.dataset_kind != UNTOUCHED_CONVERSATIONS:
        raise ValueError("native artifact requires untouched conversation activations")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite activation artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:

        def int32(values: np.ndarray, name: str) -> np.ndarray:
            limits = np.iinfo(np.int32)
            if np.any(values < limits.min) or np.any(values > limits.max):
                raise ValueError(f"{name} cannot be represented as int32")
            return np.asarray(values, dtype=np.int32)

        documents = tuple(str(value) for value in np.unique(dataset.document_ids))
        document_index = {document: index for index, document in enumerate(documents)}
        arrays = {
            "activations.npy": dataset.activations,
            "document_index.npy": np.asarray(
                [document_index[str(value)] for value in dataset.document_ids], dtype=np.int32
            ),
            "role.npy": np.asarray(
                [dataset.provenance.roles.index(str(value)) for value in dataset.roles],
                dtype=np.uint8,
            ),
            "content_token_index.npy": int32(dataset.content_token_index, "content_token_index"),
            "content_token_id.npy": int32(dataset.content_token_id, "content_token_id"),
            "sequence_token_index.npy": int32(dataset.sequence_token_index, "sequence_token_index"),
        }
        for name, values in arrays.items():
            np.save(temporary / name, values, allow_pickle=False)
        (temporary / "documents.jsonl").write_text(
            "".join(
                json.dumps({"document_index": index, "document_id": document}, sort_keys=True)
                + "\n"
                for index, document in enumerate(documents)
            ),
            encoding="utf-8",
        )
        manifest = {
            "format_version": 1,
            "provenance": asdict(dataset.provenance),
            "rows": len(dataset.roles),
            "documents": len(documents),
            "files": {name: _file_sha256(temporary / name) for name in _FILES},
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


@beartype
def load_native_activation_dataset(path: Path) -> ActivationDataset:
    """Load every checksummed array from an untouched-conversation artifact."""
    manifest = _json(path / "manifest.json")
    if not isinstance(manifest, Mapping) or manifest.get("format_version") != 1:
        raise ValueError("invalid native activation artifact manifest")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != _FILES:
        raise ValueError("native activation artifact file manifest is incomplete")
    actual = {entry.name for entry in path.iterdir() if entry.is_file()}
    if actual != _FILES | {"manifest.json"}:
        raise ValueError("native activation artifact contains unexpected files")
    for name, digest in files.items():
        if (
            not isinstance(name, str)
            or not isinstance(digest, str)
            or _file_sha256(path / name) != digest
        ):
            raise ValueError(f"native activation artifact checksum mismatch: {name}")
    try:
        raw_documents = (path / "documents.jsonl").read_text(encoding="utf-8").splitlines()
        documents = [json.loads(line) for line in raw_documents]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid native activation document mapping") from error
    if any(
        record != {"document_index": index, "document_id": record.get("document_id")}
        or not isinstance(record.get("document_id"), str)
        for index, record in enumerate(documents)
    ):
        raise ValueError("invalid native activation document mapping")
    if manifest.get("documents") != len(documents):
        raise ValueError("native activation document count does not match its mapping")
    rows = manifest.get("rows")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
        raise ValueError("native activation manifest rows must be positive")

    def array(name: str) -> np.ndarray:
        try:
            return np.load(path / name, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid native activation array: {name}") from error

    activations = array("activations.npy")
    document_indices = array("document_index.npy")
    role_codes = array("role.npy")
    content_indices = array("content_token_index.npy")
    token_ids = array("content_token_id.npy")
    sequence_indices = array("sequence_token_index.npy")
    if activations.shape[0] != rows or any(
        values.shape != (rows,)
        for values in (document_indices, role_codes, content_indices, token_ids, sequence_indices)
    ):
        raise ValueError("native activation arrays do not contain the manifest row count")
    provenance_raw = manifest.get("provenance")
    if not isinstance(provenance_raw, Mapping):
        raise ValueError("native activation manifest lacks provenance")
    provenance_values = dict(provenance_raw)
    provenance_values["layer_indices"] = tuple(provenance_values["layer_indices"])
    provenance_values["roles"] = tuple(provenance_values["roles"])
    provenance = ActivationProvenance(**provenance_values)
    if (
        document_indices.dtype != np.int32
        or role_codes.dtype != np.uint8
        or content_indices.dtype != np.int32
        or token_ids.dtype != np.int32
        or sequence_indices.dtype != np.int32
        or np.any(document_indices < 0)
        or np.any(document_indices >= len(documents))
        or np.any(role_codes >= len(provenance.roles))
    ):
        raise ValueError("native activation metadata arrays have invalid dtypes or codes")
    return ActivationDataset(
        provenance,
        activations,
        np.asarray([documents[int(index)]["document_id"] for index in document_indices]),
        np.asarray([provenance.roles[int(index)] for index in role_codes]),
        content_indices,
        token_ids,
        sequence_indices,
        np.full(rows, "", dtype=np.str_),
    )
