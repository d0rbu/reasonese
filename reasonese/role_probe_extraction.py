# ruff: noqa: I001
"""Native-template datasets and activation extraction for reasoning-role probes.

The construction follows Appendix G of *Prompt Injection as Role Confusion*: the
same neutral content is placed in reasoning and final-output roles, all control
and filler tokens are excluded, and content positions are matched exactly.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import re
import resource
import shutil
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
from beartype import beartype

from reasonese.role_probes import (
    CONTENT_TOKENS_ONLY,
    PAIRED_NEUTRAL,
    ActivationDataset,
    ActivationProvenance,
)

EXTRACTION_PROTOCOL = "role-confusion-appendix-g-reasoning-assistant-v1"
PREFIX_WEIGHTS_HASH_KIND = "sha256-filtered-index-and-shard-files-v1"
NEMOTRON_KERNEL_REVISIONS: Mapping[str, tuple[str, str]] = {
    "causal-conv1d": (
        "kernels-community/causal-conv1d",
        "19d5632c27565e020a0aa0169bc7ad44a2c7080b",
    ),
    "mamba-ssm": (
        "kernels-community/mamba-ssm",
        "a39ff24c08103278583f168091409653ada4c292",
    ),
}


class ProbeRole(StrEnum):
    """Native roles used to train and validate the role classifier."""

    SYSTEM = "system"
    USER = "user"
    TOOL = "tool"
    REASONING = "reasoning"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class NativeTemplateAdapter:
    """Pinned native-template and activation-site contract for one model."""

    name: str
    model_id: str
    model_revision: str
    architecture: str
    runtime_architecture: str
    chat_template_sha256: str
    reasoning_field: str
    role_open: str
    prefix_anchor: str
    layer_container: str
    activation_module: str
    activation_site: str
    hidden_size_path: str
    pad_token_id: int


NEMOTRON_ADAPTER = NativeTemplateAdapter(
    name="nemotron-3.5-lightning-native-v1",
    model_id="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    model_revision="a9904d24bcc1d289a1950fa9d2b978c47cf903b9",
    architecture="NemotronHForCausalLM",
    runtime_architecture="NemotronHModel",
    chat_template_sha256="58933db77d3099b4f78c55a38347a72e1ea05b97d6bd8f38775303dc0194e0a9",
    reasoning_field="reasoning_content",
    role_open="<|im_start|>assistant\n",
    prefix_anchor="<|im_start|>",
    layer_container="layers",
    activation_module="norm",
    activation_site="normalized_pre_mixer",
    hidden_size_path="config.hidden_size",
    pad_token_id=0,
)

GEMMA_ADAPTER = NativeTemplateAdapter(
    name="gemma-4-31b-native-v1",
    model_id="google/gemma-4-31B-it",
    model_revision="842da3794eaa0b77d5f08bae87a17459d91ff475",
    architecture="Gemma4ForConditionalGeneration",
    runtime_architecture="Gemma4TextModel",
    chat_template_sha256="ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
    reasoning_field="reasoning",
    role_open="<|turn>model\n",
    prefix_anchor="<|turn>",
    layer_container="layers",
    activation_module="pre_feedforward_layernorm",
    activation_site="normalized_pre_mlp",
    hidden_size_path="config.hidden_size",
    pad_token_id=0,
)

NATIVE_ADAPTERS: Mapping[str, NativeTemplateAdapter] = {
    adapter.name: adapter for adapter in (NEMOTRON_ADAPTER, GEMMA_ADAPTER)
}


@runtime_checkable
class NativeTokenizer(Protocol):
    """Tokenizer operations needed by the dataset builder."""

    chat_template: str

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: bool,
    ) -> str: ...

    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]: ...

    def decode(self, token_ids: Sequence[int], **kwargs: Any) -> str: ...


@dataclass(frozen=True)
class NeutralDocument:
    """One document-grouped neutral-text source record."""

    document_id: str
    text: str
    source: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class RoleExample:
    """One native-tagged role variant and its exact content-token mask."""

    document_index: int
    document_id: str
    role: ProbeRole
    input_ids: tuple[int, ...]
    content_positions: tuple[int, ...]
    content_token_ids: tuple[int, ...]
    filler_positions: tuple[int, ...]
    tag_positions: tuple[int, ...]
    partner_document_id: str
    content_was_truncated: bool

    def __post_init__(self) -> None:
        size = len(self.input_ids)
        if not self.content_positions:
            raise ValueError("an example must contain at least one content token")
        if len(self.content_positions) != len(self.content_token_ids):
            raise ValueError("content positions and token IDs must have equal length")
        if tuple(sorted(self.content_positions)) != self.content_positions:
            raise ValueError("content positions must be sorted")
        if any(position < 0 or position >= size for position in self.content_positions):
            raise ValueError("content position lies outside the token sequence")
        if tuple(self.input_ids[position] for position in self.content_positions) != (
            self.content_token_ids
        ):
            raise ValueError("content token IDs do not match the input sequence")
        partitions = (
            set(self.content_positions) | set(self.filler_positions) | set(self.tag_positions)
        )
        if len(partitions) != size:
            raise ValueError("content, filler, and tag positions must partition the sequence")
        if (
            set(self.content_positions) & set(self.filler_positions)
            or set(self.content_positions) & set(self.tag_positions)
            or set(self.filler_positions) & set(self.tag_positions)
        ):
            raise ValueError("content, filler, and tag positions must be disjoint")


@dataclass(frozen=True)
class RoleDataset:
    """Paired role examples and their construction provenance."""

    documents: tuple[NeutralDocument, ...]
    filler_documents: tuple[NeutralDocument, ...]
    examples: tuple[RoleExample, ...]
    source_sha256: str
    filler_source_sha256: str
    max_content_tokens: int
    max_filler_tokens: int
    max_sequence_tokens: int
    seed: int

    def __post_init__(self) -> None:
        document_ids = tuple(document.document_id for document in self.documents)
        filler_ids = tuple(document.document_id for document in self.filler_documents)
        if len(set(document_ids)) != len(document_ids) or len(set(filler_ids)) != len(filler_ids):
            raise ValueError("target and filler document IDs must each be unique")
        if set(document_ids) & set(filler_ids):
            raise ValueError("target and filler document IDs must be disjoint")
        target_text = {_sha256_text(document.text) for document in self.documents}
        filler_text = {_sha256_text(document.text) for document in self.filler_documents}
        if target_text & filler_text:
            raise ValueError("target and filler document text must be disjoint")
        grouped: dict[int, list[RoleExample]] = {}
        for example in self.examples:
            grouped.setdefault(example.document_index, []).append(example)
        if set(grouped) != set(range(len(self.documents))):
            raise ValueError("every document must have role examples")
        for document_index, pair in grouped.items():
            expected_document = self.documents[document_index].document_id
            if any(example.document_id != expected_document for example in pair):
                raise ValueError(f"document {document_index} examples have the wrong document ID")
            roles = {example.role for example in pair}
            if len(pair) != len(ProbeRole) or roles != set(ProbeRole):
                raise ValueError(f"document {document_index} must have exactly every role")
            reasoning = next(item for item in pair if item.role is ProbeRole.REASONING)
            for example in pair:
                if reasoning.content_token_ids != example.content_token_ids:
                    raise ValueError(f"document {document_index} has unequal paired content tokens")
                if reasoning.content_positions != example.content_positions:
                    raise ValueError(
                        f"document {document_index} has unequal paired token positions"
                    )
                if reasoning.partner_document_id != example.partner_document_id:
                    raise ValueError(f"document {document_index} has unequal filler partners")
            if reasoning.partner_document_id not in set(filler_ids):
                raise ValueError(f"document {document_index} uses an unknown filler partner")


@dataclass(frozen=True)
class ExtractionIdentity:
    """Identity fields that bind activations to exact local model weights."""

    weights_sha256: str
    weights_hash_kind: str
    tokenizer_id: str
    tokenizer_revision: str
    source_name: str
    model_dtype: str
    transformers_version: str
    torch_version: str
    runtime_sha256: str
    runtime: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("weights_sha256", "runtime_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        for name, value in asdict(self).items():
            if name in {"weights_sha256", "runtime_sha256", "runtime"}:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        canonical = json.dumps(self.runtime, separators=(",", ":"), sort_keys=True).encode()
        if hashlib.sha256(canonical).hexdigest() != self.runtime_sha256:
            raise ValueError("runtime_sha256 does not match the canonical runtime record")


@dataclass(frozen=True)
class PrefixCheckpointPlan:
    """Exact source-to-runtime tensor mapping for a prefix-only model."""

    max_layer: int
    key_mapping: Mapping[str, str]


def _layer_number(key: str, prefix: str) -> int | None:
    match = re.match(rf"^{re.escape(prefix)}([0-9]+)\.", key)
    return int(match.group(1)) if match else None


@beartype
def select_prefix_checkpoint_keys(
    weight_map: Mapping[str, str], adapter: NativeTemplateAdapter, *, max_layer: int
) -> PrefixCheckpointPlan:
    """Select only tensors needed to reach and capture ``max_layer``'s probe site."""
    if max_layer < 0:
        raise ValueError("max_layer must be non-negative")
    selected: dict[str, str] = {}
    seen_full_layers: set[int] = set()

    if adapter.name == NEMOTRON_ADAPTER.name:
        embedding = "backbone.embeddings.weight"
        layer_prefix = "backbone.layers."
        if embedding in weight_map:
            selected[embedding] = "embeddings.weight"
        max_key = f"backbone.layers.{max_layer}.norm.weight"
        for key in weight_map:
            layer = _layer_number(key, layer_prefix)
            if layer is not None and layer < max_layer:
                runtime_key = key.removeprefix("backbone.")
                runtime_key = re.sub(
                    r"\.experts\.[0-9]+\.(up_proj|down_proj)\.weight$",
                    r".experts.\1",
                    runtime_key,
                )
                selected[key] = runtime_key
                seen_full_layers.add(layer)
        if max_key in weight_map:
            selected[max_key] = f"layers.{max_layer}.norm.weight"
        required = {embedding, max_key}
    elif adapter.name == GEMMA_ADAPTER.name:
        embedding = "model.language_model.embed_tokens.weight"
        layer_prefix = "model.language_model.layers."
        if embedding in weight_map:
            selected[embedding] = "embed_tokens.weight"
        max_prefix = f"model.language_model.layers.{max_layer}."
        max_parts = (
            "input_layernorm.",
            "self_attn.",
            "post_attention_layernorm.",
            "pre_feedforward_layernorm.",
        )
        for key in weight_map:
            layer = _layer_number(key, layer_prefix)
            if layer is not None and layer < max_layer:
                selected[key] = key.removeprefix("model.language_model.")
                seen_full_layers.add(layer)
            elif key.startswith(max_prefix) and key.removeprefix(max_prefix).startswith(max_parts):
                selected[key] = key.removeprefix("model.language_model.")
        required = {
            embedding,
            f"{max_prefix}input_layernorm.weight",
            f"{max_prefix}post_attention_layernorm.weight",
            f"{max_prefix}pre_feedforward_layernorm.weight",
        }
        has_max_attention = any(key.startswith(f"{max_prefix}self_attn.") for key in selected)
    else:
        raise ValueError(f"unsupported native adapter: {adapter.name}")

    missing = sorted(key for key in required if key not in selected)
    if adapter.name == GEMMA_ADAPTER.name and not has_max_attention:
        missing.append(f"{max_prefix}self_attn.*")
    expected_layers = set(range(max_layer))
    if missing or seen_full_layers != expected_layers:
        details = []
        if missing:
            details.append(f"missing required keys: {missing}")
        if seen_full_layers != expected_layers:
            details.append(
                f"full prefix layers are {sorted(seen_full_layers)}, expected "
                f"{sorted(expected_layers)}"
            )
        raise ValueError("invalid prefix checkpoint plan; " + "; ".join(details))
    return PrefixCheckpointPlan(max_layer=max_layer, key_mapping=selected)


def _read_weight_map(checkpoint: Path) -> dict[str, str]:
    index_path = checkpoint / "model.safetensors.index.json"
    try:
        value = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"invalid checkpoint index: {index_path}") from error
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(filename, str) for key, filename in value.items()
    ):
        raise ValueError(f"invalid weight_map in {index_path}")
    return value


def _delete_prefix_tail(model: Any, adapter: NativeTemplateAdapter, max_layer: int) -> None:
    torch = _torch_runtime()
    layers = _resolve_attribute(model, adapter.layer_container)
    if max_layer >= len(layers):
        raise ValueError(f"max layer {max_layer} lies outside {len(layers)} model layers")
    model.layers = torch.nn.ModuleList(list(layers[: max_layer + 1]))
    final = model.layers[max_layer]
    if adapter.name == NEMOTRON_ADAPTER.name:
        del final.mixer
        del model.norm_f
    elif adapter.name == GEMMA_ADAPTER.name:
        del final.mlp
        del final.post_feedforward_layernorm
        del final.layer_scalar
        del model.norm
    else:  # pragma: no cover - callers are constrained to the two pinned adapters
        raise ValueError(f"unsupported native adapter: {adapter.name}")


def _expert_index(checkpoint_key: str) -> int:
    match = re.search(r"\.experts\.([0-9]+)\.(?:up_proj|down_proj)\.weight$", checkpoint_key)
    if match is None:
        raise ValueError(f"cannot determine packed expert index for {checkpoint_key}")
    return int(match.group(1))


def _pin_nemotron_kernel_revisions() -> None:
    """Replace moving kernel version aliases with exact, audited snapshot revisions."""
    try:
        from huggingface_hub.constants import (
            HF_HUB_CACHE,  # ty: ignore[unresolved-import, unused-ignore-comment]
        )
        from kernels import get_local_kernel  # ty: ignore[unresolved-import, unused-ignore-comment]
        from transformers.integrations.hub_kernels import (  # ty: ignore[unresolved-import, unused-ignore-comment]
            _HUB_KERNEL_MAPPING,
            _KERNEL_MODULE_MAPPING,
        )
    except ImportError as error:  # pragma: no cover - exercised by minimal installations
        raise RuntimeError("Nemotron kernel pinning requires the 'probes' extra") from error
    for name, (repository, revision) in NEMOTRON_KERNEL_REVISIONS.items():
        loaded = _KERNEL_MODULE_MAPPING.get(name)
        if loaded is not None:
            source = getattr(loaded, "__file__", "")
            if f"/snapshots/{revision}/" not in source:
                raise ValueError(f"already-loaded {name} kernel does not match pinned revision")
        _HUB_KERNEL_MAPPING[name] = {"repo_id": repository, "revision": revision}
        snapshot = (
            Path(HF_HUB_CACHE)
            / f"kernels--{repository.replace('/', '--')}"
            / "snapshots"
            / revision
        )
        if loaded is None and snapshot.is_dir():
            _KERNEL_MODULE_MAPPING[name] = get_local_kernel(snapshot)


@beartype
def load_prefix_model(
    checkpoint: Path,
    adapter: NativeTemplateAdapter,
    *,
    max_layer: int,
    execution_device: str = "cuda:0",
) -> object:
    """Load a BF16 backbone only through the last requested probe site, then CPU-offload it."""
    torch = _torch_runtime()
    try:
        from accelerate import (  # ty: ignore[unresolved-import, unused-ignore-comment]
            cpu_offload,
            init_empty_weights,
        )
        from accelerate.utils import (  # ty: ignore[unresolved-import, unused-ignore-comment]
            set_module_tensor_to_device,
        )
        from safetensors import safe_open  # ty: ignore[unresolved-import, unused-ignore-comment]
        from transformers import (  # ty: ignore[unresolved-import, unused-ignore-comment]
            AutoConfig,
            AutoModel,
        )
    except ImportError as error:  # pragma: no cover - exercised by minimal installations
        raise RuntimeError("prefix model loading requires the 'probes' extra") from error

    if adapter.name == NEMOTRON_ADAPTER.name:
        _pin_nemotron_kernel_revisions()

    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    architectures = tuple(getattr(config, "architectures", ()) or ())
    if architectures != (adapter.architecture,):
        raise ValueError(
            f"checkpoint architecture {architectures} does not match {adapter.architecture}"
        )
    runtime_config = config if adapter.name == NEMOTRON_ADAPTER.name else config.text_config
    if str(getattr(runtime_config, "dtype", None)).removeprefix("torch.") != "bfloat16":
        raise ValueError("the pinned prefix extractor requires native BF16 model weights")
    with init_empty_weights():
        model = AutoModel.from_config(runtime_config)
        _delete_prefix_tail(model, adapter, max_layer)
    if model.__class__.__name__ != adapter.runtime_architecture:
        raise ValueError(
            f"AutoModel resolved {model.__class__.__name__}, expected {adapter.runtime_architecture}"
        )

    weight_map = _read_weight_map(checkpoint)
    plan = select_prefix_checkpoint_keys(weight_map, adapter, max_layer=max_layer)
    expected = model.state_dict()
    dtype_plan = cast(Any, model)._get_dtype_plan(torch.bfloat16)

    def runtime_dtype(name: str) -> Any:
        matches = [dtype for pattern, dtype in dtype_plan.items() if re.search(pattern, name)]
        if len(matches) > 1 and len(set(matches)) != 1:
            raise ValueError(f"conflicting model dtype policies for {name}")
        return matches[0] if matches else expected[name].dtype

    runtime_sources: dict[str, list[str]] = {}
    for source, runtime in plan.key_mapping.items():
        runtime_sources.setdefault(runtime, []).append(source)
    if set(runtime_sources) != set(expected):
        missing = sorted(set(expected) - set(runtime_sources))
        extra = sorted(set(runtime_sources) - set(expected))
        raise ValueError(f"prefix runtime tensor mismatch; missing={missing}, extra={extra}")

    packed: dict[str, Any] = {}
    for runtime, sources in runtime_sources.items():
        if len(sources) > 1:
            packed[runtime] = torch.empty(
                tuple(expected[runtime].shape), dtype=runtime_dtype(runtime), device="cpu"
            )

    sources_by_file: dict[str, list[str]] = {}
    for source in plan.key_mapping:
        sources_by_file.setdefault(weight_map[source], []).append(source)
    loaded_sources: set[str] = set()
    for filename, sources in sorted(sources_by_file.items()):
        shard = checkpoint / filename
        if not shard.is_file():
            raise FileNotFoundError(f"missing selected checkpoint shard: {shard}")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for source in sources:
                if source not in available:
                    raise ValueError(f"{source} is absent from selected shard {filename}")
                runtime = plan.key_mapping[source]
                value = handle.get_tensor(source)
                target_dtype = runtime_dtype(runtime)
                if value.dtype != target_dtype:
                    raise ValueError(
                        f"{source} has dtype {value.dtype}, expected {target_dtype} "
                        f"for runtime tensor {runtime}"
                    )
                if len(runtime_sources[runtime]) == 1:
                    if tuple(value.shape) != tuple(expected[runtime].shape):
                        raise ValueError(
                            f"{source} shape {tuple(value.shape)} does not match "
                            f"{runtime} {tuple(expected[runtime].shape)}"
                        )
                    set_module_tensor_to_device(
                        model, runtime, "cpu", value=value, dtype=target_dtype
                    )
                else:
                    expert = _expert_index(source)
                    target = packed[runtime]
                    if expert >= target.shape[0] or tuple(value.shape) != tuple(target.shape[1:]):
                        raise ValueError(f"{source} cannot be packed into {runtime}")
                    target[expert].copy_(value)
                loaded_sources.add(source)
    if loaded_sources != set(plan.key_mapping):
        raise ValueError("not every selected checkpoint tensor was loaded")
    for runtime, value in packed.items():
        source_indices = sorted(_expert_index(source) for source in runtime_sources[runtime])
        if source_indices != list(range(value.shape[0])):
            raise ValueError(f"{runtime} does not contain a complete ordered expert stack")
        set_module_tensor_to_device(
            model, runtime, "cpu", value=value, dtype=runtime_dtype(runtime)
        )

    meta = [name for name, value in model.state_dict().items() if value.device.type == "meta"]
    if meta:
        raise ValueError(f"prefix model retains unloaded meta tensors: {meta}")
    model.eval()
    cpu_offload(model, execution_device=torch.device(execution_device), offload_buffers=True)
    return model


@dataclass(frozen=True)
class _RenderedText:
    text: str
    content_start: int
    content_end: int
    filler_ranges: tuple[tuple[int, int], ...]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@beartype
def validate_native_template(tokenizer: NativeTokenizer, adapter: NativeTemplateAdapter) -> None:
    """Reject a missing or changed model-native chat template."""
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ValueError(f"{adapter.model_id} tokenizer has no string chat template")
    actual = _sha256_text(template)
    if actual != adapter.chat_template_sha256:
        raise ValueError(
            f"chat template mismatch for {adapter.model_id}: expected "
            f"{adapter.chat_template_sha256}, got {actual}"
        )


def _apply_messages(
    tokenizer: NativeTokenizer,
    adapter: NativeTemplateAdapter,
    messages: list[dict[str, Any]],
) -> str:
    kwargs = (
        {"enable_thinking": True, "truncate_history_thinking": False}
        if adapter.name == NEMOTRON_ADAPTER.name
        else {"enable_thinking": True, "preserve_thinking": True}
    )
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, **kwargs
    )
    if not isinstance(rendered, str):
        raise TypeError("apply_chat_template(tokenize=False) must return a string")
    return rendered


def _apply_native_template(
    tokenizer: NativeTokenizer,
    adapter: NativeTemplateAdapter,
    *,
    reasoning: str,
    content: str,
) -> str:
    message = {"role": "assistant", "content": content, adapter.reasoning_field: reasoning}
    return _apply_messages(tokenizer, adapter, [message])


def _replace_exactly_once(value: str, marker: str, replacement: str) -> tuple[str, int, int]:
    if value.count(marker) != 1:
        raise ValueError("native template did not preserve a unique span marker")
    start = value.index(marker)
    replaced = value.replace(marker, replacement, 1)
    return replaced, start, start + len(replacement)


def _render_assistant(
    tokenizer: NativeTokenizer,
    adapter: NativeTemplateAdapter,
    content: str,
    nested_filler: str,
) -> _RenderedText:
    content_marker = "ROLE_PROBE_CONTENT_7f25a81b"
    filler_marker = "ROLE_PROBE_FILLER_96d43ac2"
    skeleton = _apply_native_template(
        tokenizer,
        adapter,
        reasoning=filler_marker,
        content=content_marker,
    )
    with_filler, filler_start, filler_end = _replace_exactly_once(
        skeleton, filler_marker, nested_filler
    )
    rendered, content_start, content_end = _replace_exactly_once(
        with_filler, content_marker, content
    )
    actual = _apply_native_template(tokenizer, adapter, reasoning=nested_filler, content=content)
    if rendered != actual:
        raise ValueError("native template transformed assistant content or filler unexpectedly")
    return _RenderedText(rendered, content_start, content_end, ((filler_start, filler_end),))


def _render_reasoning(
    tokenizer: NativeTokenizer,
    adapter: NativeTemplateAdapter,
    content: str,
    prefix_filler: str,
) -> _RenderedText:
    content_marker = "ROLE_PROBE_CONTENT_7f25a81b"
    skeleton = _apply_native_template(
        tokenizer,
        adapter,
        reasoning=content_marker,
        content="",
    )
    rendered, content_start, content_end = _replace_exactly_once(skeleton, content_marker, content)
    actual = _apply_native_template(tokenizer, adapter, reasoning=content, content="")
    if rendered != actual:
        raise ValueError("native template transformed reasoning content unexpectedly")
    if rendered.count(adapter.role_open) != 1:
        raise ValueError("native template has an ambiguous assistant-role opening")
    insertion = rendered.index(adapter.role_open)
    rendered = rendered[:insertion] + prefix_filler + rendered[insertion:]
    if insertion <= content_start:
        content_start += len(prefix_filler)
        content_end += len(prefix_filler)
    else:  # pragma: no cover - guarded by pinned templates and tested mismatch path
        raise ValueError("assistant-role opening occurs after reasoning content")
    filler_ranges = ((insertion, insertion + len(prefix_filler)),)
    return _RenderedText(rendered, content_start, content_end, filler_ranges)


def _role_messages(role: ProbeRole, content: str) -> list[dict[str, Any]]:
    if role in {ProbeRole.SYSTEM, ProbeRole.USER}:
        return [{"role": str(role), "content": content}]
    if role is ProbeRole.TOOL:
        tool_call = {
            "id": "role-probe-call",
            "type": "function",
            "function": {"name": "read_text", "arguments": {"path": "neutral.txt"}},
        }
        return [
            {"role": "assistant", "content": "", "tool_calls": [tool_call]},
            {
                "role": "tool",
                "name": "read_text",
                "tool_call_id": "role-probe-call",
                "content": content,
            },
        ]
    raise ValueError(f"{role} is not a plain external role")


def _render_plain_role(
    tokenizer: NativeTokenizer,
    adapter: NativeTemplateAdapter,
    role: ProbeRole,
    content: str,
    prefix_filler: str,
) -> _RenderedText:
    content_marker = "ROLE_PROBE_CONTENT_7f25a81b"
    skeleton = _apply_messages(tokenizer, adapter, _role_messages(role, content_marker))
    rendered, content_start, content_end = _replace_exactly_once(skeleton, content_marker, content)
    actual = _apply_messages(tokenizer, adapter, _role_messages(role, content))
    if rendered != actual:
        raise ValueError(f"native template transformed {role} content unexpectedly")
    if adapter.prefix_anchor not in rendered:
        raise ValueError(f"native template omitted the {role} role opening")
    insertion = rendered.index(adapter.prefix_anchor)
    rendered = rendered[:insertion] + prefix_filler + rendered[insertion:]
    if insertion > content_start:
        raise ValueError(f"native template places the {role} opening after its content")
    content_start += len(prefix_filler)
    content_end += len(prefix_filler)
    return _RenderedText(
        rendered,
        content_start,
        content_end,
        ((insertion, insertion + len(prefix_filler)),),
    )


def _plain_token_ids(tokenizer: NativeTokenizer, text: str) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded.get("input_ids")
    if ids is None:
        raise ValueError("tokenizer did not return input_ids")
    if ids and isinstance(ids[0], Sequence):
        raise ValueError("tokenizer returned a batched sequence for scalar text")
    return tuple(int(value) for value in ids)


def _truncate_text(tokenizer: NativeTokenizer, text: str, limit: int) -> tuple[str, bool]:
    ids = _plain_token_ids(tokenizer, text.strip())
    if not ids:
        raise ValueError("document tokenized to an empty sequence")
    truncated = len(ids) > limit
    kept = ids[:limit]
    decoded = tokenizer.decode(kept, skip_special_tokens=False).strip()
    if not decoded:
        raise ValueError("truncated document decoded to empty text")
    return decoded, truncated


def _tokenize_rendered(
    tokenizer: NativeTokenizer,
    rendered: _RenderedText,
    *,
    document_index: int,
    document_id: str,
    role: ProbeRole,
    partner_document_id: str,
    content_was_truncated: bool,
) -> RoleExample:
    encoded = tokenizer(
        rendered.text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids_raw = encoded.get("input_ids")
    offsets_raw = encoded.get("offset_mapping")
    if ids_raw is None or offsets_raw is None:
        raise ValueError("a fast tokenizer with offset_mapping is required")
    ids = tuple(int(value) for value in ids_raw)
    offsets = tuple((int(start), int(end)) for start, end in offsets_raw)
    if len(ids) != len(offsets):
        raise ValueError("token IDs and offset mappings have unequal length")

    content_positions: list[int] = []
    filler_positions: list[int] = []
    for position, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        inside_content = start >= rendered.content_start and end <= rendered.content_end
        if inside_content:
            content_positions.append(position)
            continue
        overlaps_filler = any(
            start < stop and end > begin for begin, stop in rendered.filler_ranges
        )
        if overlaps_filler:
            filler_positions.append(position)

    content = tuple(content_positions)
    filler = tuple(filler_positions)
    excluded = set(content) | set(filler)
    tags = tuple(position for position in range(len(ids)) if position not in excluded)
    return RoleExample(
        document_index=document_index,
        document_id=document_id,
        role=role,
        input_ids=ids,
        content_positions=content,
        content_token_ids=tuple(ids[position] for position in content),
        filler_positions=filler,
        tag_positions=tags,
        partner_document_id=partner_document_id,
        content_was_truncated=content_was_truncated,
    )


def _content_pairs(example: RoleExample) -> set[tuple[int, int]]:
    return set(zip(example.content_positions, example.content_token_ids, strict=True))


def _shares_positioned_content(left: RoleExample, right: RoleExample) -> bool:
    common = _content_pairs(left) & _content_pairs(right)
    required = max(1, min(len(left.content_positions), len(right.content_positions)) - 2)
    return len(common) >= required


def _harmonize_content(examples: Mapping[ProbeRole, RoleExample]) -> tuple[RoleExample, ...]:
    common = set.intersection(*(_content_pairs(example) for example in examples.values()))
    required = max(1, min(len(example.content_positions) for example in examples.values()) - 2)
    if len(common) < required:
        raise ValueError("native roles do not share enough exactly positioned content tokens")
    ordered = sorted(common)
    positions = tuple(position for position, _token_id in ordered)
    token_ids = tuple(token_id for _position, token_id in ordered)
    result = []
    for role in ProbeRole:
        example = examples[role]
        removed = set(example.content_positions) - set(positions)
        result.append(
            RoleExample(
                document_index=example.document_index,
                document_id=example.document_id,
                role=role,
                input_ids=example.input_ids,
                content_positions=positions,
                content_token_ids=token_ids,
                filler_positions=example.filler_positions,
                tag_positions=tuple(sorted(set(example.tag_positions) | removed)),
                partner_document_id=example.partner_document_id,
                content_was_truncated=example.content_was_truncated,
            )
        )
    return tuple(result)


def _prefix_candidates(
    tokenizer: NativeTokenizer, partner_text: str, maximum: int
) -> tuple[str, ...]:
    partner_ids = _plain_token_ids(tokenizer, partner_text)
    count = min(len(partner_ids), maximum)
    values = [""]
    values.extend(
        tokenizer.decode(partner_ids[:length], skip_special_tokens=False).strip()
        for length in range(1, count + 1)
    )
    return tuple(dict.fromkeys(values))


def _nested_lengths(desired: int, maximum: int) -> Iterable[int]:
    yield desired
    for distance in range(1, maximum + 1):
        lower = desired - distance
        upper = desired + distance
        if lower >= 1:
            yield lower
        if upper <= maximum:
            yield upper


def _paired_examples(
    tokenizer: NativeTokenizer,
    adapter: NativeTemplateAdapter,
    *,
    document_index: int,
    document: NeutralDocument,
    content: str,
    content_was_truncated: bool,
    partner: NeutralDocument,
    desired_filler_tokens: int,
    max_filler_tokens: int,
    max_sequence_tokens: int,
) -> tuple[RoleExample, ...]:
    candidates = _prefix_candidates(tokenizer, partner.text, max_filler_tokens + 64)
    if len(candidates) <= 1:
        raise ValueError(f"partner document {partner.document_id} has no filler tokens")

    maximum_nested = min(max_filler_tokens, len(candidates) - 1)
    desired = min(max(desired_filler_tokens, 1), maximum_nested)
    base_examples: dict[ProbeRole, RoleExample] = {}
    for role in ProbeRole:
        if role is ProbeRole.ASSISTANT:
            continue
        rendered = (
            _render_reasoning(tokenizer, adapter, content, "")
            if role is ProbeRole.REASONING
            else _render_plain_role(tokenizer, adapter, role, content, "")
        )
        base_examples[role] = _tokenize_rendered(
            tokenizer,
            rendered,
            document_index=document_index,
            document_id=document.document_id,
            role=role,
            partner_document_id=partner.document_id,
            content_was_truncated=content_was_truncated,
        )
    minimum_assistant_position = max(
        example.content_positions[0] for example in base_examples.values()
    )
    for nested_length in _nested_lengths(desired, maximum_nested):
        nested_filler = candidates[nested_length]
        assistant = _tokenize_rendered(
            tokenizer,
            _render_assistant(tokenizer, adapter, content, nested_filler),
            document_index=document_index,
            document_id=document.document_id,
            role=ProbeRole.ASSISTANT,
            partner_document_id=partner.document_id,
            content_was_truncated=content_was_truncated,
        )
        if len(assistant.input_ids) > max_sequence_tokens:
            continue
        if assistant.content_positions[0] < minimum_assistant_position:
            continue
        matched: dict[ProbeRole, RoleExample] = {ProbeRole.ASSISTANT: assistant}
        for role in ProbeRole:
            if role is ProbeRole.ASSISTANT:
                continue
            candidate_examples = (("", base_examples[role]),)
            if not _shares_positioned_content(base_examples[role], assistant):
                candidate_examples = (
                    (
                        prefix_filler,
                        _tokenize_rendered(
                            tokenizer,
                            (
                                _render_reasoning(tokenizer, adapter, content, prefix_filler)
                                if role is ProbeRole.REASONING
                                else _render_plain_role(
                                    tokenizer, adapter, role, content, prefix_filler
                                )
                            ),
                            document_index=document_index,
                            document_id=document.document_id,
                            role=role,
                            partner_document_id=partner.document_id,
                            content_was_truncated=content_was_truncated,
                        ),
                    )
                    for prefix_filler in candidates[1:]
                )
            for _prefix_filler, example in candidate_examples:
                if len(example.input_ids) > max_sequence_tokens:
                    continue
                if _shares_positioned_content(example, assistant):
                    matched[role] = example
                    break
            if role not in matched:
                break
        if set(matched) == set(ProbeRole):
            return _harmonize_content(matched)
    raise ValueError(
        f"could not exactly position-match all native roles for {document.document_id}"
    )


def _source_digest(documents: Sequence[NeutralDocument]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        payload = json.dumps(asdict(document), sort_keys=True, separators=(",", ":"))
        digest.update(payload.encode())
        digest.update(b"\n")
    return digest.hexdigest()


@beartype
def load_neutral_documents(path: Path, *, limit: int | None = None) -> tuple[NeutralDocument, ...]:
    """Load strict JSONL neutral documents without silently skipping malformed rows."""
    documents: list[NeutralDocument] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and len(documents) >= limit:
                break
            try:
                value = json.loads(line)
                documents.append(
                    NeutralDocument(
                        document_id=value["id"], text=value["text"], source=value["source"]
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid neutral document at {path}:{line_number}") from error
    if len(documents) < 2:
        raise ValueError("role-probe construction requires at least two documents")
    if len({document.document_id for document in documents}) != len(documents):
        raise ValueError("neutral document IDs must be unique")
    return tuple(documents)


@beartype
def build_role_dataset(
    documents: tuple[NeutralDocument, ...],
    filler_documents: tuple[NeutralDocument, ...],
    tokenizer: NativeTokenizer,
    adapter: NativeTemplateAdapter,
    *,
    max_content_tokens: int,
    max_filler_tokens: int,
    max_sequence_tokens: int,
    seed: int,
) -> RoleDataset:
    """Construct exactly paired, position-controlled native-role examples."""
    if len(documents) < 2 or not filler_documents:
        raise ValueError("role-probe construction requires target and filler documents")
    document_ids = {document.document_id for document in documents}
    filler_ids = {document.document_id for document in filler_documents}
    if document_ids & filler_ids:
        raise ValueError("target and filler document IDs must be disjoint")
    target_text_digests = {_sha256_text(document.text) for document in documents}
    filler_text_digests = {_sha256_text(document.text) for document in filler_documents}
    if target_text_digests & filler_text_digests:
        raise ValueError("target and filler document text must be disjoint")
    if min(max_content_tokens, max_filler_tokens, max_sequence_tokens) < 1:
        raise ValueError("token limits must be positive")
    validate_native_template(tokenizer, adapter)

    generator = np.random.default_rng(seed)
    filler_order = generator.permutation(len(filler_documents))
    fractions = generator.beta(0.5, 4.0, size=len(documents))
    desired_lengths = np.maximum(1, (fractions * max_filler_tokens).astype(int))

    examples: list[RoleExample] = []
    for document_index, document in enumerate(documents):
        content, truncated = _truncate_text(tokenizer, document.text, max_content_tokens)
        partner = filler_documents[int(filler_order[document_index % len(filler_order)])]
        examples.extend(
            _paired_examples(
                tokenizer,
                adapter,
                document_index=document_index,
                document=document,
                content=content,
                content_was_truncated=truncated,
                partner=partner,
                desired_filler_tokens=int(desired_lengths[document_index]),
                max_filler_tokens=max_filler_tokens,
                max_sequence_tokens=max_sequence_tokens,
            )
        )
    return RoleDataset(
        documents=documents,
        filler_documents=filler_documents,
        examples=tuple(examples),
        source_sha256=_source_digest(documents),
        filler_source_sha256=_source_digest(filler_documents),
        max_content_tokens=max_content_tokens,
        max_filler_tokens=max_filler_tokens,
        max_sequence_tokens=max_sequence_tokens,
        seed=seed,
    )


def _resolve_attribute(value: object, dotted: str) -> Any:
    current = value
    for component in dotted.split("."):
        current = getattr(current, component)
    return current


class _PrefixComplete(RuntimeError):
    pass


def _torch_runtime() -> Any:
    try:
        import torch  # ty: ignore[unresolved-import, unused-ignore-comment]
    except ImportError as error:  # pragma: no cover - exercised by minimal installations
        raise RuntimeError("role-probe extraction requires the 'probes' extra") from error
    return torch


def _input_device(model: Any) -> Any:
    embedding = model.get_input_embeddings()
    hook = getattr(embedding, "_hf_hook", None)
    execution_device = getattr(hook, "execution_device", None)
    if execution_device is not None:
        return execution_device
    device = embedding.weight.device
    if device.type == "meta":
        raise ValueError("input embedding remains on meta and has no execution-device hook")
    return device


@beartype
def capture_token_activations(
    model: object,
    adapter: NativeTemplateAdapter,
    *,
    input_ids: tuple[int, ...],
    token_positions: tuple[int, ...],
    layers: tuple[int, ...],
) -> np.ndarray:
    """Capture selected content-token states and stop at the final requested site."""
    return _capture_sequences(
        model,
        adapter,
        input_ids=(input_ids,),
        token_positions=(token_positions,),
        layers=layers,
    )[0]


def _capture_sequences(
    model: Any,
    adapter: NativeTemplateAdapter,
    *,
    input_ids: tuple[tuple[int, ...], ...],
    token_positions: tuple[tuple[int, ...], ...],
    layers: tuple[int, ...],
) -> np.ndarray:
    torch = _torch_runtime()
    layer_modules = _resolve_attribute(model, adapter.layer_container)
    if model.__class__.__name__ != adapter.runtime_architecture:
        raise ValueError(
            f"adapter {adapter.name} requires {adapter.runtime_architecture}, got "
            f"{model.__class__.__name__}"
        )
    if not layers or tuple(sorted(set(layers))) != layers:
        raise ValueError("layers must be a non-empty, sorted, unique tuple")
    if layers[0] < 0 or layers[-1] >= len(layer_modules):
        raise ValueError("probe layer lies outside the loaded model prefix")

    if not input_ids or not token_positions:
        raise ValueError("input IDs and token positions must be non-empty")
    if len(input_ids) != len(token_positions):
        raise ValueError("input IDs and token-position batches must have equal length")
    content_lengths = {len(positions) for positions in token_positions}
    if content_lengths == {0} or len(content_lengths) != 1:
        raise ValueError("batched token-position rows must have one equal positive length")
    if any(
        position < 0 or position >= len(sequence)
        for sequence, positions in zip(input_ids, token_positions, strict=True)
        for position in positions
    ):
        raise ValueError("token position lies outside input IDs")
    captured: dict[int, Any] = {}
    handles = []
    final_layer = layers[-1]

    def make_hook(layer_index: int) -> Any:
        def hook(_module: object, _inputs: object, output: Any) -> None:
            if not torch.is_tensor(output) or output.ndim != 3 or output.shape[0] != len(input_ids):
                raise ValueError("probe-site hook output has an unexpected batch shape")
            selected = [
                output[row].index_select(
                    0, torch.tensor(positions, dtype=torch.long, device=output.device)
                )
                for row, positions in enumerate(token_positions)
            ]
            captured[layer_index] = torch.stack(selected).detach().float().cpu()
            if layer_index == final_layer:
                raise _PrefixComplete

        return hook

    for layer_index in layers:
        module = getattr(layer_modules[layer_index], adapter.activation_module)
        handles.append(module.register_forward_hook(make_hook(layer_index)))

    device = _input_device(model)
    max_length = max(len(sequence) for sequence in input_ids)
    input_tensor = torch.full(
        (len(input_ids), max_length),
        adapter.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_tensor)
    for row, sequence in enumerate(input_ids):
        length = len(sequence)
        input_tensor[row, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row, :length] = 1
    try:
        with torch.inference_mode():
            model(input_ids=input_tensor, attention_mask=attention_mask, use_cache=False)
    except _PrefixComplete:
        pass
    else:
        raise RuntimeError("prefix model ran past the final requested activation site")
    finally:
        for handle in handles:
            handle.remove()

    if set(captured) != set(layers):
        raise RuntimeError(f"captured layers {sorted(captured)}, expected {list(layers)}")
    return torch.stack([captured[layer] for layer in layers], dim=2).numpy()


def _capture_one(
    model: object,
    adapter: NativeTemplateAdapter,
    example: RoleExample,
    layers: tuple[int, ...],
) -> np.ndarray:
    return _capture_sequences(
        model,
        adapter,
        input_ids=(example.input_ids,),
        token_positions=(example.content_positions,),
        layers=layers,
    )[0]


def _capture_role_group(
    model: object,
    adapter: NativeTemplateAdapter,
    examples: tuple[RoleExample, ...],
    layers: tuple[int, ...],
) -> np.ndarray:
    return _capture_sequences(
        model,
        adapter,
        input_ids=tuple(example.input_ids for example in examples),
        token_positions=tuple(example.content_positions for example in examples),
        layers=layers,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_record(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    module_name = getattr(value, "__module__", None)
    if not isinstance(module_name, str) or not module_name:
        raise ValueError("runtime callable has no module identity")
    module = sys.modules.get(module_name)
    source = inspect.getsourcefile(value)
    if source is None and module is not None:
        source = getattr(module, "__file__", None)
    if not isinstance(source, str) or not Path(source).is_file():
        raise ValueError(f"runtime callable {module_name} has no hashable module file")
    record = {
        "module": module_name,
        "qualname": str(getattr(value, "__qualname__", getattr(value, "__name__", ""))),
        "module_file_sha256": _file_sha256(Path(source)),
    }
    snapshot = re.search(r"/snapshots/([0-9a-f]{40})/", source)
    if snapshot is not None:
        record["snapshot_revision"] = snapshot.group(1)
    return record


def _nemotron_runtime(model: object) -> dict[str, Any]:
    module = sys.modules.get(model.__class__.__module__)
    if module is None:
        raise ValueError("Nemotron modeling module is not loaded")
    kernel_names = (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "selective_state_update",
        "mamba_chunk_scan_combined",
        "mamba_split_conv1d_scan_combined",
    )
    fast_path = bool(getattr(module, "is_fast_path_available", False))
    kernels = {name: _module_record(getattr(module, name, None)) for name in kernel_names}
    if fast_path and any(value is None for value in kernels.values()):
        raise ValueError("Nemotron reports a fast path without complete kernel identities")
    config = cast(Any, model).config
    configured = bool(getattr(config, "use_mamba_kernels", False))
    selected = configured and fast_path and _input_device(model).type == "cuda"
    kernel_roots = {
        record["module"].split(".", 1)[0] for record in kernels.values() if record is not None
    }
    binaries = {}
    for name, loaded_module in sys.modules.items():
        source = getattr(loaded_module, "__file__", None)
        if (
            any(name == root or name.startswith(f"{root}.") for root in kernel_roots)
            and isinstance(source, str)
            and source.endswith(".so")
        ):
            binaries[name] = _file_sha256(Path(source))
    if selected and not binaries:
        raise ValueError("selected Nemotron fast path has no hashable compiled kernel binary")
    return {
        "configured_use_mamba_kernels": configured,
        "chunk_size": int(config.chunk_size),
        "fast_path_available": fast_path,
        "fast_path_selected": selected,
        "kernels_distribution_version": _distribution_version("kernels"),
        "einops_distribution_version": _distribution_version("einops"),
        "callables": kernels,
        "compiled_binaries_sha256": binaries,
    }


def _uses_nemotron_fast_path(model: object, adapter: NativeTemplateAdapter) -> bool:
    module = sys.modules.get(model.__class__.__module__)
    return (
        adapter.name == NEMOTRON_ADAPTER.name
        and module is not None
        and bool(getattr(module, "is_fast_path_available", False))
        and _input_device(model).type == "cuda"
        and bool(getattr(cast(Any, model).config, "use_mamba_kernels", False))
    )


@beartype
def model_runtime_identity(
    model: object, adapter: NativeTemplateAdapter, *, checkpoint: Path
) -> tuple[str, Mapping[str, Any]]:
    """Fingerprint the exact local numerical runtime used for activation extraction."""
    torch = _torch_runtime()
    try:
        import transformers  # ty: ignore[unresolved-import, unused-ignore-comment]
    except ImportError as error:  # pragma: no cover - exercised by minimal installations
        raise RuntimeError("runtime identity requires the 'probes' extra") from error
    modeling_module = sys.modules.get(model.__class__.__module__)
    modeling_file = getattr(modeling_module, "__file__", None)
    if not isinstance(modeling_file, str) or not Path(modeling_file).is_file():
        raise ValueError("model runtime has no hashable Transformers implementation")
    config_path = checkpoint / "config.json"
    device = _input_device(model)
    device_record: dict[str, Any] = {"type": str(device.type)}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        device_record.update(
            {
                "name": torch.cuda.get_device_name(index),
                "compute_capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    runtime: dict[str, Any] = {
        "format_version": 1,
        "adapter": adapter.name,
        "model_config_sha256": _file_sha256(config_path),
        "model_dtype_plan": {
            name: str(dtype).removeprefix("torch.")
            for name, dtype in sorted(cast(Any, model)._get_dtype_plan(torch.bfloat16).items())
        },
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "transformers_modeling_module": model.__class__.__module__,
        "transformers_modeling_sha256": _file_sha256(Path(modeling_file)),
        "attention_implementation": str(
            getattr(cast(Any, model).config, "_attn_implementation", None)
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "execution_device": device_record,
        "nemotron_mamba": (
            _nemotron_runtime(model) if adapter.name == NEMOTRON_ADAPTER.name else None
        ),
    }
    canonical = json.dumps(runtime, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest(), runtime


@beartype
def validate_prefix_checkpoint_identity(checkpoint: Path, manifest: Mapping[str, Any]) -> None:
    """Recompute the downloader's exact filtered-index and shard identity."""
    if manifest.get("weights_hash_kind") != PREFIX_WEIGHTS_HASH_KIND:
        raise ValueError(
            f"unsupported prefix weights hash kind: {manifest.get('weights_hash_kind')}"
        )
    weight_map = _read_weight_map(checkpoint)
    shard_names = sorted(set(weight_map.values()))
    shard_hashes = {name: _file_sha256(checkpoint / name) for name in shard_names}
    identity = {
        "index_sha256": _file_sha256(checkpoint / "model.safetensors.index.json"),
        "shard_sha256": shard_hashes,
    }
    canonical = json.dumps(identity, separators=(",", ":"), sort_keys=True).encode()
    actual = hashlib.sha256(canonical).hexdigest()
    if manifest.get("weights_sha256") != actual:
        raise ValueError("prefix checkpoint weight identity does not match its files")

    shard_records = manifest.get("shards")
    if not isinstance(shard_records, list):
        raise ValueError("prefix checkpoint manifest lacks shard records")
    recorded_hashes: dict[str, str] = {}
    for record in shard_records:
        if not isinstance(record, dict):
            raise ValueError("invalid prefix checkpoint shard record")
        filename = record.get("filename")
        digest = record.get("sha256")
        if (
            not isinstance(filename, str)
            or not isinstance(digest, str)
            or filename in recorded_hashes
        ):
            raise ValueError("invalid prefix checkpoint shard record")
        recorded_hashes[filename] = digest
    if recorded_hashes != shard_hashes:
        raise ValueError("prefix checkpoint shard records do not match its files")

    auxiliary_records = manifest.get("auxiliary_files")
    if not isinstance(auxiliary_records, list):
        raise ValueError("prefix checkpoint manifest lacks auxiliary file records")
    for record in auxiliary_records:
        if not isinstance(record, dict):
            raise ValueError("invalid prefix checkpoint auxiliary record")
        filename = record.get("filename")
        size = record.get("bytes")
        digest = record.get("sha256")
        if (
            not isinstance(filename, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
        ):
            raise ValueError("invalid prefix checkpoint auxiliary record")
        path = checkpoint / filename
        if path.stat().st_size != size or _file_sha256(path) != digest:
            raise ValueError(f"prefix checkpoint auxiliary file does not match: {filename}")


_ACTIVATION_FILES = frozenset(
    {
        "activations.npy",
        "document_index.npy",
        "filler_document_index.npy",
        "role.npy",
        "content_token_index.npy",
        "content_token_id.npy",
        "sequence_token_index.npy",
        "sequences.jsonl",
        "target_documents.jsonl",
        "filler_documents.jsonl",
    }
)


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid JSON object: {path}")
    return value


def _jsonl_objects(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSONL: {path}") from error
    if not values or not all(isinstance(value, dict) for value in values):
        raise ValueError(f"invalid JSONL: {path}")
    return values


def _indexed_documents(
    records: list[dict[str, Any]], *, index_name: str
) -> tuple[tuple[str, ...], set[str]]:
    expected_keys = {index_name, "document_id", "source", "text_sha256"}
    if any(set(record) != expected_keys for record in records):
        raise ValueError(f"invalid document mapping fields for {index_name}")
    if [record[index_name] for record in records] != list(range(len(records))):
        raise ValueError(f"{index_name} values must be contiguous and ordered")
    document_ids = tuple(record["document_id"] for record in records)
    text_digests = {record["text_sha256"] for record in records}
    if not all(
        isinstance(value, str) and value.strip() == value and value for value in document_ids
    ):
        raise ValueError("document IDs must be non-empty and trimmed")
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("document mapping contains duplicate IDs")
    for record in records:
        if not isinstance(record["source"], str) or not record["source"].strip():
            raise ValueError("document sources must be non-empty strings")
        if not isinstance(record["text_sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", record["text_sha256"]
        ):
            raise ValueError("document text_sha256 must be a lowercase SHA-256 digest")
    if len(text_digests) != len(records):
        raise ValueError("document mapping contains duplicate source text")
    return document_ids, text_digests


def _load_integer_array(path: Path, *, dtype: np.dtype[Any], rows: int) -> np.ndarray:
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid NumPy array: {path}") from error
    if value.dtype != dtype or value.shape != (rows,):
        raise ValueError(f"{path.name} must have shape ({rows},) and dtype {dtype.name}")
    return value


@beartype
def load_activation_dataset(path: Path) -> ActivationDataset:
    """Load and cross-check every file in a native neutral-role activation artifact."""
    manifest = _json_object(path / "manifest.json")
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported activation artifact format_version")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != _ACTIVATION_FILES:
        raise ValueError("activation artifact file manifest is incomplete")
    actual_names = {item.name for item in path.iterdir() if item.is_file()}
    if actual_names != _ACTIVATION_FILES | {"manifest.json"}:
        raise ValueError("activation artifact contains missing or unexpected files")
    for name, expected_digest in files.items():
        if not isinstance(expected_digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_digest
        ):
            raise ValueError(f"invalid file digest for {name}")
        if _file_sha256(path / name) != expected_digest:
            raise ValueError(f"activation artifact checksum mismatch: {name}")

    roles_value = manifest.get("roles")
    if not isinstance(roles_value, dict) or not all(
        isinstance(name, str) and isinstance(code, int) and not isinstance(code, bool)
        for name, code in roles_value.items()
    ):
        raise ValueError("manifest roles must map names to integer codes")
    role_items = cast(dict[str, int], roles_value)
    roles_by_code = tuple(
        name for name, _code in sorted(role_items.items(), key=lambda item: item[1])
    )
    if sorted(role_items.values()) != list(range(len(role_items))):
        raise ValueError("manifest role codes must be contiguous from zero")

    targets, target_texts = _indexed_documents(
        _jsonl_objects(path / "target_documents.jsonl"), index_name="document_index"
    )
    fillers, filler_texts = _indexed_documents(
        _jsonl_objects(path / "filler_documents.jsonl"),
        index_name="filler_document_index",
    )
    if set(targets) & set(fillers) or target_texts & filler_texts:
        raise ValueError("target and filler document pools must have disjoint IDs and text")
    if manifest.get("documents") != len(targets):
        raise ValueError("manifest target document count does not match its mapping")
    if manifest.get("filler_documents") != len(fillers):
        raise ValueError("manifest filler document count does not match its mapping")

    try:
        activations = np.load(path / "activations.npy", mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("invalid activation array") from error
    if activations.ndim != 3:
        raise ValueError("activations.npy must have shape [tokens, layers, hidden]")
    rows = activations.shape[0]
    document_index = _load_integer_array(
        path / "document_index.npy", dtype=np.dtype(np.int32), rows=rows
    )
    filler_document_index = _load_integer_array(
        path / "filler_document_index.npy", dtype=np.dtype(np.int32), rows=rows
    )
    role_codes = _load_integer_array(path / "role.npy", dtype=np.dtype(np.uint8), rows=rows)
    content_token_index = _load_integer_array(
        path / "content_token_index.npy", dtype=np.dtype(np.int32), rows=rows
    )
    content_token_id = _load_integer_array(
        path / "content_token_id.npy", dtype=np.dtype(np.int32), rows=rows
    )
    sequence_token_index = _load_integer_array(
        path / "sequence_token_index.npy", dtype=np.dtype(np.int32), rows=rows
    )
    if (
        np.any(document_index < 0)
        or np.any(document_index >= len(targets))
        or np.any(filler_document_index < 0)
        or np.any(filler_document_index >= len(fillers))
        or np.any(role_codes >= len(roles_by_code))
    ):
        raise ValueError("activation artifact contains an out-of-range metadata index")

    sequence_records = _jsonl_objects(path / "sequences.jsonl")
    if manifest.get("sequences") != len(sequence_records):
        raise ValueError("manifest sequence count does not match sequences.jsonl")
    cursor = 0
    for sequence_index, record in enumerate(sequence_records):
        required = {
            "sequence_index",
            "document_index",
            "document_id",
            "role",
            "partner_document_id",
            "filler_document_index",
            "sequence_tokens",
            "input_ids_sha256",
            "content_start",
            "content_stop",
            "content_tokens",
            "masked_control_tokens",
            "masked_filler_tokens",
            "content_was_truncated",
        }
        if set(record) != required or record["sequence_index"] != sequence_index:
            raise ValueError("invalid or unordered sequence record")
        count = record["content_tokens"]
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("sequence content_tokens must be positive")
        stop = cursor + count
        document = record["document_index"]
        filler = record["filler_document_index"]
        role_name = record["role"]
        if (
            not isinstance(document, int)
            or isinstance(document, bool)
            or not 0 <= document < len(targets)
            or not isinstance(filler, int)
            or isinstance(filler, bool)
            or not 0 <= filler < len(fillers)
            or role_name not in role_items
            or record["document_id"] != targets[document]
            or record["partner_document_id"] != fillers[filler]
            or stop > rows
            or not np.all(document_index[cursor:stop] == document)
            or not np.all(filler_document_index[cursor:stop] == filler)
            or not np.all(role_codes[cursor:stop] == role_items[role_name])
            or not np.array_equal(content_token_index[cursor:stop], np.arange(count))
            or record["content_start"] != int(sequence_token_index[cursor])
            or record["content_stop"] != int(sequence_token_index[stop - 1]) + 1
        ):
            raise ValueError("sequence record does not match activation row metadata")
        cursor = stop
    if cursor != rows or manifest.get("activation_rows") != rows:
        raise ValueError("sequence records do not cover every activation row exactly once")

    runtime = manifest.get("runtime")
    runtime_sha256 = manifest.get("runtime_sha256")
    if not isinstance(runtime, dict) or not isinstance(runtime_sha256, str):
        raise ValueError("activation manifest lacks its runtime identity")
    canonical_runtime = json.dumps(runtime, separators=(",", ":"), sort_keys=True).encode()
    if hashlib.sha256(canonical_runtime).hexdigest() != runtime_sha256:
        raise ValueError("activation runtime identity does not match its record")
    provenance = ActivationProvenance(
        dataset_kind=manifest["dataset_kind"],
        model_id=manifest["model_id"],
        model_revision=manifest["model_revision"],
        weights_sha256=manifest["weights_sha256"],
        weights_hash_kind=manifest["weights_hash_kind"],
        tokenizer_id=manifest["tokenizer_id"],
        tokenizer_revision=manifest["tokenizer_revision"],
        chat_template_sha256=manifest["chat_template_sha256"],
        native_template_adapter=manifest["native_template_adapter"],
        activation_site=manifest["activation_site"],
        runtime_sha256=runtime_sha256,
        model_dtype=manifest["model_dtype"],
        activation_dtype=manifest["activation_dtype"],
        layer_indices=tuple(manifest["layer_indices"]),
        hidden_size=manifest["hidden_size"],
        roles=roles_by_code,
        source_name=manifest["source_name"],
        source_sha256=manifest["source_sha256"],
        extraction_protocol=manifest["extraction_protocol"],
        content_mask=manifest["content_mask"],
        masked_control_tokens=manifest["masked_control_tokens"],
        masked_filler_tokens=manifest["masked_filler_tokens"],
        filler_pool_kind=manifest["filler_pool_kind"],
        filler_source_sha256=manifest["filler_source_sha256"],
        filler_documents=manifest["filler_documents"],
    )
    return ActivationDataset(
        provenance=provenance,
        activations=activations,
        document_ids=np.asarray([targets[int(index)] for index in document_index]),
        roles=np.asarray([roles_by_code[int(code)] for code in role_codes]),
        content_token_index=content_token_index,
        content_token_id=content_token_id,
        sequence_token_index=sequence_token_index,
        filler_document_ids=np.asarray([fillers[int(index)] for index in filler_document_index]),
    )


def _cuda_runtime() -> Any | None:
    """Return CUDA-enabled torch when available without making extraction depend on it."""
    try:
        import torch  # ty: ignore[unresolved-import, unused-ignore-comment]
    except ImportError:  # pragma: no cover - ordinary extraction installs the probes extra
        return None
    return torch if torch.cuda.is_available() else None


@beartype
def extract_role_activations(
    model: object,
    dataset: RoleDataset,
    adapter: NativeTemplateAdapter,
    *,
    layers: tuple[int, ...],
    identity: ExtractionIdentity,
    output: Path,
    activation_dtype: str = "float32",
    model_load_seconds: float | None = None,
    batch_size: int | None = None,
) -> Path:
    """Extract activations into an atomic directory, refusing to overwrite artifacts."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite activation artifact: {output}")
    if activation_dtype not in {"float16", "float32"}:
        raise ValueError("activation_dtype must be float16 or float32")
    if not layers or tuple(sorted(set(layers))) != layers:
        raise ValueError("layers must be a non-empty, sorted, unique tuple")
    effective_batch_size = (
        (
            len(ProbeRole)
            if adapter.name != NEMOTRON_ADAPTER.name or _uses_nemotron_fast_path(model, adapter)
            else 2
        )
        if batch_size is None
        else batch_size
    )
    if effective_batch_size < 1:
        raise ValueError("batch_size must be positive")
    hidden_size = int(_resolve_attribute(model, adapter.hidden_size_path))
    rows = sum(len(example.content_positions) for example in dataset.examples)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        cuda = _cuda_runtime()
        if cuda is not None:
            cuda.cuda.synchronize()
            cuda.cuda.reset_peak_memory_stats()
        extraction_started = time.perf_counter()
        batch_seconds: list[float] = []
        activations = np.lib.format.open_memmap(
            temporary / "activations.npy",
            mode="w+",
            dtype=np.dtype(activation_dtype),
            shape=(rows, len(layers), hidden_size),
        )
        document_index = np.empty(rows, dtype=np.int32)
        filler_document_index = np.empty(rows, dtype=np.int32)
        role = np.empty(rows, dtype=np.uint8)
        content_token_index = np.empty(rows, dtype=np.int32)
        content_token_id = np.empty(rows, dtype=np.int32)
        sequence_token_index = np.empty(rows, dtype=np.int32)
        sequence_records: list[dict[str, Any]] = []
        cursor = 0
        role_codes = {probe_role: index for index, probe_role in enumerate(ProbeRole)}
        filler_indices = {
            document.document_id: index for index, document in enumerate(dataset.filler_documents)
        }
        for document in dataset.documents:
            group = tuple(
                example
                for example in dataset.examples
                if example.document_id == document.document_id
            )
            group_parts = []
            for start in range(0, len(group), effective_batch_size):
                batch_started = time.perf_counter()
                group_parts.append(
                    _capture_role_group(
                        model,
                        adapter,
                        group[start : start + effective_batch_size],
                        layers,
                    )
                )
                if cuda is not None:
                    cuda.cuda.synchronize()
                batch_seconds.append(time.perf_counter() - batch_started)
            group_values = np.concatenate(group_parts)
            for example, values in zip(group, group_values, strict=True):
                count = len(example.content_positions)
                expected_shape = (count, len(layers), hidden_size)
                if values.shape != expected_shape:
                    raise ValueError(
                        f"activation shape {values.shape} does not match {expected_shape}"
                    )
                stop = cursor + count
                activations[cursor:stop] = values.astype(activation_dtype, copy=False)
                document_index[cursor:stop] = example.document_index
                filler_document_index[cursor:stop] = filler_indices[example.partner_document_id]
                role[cursor:stop] = role_codes[example.role]
                content_token_index[cursor:stop] = np.arange(count, dtype=np.int32)
                content_token_id[cursor:stop] = example.content_token_ids
                sequence_token_index[cursor:stop] = example.content_positions
                sequence_records.append(
                    {
                        "sequence_index": len(sequence_records),
                        "document_index": example.document_index,
                        "document_id": example.document_id,
                        "role": str(example.role),
                        "partner_document_id": example.partner_document_id,
                        "filler_document_index": filler_indices[example.partner_document_id],
                        "sequence_tokens": len(example.input_ids),
                        "input_ids_sha256": hashlib.sha256(
                            np.asarray(example.input_ids, dtype="<i4").tobytes()
                        ).hexdigest(),
                        "content_start": example.content_positions[0],
                        "content_stop": example.content_positions[-1] + 1,
                        "content_tokens": count,
                        "masked_control_tokens": len(example.tag_positions),
                        "masked_filler_tokens": len(example.filler_positions),
                        "content_was_truncated": example.content_was_truncated,
                    }
                )
                cursor = stop
        activations.flush()
        del activations
        arrays = {
            "document_index.npy": document_index,
            "filler_document_index.npy": filler_document_index,
            "role.npy": role,
            "content_token_index.npy": content_token_index,
            "content_token_id.npy": content_token_id,
            "sequence_token_index.npy": sequence_token_index,
        }
        for name, value in arrays.items():
            np.save(temporary / name, value, allow_pickle=False)
        with (temporary / "sequences.jsonl").open("w", encoding="utf-8") as handle:
            for record in sequence_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        with (temporary / "target_documents.jsonl").open("w", encoding="utf-8") as handle:
            for index, document in enumerate(dataset.documents):
                handle.write(
                    json.dumps(
                        {
                            "document_index": index,
                            "document_id": document.document_id,
                            "source": document.source,
                            "text_sha256": _sha256_text(document.text),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        with (temporary / "filler_documents.jsonl").open("w", encoding="utf-8") as handle:
            for index, document in enumerate(dataset.filler_documents):
                handle.write(
                    json.dumps(
                        {
                            "filler_document_index": index,
                            "document_id": document.document_id,
                            "source": document.source,
                            "text_sha256": _sha256_text(document.text),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

        file_names = [
            "activations.npy",
            *arrays,
            "sequences.jsonl",
            "target_documents.jsonl",
            "filler_documents.jsonl",
        ]
        extraction_seconds = time.perf_counter() - extraction_started
        runtime_metrics = {
            "model_load_seconds": model_load_seconds,
            "extraction_seconds": extraction_seconds,
            "documents": len(dataset.documents),
            "forward_batches": len(batch_seconds),
            "configured_batch_size": effective_batch_size,
            "role_sequences_per_document": len(ProbeRole),
            "batch_seconds": batch_seconds,
            "activation_rows_per_second": rows / extraction_seconds,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "peak_rss_measurement": "process-ru_maxrss-linux-kib",
            "peak_cuda_allocated_bytes": (
                cuda.cuda.max_memory_allocated() if cuda is not None else None
            ),
            "peak_cuda_reserved_bytes": (
                cuda.cuda.max_memory_reserved() if cuda is not None else None
            ),
        }
        manifest = {
            "format_version": 1,
            "dataset_kind": PAIRED_NEUTRAL,
            "model_id": adapter.model_id,
            "model_revision": adapter.model_revision,
            "weights_sha256": identity.weights_sha256,
            "weights_hash_kind": identity.weights_hash_kind,
            "model_dtype": identity.model_dtype,
            "tokenizer_id": identity.tokenizer_id,
            "tokenizer_revision": identity.tokenizer_revision,
            "chat_template_sha256": adapter.chat_template_sha256,
            "native_template_adapter": adapter.name,
            "activation_site": adapter.activation_site,
            "activation_dtype": activation_dtype,
            "layer_indices": list(layers),
            "hidden_size": hidden_size,
            "roles": {str(key): value for key, value in role_codes.items()},
            "source_name": identity.source_name,
            "source_sha256": dataset.source_sha256,
            "filler_source_sha256": dataset.filler_source_sha256,
            "filler_pool_kind": "dedicated-disjoint-documents",
            "extraction_protocol": EXTRACTION_PROTOCOL,
            "content_mask": CONTENT_TOKENS_ONLY,
            "documents": len(dataset.documents),
            "filler_documents": len(dataset.filler_documents),
            "sequences": len(dataset.examples),
            "activation_rows": rows,
            "max_content_tokens": dataset.max_content_tokens,
            "max_filler_tokens": dataset.max_filler_tokens,
            "max_sequence_tokens": dataset.max_sequence_tokens,
            "seed": dataset.seed,
            "masked_control_tokens": sum(len(item.tag_positions) for item in dataset.examples),
            "masked_filler_tokens": sum(len(item.filler_positions) for item in dataset.examples),
            "transformers_version": identity.transformers_version,
            "torch_version": identity.torch_version,
            "runtime_sha256": identity.runtime_sha256,
            "runtime": identity.runtime,
            "runtime_metrics": runtime_metrics,
            "files": {name: _file_sha256(temporary / name) for name in file_names},
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output
