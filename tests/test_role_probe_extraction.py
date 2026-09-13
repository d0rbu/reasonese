"""Role-probe native dataset and prefix-extraction contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest

import reasonese.extract_role_activations as extraction_cli
import reasonese.role_probe_extraction as extraction
from reasonese.role_probe_extraction import (
    NEMOTRON_ADAPTER,
    ExtractionIdentity,
    NativeTemplateAdapter,
    NeutralDocument,
    ProbeRole,
    build_role_dataset,
    extract_role_activations,
    load_activation_dataset,
    load_neutral_documents,
    select_prefix_checkpoint_keys,
    validate_native_template,
    validate_prefix_checkpoint_identity,
)
from reasonese.role_probes import CONTENT_TOKENS_ONLY, PAIRED_NEUTRAL


class CharacterTokenizer:
    """Small exact-offset tokenizer with the pinned template identity."""

    def __init__(self, template: str) -> None:
        self.chat_template = template

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: bool,
    ) -> str:
        assert not tokenize
        assert not add_generation_prompt
        assert kwargs == {"enable_thinking": True, "truncate_history_thinking": False}
        message = conversation[-1]
        if message["role"] in {"system", "user"}:
            return f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        if message["role"] == "tool":
            return (
                "<|im_start|>assistant\n<think></think><tool_call>read_text</tool_call>"
                "<|im_end|>\n<|im_start|>user\n<tool_response>\n"
                f"{message['content']}\n</tool_response><|im_end|>\n"
            )
        reasoning = message["reasoning_content"].strip()
        content = message["content"].strip()
        return f"<|im_start|>assistant\n<think>\n{reasoning}</think>{content}<|im_end|>\n"

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        assert kwargs.get("add_special_tokens") is False
        result: dict[str, Any] = {"input_ids": [ord(character) for character in text]}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result

    def decode(self, token_ids: Sequence[int], **kwargs: Any) -> str:
        assert kwargs == {"skip_special_tokens": False}
        return "".join(chr(token_id) for token_id in token_ids)


def _test_runtime_identity() -> tuple[str, dict[str, object]]:
    runtime: dict[str, object] = {"runtime": "test"}
    digest = hashlib.sha256(
        json.dumps(runtime, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return digest, runtime


@pytest.fixture
def native_template() -> str:
    return "synthetic native template"


@pytest.fixture
def test_adapter(native_template: str) -> NativeTemplateAdapter:
    digest = hashlib.sha256(native_template.encode()).hexdigest()
    return replace(NEMOTRON_ADAPTER, chat_template_sha256=digest)


@pytest.fixture
def documents() -> tuple[NeutralDocument, ...]:
    return (
        NeutralDocument("doc-a", "Alpha neutral document text. " * 8, "c4"),
        NeutralDocument("doc-b", "Beta reference passage for filler. " * 8, "c4"),
        NeutralDocument("doc-c", "Gamma factual prose for testing. " * 8, "c4"),
    )


@pytest.fixture
def filler_documents() -> tuple[NeutralDocument, ...]:
    return (
        NeutralDocument("filler-a", "Delta unrelated filler source. " * 8, "c4"),
        NeutralDocument("filler-b", "Epsilon separate filler source. " * 8, "c4"),
        NeutralDocument("filler-c", "Zeta additional filler source. " * 8, "c4"),
    )


def test_native_template_is_exact(
    native_template: str, test_adapter: NativeTemplateAdapter
) -> None:
    tokenizer = CharacterTokenizer(native_template)
    validate_native_template(tokenizer, test_adapter)
    tokenizer.chat_template += "\n"
    with pytest.raises(ValueError, match="chat template mismatch"):
        validate_native_template(tokenizer, test_adapter)


def test_role_dataset_has_exact_content_and_position_controls(
    native_template: str,
    test_adapter: NativeTemplateAdapter,
    documents: tuple[NeutralDocument, ...],
    filler_documents: tuple[NeutralDocument, ...],
) -> None:
    dataset = build_role_dataset(
        documents,
        filler_documents,
        CharacterTokenizer(native_template),
        test_adapter,
        max_content_tokens=18,
        max_filler_tokens=80,
        max_sequence_tokens=500,
        seed=7,
    )

    assert len(dataset.examples) == len(ProbeRole) * len(documents)
    assert any(example.content_was_truncated for example in dataset.examples)
    for document_index in range(len(documents)):
        variants = [item for item in dataset.examples if item.document_index == document_index]
        assert {item.role for item in variants} == set(ProbeRole)
        assert len({item.content_positions for item in variants}) == 1
        assert len({item.content_token_ids for item in variants}) == 1
        assert variants[0].partner_document_id != variants[0].document_id
        for example in variants:
            assert set(example.content_positions).isdisjoint(example.filler_positions)
            assert set(example.content_positions).isdisjoint(example.tag_positions)
            assert set(example.filler_positions).isdisjoint(example.tag_positions)
            assert len(example.content_positions) + len(example.filler_positions) + len(
                example.tag_positions
            ) == len(example.input_ids)


def test_role_dataset_rejects_target_as_filler(
    native_template: str,
    test_adapter: NativeTemplateAdapter,
    documents: tuple[NeutralDocument, ...],
) -> None:
    with pytest.raises(ValueError, match="target and filler document IDs must be disjoint"):
        build_role_dataset(
            documents,
            documents,
            CharacterTokenizer(native_template),
            test_adapter,
            max_content_tokens=8,
            max_filler_tokens=8,
            max_sequence_tokens=500,
            seed=0,
        )


def test_role_dataset_rejects_filler_with_duplicate_target_text(
    native_template: str,
    test_adapter: NativeTemplateAdapter,
    documents: tuple[NeutralDocument, ...],
) -> None:
    duplicate = NeutralDocument("different-id", documents[0].text, "c4")
    with pytest.raises(ValueError, match="target and filler document text must be disjoint"):
        build_role_dataset(
            documents,
            (duplicate,),
            CharacterTokenizer(native_template),
            test_adapter,
            max_content_tokens=8,
            max_filler_tokens=8,
            max_sequence_tokens=500,
            seed=0,
        )


def test_load_documents_fails_closed(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"id":"one","text":"One","source":"c4"}\n{"id":"two","text":"Two","source":"c4"}\n',
        encoding="utf-8",
    )
    assert [item.document_id for item in load_neutral_documents(corpus)] == ["one", "two"]
    corpus.write_text('{"id":"one","text":"One"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid neutral document"):
        load_neutral_documents(corpus)


def test_extraction_cli_manifest_and_layer_parsing_fail_closed(tmp_path: Path) -> None:
    assert extraction_cli._layers("0,2,26") == (0, 2, 26)
    with pytest.raises(Exception, match="non-negative, sorted, and unique"):
        extraction_cli._layers("2,1")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "prefix-checkpoint-manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid prefix checkpoint manifest"):
        extraction_cli._manifest(checkpoint)


def test_extraction_cli_wires_prefix_and_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    corpus = tmp_path / "corpus.jsonl"
    output = tmp_path / "output"
    identity: dict[str, object] = {}
    documents = tuple(
        NeutralDocument(f"doc-{index}", f"neutral text {index}", "c4") for index in range(3)
    )
    manifest = {
        "adapter": NEMOTRON_ADAPTER.name,
        "model_id": NEMOTRON_ADAPTER.model_id,
        "revision": NEMOTRON_ADAPTER.model_revision,
        "max_layer": 2,
        "weights_sha256": "a" * 64,
        "weights_hash_kind": extraction.PREFIX_WEIGHTS_HASH_KIND,
    }
    monkeypatch.setattr(extraction_cli, "_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        extraction_cli, "validate_prefix_checkpoint_identity", lambda _path, _manifest: None
    )
    monkeypatch.setattr(extraction_cli, "load_neutral_documents", lambda _path, limit: documents)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: object(),
    )
    dataset = object()
    monkeypatch.setattr(extraction_cli, "build_role_dataset", lambda *_args, **_kwargs: dataset)
    model = object()
    monkeypatch.setattr(extraction_cli, "load_prefix_model", lambda *_args, **_kwargs: model)
    runtime_sha256, runtime = _test_runtime_identity()
    monkeypatch.setattr(
        extraction_cli,
        "model_runtime_identity",
        lambda *_args, **_kwargs: (runtime_sha256, runtime),
    )

    def fake_extract(*args: object, **kwargs: object) -> Path:
        identity.update(kwargs)
        assert args == (model, dataset, NEMOTRON_ADAPTER)
        return output

    monkeypatch.setattr(extraction_cli, "extract_role_activations", fake_extract)
    extraction_cli.main(
        [
            "--adapter",
            NEMOTRON_ADAPTER.name,
            "--checkpoint",
            str(checkpoint),
            "--corpus",
            str(corpus),
            "--output",
            str(output),
            "--layers",
            "0,2",
            "--documents",
            "2",
            "--filler-documents",
            "1",
        ]
    )
    assert identity["layers"] == (0, 2)
    assert identity["batch_size"] is None
    assert identity["activation_dtype"] == "float32"
    assert isinstance(identity["identity"], ExtractionIdentity)
    assert identity["identity"].runtime_sha256 == runtime_sha256
    assert capsys.readouterr().out.strip() == str(output)


def test_nemotron_prefix_key_plan_packs_experts() -> None:
    weights = {
        "backbone.embeddings.weight": "one.safetensors",
        "backbone.layers.0.norm.weight": "one.safetensors",
        "backbone.layers.0.mixer.experts.0.up_proj.weight": "one.safetensors",
        "backbone.layers.0.mixer.experts.1.up_proj.weight": "one.safetensors",
        "backbone.layers.1.norm.weight": "two.safetensors",
        "backbone.layers.1.mixer.D": "two.safetensors",
        "backbone.norm_f.weight": "tail.safetensors",
        "lm_head.weight": "tail.safetensors",
    }
    plan = select_prefix_checkpoint_keys(weights, NEMOTRON_ADAPTER, max_layer=1)
    assert plan.key_mapping == {
        "backbone.embeddings.weight": "embeddings.weight",
        "backbone.layers.0.norm.weight": "layers.0.norm.weight",
        "backbone.layers.0.mixer.experts.0.up_proj.weight": ("layers.0.mixer.experts.up_proj"),
        "backbone.layers.0.mixer.experts.1.up_proj.weight": ("layers.0.mixer.experts.up_proj"),
        "backbone.layers.1.norm.weight": "layers.1.norm.weight",
    }
    assert not any("norm_f" in key or "lm_head" in key for key in plan.key_mapping)


def test_prefix_key_plan_rejects_layer_gap() -> None:
    weights = {
        "backbone.embeddings.weight": "one.safetensors",
        "backbone.layers.0.norm.weight": "one.safetensors",
        "backbone.layers.2.norm.weight": "two.safetensors",
    }
    with pytest.raises(ValueError, match="full prefix layers"):
        select_prefix_checkpoint_keys(weights, NEMOTRON_ADAPTER, max_layer=2)


def test_prefix_checkpoint_identity_is_recomputed(tmp_path: Path) -> None:
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps({"weight_map": {"a": "one.safetensors", "b": "two.safetensors"}}),
        encoding="utf-8",
    )
    (tmp_path / "one.safetensors").write_bytes(b"one")
    (tmp_path / "two.safetensors").write_bytes(b"two")
    (tmp_path / "tokenizer.json").write_bytes(b"tokenizer")
    shard_hashes = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ("one.safetensors", "two.safetensors")
    }
    identity = {
        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "shard_sha256": shard_hashes,
    }
    weights_sha256 = hashlib.sha256(
        json.dumps(identity, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    manifest = {
        "weights_hash_kind": extraction.PREFIX_WEIGHTS_HASH_KIND,
        "weights_sha256": weights_sha256,
        "shards": [
            {"filename": name, "sha256": digest} for name, digest in shard_hashes.items()
        ],
        "auxiliary_files": [
            {
                "filename": "tokenizer.json",
                "bytes": 9,
                "sha256": hashlib.sha256(b"tokenizer").hexdigest(),
            }
        ],
    }
    validate_prefix_checkpoint_identity(tmp_path, manifest)
    (tmp_path / "one.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="weight identity"):
        validate_prefix_checkpoint_identity(tmp_path, manifest)


def test_nemotron_kernel_revisions_load_exact_offline_snapshots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import kernels
    from huggingface_hub import constants
    from transformers.integrations import hub_kernels

    module_mapping: dict[str, ModuleType | None] = {}
    hub_mapping: dict[str, dict[str, str]] = {}
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(hub_kernels, "_KERNEL_MODULE_MAPPING", module_mapping)
    monkeypatch.setattr(hub_kernels, "_HUB_KERNEL_MAPPING", hub_mapping)

    def fake_local_kernel(snapshot: Path) -> ModuleType:
        module = ModuleType(snapshot.parent.parent.parent.name)
        module.__file__ = str(snapshot / "build" / "test" / "__init__.py")
        return module

    monkeypatch.setattr(kernels, "get_local_kernel", fake_local_kernel)
    for repository, revision in extraction.NEMOTRON_KERNEL_REVISIONS.values():
        (tmp_path / f"kernels--{repository.replace('/', '--')}" / "snapshots" / revision).mkdir(
            parents=True
        )
    extraction._pin_nemotron_kernel_revisions()
    assert set(module_mapping) == set(extraction.NEMOTRON_KERNEL_REVISIONS)
    for name, (repository, revision) in extraction.NEMOTRON_KERNEL_REVISIONS.items():
        assert hub_mapping[name] == {"repo_id": repository, "revision": revision}
        loaded = module_mapping[name]
        assert loaded is not None
        assert f"/snapshots/{revision}/" in str(loaded.__file__)


def test_loaded_native_prefix_exactly_matches_full_model(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    try:
        from safetensors.torch import save_file
        from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
        from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHModel
    except ImportError:
        pytest.skip("pinned role-probe runtime is not installed")
    config = NemotronHConfig(
        vocab_size=64,
        hidden_size=32,
        layers_block_type=["attention", "attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        use_mamba_kernels=False,
        dtype="bfloat16",
        architectures=["NemotronHForCausalLM"],
    )
    config.save_pretrained(tmp_path)
    full = NemotronHModel(config).to(dtype=torch.bfloat16).eval()  # ty: ignore[missing-argument]
    source = {
        f"backbone.{name}": value.detach().cpu()
        for name, value in full.state_dict().items()
        if name == "embeddings.weight" or name.startswith("layers.0.") or name == "layers.1.norm.weight"
    }
    shard_name = "model-00001-of-00001.safetensors"
    save_file(source, tmp_path / shard_name)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(source, shard_name)}), encoding="utf-8"
    )

    loaded = extraction.load_prefix_model(
        tmp_path, NEMOTRON_ADAPTER, max_layer=1, execution_device="cpu"
    )
    request = {
        "input_ids": (1, 2, 3, 4),
        "token_positions": (1, 2),
        "layers": (0, 1),
    }
    expected = extraction.capture_token_activations(full, NEMOTRON_ADAPTER, **request)
    actual = extraction.capture_token_activations(loaded, NEMOTRON_ADAPTER, **request)
    assert np.array_equal(actual, expected)
    assert all(
        parameter.device.type == "meta" for parameter in cast(Any, loaded).parameters()
    )
    runtime_sha256, runtime = extraction.model_runtime_identity(
        loaded, NEMOTRON_ADAPTER, checkpoint=tmp_path
    )
    assert hashlib.sha256(
        json.dumps(runtime, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest() == runtime_sha256
    assert runtime["nemotron_mamba"]["fast_path_selected"] is False


def test_artifact_writer_is_atomic_and_complete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    native_template: str,
    test_adapter: NativeTemplateAdapter,
    documents: tuple[NeutralDocument, ...],
    filler_documents: tuple[NeutralDocument, ...],
) -> None:
    dataset = build_role_dataset(
        documents[:2],
        filler_documents[:2],
        CharacterTokenizer(native_template),
        test_adapter,
        max_content_tokens=8,
        max_filler_tokens=80,
        max_sequence_tokens=500,
        seed=2,
    )

    class Config:
        hidden_size = 3

    class Model:
        config = Config()

    def fake_capture(
        model: object, adapter: object, examples: tuple[Any, ...], layers: tuple[int, ...]
    ) -> np.ndarray:
        del model, adapter
        count = len(examples[0].content_positions)
        return np.arange(len(examples) * count * len(layers) * 3, dtype=np.float32).reshape(
            len(examples), count, len(layers), 3
        )

    monkeypatch.setattr(extraction, "_capture_role_group", fake_capture)
    output = tmp_path / "artifact"
    extract_role_activations(
        Model(),
        dataset,
        test_adapter,
        layers=(0, 2),
        identity=ExtractionIdentity(
            weights_sha256="a" * 64,
            weights_hash_kind="test-manifest",
            tokenizer_id=NEMOTRON_ADAPTER.model_id,
            tokenizer_revision=NEMOTRON_ADAPTER.model_revision,
            source_name="test-c4",
            model_dtype="bfloat16",
            transformers_version="test",
            torch_version="test",
            runtime_sha256=_test_runtime_identity()[0],
            runtime=_test_runtime_identity()[1],
        ),
        output=output,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    activations = np.load(output / "activations.npy", allow_pickle=False)
    assert activations.shape == (80, 2, 3)
    assert manifest["activation_site"] == "normalized_pre_mixer"
    assert manifest["dataset_kind"] == PAIRED_NEUTRAL
    assert manifest["content_mask"] == CONTENT_TOKENS_ONLY
    assert manifest["filler_pool_kind"] == "dedicated-disjoint-documents"
    assert manifest["filler_documents"] == 2
    assert manifest["roles"] == {
        "assistant": 4,
        "reasoning": 3,
        "system": 0,
        "tool": 2,
        "user": 1,
    }
    assert set(manifest["files"]) == {
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
    target_documents = [
        json.loads(line)
        for line in (output / "target_documents.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    filler_documents_json = [
        json.loads(line)
        for line in (output / "filler_documents.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["text_sha256"] for row in target_documents}.isdisjoint(
        row["text_sha256"] for row in filler_documents_json
    )
    assert manifest["runtime_metrics"]["documents"] == 2
    assert manifest["runtime_metrics"]["forward_batches"] == 6
    assert manifest["runtime_metrics"]["configured_batch_size"] == 2
    assert manifest["runtime_metrics"]["role_sequences_per_document"] == 5
    loaded = load_activation_dataset(output)
    assert loaded.activations.shape == (80, 2, 3)
    assert loaded.provenance.model_dtype == "bfloat16"
    assert set(loaded.document_ids) == {"doc-a", "doc-b"}
    assert set(loaded.filler_document_ids) == {"filler-a", "filler-b"}
    unexpected = output / "unexpected.txt"
    unexpected.write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or unexpected files"):
        load_activation_dataset(output)
    unexpected.unlink()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        extract_role_activations(
            Model(),
            dataset,
            NEMOTRON_ADAPTER,
            layers=(0,),
            identity=ExtractionIdentity(
                weights_sha256="a" * 64,
                weights_hash_kind="test",
                tokenizer_id="test",
                tokenizer_revision="test",
                source_name="test",
                model_dtype="test",
                transformers_version="test",
                torch_version="test",
                runtime_sha256=_test_runtime_identity()[0],
                runtime=_test_runtime_identity()[1],
            ),
            output=output,
        )


def test_prefix_capture_equals_full_forward_and_stops_tail(
    native_template: str, test_adapter: NativeTemplateAdapter
) -> None:
    torch = pytest.importorskip("torch")

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = torch.nn.LayerNorm(4)
            self.mixer = torch.nn.Linear(4, 4, bias=False)
            self.calls = 0

        def forward(self, hidden: Any, **kwargs: Any) -> Any:
            del kwargs
            self.calls += 1
            return hidden + self.mixer(self.norm(hidden))

    class Config:
        hidden_size = 4

    class NemotronHModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = Config()
            self.embeddings = torch.nn.Embedding(256, 4)
            self.layers = torch.nn.ModuleList([Block(), Block(), Block()])

        def get_input_embeddings(self) -> Any:
            return self.embeddings

        def forward(self, input_ids: Any, **kwargs: Any) -> Any:
            del kwargs
            hidden = self.embeddings(input_ids)
            for layer in self.layers:
                hidden = layer(hidden)
            return hidden

    model = NemotronHModel().eval()
    dataset = build_role_dataset(
        (
            NeutralDocument("one", "abcd filler passage has enough characters " * 8, "c4"),
            NeutralDocument("two", "wxyz filler passage has enough characters " * 8, "c4"),
        ),
        (
            NeutralDocument("three", "other source passage has enough characters " * 8, "c4"),
            NeutralDocument("four", "separate source passage has enough characters " * 8, "c4"),
        ),
        CharacterTokenizer(native_template),
        test_adapter,
        max_content_tokens=4,
        max_filler_tokens=80,
        max_sequence_tokens=500,
        seed=0,
    )
    example = dataset.examples[0]
    expected: dict[int, Any] = {}
    handles = [
        model.layers[index].norm.register_forward_hook(
            lambda _module, _inputs, value, index=index: expected.__setitem__(
                index, value[0, list(example.content_positions)].detach().float()
            )
        )
        for index in (0, 1)
    ]
    with torch.inference_mode():
        model(
            input_ids=torch.tensor([example.input_ids]),
            attention_mask=torch.ones(1, len(example.input_ids)),
            use_cache=False,
        )
    for handle in handles:
        handle.remove()
    for layer in model.layers:
        layer.calls = 0

    actual = extraction._capture_one(model, test_adapter, example, (0, 1))
    assert np.array_equal(actual, torch.stack([expected[0], expected[1]], dim=1).numpy())
    assert [layer.calls for layer in model.layers] == [1, 1, 0]

    group = tuple(item for item in dataset.examples if item.document_index == 0)
    singles = np.stack(
        [extraction._capture_one(model, test_adapter, item, (0, 1)) for item in group]
    )
    for layer in model.layers:
        layer.calls = 0
    batched = extraction._capture_role_group(model, test_adapter, group, (0, 1))
    assert np.array_equal(batched, singles)
    assert [layer.calls for layer in model.layers] == [1, 1, 0]


def test_native_prefix_batch_matches_unbatched() -> None:
    pytest.importorskip("torch")
    try:
        from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
        from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
        from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHModel
    except ImportError:
        pytest.skip("pinned role-probe runtime is not installed")
    cases = (
        (
            NemotronHModel(
                NemotronHConfig(
                    vocab_size=64,
                    hidden_size=32,
                    num_hidden_layers=3,  # ty: ignore[unknown-argument]
                    layers_block_type=["attention"] * 3,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=8,
                    intermediate_size=64,
                    use_mamba_kernels=False,
                )
            ),
            NEMOTRON_ADAPTER,
        ),
        (
            Gemma4TextModel(
                Gemma4TextConfig(
                    vocab_size=64,
                    hidden_size=32,
                    num_hidden_layers=3,
                    layer_types=["full_attention"] * 3,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    num_global_key_value_heads=1,
                    head_dim=8,
                    global_head_dim=8,
                    intermediate_size=64,
                    max_position_embeddings=128,
                    sliding_window=32,
                )
            ),
            extraction.GEMMA_ADAPTER,
        ),
    )
    input_ids = ((1, 2, 3, 4, 5), (1, 2, 3, 4, 5, 6, 7))
    positions = ((1, 2), (1, 2))
    for model, adapter in cases:
        model.eval()
        singles = np.stack(
            [
                extraction.capture_token_activations(
                    model,
                    adapter,
                    input_ids=sequence,
                    token_positions=selected,
                    layers=(0, 1),
                )
                for sequence, selected in zip(input_ids, positions, strict=True)
            ]
        )
        batched = extraction._capture_sequences(
            model,
            adapter,
            input_ids=input_ids,
            token_positions=positions,
            layers=(0, 1),
        )
        assert np.array_equal(batched, singles)
