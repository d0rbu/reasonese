"""Extract local native-role activations from a prefix checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

from reasonese.role_probe_extraction import (
    NATIVE_ADAPTERS,
    ExtractionIdentity,
    NativeTokenizer,
    build_role_dataset,
    extract_role_activations,
    load_neutral_documents,
    load_prefix_model,
    model_runtime_identity,
    validate_prefix_checkpoint_identity,
)


def _layers(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from error
    if not result or tuple(sorted(set(result))) != result or result[0] < 0:
        raise argparse.ArgumentTypeError("layers must be non-negative, sorted, and unique")
    return result


def _manifest(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint / "prefix-checkpoint-manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid prefix checkpoint manifest: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid prefix checkpoint manifest: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=sorted(NATIVE_ADAPTERS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=_layers, required=True)
    parser.add_argument("--documents", type=int, default=16)
    parser.add_argument("--filler-documents", type=int, default=16)
    parser.add_argument("--max-content-tokens", type=int, default=256)
    parser.add_argument("--max-filler-tokens", type=int, default=128)
    parser.add_argument("--max-sequence-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execution-device", default="cuda:0")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="sequence microbatch (default: 2 for Nemotron, 5 for Gemma)",
    )
    parser.add_argument("--activation-dtype", choices=("float16", "float32"), default="float32")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.documents < 2:
        raise ValueError("--documents must be at least two")
    if args.filler_documents < 1:
        raise ValueError("--filler-documents must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    adapter = NATIVE_ADAPTERS[args.adapter]
    checkpoint_manifest = _manifest(args.checkpoint)
    required = {
        "adapter": adapter.name,
        "model_id": adapter.model_id,
        "revision": adapter.model_revision,
        "max_layer": args.layers[-1],
    }
    for key, expected in required.items():
        if checkpoint_manifest.get(key) != expected:
            raise ValueError(
                f"prefix checkpoint manifest {key}={checkpoint_manifest.get(key)!r}, "
                f"expected {expected!r}"
            )
    try:
        weights_sha256 = checkpoint_manifest["weights_sha256"]
        weights_hash_kind = checkpoint_manifest["weights_hash_kind"]
    except KeyError as error:
        raise ValueError("prefix checkpoint manifest lacks its exact weight identity") from error
    validate_prefix_checkpoint_identity(args.checkpoint, checkpoint_manifest)

    try:
        import torch  # ty: ignore[unresolved-import, unused-ignore-comment]
        import transformers  # ty: ignore[unresolved-import, unused-ignore-comment]
        from transformers import AutoTokenizer  # ty: ignore[unresolved-import, unused-ignore-comment]
    except ImportError as error:  # pragma: no cover - exercised by minimal installations
        raise RuntimeError("role-probe extraction requires the 'probes' extra") from error
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, use_fast=True)
    required_documents = args.documents + args.filler_documents
    corpus_records = load_neutral_documents(args.corpus, limit=required_documents)
    if len(corpus_records) != required_documents:
        raise ValueError("corpus does not contain the requested target and filler documents")
    documents = corpus_records[: args.documents]
    filler_documents = corpus_records[args.documents : required_documents]
    dataset = build_role_dataset(
        documents,
        filler_documents,
        cast(NativeTokenizer, tokenizer),
        adapter,
        max_content_tokens=args.max_content_tokens,
        max_filler_tokens=args.max_filler_tokens,
        max_sequence_tokens=args.max_sequence_tokens,
        seed=args.seed,
    )
    load_started = time.perf_counter()
    model = load_prefix_model(
        args.checkpoint,
        adapter,
        max_layer=args.layers[-1],
        execution_device=args.execution_device,
    )
    model_load_seconds = time.perf_counter() - load_started
    runtime_sha256, runtime = model_runtime_identity(
        model, adapter, checkpoint=args.checkpoint
    )
    output = extract_role_activations(
        model,
        dataset,
        adapter,
        layers=args.layers,
        identity=ExtractionIdentity(
            weights_sha256=weights_sha256,
            weights_hash_kind=weights_hash_kind,
            tokenizer_id=adapter.model_id,
            tokenizer_revision=adapter.model_revision,
            source_name=args.corpus.name,
            model_dtype="bfloat16",
            transformers_version=transformers.__version__,
            torch_version=torch.__version__,
            runtime_sha256=runtime_sha256,
            runtime=runtime,
        ),
        output=args.output,
        activation_dtype=args.activation_dtype,
        model_load_seconds=model_load_seconds,
        batch_size=args.batch_size,
    )
    print(output)


if __name__ == "__main__":  # pragma: no cover
    main()
