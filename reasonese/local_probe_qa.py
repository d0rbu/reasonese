"""Concrete, cache-aware local activation-probe QA scorer."""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from beartype import beartype

from reasonese.axes import Assistant
from reasonese.probe_qa import (
    ProbeExpectation,
    ProbeQaRequest,
    ProbeQaVerdict,
    probe_expectation,
)
from reasonese.probe_rendering import SERVER_TOOL_LIMITATION, render_collector_probe_context
from reasonese.role_probe_extraction import (
    NATIVE_ADAPTERS,
    NativeTemplateAdapter,
    ProbeRole,
    capture_token_activations,
    load_prefix_model,
    model_runtime_identity,
    validate_prefix_checkpoint_identity,
)
from reasonese.role_probes import RoleProbe, load_role_probe, project_role

_FORMAT_VERSION = 1
_ASSISTANT_ADAPTERS = {
    Assistant.NEMOTRON_3_5_LIGHTNING: "nemotron-3.5-lightning-native-v1",
    Assistant.GEMMA_4_31B_IT: "gemma-4-31b-native-v1",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid JSON object: {path}")
    return value


@beartype
@dataclass(frozen=True, slots=True)
class ProbeBundle:
    """One assistant's qualified probe and exact local prefix checkpoint."""

    assistant: Assistant
    adapter: NativeTemplateAdapter
    checkpoint: Path
    probe_path: Path


def load_probe_bundles(path: Path) -> tuple[ProbeBundle, ...]:
    """Load a small path-relative assistant bundle configuration."""
    raw = _json_object(path)
    if set(raw) != {"bundles"} or not isinstance(raw["bundles"], list):
        raise ValueError("role-probe config must contain only a bundles list")
    bundles: list[ProbeBundle] = []
    for value in raw["bundles"]:
        if not isinstance(value, dict) or set(value) != {
            "assistant",
            "adapter",
            "checkpoint",
            "probe",
        }:
            raise ValueError("invalid role-probe bundle record")
        try:
            assistant = Assistant(value["assistant"])
            adapter = NATIVE_ADAPTERS[value["adapter"]]
        except (KeyError, ValueError) as error:
            raise ValueError(
                "role-probe bundle names an unsupported assistant or adapter"
            ) from error
        if _ASSISTANT_ADAPTERS.get(assistant) != adapter.name:
            raise ValueError("role-probe assistant does not match its native adapter")
        checkpoint = value["checkpoint"]
        probe = value["probe"]
        if not isinstance(checkpoint, str) or not isinstance(probe, str):
            raise ValueError("role-probe bundle paths must be strings")
        bundles.append(
            ProbeBundle(
                assistant,
                adapter,
                (path.parent / checkpoint).resolve(),
                (path.parent / probe).resolve(),
            )
        )
    if not bundles or len({bundle.assistant for bundle in bundles}) != len(bundles):
        raise ValueError("role-probe config requires distinct assistant bundles")
    return tuple(bundles)


@dataclass(frozen=True, slots=True)
class _PreparedBundle:
    bundle: ProbeBundle
    probe: RoleProbe
    probe_sha256: str
    checkpoint_manifest: dict[str, Any]


class _VerdictCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: dict[str, dict[str, Any]] = {}
        if not path.exists():
            return
        raw = _json_object(path)
        if set(raw) != {"format_version", "records"} or raw["format_version"] != _FORMAT_VERSION:
            raise ValueError("unsupported local probe-QA cache")
        records = raw["records"]
        if not isinstance(records, dict) or any(
            not isinstance(key, str) or not isinstance(value, dict)
            for key, value in records.items()
        ):
            raise ValueError("invalid local probe-QA cache records")
        self.records = cast(dict[str, dict[str, Any]], records)

    def verdict(self, key: str, request: ProbeQaRequest, probe: RoleProbe) -> ProbeQaVerdict | None:
        raw = self.records.get(key)
        if raw is None:
            return None
        if set(raw) != {
            "role_probabilities",
            "reasoning_probability",
            "complies",
            "issue",
            "masked_boundary_tokens",
        }:
            raise ValueError("invalid local probe-QA cache verdict")
        role_probabilities = raw["role_probabilities"]
        if not isinstance(role_probabilities, list):
            raise ValueError("invalid cached role probabilities")
        try:
            probabilities = tuple((str(row[0]), float(row[1])) for row in role_probabilities)
            reasoning = float(raw["reasoning_probability"])
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError("invalid cached role probabilities") from error
        complies = raw["complies"]
        issue = raw["issue"]
        if (complies is not None and not isinstance(complies, bool)) or (
            issue is not None and not isinstance(issue, str)
        ):
            raise ValueError("invalid cached probe decision")
        if tuple(role for role, _ in probabilities) != probe.provenance.roles:
            raise ValueError("cached role probabilities do not match the qualified probe")
        expected_complies, expected_issue = _decision(request, probe, reasoning)
        if (complies, issue) != (expected_complies, expected_issue):
            raise ValueError("cached probe decision does not match its probability and threshold")
        return ProbeQaVerdict(
            request,
            key,
            probabilities,
            reasoning,
            probe_expectation(request.spec),
            complies,
            issue,
            raw["masked_boundary_tokens"],
        )

    def put(self, verdict: ProbeQaVerdict) -> None:
        self.records[verdict.context_fingerprint] = {
            "role_probabilities": [list(row) for row in verdict.role_probabilities],
            "reasoning_probability": verdict.reasoning_probability,
            "complies": verdict.complies,
            "issue": verdict.issue,
            "masked_boundary_tokens": verdict.masked_boundary_tokens,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    {"format_version": _FORMAT_VERSION, "records": self.records},
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                handle.write("\n")
            os.replace(name, self.path)
        except BaseException:
            Path(name).unlink(missing_ok=True)
            raise


def _validate_probe(bundle: ProbeBundle, probe: RoleProbe) -> None:
    adapter = bundle.adapter
    provenance = probe.provenance
    if not probe.qa_eligible:
        raise ValueError(f"role probe for {bundle.assistant} is not QA-eligible")
    if (
        provenance.model_id != adapter.model_id
        or provenance.model_revision != adapter.model_revision
        or provenance.tokenizer_id != adapter.model_id
        or provenance.tokenizer_revision != adapter.model_revision
        or provenance.chat_template_sha256 != adapter.chat_template_sha256
        or provenance.native_template_adapter != adapter.name
        or provenance.activation_site != adapter.activation_site
        or provenance.activation_dtype != "float32"
        or provenance.roles != tuple(str(role) for role in ProbeRole)
    ):
        raise ValueError(f"role probe for {bundle.assistant} does not match its native adapter")


def _decision(
    request: ProbeQaRequest, probe: RoleProbe, reasoning_probability: float
) -> tuple[bool | None, str | None]:
    expectation = probe_expectation(request.spec)
    if expectation is ProbeExpectation.DESCRIPTIVE:
        return None, None
    assert probe.qualification is not None
    threshold = probe.qualification.calibration.threshold
    if threshold is None:
        raise ValueError("QA-eligible role probe has no calibration threshold")
    classified_reasoning = reasoning_probability >= threshold
    complies = (
        classified_reasoning
        if expectation is ProbeExpectation.REASONING
        else not classified_reasoning
    )
    if complies:
        return True, None
    return False, (
        f"mean P(reasoning)={reasoning_probability:.8f} conflicts with "
        f"{expectation} at frozen threshold {threshold:.8f}"
    )


class LocalProbeQaScorer:
    """Score exact collector contexts with qualified local activation probes."""

    def __init__(self, config: Path, cache: Path, *, execution_device: str = "cuda:0") -> None:
        self.bundles = {bundle.assistant: bundle for bundle in load_probe_bundles(config)}
        self.cache = _VerdictCache(cache)
        self.execution_device = execution_device
        self._prepared: dict[Assistant, _PreparedBundle] = {}

    def preflight(self, assistants: tuple[Assistant, ...]) -> None:
        """Validate every static artifact and checkpoint before any provider call."""
        try:
            importlib.import_module("torch")
            importlib.import_module("transformers")
        except ImportError as error:
            raise RuntimeError("local role-probe QA requires the 'probes' extra") from error
        for assistant in dict.fromkeys(assistants):
            bundle = self.bundles.get(assistant)
            if bundle is None:
                raise ValueError(f"no local role-probe bundle is configured for {assistant}")
            if assistant in self._prepared:
                continue
            probe = load_role_probe(bundle.probe_path)
            _validate_probe(bundle, probe)
            manifest = _json_object(bundle.checkpoint / "prefix-checkpoint-manifest.json")
            expected = {
                "adapter": bundle.adapter.name,
                "model_id": bundle.adapter.model_id,
                "revision": bundle.adapter.model_revision,
                "max_layer": max(probe.provenance.layer_indices),
                "weights_sha256": probe.provenance.weights_sha256,
                "weights_hash_kind": probe.provenance.weights_hash_kind,
            }
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise ValueError(f"prefix checkpoint does not match the probe for {assistant}")
            validate_prefix_checkpoint_identity(bundle.checkpoint, manifest)
            prepared = _PreparedBundle(bundle, probe, _file_sha256(bundle.probe_path), manifest)
            model = self._load_validated_model(prepared)
            del model
            self._release_device_cache()
            self._prepared[assistant] = prepared

    def _load_validated_model(self, prepared: _PreparedBundle) -> object:
        try:
            model = load_prefix_model(
                prepared.bundle.checkpoint,
                prepared.bundle.adapter,
                max_layer=max(prepared.probe.training.layer_indices),
                execution_device=self.execution_device,
            )
        except BaseException:
            self._release_device_cache()
            raise
        try:
            validate_prefix_checkpoint_identity(
                prepared.bundle.checkpoint, prepared.checkpoint_manifest
            )
            runtime_sha256, _ = model_runtime_identity(
                model, prepared.bundle.adapter, checkpoint=prepared.bundle.checkpoint
            )
            if runtime_sha256 != prepared.probe.provenance.runtime_sha256:
                raise ValueError("local scorer runtime does not match qualified probe provenance")
        except BaseException:
            del model
            self._release_device_cache()
            raise
        return model

    @staticmethod
    def _release_device_cache() -> None:
        gc.collect()
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _context_key(
        prepared: _PreparedBundle,
        request: ProbeQaRequest,
        input_ids: tuple[int, ...],
        positions: tuple[int, ...],
        content_ids: tuple[int, ...],
        render_config_sha256: str,
    ) -> str:
        return _canonical_sha256(
            {
                "probe_sha256": prepared.probe_sha256,
                "runtime_sha256": prepared.probe.provenance.runtime_sha256,
                "weights_sha256": prepared.probe.provenance.weights_sha256,
                "adapter": prepared.bundle.adapter.name,
                "assistant": str(request.setup.matchup.assistant),
                "study_id": request.study_id,
                "permutation": request.permutation,
                "position": request.position,
                "render_config_sha256": render_config_sha256,
                "input_ids": input_ids,
                "token_positions": positions,
                "content_token_ids": content_ids,
            }
        )

    def check(self, requests: tuple[ProbeQaRequest, ...]) -> tuple[ProbeQaVerdict, ...]:
        """Score requests by assistant while keeping only one prefix model resident."""
        if not requests:
            return ()
        self.preflight(tuple(dict.fromkeys(row.setup.matchup.assistant for row in requests)))
        verdicts: dict[ProbeQaRequest, ProbeQaVerdict] = {}
        grouped: dict[Assistant, list[ProbeQaRequest]] = defaultdict(list)
        for request in requests:
            grouped[request.setup.matchup.assistant].append(request)
        for assistant, assistant_requests in grouped.items():
            self._check_assistant(self._prepared[assistant], assistant_requests, verdicts)
        return tuple(verdicts[request] for request in requests)

    def _check_assistant(
        self,
        prepared: _PreparedBundle,
        requests: list[ProbeQaRequest],
        verdicts: dict[ProbeQaRequest, ProbeQaVerdict],
    ) -> None:
        transformers = importlib.import_module("transformers")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            prepared.bundle.checkpoint, local_files_only=True, use_fast=True
        )
        rendered: dict[ProbeQaRequest, Any] = {}
        keys: dict[ProbeQaRequest, str] = {}
        missing: list[ProbeQaRequest] = []
        for request in requests:
            context = render_collector_probe_context(
                tokenizer,
                prepared.bundle.adapter,
                request.setup,
                request.position,
            )
            rendered[request] = context
            key = self._context_key(
                prepared,
                request,
                context.input_ids,
                context.token_positions[0],
                context.content_token_ids[0],
                context.render_config_sha256,
            )
            keys[request] = key
            cached = self.cache.verdict(key, request, prepared.probe)
            if cached is None:
                missing.append(request)
            else:
                verdicts[request] = cached
        if not missing:
            return

        model = self._load_validated_model(prepared)
        try:
            by_context: dict[tuple[str, int], list[ProbeQaRequest]] = defaultdict(list)
            for request in missing:
                by_context[(request.study_id, request.permutation)].append(request)
            for unsorted_requests in by_context.values():
                context_requests = sorted(unsorted_requests, key=lambda request: request.position)
                contexts = [rendered[request] for request in context_requests]
                if len({context.input_ids for context in contexts}) != 1:
                    raise ValueError(
                        "target spans in one ordered setup rendered different contexts"
                    )
                positions = tuple(
                    position for context in contexts for position in context.token_positions[0]
                )
                captured = capture_token_activations(
                    model,
                    prepared.bundle.adapter,
                    input_ids=contexts[0].input_ids,
                    token_positions=positions,
                    layers=(prepared.probe.selected_layer_index,),
                )
                if (
                    captured.ndim != 3
                    or captured.shape[0] != len(positions)
                    or captured.shape[1] != 1
                    or captured.dtype.name != prepared.probe.provenance.activation_dtype
                ):
                    raise ValueError("local role-probe capture returned an invalid shape")
                cursor = 0
                for request, context in zip(context_requests, contexts, strict=True):
                    count = len(context.token_positions[0])
                    activations = np.asarray(captured[cursor : cursor + count, 0])
                    projection = project_role(
                        prepared.probe,
                        activations,
                        provenance=prepared.probe.provenance,
                        layer_index=prepared.probe.selected_layer_index,
                    )
                    complies, issue = _decision(
                        request, prepared.probe, projection.mean_reasoning_probability
                    )
                    verdict = ProbeQaVerdict(
                        request,
                        keys[request],
                        tuple(zip(projection.roles, projection.mean_probabilities, strict=True)),
                        projection.mean_reasoning_probability,
                        probe_expectation(request.spec),
                        complies,
                        issue,
                        context.masked_boundary_tokens,
                    )
                    verdicts[request] = verdict
                    self.cache.put(verdict)
                    cursor += count
                if cursor != captured.shape[0]:
                    raise ValueError("captured activations did not map to every target span")
        finally:
            del model
            self._release_device_cache()

    @property
    def limitations(self) -> tuple[str, ...]:
        return (
            SERVER_TOOL_LIMITATION,
            "Local open-weight activations do not establish hosted checkpoint, quantization, "
            "template, or provider-transform parity.",
        )
