"""Concrete, cache-aware local activation-probe QA scorer."""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from beartype import beartype

from reasonese.axes import Assistant, Channel
from reasonese.probe_qa import (
    ProbeExpectation,
    ProbeQaRequest,
    ProbeQaVerdict,
    probe_expectation,
)
from reasonese.probe_rendering import (
    SERVER_TOOL_LIMITATION,
    RenderedProbeContext,
    render_collector_probe_context,
)
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
CAPTURE_POLICY = "segment_prefix_capture_v2_nemotron_cumsum_h1"
_ASSISTANT_ADAPTERS = {
    Assistant.NEMOTRON_3_5_LIGHTNING: "nemotron-3.5-lightning-native-v1",
    Assistant.GEMMA_4_31B_IT: "gemma-4-31b-native-v1",
}
_NEMOTRON_CUMSUM_HEAD_TILES = (1, 2, 4, 8, 16, 32, 64)


def _nemotron_cumsum_autotuner(model: object) -> Any:
    """Return the pinned Mamba cumsum autotuner or fail on runtime drift."""
    autotuner_type = importlib.import_module("triton.runtime.autotuner").Autotuner
    model_module = sys.modules.get(model.__class__.__module__)
    combined = getattr(model_module, "mamba_chunk_scan_combined", None)
    module_name = getattr(combined, "__module__", "")
    marker = ".ops.triton."
    if not module_name.startswith("_mamba_ssm_cuda_") or marker not in module_name:
        raise RuntimeError("validated Nemotron model lacks its pinned Mamba Triton callable")
    kernel_module = importlib.import_module(
        f"{module_name.partition(marker)[0]}{marker}ssd_chunk_state"
    )
    value = getattr(kernel_module, "_chunk_cumsum_fwd_kernel", None)
    visited: set[int] = set()
    for _ in range(5):
        if id(value) in visited:
            break
        visited.add(id(value))
        if isinstance(value, autotuner_type):
            return value
        value = getattr(value, "fn", None)
    raise RuntimeError("validated Nemotron model lacks its cumsum Triton autotuner")


@contextmanager
def _pinned_nemotron_cumsum(
    model: object, adapter: NativeTemplateAdapter, execution_device: str
) -> Any:
    """Use the registered H1 cumsum config for one Nemotron CUDA capture scope."""
    if adapter.name != "nemotron-3.5-lightning-native-v1" or not execution_device.startswith(
        "cuda"
    ):
        yield
        return
    tuner = _nemotron_cumsum_autotuner(model)
    configs = tuner.configs
    records = [config.all_kwargs() for config in configs]
    expected = [
        {
            "BLOCK_SIZE_H": head_tile,
            "num_ctas": 1,
            "num_stages": 3,
            "num_warps": 4,
        }
        for head_tile in _NEMOTRON_CUMSUM_HEAD_TILES
    ]
    if records != expected:
        raise RuntimeError("Nemotron cumsum Triton configs differ from the pinned runtime")
    selected = configs[0]
    if selected.pre_hook is not None:
        raise RuntimeError("Nemotron cumsum H1 config unexpectedly has a pre-hook")
    absent = object()
    best_config = getattr(tuner, "best_config", absent)
    nargs = getattr(tuner, "nargs", absent)
    tuner.configs = [selected]
    try:
        yield
    finally:
        tuner.configs = configs
        for name, value in (("best_config", best_config), ("nargs", nargs)):
            if value is absent:
                if hasattr(tuner, name):
                    delattr(tuner, name)
            else:
                setattr(tuner, name, value)


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
class FramingThresholds:
    """Separate acceptance cutoffs; overlap is tolerance, not a unique role label."""

    reasoning_minimum: float
    nonreasoning_maximum: float

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or not 0 <= value <= 1
            for value in (self.reasoning_minimum, self.nonreasoning_maximum)
        ):
            raise ValueError("framing thresholds must be finite probabilities")


@beartype
@dataclass(frozen=True, slots=True)
class ProbeBundle:
    """One assistant's qualified probe and exact local prefix checkpoint."""

    assistant: Assistant
    adapter: NativeTemplateAdapter
    checkpoint: Path
    probe_path: Path
    framing_thresholds: tuple[tuple[Channel, FramingThresholds], ...] = ()

    def __post_init__(self) -> None:
        if self.framing_thresholds and (
            len(self.framing_thresholds) != len(Channel)
            or {channel for channel, _ in self.framing_thresholds} != set(Channel)
        ):
            raise ValueError("framing thresholds must cover each channel exactly once")

    def thresholds_for(self, channel: Channel) -> FramingThresholds | None:
        return dict(self.framing_thresholds).get(channel)

    def threshold_identity(self) -> dict[str, object]:
        return {
            str(channel): {
                "reasoning_minimum": cutoffs.reasoning_minimum,
                "nonreasoning_maximum": cutoffs.nonreasoning_maximum,
            }
            for channel, cutoffs in self.framing_thresholds
        }


def load_probe_bundles(path: Path) -> tuple[ProbeBundle, ...]:
    """Load a small path-relative assistant bundle configuration."""
    raw = _json_object(path)
    if set(raw) != {"bundles"} or not isinstance(raw["bundles"], list):
        raise ValueError("role-probe config must contain only a bundles list")
    bundles: list[ProbeBundle] = []
    for value in raw["bundles"]:
        required = {"assistant", "adapter", "checkpoint", "probe"}
        if (
            not isinstance(value, dict) or not required <= set(value)
            or set(value) - required - {"framing_thresholds"}
        ):
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
        thresholds: list[tuple[Channel, FramingThresholds]] = []
        if "framing_thresholds" in value:
            raw_thresholds = value["framing_thresholds"]
            if not isinstance(raw_thresholds, dict) or set(raw_thresholds) != set(Channel):
                raise ValueError("framing thresholds must cover every channel")
            for channel in Channel:
                raw_pair = raw_thresholds[str(channel)]
                if not isinstance(raw_pair, dict) or set(raw_pair) != {
                    "reasoning_minimum", "nonreasoning_maximum"
                }:
                    raise ValueError("invalid framing threshold pair")
                if any(type(v) not in (int, float) for v in raw_pair.values()):
                    raise ValueError("framing thresholds must be numeric probabilities")
                thresholds.append(
                    (channel, FramingThresholds(
                        float(raw_pair["reasoning_minimum"]),
                        float(raw_pair["nonreasoning_maximum"]),
                    ))
                )
        bundles.append(
            ProbeBundle(
                assistant,
                adapter,
                (path.parent / checkpoint).resolve(),
                (path.parent / probe).resolve(),
                tuple(thresholds),
            )
        )
    if not bundles or len({bundle.assistant for bundle in bundles}) != len(bundles):
        raise ValueError("role-probe config requires distinct assistant bundles")
    return tuple(bundles)


def probe_bundle_identity(path: Path) -> dict[str, object]:
    """Describe the configured scoring artifacts without loading model frameworks."""
    if not path.is_file():
        raise ValueError(f"role-probe bundle does not exist: {path}")
    bundles = load_probe_bundles(path)
    return {
        "capture_policy": CAPTURE_POLICY,
        "config": {"path": str(path.resolve()), "sha256": _file_sha256(path)},
        "bundles": [
            {
                "assistant": str(bundle.assistant),
                "adapter": bundle.adapter.name,
                "checkpoint": str(bundle.checkpoint),
                "checkpoint_manifest": (
                    {
                        "path": str((bundle.checkpoint / "prefix-checkpoint-manifest.json").resolve()),
                        "sha256": _file_sha256(
                            bundle.checkpoint / "prefix-checkpoint-manifest.json"
                        ),
                        "identity": _json_object(
                            bundle.checkpoint / "prefix-checkpoint-manifest.json"
                        ),
                    }
                    if (bundle.checkpoint / "prefix-checkpoint-manifest.json").is_file()
                    else None
                ),
                "probe": str(bundle.probe_path),
                "probe_sha256": (
                    _file_sha256(bundle.probe_path) if bundle.probe_path.is_file() else None
                ),
                "framing_thresholds": bundle.threshold_identity(),
            }
            for bundle in bundles
        ],
    }


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

    def verdict(
        self, key: str, request: ProbeQaRequest, probe: RoleProbe,
        thresholds: FramingThresholds | None = None,
    ) -> ProbeQaVerdict | None:
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
        expected_complies, expected_issue = _decision(request, probe, reasoning, thresholds)
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
    request: ProbeQaRequest, probe: RoleProbe, reasoning_probability: float,
    thresholds: FramingThresholds | None = None,
) -> tuple[bool | None, str | None]:
    expectation = probe_expectation(request.spec)
    if expectation is ProbeExpectation.DESCRIPTIVE:
        return None, None
    assert probe.qualification is not None
    threshold = probe.qualification.calibration.threshold
    if threshold is None:
        raise ValueError("QA-eligible role probe has no calibration threshold")
    if thresholds is not None:
        threshold = (
            thresholds.reasoning_minimum
            if expectation is ProbeExpectation.REASONING
            else thresholds.nonreasoning_maximum
        )
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
            with _pinned_nemotron_cumsum(
                model, prepared.bundle.adapter, self.execution_device
            ):
                pass
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
    def _target_prefix(context: RenderedProbeContext) -> RenderedProbeContext:
        """Keep context through the scored span, omitting only later suffix tokens."""
        if len(context.token_positions) != 1:
            raise ValueError("collector probe context must contain exactly one measured span")
        positions = context.token_positions[0]
        last_position = positions[-1]
        return RenderedProbeContext(
            context.input_ids[: last_position + 1],
            (positions,),
            (context.content_token_ids[0],),
            context.render_config_sha256,
            context.masked_boundary_tokens,
        )

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
                "capture_policy": CAPTURE_POLICY,
                **(
                    {"framing_thresholds": prepared.bundle.threshold_identity()}
                    if prepared.bundle.framing_thresholds else {}
                ),
                "probe_sha256": prepared.probe_sha256,
                "runtime_sha256": prepared.probe.provenance.runtime_sha256,
                "weights_sha256": prepared.probe.provenance.weights_sha256,
                "adapter": prepared.bundle.adapter.name,
                "assistant": str(request.setup.matchup.assistant),
                "study_id": request.study_id,
                "permutation": request.permutation,
                "position": request.position,
                "render_config_sha256": render_config_sha256,
                "prefix_input_ids": input_ids,
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
        rendered: dict[ProbeQaRequest, RenderedProbeContext] = {}
        keys: dict[ProbeQaRequest, str] = {}
        missing: list[ProbeQaRequest] = []
        for request in requests:
            context = render_collector_probe_context(
                tokenizer,
                prepared.bundle.adapter,
                request.setup,
                request.position,
            )
            prefix = self._target_prefix(context)
            rendered[request] = prefix
            key = self._context_key(
                prepared,
                request,
                prefix.input_ids,
                prefix.token_positions[0],
                prefix.content_token_ids[0],
                prefix.render_config_sha256,
            )
            keys[request] = key
            cached = self.cache.verdict(
                key, request, prepared.probe, prepared.bundle.thresholds_for(request.spec.channel)
            )
            if cached is None:
                missing.append(request)
            else:
                verdicts[request] = cached
        if not missing:
            return

        model = self._load_validated_model(prepared)
        try:
            with _pinned_nemotron_cumsum(
                model, prepared.bundle.adapter, self.execution_device
            ):
                for request in missing:
                    context = rendered[request]
                    positions = context.token_positions[0]
                    captured = capture_token_activations(
                        model,
                        prepared.bundle.adapter,
                        input_ids=context.input_ids,
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
                    activations = np.asarray(captured[:, 0])
                    projection = project_role(
                        prepared.probe,
                        activations,
                        provenance=prepared.probe.provenance,
                        layer_index=prepared.probe.selected_layer_index,
                    )
                    complies, issue = _decision(
                        request, prepared.probe, projection.mean_reasoning_probability,
                        prepared.bundle.thresholds_for(request.spec.channel),
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
