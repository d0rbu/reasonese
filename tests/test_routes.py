"""Route policy, narrow fingerprint equivalence, and cache provenance contracts."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from reasonese.axes import Assistant, Author, Framing
from reasonese.cache import YamlMessageCache, YamlTraceCache, trace_from_dict, trace_to_dict
from reasonese.collect_data import collect_study
from reasonese.conversation import (
    ConversationTrace,
    GeneratedMessage,
    GeneratedText,
    ToolStep,
    construct_conversation,
)
from reasonese.judging import fingerprint_traces, trace_fingerprint
from reasonese.openrouter import (
    CompletionTransport,
    ModelRoute,
    OpenRouterClient,
    OpenRouterModelId,
    RoutePreference,
    RouteProvenance,
    canonical_model_id,
    completion_provenance,
    model_route,
    provenance_from_dict,
    select_route,
)
from reasonese.routing import CollectionRouting
from reasonese.runner import materialize_specs
from reasonese.study import build_trials
from reasonese.study_cache import SqliteStudyCache
from tests.test_judging import _trace
from tests.test_openrouter import FakeTransport, _batch_result, _chat
from tests.test_study_orchestration import (
    _assistant_responses,
    _judge_batch,
    _manual_library,
    _message_qa_batch,
    _study,
)


@pytest.mark.parametrize("assistant", tuple(Assistant))
def test_complete_registry_and_route_matrix(assistant: Assistant) -> None:
    author = Author(assistant.value)
    registered = model_route(assistant)
    assert registered == model_route(author)
    free = assistant in (
        Assistant.INKLING, Assistant.INKLING_SMALL,
        Assistant.GEMMA_4_31B_IT, Assistant.NEMOTRON_3_5_LIGHTNING,
    )
    assert (registered.free_model_id is not None) is free
    for preference in RoutePreference:
        route = select_route(assistant, preference)
        assert select_route(author, preference) == route
        expected = (
            registered.free_model_id
            if preference is RoutePreference.FREE and free
            else registered.model_id
        )
        assert route.model_id == expected
        assert route.batch_model_id == (
            registered.batch_model_id if preference is RoutePreference.BATCH else None
        )
        for prefer_batch in (False, True):
            for body in ({"messages": []}, {"tools": [{"type": "openrouter:web_search"}]}):
                result = completion_provenance(route, (body,), prefer_batch=prefer_batch)
                batch = (
                    preference is RoutePreference.BATCH
                    and prefer_batch
                    and registered.batch_model_id is not None
                    and "tools" not in body
                )
                assert result.transport is (
                    CompletionTransport.BATCH if batch else CompletionTransport.SYNC
                )
                assert result.requested_model_id == expected
    for slug in (registered.free_model_id, registered.batch_model_id):
        if slug is not None:
            assert canonical_model_id(slug) == registered.model_id


def test_route_boundaries() -> None:
    with pytest.raises(ValueError, match="user author"):
        select_route(Author.USER, RoutePreference.FREE)
    with pytest.raises(BeartypeCallHintParamViolation):
        CollectionRouting(preference=cast(RoutePreference, "free"))
    assert ModelRoute(OpenRouterModelId.parse("example/model"), None).free_model_id is None


@pytest.mark.parametrize(
    ("slug", "expected"),
    [
        ("thinkingmachines/inkling:free", "thinkingmachines/inkling"),
        ("thinkingmachines/inkling:batch", "thinkingmachines/inkling"),
        ("thinkingmachines/inkling-small:free", "thinkingmachines/inkling-small"),
        ("x/y", "x/y"),
        ("x/y:online", "x/y:online"),
        ("x/y:free:online", "x/y:free:online"),
        ("x/y:free:batch", "x/y:free"),
        ("x/y:FREE", "x/y:FREE"),
        ("x/:free/y", "x/:free/y"),
    ],
)
def test_suffix_stripping_is_only_one_terminal_suffix(slug: str, expected: str) -> None:
    assert canonical_model_id(slug) == expected


@pytest.mark.parametrize("suffix", ("", ":free", ":batch"))
@pytest.mark.parametrize("with_steps", (False, True))
def test_fingerprints_normalize_only_response_model(suffix: str, with_steps: bool) -> None:
    raw = {
        **_trace().response,
        "model": "thinkingmachines/inkling",
        "nested": {"model": "x/y:free"},
        "reasoning": "雪",
    }
    paid = replace(
        _trace(), response=raw, tool_steps=(ToolStep(copy.deepcopy(raw), ()),) if with_steps else ()
    )
    variant = replace(
        paid,
        response={**raw, "model": raw["model"] + suffix},
        tool_steps=tuple(
            replace(step, response={**step.response, "model": raw["model"] + suffix})
            for step in paid.tool_steps
        ),
        provenance=RouteProvenance(
            OpenRouterModelId.parse(raw["model"] + suffix), CompletionTransport.SYNC
        ),
    )
    before = copy.deepcopy(variant)
    expected = trace_fingerprint(paid)
    assert trace_fingerprint(variant) == expected
    assert [item.fingerprint for item in fingerprint_traces((paid, variant, paid))] == [
        expected
    ] * 3
    assert variant == before
    for field, value in (
        ("id", "another id"),
        ("usage", {"tokens": 7}),
        ("reasoning", "changed"),
        ("nested", {"model": "x/y"}),
        ("model", "thinkingmachines/inkling-small"),
    ):
        different = replace(variant, response={**variant.response, field: value})
        assert trace_fingerprint(different) != expected
        assert fingerprint_traces((different,))[0].fingerprint == trace_fingerprint(different)


@pytest.mark.parametrize("model", (None, 12, {"nested": "x/y:free"}))
def test_legacy_non_string_model_metadata_is_preserved(model: object) -> None:
    trace = replace(_trace(), response={**_trace().response, "model": model})
    assert fingerprint_traces((trace,))[0].fingerprint == trace_fingerprint(trace)
    assert trace_from_dict(trace_to_dict(trace)) == trace


@pytest.mark.parametrize(
    "raw",
    (
        {},
        {"extra": 1},
        [],
        {"requested_model_id": "x/y", "transport": "invalid"},
        {"requested_model_id": "bad", "transport": "sync"},
    ),
)
def test_invalid_provenance_fails_closed(raw: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        provenance_from_dict(raw)


def test_raw_provenance_round_trips_both_cache_backends(tmp_path: Path) -> None:
    study = _study()
    trial = build_trials(study)[0]
    messages = tuple(
        GeneratedMessage(spec, GeneratedText.parse(str(spec.instruction)), None)
        for spec in trial.matchup.inputs
    )
    route = RouteProvenance(
        OpenRouterModelId.parse("thinkingmachines/inkling:free"), CompletionTransport.SYNC
    )
    raw = {"model": "thinkingmachines/inkling:free", **_chat("result")}
    trace = ConversationTrace(
        construct_conversation(trial.matchup, messages), raw, (ToolStep(raw, ()),), route
    )
    yaml = YamlTraceCache(tmp_path / "trace.yaml")
    sql = SqliteStudyCache(tmp_path / "collection.sqlite3")
    yaml.put(trace)
    sql.put_traces(((trial.trial_id, trace),))
    assert yaml.get(trial.matchup) == trace == sql.load_traces((trial,))[trial.trial_id]
    loaded = yaml.get(trial.matchup)
    assert loaded is not None and loaded.response == raw
    message = replace(messages[0], response=raw, provenance=route)
    cache = YamlMessageCache(tmp_path / "messages.yaml")
    cache.put_many((message,))
    assert cache.get(message.spec) == message
    legacy = replace(trace, provenance=None)
    assert "provenance" not in trace_to_dict(legacy)
    assert trace_from_dict(trace_to_dict(legacy)).provenance is None


@pytest.mark.parametrize("preference", tuple(RoutePreference))
def test_cold_collection_rejects_before_any_provider_call(
    tmp_path: Path, preference: RoutePreference
) -> None:
    study = _study()
    transport = FakeTransport()
    with pytest.raises(ValueError, match="--allow-paid"):
        collect_study(
            study,
            tmp_path / "out",
            OpenRouterClient(transport),
            _manual_library(tmp_path, study),
            prefer_batch=True,
            routing=CollectionRouting(preference),
        )
    assert transport.post_calls == []


def test_free_collection_preserves_route_on_cache_only_paid_preference(tmp_path: Path) -> None:
    study = _study()
    manual = _manual_library(tmp_path, study)
    transport = FakeTransport(
        posts=[
            _message_qa_batch(2),
            *_assistant_responses(2),
            _judge_batch((True, False, False, True)),
        ]
    )
    cold_routing = CollectionRouting(allow_paid=True)
    cold = collect_study(
        study,
        tmp_path / "out",
        OpenRouterClient(transport),
        manual,
        prefer_batch=True,
        routing=cold_routing,
    )
    assert transport.post_calls[1][1]["model"] == "thinkingmachines/inkling:free"
    warm_routing = CollectionRouting(RoutePreference.PAID)
    warm = collect_study(
        study, tmp_path / "out", None, manual, prefer_batch=True, routing=warm_routing
    )
    assert warm.observations == cold.observations
    assistant = [row for row in warm_routing.summary() if row["stage"] == "assistant"]
    assert assistant == [
        {
            "stage": "assistant",
            "model": "Inkling",
            "source": "cache",
            "requested_model_id": "thinkingmachines/inkling:free",
            "transport": "sync",
            "response_model": None,
            "count": 2,
        }
    ]
    assert len(transport.post_calls) == 4
    # A warm trace with missing QA is still paid work, and must not contact the provider.
    (tmp_path / "out" / "message_qa.yaml").unlink()
    denied = FakeTransport()
    with pytest.raises(ValueError, match="--allow-paid.*message-QA"):
        collect_study(study, tmp_path / "out", OpenRouterClient(denied), manual, prefer_batch=True)
    assert denied.post_calls == []


@pytest.mark.parametrize(
    ("author", "slug"),
    [
        (Author.GEMMA_4_31B_IT, "google/gemma-4-31b-it"),
        (Author.NEMOTRON_3_5_LIGHTNING, "nvidia/nemotron-3.5-lightning"),
    ],
)
@pytest.mark.parametrize("preference", tuple(RoutePreference))
def test_authoring_provenance_tracks_actual_endpoint(
    tmp_path: Path, preference: RoutePreference, author: Author, slug: str
) -> None:
    study = _study()
    spec = replace(study.inputs[0], author=author)
    batch = preference is RoutePreference.BATCH and author is Author.GEMMA_4_31B_IT
    raw = _chat("authored")
    transport = FakeTransport(
        posts=[
            {"id": "b", "status": "completed", "results": [_batch_result("request-0", "authored")]}
        ]
        if batch
        else [raw]
    )
    routing = CollectionRouting(preference, True)
    cache = YamlMessageCache(tmp_path / "messages.yaml")
    manual = _manual_library(tmp_path, study)
    messages = materialize_specs(
        (spec, spec), OpenRouterClient(transport), cache, manual, prefer_batch=True, routing=routing
    )
    route = messages[0].provenance
    assert route is not None
    assert route.transport is (CompletionTransport.BATCH if batch else CompletionTransport.SYNC)
    assert route.requested_model_id == (
        slug + ":free" if preference is RoutePreference.FREE else slug
    )
    assert transport.post_calls[0][1]["model"] == route.requested_model_id
    assert messages[0].response == raw
    assert messages[0] == messages[1] == cache.get(spec)
    # Different preference reuses the exact original author response and provenance.
    again = materialize_specs(
        (spec,), OpenRouterClient(FakeTransport()), cache, manual, prefer_batch=False
    )
    assert again == messages[:1]


def test_exact_legacy_fingerprint_and_filtered_planner_bytes() -> None:
    """Golden digests captured from main 2f8c634, independent of the new helpers."""
    import hashlib
    import json

    from reasonese.instructions import load_instruction_pairs
    from reasonese.matchup import prompt_spec_to_dict
    from reasonese.planning import build_pair_specs

    assert (
        trace_fingerprint(_trace())
        == "701e4aa97d671c34d63690be30fb6d3506bec471a0afb20285dd25f29daa75c0"
    )
    specs = [
        prompt_spec_to_dict(spec)
        for pair in build_pair_specs(load_instruction_pairs(Path("configs/instruction_pairs.yaml")))
        for spec in pair.first + pair.second
        if spec.author not in (Author.GEMMA_4_31B_IT, Author.NEMOTRON_3_5_LIGHTNING)
        and spec.framing not in (Framing.COMPRESSED_NORMAL, Framing.COMPRESSED_PERSUASIVE)
    ]
    assert (
        hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()
        == "44506ce15129a9d6f5159b9feef4ef0bd8f5e2c8ab14a2fd3a8092f5fc4d6221"
    )


@pytest.mark.parametrize("command", ("conversation", "study", "suite"))
@pytest.mark.parametrize("preference", tuple(RoutePreference))
def test_collection_clis_refuse_cold_work_even_with_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, preference: RoutePreference
) -> None:
    from reasonese.collect_data import main as collect_data_cli
    from reasonese.collect_studies import main as collect_studies_cli
    from reasonese.io import write_study_suite
    from reasonese.run_conversation import main as conversation_cli
    from tests.test_study_orchestration import _write_study

    study = _study()
    manual = _manual_library(tmp_path, study)
    transport = FakeTransport()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    common = ["--route", str(preference), "--user-messages", str(manual.root)]
    if command == "conversation":
        import yaml

        from reasonese.matchup import matchup_to_dict

        config = tmp_path / "matchup.yaml"
        config.write_text(yaml.safe_dump(matchup_to_dict(build_trials(study)[0].matchup)))
        fn = conversation_cli
        module = "reasonese.run_conversation"
        args = [
            "--matchup",
            str(config),
            "--message-cache",
            str(tmp_path / "messages.yaml"),
            "--trace-cache",
            str(tmp_path / "traces.yaml"),
            "--message-qa-cache",
            str(tmp_path / "qa.yaml"),
        ]
    elif command == "study":
        config = tmp_path / "study.yaml"
        _write_study(config, study)
        fn = collect_data_cli
        module = "reasonese.collect_data"
        args = ["--study", str(config), "--output", str(tmp_path / "out")]
    else:
        config = tmp_path / "suite.yaml"
        write_study_suite(config, (study,))
        fn = collect_studies_cli
        module = "reasonese.collect_studies"
        args = ["--suite", str(config), "--output", str(tmp_path / "out")]
    monkeypatch.setattr(f"{module}.RequestsTransport", lambda key: transport)
    with pytest.raises(SystemExit, match="2"):
        fn(args + common)
    assert not transport.post_calls
    if preference is RoutePreference.BATCH:
        with pytest.raises(SystemExit, match="2"):
            fn(args + common + ["--allow-paid", "--no-batch"])
        assert not transport.post_calls


def test_suffix_legacy_judgments_miss_without_recollecting_traces(tmp_path: Path) -> None:
    import hashlib
    import json

    from reasonese.judging import TraceFingerprint

    study = _study()
    manual = _manual_library(tmp_path, study)
    assistant_responses = [
        {**response, "model": "thinkingmachines/inkling:free"}
        for response in _assistant_responses(2)
    ]
    transport = FakeTransport(
        posts=[_message_qa_batch(2), *assistant_responses, _judge_batch((True, False, False, True))]
    )
    output = tmp_path / "out"
    cold = collect_study(
        study,
        output,
        OpenRouterClient(transport),
        manual,
        prefer_batch=True,
        routing=CollectionRouting(allow_paid=True),
    )
    cache = SqliteStudyCache(output / "collection.sqlite3")
    traces = cache.load_traces(cold.trials)
    judgments = cache.load_judgments(cold.trials)
    legacy = []
    for trial in cold.trials:
        raw = trace_to_dict(traces[trial.trial_id])
        raw.pop("provenance")
        fingerprint = hashlib.sha256(
            json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        assert fingerprint != trace_fingerprint(traces[trial.trial_id])
        legacy.append(
            (
                trial.trial_id,
                replace(
                    judgments[trial.trial_id], trace_fingerprint=TraceFingerprint.parse(fingerprint)
                ),
            )
        )
    cache.put_judgments(tuple(legacy))
    denied = FakeTransport()
    with pytest.raises(ValueError, match="--allow-paid.*judgments"):
        collect_study(study, output, OpenRouterClient(denied), manual, prefer_batch=True)
    assert denied.post_calls == []
    assert cache.load_traces(cold.trials) == traces
    allowed = FakeTransport(posts=[_judge_batch((True, False, False, True))])
    resumed = collect_study(
        study,
        output,
        OpenRouterClient(allowed),
        manual,
        prefer_batch=True,
        routing=CollectionRouting(allow_paid=True),
    )
    assert resumed.trace_cache_hits == 2 and resumed.judgment_cache_hits == 0
    assert resumed.observations == cold.observations
    assert len(allowed.post_calls) == 1 and allowed.post_calls[0][0] == "/api/beta/batches"


def test_mixed_author_group_denial_happens_before_free_submission(tmp_path: Path) -> None:
    study = _study()
    specs = (
        replace(study.inputs[0], author=Author.INKLING),
        replace(study.inputs[1], author=Author.QWEN3_8_FLASH),
    )
    transport = FakeTransport()
    with pytest.raises(ValueError, match="--allow-paid.*authoring"):
        materialize_specs(
            specs,
            OpenRouterClient(transport),
            YamlMessageCache(tmp_path / "messages.yaml"),
            _manual_library(tmp_path, study),
            prefer_batch=True,
        )
    assert transport.post_calls == []


def test_failed_free_author_request_never_retries_paid(tmp_path: Path) -> None:
    class FailedTransport(FakeTransport):
        def post_json(self, path: str, body: dict) -> dict:
            self.post_calls.append((path, body))
            raise RuntimeError("free route unavailable")

    study = _study()
    spec = replace(study.inputs[0], author=Author.GEMMA_4_31B_IT)
    transport = FailedTransport()
    with pytest.raises(RuntimeError, match="free route unavailable"):
        materialize_specs(
            (spec,),
            OpenRouterClient(transport),
            YamlMessageCache(tmp_path / "messages.yaml"),
            _manual_library(tmp_path, study),
            prefer_batch=True,
            routing=CollectionRouting(allow_paid=True),
        )
    assert len(transport.post_calls) == 1
    assert transport.post_calls[0][1]["model"] == "google/gemma-4-31b-it:free"


def test_author_provenance_is_resolved_once_per_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reasonese.runner as runner

    study = _study()
    specs = tuple(replace(spec, author=Author.GEMMA_4_31B_IT) for spec in study.inputs)
    calls = []
    original = runner.completion_provenance

    def counted(
        route: ModelRoute, bodies: tuple[dict, ...], *, prefer_batch: bool
    ) -> RouteProvenance:
        calls.append(len(bodies))
        return original(route, bodies, prefer_batch=prefer_batch)

    monkeypatch.setattr(runner, "completion_provenance", counted)
    messages = materialize_specs(
        specs,
        OpenRouterClient(FakeTransport(posts=[_chat("one"), _chat("two")]), sync_workers=1),
        YamlMessageCache(tmp_path / "messages.yaml"),
        _manual_library(tmp_path, study),
        prefer_batch=True,
    )
    assert len(messages) == 2
    assert calls == [2]
    assert messages[0].provenance is messages[1].provenance
