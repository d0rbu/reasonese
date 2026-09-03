from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import yaml
from beartype.roar import BeartypeCallHintParamViolation

from reasonese.axes import Assistant, Author, Channel, Framing, Instruction
from reasonese.cache import YamlMessageCache
from reasonese.collect_data import CollectionResult, CollectionTask, collect_studies, collect_study
from reasonese.collect_data import main as collect_data
from reasonese.collect_studies import collection_tasks
from reasonese.collect_studies import main as collect_studies_cli
from reasonese.config import load_study
from reasonese.conversation import (
    ConversationSetup,
    ConversationTrace,
    GeneratedMessage,
    GeneratedText,
    construct_conversation,
)
from reasonese.judging import (
    FingerprintedTrace,
    InstructionVerdict,
    InstructionVerdicts,
    Judgment,
    judge_traces,
    trace_fingerprint,
)
from reasonese.manual_messages import ManualMessageLibrary, ManualMessageSnapshot
from reasonese.matchup import Matchup
from reasonese.message_qa import MessageQaVerdict
from reasonese.message_qa_cache import YamlMessageQaCache
from reasonese.observations import (
    cell_id,
    observation_to_dict,
    observations_from_trial,
    observations_from_trials,
    write_observations,
)
from reasonese.openrouter import JsonObject, OpenRouterClient
from reasonese.planning import PromptSpec
from reasonese.study import (
    Cell,
    PositiveInteger,
    Study,
    StudyInputs,
    build_trials,
    make_study,
    observations_per_cell,
    observations_per_cell_position,
    study_cells,
    study_fingerprint,
    study_from_dict,
    study_to_dict,
    trial_count,
)
from reasonese.study_cache import SqliteStudyCache


def _spec(text: str, channel: Channel, author: Author = Author.USER) -> PromptSpec:
    return PromptSpec(Instruction.parse(text), Framing.NORMAL, channel, author)


def _study(rollouts: int = 1, assistant: Assistant = Assistant.INKLING) -> Study:
    return make_study(
        (
            _spec("Name the capital of France.", Channel.README),
            _spec("What is two plus two?", Channel.USER),
        ),
        assistant,
        rollouts,
    )


def _manual_library(tmp_path: Path, *studies: Study) -> ManualMessageLibrary:
    root = tmp_path / "manual"
    instructions = tuple(
        dict.fromkeys(
            spec.instruction
            for study in studies
            for spec in study.inputs
            if spec.author is Author.USER
        )
    )
    for index, instruction in enumerate(instructions):
        directory = root / f"instruction-{index}"
        directory.mkdir(parents=True)
        (directory / "instruction.txt").write_text(str(instruction))
        for framing in Framing:
            (directory / f"{framing}.txt").write_text(str(instruction))
    return ManualMessageLibrary(root)


def _chat(content: str, response_id: str) -> JsonObject:
    return {
        "id": response_id,
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }


def _batch_result(custom_id: str, response: JsonObject) -> JsonObject:
    return {
        "custom_id": custom_id,
        "response": {"status_code": 200, "body": response},
        "error": None,
    }


def _assistant_responses(count: int) -> list[JsonObject]:
    return [_chat(f"answer {index}", f"assistant-{index}") for index in range(count)]


def _tool_chat() -> JsonObject:
    return {
        "id": "tool-response",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "live-read",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"README.md"}',
                            },
                        }
                    ],
                }
            }
        ],
    }


def _judge_batch(values: tuple[bool, ...]) -> JsonObject:
    return {
        "id": "judge-batch",
        "status": "completed",
        "results": [
            _batch_result(
                f"request-{index}",
                _chat(json.dumps({"completed": value}), f"judge-{index}"),
            )
            for index, value in enumerate(values)
        ],
    }


def _message_qa_batch(count: int) -> JsonObject:
    return {
        "id": "message-qa-batch",
        "status": "completed",
        "results": [
            _batch_result(
                f"request-{index}",
                _chat(json.dumps({"complies": True, "issues": []}), f"message-qa-{index}"),
            )
            for index in range(count)
        ],
    }


class FakeTransport:
    def __init__(
        self,
        posts: list[JsonObject],
        gets: list[JsonObject] | None = None,
    ) -> None:
        self.posts = posts
        self.gets = gets or []
        self.post_calls: list[tuple[str, JsonObject]] = []
        self.calls: list[tuple[str, str]] = []

    def post_json(self, path: str, body: JsonObject) -> JsonObject:
        self.post_calls.append((path, body))
        self.calls.append(("POST", path))
        return self.posts.pop(0)

    def get_json(self, path: str) -> JsonObject:
        self.calls.append(("GET", path))
        if not self.gets:
            raise AssertionError(f"unexpected GET {path}")
        return self.gets.pop(0)


def test_two_input_study_has_two_permutations_and_two_scores_per_cell() -> None:
    study = _study()
    trials = build_trials(study)

    assert isinstance(study.inputs, StudyInputs)
    assert isinstance(study.rollouts_per_permutation, PositiveInteger)
    assert len(trials) == 2
    assert trial_count(study) == 2
    assert observations_per_cell(study) == 2
    assert observations_per_cell_position(study) == 1
    first, second = study.inputs
    assert {trial.matchup.inputs for trial in trials} == {(first, second), (second, first)}
    assert len({trial.trial_id for trial in trials}) == 2


def test_trials_reuse_one_validated_matchup_per_permutation() -> None:
    trials = build_trials(_study(2))

    assert trials[0].matchup is trials[1].matchup
    assert trials[2].matchup is trials[3].matchup
    assert trials[0].matchup is not trials[2].matchup


def test_study_cells_pair_each_input_with_the_assistant() -> None:
    study = _study()
    assert study_cells(study) == tuple(Cell(spec, study.assistant) for spec in study.inputs)
    assert cell_id(study_cells(study)[0]) == cell_id(study_cells(study)[0])
    assert cell_id(Cell(study.inputs[0], Assistant.INKLING)) != cell_id(
        Cell(study.inputs[0], Assistant.INKLING_SMALL)
    )


@pytest.mark.parametrize(
    ("inputs", "rollouts", "error"),
    [
        ((_spec("Only", Channel.USER),), 1, "exactly two"),
        (
            (
                _spec("A", Channel.SYSTEM),
                _spec("B", Channel.USER),
                _spec("C", Channel.USER),
            ),
            1,
            "exactly two",
        ),
        (
            (_spec("Same", Channel.USER), _spec("Same", Channel.USER)),
            1,
            "distinct",
        ),
        (
            (_spec("A", Channel.SYSTEM), _spec("B", Channel.README)),
            1,
            "user message",
        ),
        (
            (_spec("A", Channel.SYSTEM), _spec("B", Channel.USER)),
            0,
            "at least one",
        ),
    ],
)
def test_invalid_studies_are_rejected(
    inputs: tuple[PromptSpec, ...], rollouts: int, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        make_study(inputs, Assistant.INKLING, rollouts)


def test_study_yaml_round_trip_and_example() -> None:
    study = _study(2)
    assert study_from_dict(study_to_dict(study)) == study
    example = load_study(Path("configs/example_study.yaml"))
    assert len(example.inputs) == 2
    assert example.rollouts_per_permutation == 1


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ([], "study must be a mapping"),
        ({"assistant": "Inkling"}, "study fields"),
        (
            {"assistant": "Inkling", "rollouts_per_permutation": 1, "inputs": "bad"},
            "inputs must be a list",
        ),
        (
            {"assistant": "Inkling", "rollouts_per_permutation": True, "inputs": []},
            "must be an integer",
        ),
    ],
)
def test_study_yaml_validation(raw: object, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        study_from_dict(raw)


def test_study_fingerprint_is_stable_and_sensitive_to_sampling() -> None:
    assert study_fingerprint(_study()) == study_fingerprint(_study())
    assert study_fingerprint(_study()) != study_fingerprint(_study(2))


def _trace_for_trial(study: Study, trial_index: int, content: str = "answer") -> ConversationTrace:
    trial = build_trials(study)[trial_index]
    generated = tuple(
        GeneratedMessage(spec, GeneratedText.parse(str(spec.instruction)), None)
        for spec in trial.matchup.inputs
    )
    from reasonese.conversation import construct_conversation

    return ConversationTrace(
        construct_conversation(trial.matchup, generated),
        _chat(content, f"assistant-{trial_index}"),
    )


def _judgment_for_trace(trace: ConversationTrace, values: tuple[bool, ...]) -> Judgment:
    verdicts = InstructionVerdicts.parse(
        tuple(
            InstructionVerdict(spec, value, _chat(json.dumps({"completed": value}), str(index)))
            for index, (spec, value) in enumerate(
                zip(trace.setup.matchup.inputs, values, strict=True)
            )
        )
    )
    return Judgment(trace.setup.matchup, trace_fingerprint(trace), verdicts)


def test_sqlite_study_cache_batches_round_trips_and_replaces(tmp_path: Path) -> None:
    study = _study()
    trials = build_trials(study)
    traces = tuple(_trace_for_trial(study, index) for index in range(len(trials)))
    cache = SqliteStudyCache(tmp_path / "nested" / "collection.sqlite3")

    cache.put_traces(
        tuple(
            (trial.trial_id, trace)
            for trial, trace in zip(trials, traces, strict=True)
        )
    )
    assert cache.load_traces(trials) == {
        trial.trial_id: trace for trial, trace in zip(trials, traces, strict=True)
    }

    replacement = _trace_for_trial(study, 0, "replacement")
    cache.put_traces(((trials[0].trial_id, replacement),))
    judgments = (
        _judgment_for_trace(replacement, (True, False)),
        _judgment_for_trace(traces[1], (False, True)),
    )
    cache.put_judgments(
        tuple((trial.trial_id, judgment) for trial, judgment in zip(trials, judgments, strict=True))
    )

    assert cache.load_traces(trials)[trials[0].trial_id] == replacement
    assert cache.load_judgments(trials) == {
        trial.trial_id: judgment for trial, judgment in zip(trials, judgments, strict=True)
    }
    cache.put_traces(())
    cache.put_judgments(())


def test_sqlite_trace_cache_reuses_setups_across_rollouts(tmp_path: Path) -> None:
    study = _study(2)
    trials = build_trials(study)
    traces = tuple(_trace_for_trial(study, index) for index in range(len(trials)))
    cache = SqliteStudyCache(tmp_path / "collection.sqlite3")
    cache.put_traces(tuple(zip((trial.trial_id for trial in trials), traces, strict=True)))

    loaded = cache.load_traces(trials)

    assert loaded[trials[0].trial_id].setup is loaded[trials[1].trial_id].setup
    assert loaded[trials[2].trial_id].setup is loaded[trials[3].trial_id].setup
    assert loaded[trials[0].trial_id].setup is not loaded[trials[2].trial_id].setup


@pytest.mark.parametrize(
    ("table", "payload", "error"),
    [
        ("traces", "not JSON", "not valid JSON"),
        ("judgments", sqlite3.Binary(b"{}"), "payload must be text"),
    ],
)
def test_sqlite_study_cache_rejects_corrupt_payloads(
    tmp_path: Path,
    table: str,
    payload: object,
    error: str,
) -> None:
    trials = build_trials(_study())
    cache = SqliteStudyCache(tmp_path / "collection.sqlite3")
    cache.load_traces(trials)
    with sqlite3.connect(cache.path) as connection:
        connection.execute(
            f"INSERT INTO {table} (trial_id, payload) VALUES (?, ?)",
            (str(trials[0].trial_id), payload),
        )

    with pytest.raises(ValueError, match=error):
        if table == "traces":
            cache.load_traces(trials)
        else:
            cache.load_judgments(trials)


def test_sqlite_study_cache_rejects_records_for_the_wrong_trial(tmp_path: Path) -> None:
    study = _study()
    trials = build_trials(study)
    traces = tuple(_trace_for_trial(study, index) for index in range(len(trials)))
    cache = SqliteStudyCache(tmp_path / "collection.sqlite3")

    cache.put_traces(((trials[0].trial_id, traces[1]),))
    with pytest.raises(ValueError, match="trace matchup does not match"):
        cache.load_traces(trials)

    cache.put_judgments(
        ((trials[0].trial_id, _judgment_for_trace(traces[1], (True, False))),)
    )
    with pytest.raises(ValueError, match="judgment matchup does not match"):
        cache.load_judgments(trials)


def test_judge_traces_flattens_multiple_conversations_into_one_batch() -> None:
    study = _study()
    traces = (_trace_for_trial(study, 0), _trace_for_trial(study, 1))
    transport = FakeTransport([_judge_batch((True, False, False, True))])

    judgments = judge_traces(traces, OpenRouterClient(transport))

    assert [[item.completed for item in judgment.verdicts] for judgment in judgments] == [
        [True, False],
        [False, True],
    ]
    assert len(transport.post_calls) == 1
    assert len(transport.post_calls[0][1]["requests"]) == 4
    assert judge_traces((), OpenRouterClient(FakeTransport([]))) == ()


def test_observations_join_trial_trace_and_judgment_in_position_order() -> None:
    study = _study()
    trial = build_trials(study)[0]
    trace = _trace_for_trial(study, 0)
    judgment = _judgment_for_trace(trace, (True, False))

    observations = observations_from_trial(trial, trace, judgment)
    row = observation_to_dict(observations[0])

    assert [observation.position for observation in observations] == [1, 2]
    assert [observation.completed for observation in observations] == [True, False]
    assert row["instruction"] == "Name the capital of France."
    assert row["assistant"] == "Inkling"
    assert row["assistant_response_id"] == "assistant-0"
    assert row["judge_response_id"] == "0"


def test_batch_observations_match_individual_conversion_and_validate_lengths() -> None:
    study = _study()
    trials = build_trials(study)
    traces = tuple(_trace_for_trial(study, index) for index in range(len(trials)))
    fingerprinted = tuple(FingerprintedTrace(trace) for trace in traces)
    judgments = tuple(
        _judgment_for_trace(trace, values)
        for trace, values in zip(traces, ((True, False), (False, True)), strict=True)
    )

    batched = observations_from_trials(trials, fingerprinted, judgments)
    individual = tuple(
        observation
        for trial, trace, judgment in zip(trials, traces, judgments, strict=True)
        for observation in observations_from_trial(trial, trace, judgment)
    )

    assert batched == individual
    assert [item.fingerprint for item in fingerprinted] == [
        trace_fingerprint(trace) for trace in traces
    ]
    with pytest.raises(ValueError, match="equal lengths"):
        observations_from_trials(trials, fingerprinted, judgments[:1])

    with pytest.raises(BeartypeCallHintParamViolation):
        replace(batched[0], completed=cast(bool, 1))


def test_observations_preserve_missing_provider_ids() -> None:
    study = _study()
    trial = build_trials(study)[0]
    trace = _trace_for_trial(study, 0)
    trace = ConversationTrace(trace.setup, {"choices": trace.response["choices"]})
    judgment = _judgment_for_trace(trace, (True, True))
    verdict = judgment.verdicts[0]
    verdicts = InstructionVerdicts.parse(
        (
            InstructionVerdict(
                verdict.spec, verdict.completed, {"choices": verdict.response["choices"]}
            ),
            judgment.verdicts[1],
        )
    )
    judgment = Judgment(judgment.matchup, judgment.trace_fingerprint, verdicts)

    observations = observations_from_trial(trial, trace, judgment)
    assert observations[0].assistant_response_id is None
    assert observations[0].judge_response_id is None


def test_observations_reject_mismatched_trace_judgment_or_fingerprint() -> None:
    study = _study()
    trials = build_trials(study)
    first_trace = _trace_for_trial(study, 0)
    second_trace = _trace_for_trial(study, 1)
    first_judgment = _judgment_for_trace(first_trace, (True, False))
    second_judgment = _judgment_for_trace(second_trace, (False, True))

    with pytest.raises(ValueError, match="trace matchup"):
        observations_from_trial(trials[0], second_trace, second_judgment)
    with pytest.raises(ValueError, match="judgment matchup"):
        observations_from_trial(trials[0], first_trace, second_judgment)
    bad_fingerprint = Judgment(
        first_judgment.matchup,
        trace_fingerprint(ConversationTrace(first_trace.setup, _chat("different", "other"))),
        first_judgment.verdicts,
    )
    with pytest.raises(ValueError, match="fingerprint"):
        observations_from_trial(trials[0], first_trace, bad_fingerprint)


def test_write_observations_emits_jsonl(tmp_path: Path) -> None:
    study = _study()
    trace = _trace_for_trial(study, 0)
    observations = observations_from_trial(
        build_trials(study)[0], trace, _judgment_for_trace(trace, (True, False))
    )
    path = tmp_path / "nested" / "observations.jsonl"
    write_observations(path, observations)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["completed"] for row in rows] == [True, False]


def test_collect_study_batches_trials_and_judgments_then_resumes_without_a_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _study(2)
    constructed_setups = []
    manual_match_calls = 0

    def track_construction(
        matchup: Matchup,
        generated_messages: tuple[GeneratedMessage, ...],
    ) -> ConversationSetup:
        setup = construct_conversation(matchup, generated_messages)
        constructed_setups.append(setup)
        return setup

    monkeypatch.setattr("reasonese.collect_data.construct_conversation", track_construction)
    original_matches = ManualMessageSnapshot.matches

    def track_manual_match(
        self: ManualMessageSnapshot,
        setup: ConversationSetup,
    ) -> bool:
        nonlocal manual_match_calls
        manual_match_calls += 1
        return original_matches(self, setup)

    monkeypatch.setattr(ManualMessageSnapshot, "matches", track_manual_match)
    transport = FakeTransport(
        [
            _message_qa_batch(2),
            *_assistant_responses(4),
            _judge_batch((True, False, False, True, True, False, False, True)),
        ]
    )
    output = tmp_path / "collection"
    manual = _manual_library(tmp_path, study)

    cold = collect_study(
        study,
        output,
        OpenRouterClient(transport),
        manual,
        prefer_batch=True,
    )
    warm = collect_study(study, output, None, manual, prefer_batch=True)

    assert len(cold.trials) == 4
    assert len(cold.observations) == 8
    assert cold.trace_cache_hits == 0
    assert cold.judgment_cache_hits == 0
    assert warm.trace_cache_hits == 4
    assert warm.judgment_cache_hits == 4
    assert warm.observations == cold.observations
    assert len(constructed_setups) == 2
    assert manual_match_calls == 2
    assert {setup.matchup for setup in constructed_setups} == {
        trial.matchup for trial in cold.trials
    }
    assert len(transport.post_calls) == 6
    assert transport.post_calls[0][1]["model"] == "openai/gpt-5.6-luna"
    assert all(call[0] == "/api/v1/chat/completions" for call in transport.post_calls[1:5])
    assert all(call[1]["model"] == "thinkingmachines/inkling" for call in transport.post_calls[1:5])
    first_request = transport.post_calls[1][1]
    assert first_request["temperature"] == 0.7
    assert first_request["parallel_tool_calls"] is False
    assert any(tool["type"] == "openrouter:web_search" for tool in first_request["tools"])
    assert transport.post_calls[5][1]["model"] == "openai/gpt-5.6-luna"
    assert len(transport.post_calls[5][1]["requests"]) == 8
    rows = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    assert len(rows) == 8
    assert (output / "study.yaml").exists()
    assert (output / "collection.sqlite3").is_file()

    first_directory = next(directory for directory in manual.root.iterdir() if directory.is_dir())
    (first_directory / "normal.txt").write_text("Changed manual instruction.")
    with pytest.raises(ValueError, match="conversation trials"):
        collect_study(study, output, None, manual, prefer_batch=True)


def test_collect_study_preserves_distinct_verdicts_for_identical_rollout_traces(
    tmp_path: Path,
) -> None:
    study = _study(2)
    identical_response = _chat("same answer", "same-assistant-response")
    transport = FakeTransport(
        [
            _message_qa_batch(2),
            *(identical_response for _ in range(4)),
            _judge_batch((True, False, False, True, True, True, False, False)),
        ]
    )
    output = tmp_path / "identical-rollouts"
    manual = _manual_library(tmp_path, study)

    cold = collect_study(
        study,
        output,
        OpenRouterClient(transport),
        manual,
        prefer_batch=True,
    )
    warm = collect_study(study, output, None, manual, prefer_batch=True)

    assert warm.observations == cold.observations
    assert [observation.completed for observation in warm.observations] == [
        True,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
    ]


def test_collect_studies_batches_across_tasks_and_matches_independent_outcomes(
    tmp_path: Path,
) -> None:
    first = _study()
    shared = first.inputs[0]
    second = make_study(
        (shared, _spec("What is three plus two?", Channel.USER)),
        Assistant.INKLING,
        1,
    )
    studies = (first, second)
    manual = _manual_library(tmp_path, *studies)
    tasks = tuple(
        CollectionTask(study, tmp_path / "suite" / f"study-{index}")
        for index, study in enumerate(studies, start=1)
    )
    suite_transport = FakeTransport(
        [
            _message_qa_batch(3),
            *_assistant_responses(4),
            _judge_batch((True, False, False, True, True, False, False, True)),
        ]
    )

    suite_results = collect_studies(
        tasks,
        OpenRouterClient(suite_transport),
        manual,
        YamlMessageCache(tmp_path / "suite" / "generated_messages.yaml"),
        YamlMessageQaCache(tmp_path / "suite" / "message_qa.yaml"),
        prefer_batch=True,
    )

    independent_results = tuple(
        collect_study(
            study,
            tmp_path / "independent" / f"study-{index}",
            OpenRouterClient(
                FakeTransport(
                    [
                        _message_qa_batch(2),
                        *_assistant_responses(2),
                        _judge_batch((True, False, False, True)),
                    ]
                )
            ),
            manual,
            prefer_batch=True,
        )
        for index, study in enumerate(studies, start=1)
    )

    def numeric_outcomes(result: CollectionResult) -> list[tuple[object, ...]]:
        return [
            (
                observation.trial_id,
                observation.cell_id,
                observation.permutation,
                observation.rollout,
                observation.position,
                observation.completed,
            )
            for observation in result.observations
        ]

    assert [numeric_outcomes(result) for result in suite_results] == [
        numeric_outcomes(result) for result in independent_results
    ]
    assert len(suite_transport.post_calls) == 6
    assert len(suite_transport.post_calls[0][1]["requests"]) == 3
    assert all(call[0] == "/api/v1/chat/completions" for call in suite_transport.post_calls[1:5])
    assert len(suite_transport.post_calls[5][1]["requests"]) == 8

    warm_results = collect_studies(
        tasks,
        None,
        manual,
        YamlMessageCache(tmp_path / "suite" / "generated_messages.yaml"),
        YamlMessageQaCache(tmp_path / "suite" / "message_qa.yaml"),
        prefer_batch=True,
    )
    assert [result.observations for result in warm_results] == [
        result.observations for result in suite_results
    ]
    assert [result.trace_cache_hits for result in warm_results] == [2, 2]
    assert [result.judgment_cache_hits for result in warm_results] == [2, 2]


def test_collect_studies_runs_mixed_assistant_models_through_sync_requests(
    tmp_path: Path,
) -> None:
    first = _study(assistant=Assistant.INKLING)
    second = _study(assistant=Assistant.INKLING_SMALL)
    tasks = (
        CollectionTask(first, tmp_path / "mixed" / "inkling"),
        CollectionTask(second, tmp_path / "mixed" / "inkling-small"),
    )
    transport = FakeTransport(
        [
            _message_qa_batch(2),
            *_assistant_responses(4),
            _judge_batch((True, False, False, True, True, False, False, True)),
        ]
    )

    results = collect_studies(
        tasks,
        OpenRouterClient(transport),
        _manual_library(tmp_path, first, second),
        YamlMessageCache(tmp_path / "mixed" / "generated_messages.yaml"),
        YamlMessageQaCache(tmp_path / "mixed" / "message_qa.yaml"),
        prefer_batch=True,
    )

    assert [len(result.observations) for result in results] == [4, 4]
    assert transport.post_calls[0][0] == "/api/beta/batches"
    assert all(call[0] == "/api/v1/chat/completions" for call in transport.post_calls[1:5])
    assert sorted(call[1]["model"] for call in transport.post_calls[1:5]) == [
        "thinkingmachines/inkling",
        "thinkingmachines/inkling",
        "thinkingmachines/inkling-small",
        "thinkingmachines/inkling-small",
    ]
    assert transport.post_calls[5][0] == "/api/beta/batches"
    assert len(transport.post_calls[5][1]["requests"]) == 8


def test_collect_study_runs_each_active_tool_round(tmp_path: Path) -> None:
    study = _study()
    transport = FakeTransport(
        [
            _message_qa_batch(2),
            _tool_chat(),
            _chat("used the file", "assistant-0"),
            _chat("already done", "assistant-1"),
            _judge_batch((True, False, False, True)),
        ]
    )

    result = collect_study(
        study,
        tmp_path / "tool-collection",
        OpenRouterClient(transport, sync_workers=1),
        _manual_library(tmp_path, study),
        prefer_batch=True,
    )

    cached_by_trial = SqliteStudyCache(
        tmp_path / "tool-collection" / "collection.sqlite3"
    ).load_traces(result.trials)
    cached_traces = tuple(cached_by_trial[trial.trial_id] for trial in result.trials)
    assert [len(trace.tool_steps) for trace in cached_traces] == [1, 0]
    assert all(call[0] == "/api/v1/chat/completions" for call in transport.post_calls[1:4])
    followup = transport.post_calls[2][1]["messages"]
    assert followup[-1] == {
        "role": "tool",
        "tool_call_id": "live-read",
        "content": "Name the capital of France.",
    }
    assert len(transport.post_calls[4][1]["requests"]) == 4


def test_collect_study_requires_key_only_for_missing_work(tmp_path: Path) -> None:
    study = _study()
    manual = _manual_library(tmp_path, study)
    with pytest.raises(ValueError, match="conversation trials"):
        collect_study(study, tmp_path / "empty", None, manual, prefer_batch=True)

    output = tmp_path / "traces-only"
    trials = build_trials(study)
    SqliteStudyCache(output / "collection.sqlite3").put_traces(
        tuple(
            (trial.trial_id, _trace_for_trial(study, trial_index))
            for trial_index, trial in enumerate(trials)
        )
    )
    YamlMessageQaCache(output / "message_qa.yaml").put_many(
        tuple(
            MessageQaVerdict(
                spec,
                GeneratedText.parse(str(spec.instruction)),
                True,
                (),
                _chat(json.dumps({"complies": True, "issues": []}), f"qa-{index}"),
            )
            for index, spec in enumerate(study.inputs)
        )
    )
    with pytest.raises(ValueError, match="uncached judgments"):
        collect_study(study, output, None, manual, prefer_batch=True)


def _write_study(path: Path, study: Study) -> None:
    path.write_text(yaml.safe_dump(study_to_dict(study), sort_keys=False))


def test_collect_data_cli_runs_then_reports_warm_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    study_path = tmp_path / "study.yaml"
    output = tmp_path / "output"
    _write_study(study_path, _study())
    manual = _manual_library(tmp_path, _study())
    transport = FakeTransport(
        [
            _message_qa_batch(2),
            *_assistant_responses(2),
            _judge_batch((True, False, False, True)),
        ]
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("reasonese.collect_data.RequestsTransport", lambda key: transport)
    args = [
        "--study",
        str(study_path),
        "--output",
        str(output),
        "--user-messages",
        str(manual.root),
    ]

    assert collect_data(args) == 0
    cold = json.loads(capsys.readouterr().out)
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert collect_data(args) == 0
    warm = json.loads(capsys.readouterr().out)

    assert cold["trials"] == 2
    assert cold["observations"] == 4
    assert warm["trace_cache_hits"] == 2
    assert warm["judgment_cache_hits"] == 2
    assert len(transport.post_calls) == 4


def test_collect_studies_cli_batches_tasks_then_reports_warm_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = _study()
    second = make_study(
        (first.inputs[0], _spec("What is three plus two?", Channel.USER)),
        Assistant.INKLING,
        1,
    )
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    _write_study(first_path, first)
    _write_study(second_path, second)
    manual = _manual_library(tmp_path, first, second)
    output = tmp_path / "suite-output"
    transport = FakeTransport(
        [
            _message_qa_batch(3),
            *_assistant_responses(4),
            _judge_batch((True, False, False, True, True, False, False, True)),
        ]
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("reasonese.collect_studies.RequestsTransport", lambda key: transport)
    args = [
        "--study",
        str(first_path),
        "--study",
        str(second_path),
        "--output",
        str(output),
        "--user-messages",
        str(manual.root),
    ]

    assert collect_studies_cli(args) == 0
    cold = json.loads(capsys.readouterr().out)
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert collect_studies_cli(args) == 0
    warm = json.loads(capsys.readouterr().out)

    assert cold["trials"] == 4
    assert cold["observations"] == 8
    assert [item["output"] for item in cold["studies"]] == [
        str(output / "first"),
        str(output / "second"),
    ]
    assert [item["trace_cache_hits"] for item in warm["studies"]] == [2, 2]
    assert [item["judgment_cache_hits"] for item in warm["studies"]] == [2, 2]
    assert len(transport.post_calls) == 6


def test_collection_tasks_require_paths_with_distinct_stems(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        collection_tasks((), tmp_path)
    with pytest.raises(ValueError, match="distinct stems"):
        collection_tasks(
            (tmp_path / "one" / "same.yaml", tmp_path / "two" / "same.yaml"),
            tmp_path,
        )


def test_collect_studies_requires_distinct_tasks(tmp_path: Path) -> None:
    study = _study()
    manual = _manual_library(tmp_path, study)
    message_cache = YamlMessageCache(tmp_path / "messages.yaml")
    qa_cache = YamlMessageQaCache(tmp_path / "qa.yaml")
    with pytest.raises(ValueError, match="at least one"):
        collect_studies((), None, manual, message_cache, qa_cache, prefer_batch=True)
    with pytest.raises(ValueError, match="output directories"):
        collect_studies(
            (
                CollectionTask(study, tmp_path / "same"),
                CollectionTask(
                    make_study(
                        (study.inputs[0], _spec("Different.", Channel.USER)),
                        study.assistant,
                        1,
                    ),
                    tmp_path / "same",
                ),
            ),
            None,
            manual,
            message_cache,
            qa_cache,
            prefer_batch=True,
        )
    with pytest.raises(ValueError, match="studies"):
        collect_studies(
            (
                CollectionTask(study, tmp_path / "first"),
                CollectionTask(study, tmp_path / "second"),
            ),
            None,
            manual,
            message_cache,
            qa_cache,
            prefer_batch=True,
        )


def test_collect_data_cli_reports_missing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study_path = tmp_path / "study.yaml"
    study = _study()
    _write_study(study_path, study)
    manual = _manual_library(tmp_path, study)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="2"):
        collect_data(
            [
                "--study",
                str(study_path),
                "--output",
                str(tmp_path / "out"),
                "--user-messages",
                str(manual.root),
            ]
        )


def test_refined_study_types_reject_bad_values() -> None:
    with pytest.raises(TypeError):
        PositiveInteger.parse(0)
    with pytest.raises(TypeError):
        StudyInputs.parse((_spec("Only", Channel.USER),))
