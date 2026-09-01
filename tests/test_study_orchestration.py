from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from reasonese.axes import Assistant, Author, Channel, Framing, Instruction
from reasonese.cache import YamlTraceCache
from reasonese.collect_data import collect_study
from reasonese.collect_data import main as collect_data
from reasonese.config import load_study
from reasonese.conversation import ConversationTrace, GeneratedMessage, GeneratedText
from reasonese.judging import (
    InstructionVerdict,
    InstructionVerdicts,
    Judgment,
    judge_traces,
    trace_fingerprint,
)
from reasonese.manual_messages import ManualMessageLibrary
from reasonese.observations import (
    cell_id,
    observation_to_dict,
    observations_from_trial,
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


def _manual_library(tmp_path: Path, study: Study) -> ManualMessageLibrary:
    root = tmp_path / "manual"
    instructions = tuple(
        dict.fromkeys(spec.instruction for spec in study.inputs if spec.author is Author.USER)
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


def _assistant_batch(count: int) -> JsonObject:
    return {
        "id": "assistant-batch",
        "status": "completed",
        "results": [
            _batch_result(f"request-{index}", _chat(f"answer {index}", f"assistant-{index}"))
            for index in range(count)
        ],
    }


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


def _assistant_batch_with_tool_and_final() -> JsonObject:
    return {
        "id": "assistant-batch",
        "status": "completed",
        "results": [
            _batch_result("request-0", _tool_chat()),
            _batch_result("request-1", _chat("already done", "assistant-1")),
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


class FakeTransport:
    def __init__(self, posts: list[JsonObject]) -> None:
        self.posts = posts
        self.post_calls: list[tuple[str, JsonObject]] = []

    def post_json(self, path: str, body: JsonObject) -> JsonObject:
        self.post_calls.append((path, body))
        return self.posts.pop(0)

    def get_json(self, path: str) -> JsonObject:
        raise AssertionError(f"unexpected GET {path}")


def test_two_input_study_has_two_permutations_and_two_scores_per_cell() -> None:
    study = _study()
    trials = build_trials(study)

    assert isinstance(study.inputs, StudyInputs)
    assert isinstance(study.rollouts_per_permutation, PositiveInteger)
    assert len(trials) == 2
    assert trial_count(study) == 2
    assert observations_per_cell(study) == 2
    assert observations_per_cell_position(study) == 1
    assert {trial.matchup.inputs for trial in trials} == set(itertools.permutations(study.inputs))
    assert len({trial.trial_id for trial in trials}) == 2


def test_three_input_study_balances_every_cell_over_positions_and_rollouts() -> None:
    study = make_study(
        (
            _spec("A", Channel.SYSTEM),
            _spec("B", Channel.USER),
            _spec("C", Channel.USER),
        ),
        Assistant.QWEN3_8_FLASH,
        2,
    )
    trials = build_trials(study)
    position_counts = Counter(
        (spec, position)
        for trial in trials
        for position, spec in enumerate(trial.matchup.inputs, start=1)
    )

    assert len(trials) == 12
    assert trial_count(study) == 12
    assert observations_per_cell(study) == 12
    assert observations_per_cell_position(study) == 4
    assert set(position_counts.values()) == {4}
    assert Counter(trial.rollout for trial in trials) == {1: 6, 2: 6}


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
        ((_spec("Only", Channel.USER),), 1, "at least two"),
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
) -> None:
    study = _study(2)
    transport = FakeTransport(
        [
            _assistant_batch(4),
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
    assert len(transport.post_calls) == 2
    assert transport.post_calls[0][1]["model"] == "thinkingmachines/inkling"
    assert len(transport.post_calls[0][1]["requests"]) == 4
    first_request = transport.post_calls[0][1]["requests"][0]["body"]
    assert first_request["temperature"] == 0.7
    assert first_request["parallel_tool_calls"] is False
    assert any(tool["type"] == "openrouter:web_search" for tool in first_request["tools"])
    assert transport.post_calls[1][1]["model"] == "openai/gpt-5.6-luna"
    assert len(transport.post_calls[1][1]["requests"]) == 8
    rows = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    assert len(rows) == 8
    assert (output / "study.yaml").exists()
    assert len(list((output / "trials").glob("*/trace.yaml"))) == 4

    first_directory = next(directory for directory in manual.root.iterdir() if directory.is_dir())
    (first_directory / "normal.txt").write_text("Changed manual instruction.")
    with pytest.raises(ValueError, match="conversation trials"):
        collect_study(study, output, None, manual, prefer_batch=True)


def test_collect_study_batches_each_active_tool_round(tmp_path: Path) -> None:
    study = _study()
    transport = FakeTransport(
        [
            _assistant_batch_with_tool_and_final(),
            {
                "id": "assistant-followup",
                "status": "completed",
                "results": [_batch_result("request-0", _chat("used the file", "assistant-0"))],
            },
            _judge_batch((True, False, False, True)),
        ]
    )

    result = collect_study(
        study,
        tmp_path / "tool-collection",
        OpenRouterClient(transport),
        _manual_library(tmp_path, study),
        prefer_batch=True,
    )

    cached_traces = tuple(
        YamlTraceCache(
            tmp_path / "tool-collection" / "trials" / str(trial.trial_id) / "trace.yaml"
        ).load()[0]
        for trial in result.trials
    )
    assert [len(trace.tool_steps) for trace in cached_traces] == [1, 0]
    assert len(transport.post_calls[0][1]["requests"]) == 2
    assert len(transport.post_calls[1][1]["requests"]) == 1
    followup = transport.post_calls[1][1]["requests"][0]["body"]["messages"]
    assert followup[-1] == {
        "role": "tool",
        "tool_call_id": "live-read",
        "content": "Name the capital of France.",
    }
    assert len(transport.post_calls[2][1]["requests"]) == 4


def test_collect_study_requires_key_only_for_missing_work(tmp_path: Path) -> None:
    study = _study()
    manual = _manual_library(tmp_path, study)
    with pytest.raises(ValueError, match="conversation trials"):
        collect_study(study, tmp_path / "empty", None, manual, prefer_batch=True)

    output = tmp_path / "traces-only"
    for trial_index, trial in enumerate(build_trials(study)):
        trace = _trace_for_trial(study, trial_index)
        YamlTraceCache(output / "trials" / str(trial.trial_id) / "trace.yaml").put(trace)
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
    transport = FakeTransport([_assistant_batch(2), _judge_batch((True, False, False, True))])
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
    assert len(transport.post_calls) == 2


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
