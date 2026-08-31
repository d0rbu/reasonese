from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from reasonese.config import (
    ExperimentConfig,
    SimulationConfig,
    load_experiment_config,
    load_simulation_config,
)
from reasonese.design import build_trials, expected_trial_count
from reasonese.schemas import Condition
from reasonese.types import parse_probability

ROOT = Path(__file__).parent.parent


def test_pilot_config_builds_complete_balanced_design() -> None:
    config = load_experiment_config(ROOT / "configs/pilot.toml")
    trials = build_trials(config)

    assert config.name == "representation_pilot"
    assert len(config.conditions) == 14
    assert len(trials) == expected_trial_count(config) == 728
    assert len({trial.trial_id for trial in trials}) == len(trials)
    assert {trial.design_id for trial in trials} == {config.design_id}

    per_block = Counter((trial.pair_id, trial.code_pair_id, trial.repetition) for trial in trials)
    assert set(per_block.values()) == {4}
    for block in per_block:
        blocked = [
            trial
            for trial in trials
            if (trial.pair_id, trial.code_pair_id, trial.repetition) == block
        ]
        first_counts = Counter(trial.first_condition for trial in blocked)
        assert set(first_counts.values()) == {2}
        condition_target_counts = Counter(
            (condition, target)
            for trial in blocked
            for condition, target in trial.condition_to_target.items()
        )
        assert set(condition_target_counts.values()) == {2}


def test_design_is_deterministic_and_seed_controls_order(
    experiment_config: ExperimentConfig,
) -> None:
    original = build_trials(experiment_config)
    repeated = build_trials(experiment_config)
    reseeded = build_trials(
        ExperimentConfig(
            name=experiment_config.name,
            seed=12,
            repetitions=experiment_config.repetitions,
            system_prompt=experiment_config.system_prompt,
            user_preamble=experiment_config.user_preamble,
            response_code_pairs=experiment_config.response_code_pairs,
            conditions=experiment_config.conditions,
        )
    )
    assert original == repeated
    assert {trial.trial_id for trial in original} != {trial.trial_id for trial in reseeded}
    assert [trial.pair_id for trial in original] != [trial.pair_id for trial in reseeded]


def test_rendered_trial_contains_conditions(experiment_config: ExperimentConfig) -> None:
    trial = build_trials(experiment_config)[0]
    assert trial.system_prompt == experiment_config.system_prompt
    assert experiment_config.user_preamble in trial.user_prompt
    assert "[DIRECTIVE 1]" in trial.user_prompt
    assert "[DIRECTIVE 2]" in trial.user_prompt
    assert trial.first_target in trial.user_prompt
    assert trial.second_target in trial.user_prompt


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": "Bad"}, "experiment name"),
        ({"seed": -1}, "seed"),
        ({"seed": 1.5}, "seed"),
        ({"repetitions": 0}, "repetitions"),
        ({"repetitions": 1.5}, "repetitions"),
        ({"system_prompt": ""}, "prompt"),
        ({"response_code_pairs": ()}, "at least one"),
        ({"response_code_pairs": (("KITE", "KITE"),)}, "distinct"),
        ({"conditions": ()}, "at least two"),
    ],
)
def test_experiment_config_rejects_invalid_state(
    experiment_config: ExperimentConfig,
    overrides: dict[str, object],
    message: str,
) -> None:
    values = {
        "name": experiment_config.name,
        "seed": experiment_config.seed,
        "repetitions": experiment_config.repetitions,
        "system_prompt": experiment_config.system_prompt,
        "user_preamble": experiment_config.user_preamble,
        "response_code_pairs": experiment_config.response_code_pairs,
        "conditions": experiment_config.conditions,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        ExperimentConfig(**values)


def test_experiment_config_rejects_duplicate_conditions(
    experiment_config: ExperimentConfig,
) -> None:
    duplicate = (experiment_config.conditions[0], experiment_config.conditions[0])
    with pytest.raises(ValueError, match="unique"):
        ExperimentConfig(
            name="duplicate",
            seed=0,
            repetitions=1,
            system_prompt="System.",
            user_preamble="User.",
            response_code_pairs=(("KITE", "MOSS"),),
            conditions=duplicate,
        )


def test_load_experiment_config_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("schema_version = 1\nunknown = true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level"):
        load_experiment_config(path)


@pytest.mark.parametrize(
    "body",
    [
        "schema_version = 2\n[experiment]\n",
        "not valid toml = [",
    ],
)
def test_load_experiment_config_rejects_version_and_bad_toml(
    tmp_path: Path, body: str
) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiment_config(path)


def test_load_simulation_config() -> None:
    config = load_simulation_config(ROOT / "configs/synthetic_demo.toml")
    assert config.model_id == "synthetic_demo"
    assert config.invalid_rate == pytest.approx(0.02)
    assert len(config.strengths) == 14


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            SimulationConfig(
                model_id="model",
                seed=0,
                invalid_rate=parse_probability(0),
                first_position_bias=0.0,
                strengths={"plain": 0.0},
            ),
            "",
        ),
    ],
)
def test_simulation_config_valid(config: SimulationConfig, message: str) -> None:
    assert config.model_id == "model"
    assert message == ""


def test_simulation_config_rejects_invalid_state() -> None:
    with pytest.raises(ValueError, match="seed"):
        SimulationConfig("model", -1, parse_probability(0), 0.0, {"plain": 0.0})
    with pytest.raises(ValueError, match="finite"):
        SimulationConfig("model", 0, parse_probability(0), float("inf"), {"plain": 0.0})
    with pytest.raises(ValueError, match="must not be empty"):
        SimulationConfig("model", 0, parse_probability(0), 0.0, {})
    with pytest.raises(ValueError, match="finite"):
        SimulationConfig("model", 0, parse_probability(0), 0.0, {"plain": float("nan")})


def test_simulation_config_preserves_exact_model_identifier() -> None:
    config = SimulationConfig(
        "Provider/Model-v1.2", 0, parse_probability(0), 0.0, {"plain": 0.0}
    )
    assert config.model_id == "Provider/Model-v1.2"


def test_experiment_config_rejects_bad_condition_and_code() -> None:
    good = Condition("plain", "representation", "Return {target}.", "Plain.")
    other = Condition("other", "representation", "Return {target}.", "Other.")
    with pytest.raises(ValueError, match="response code"):
        ExperimentConfig("test", 0, 1, "S", "U", (("bad", "MOSS"),), (good, other))
