"""A synthetic backend for exercising the pipeline without model-provider calls."""

from __future__ import annotations

import math
import random

from reasonese.config import SimulationConfig
from reasonese.schemas import SCHEMA_VERSION, ResponseRecord, Trial


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def simulate_responses(
    trials: list[Trial],
    config: SimulationConfig,
) -> list[ResponseRecord]:
    """Sample reproducible responses from known latent strengths and position bias."""
    condition_ids = {
        condition_id
        for trial in trials
        for condition_id in (trial.first_condition, trial.second_condition)
    }
    missing_strengths = sorted(condition_ids - set(config.strengths))
    if missing_strengths:
        raise ValueError(f"simulation config is missing strengths for {missing_strengths}")

    responses: list[ResponseRecord] = []
    for trial in trials:
        generator = random.Random(f"{config.seed}:{trial.trial_id}")
        if generator.random() < config.invalid_rate:
            response_text = "INVALID"
        else:
            first_logit = (
                config.strengths[trial.first_condition]
                - config.strengths[trial.second_condition]
                + config.first_position_bias
            )
            response_text = (
                trial.first_target
                if generator.random() < _sigmoid(first_logit)
                else trial.second_target
            )
        responses.append(
            ResponseRecord(
                schema_version=SCHEMA_VERSION,
                trial_id=trial.trial_id,
                model_id=config.model_id,
                source="synthetic",
                response_text=response_text,
            )
        )
    return responses
