"""Deterministic construction of counterbalanced pairwise trials."""

from __future__ import annotations

import hashlib
import itertools
import json
import random

from reasonese.config import ExperimentConfig
from reasonese.schemas import SCHEMA_VERSION, Condition, Trial


def expected_trial_count(config: ExperimentConfig) -> int:
    """Return the complete-design size before allocating trial records."""
    condition_pairs = len(config.conditions) * (len(config.conditions) - 1) // 2
    return condition_pairs * len(config.response_code_pairs) * 4 * config.repetitions


def _trial_id(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def _render_user_prompt(
    preamble: str,
    first_condition: Condition,
    first_target: str,
    second_condition: Condition,
    second_target: str,
) -> str:
    return (
        f"{preamble.strip()}\n\n"
        f"[DIRECTIVE 1]\n{first_condition.render(first_target)}\n\n"
        f"[DIRECTIVE 2]\n{second_condition.render(second_target)}"
    )


def build_trials(config: ExperimentConfig) -> list[Trial]:
    """Build all order and target-assignment counterbalances, then shuffle deterministically."""
    trials: list[Trial] = []
    design_id = config.design_id

    for condition_a, condition_b in itertools.combinations(config.conditions, 2):
        pair_id = f"{condition_a.id}__{condition_b.id}"
        for code_index, (code_a, code_b) in enumerate(config.response_code_pairs):
            code_pair_id = f"codes_{code_index:02d}_{code_a}_{code_b}"
            for repetition in range(config.repetitions):
                for target_flip in (False, True):
                    targets = (
                        {condition_a.id: code_b, condition_b.id: code_a}
                        if target_flip
                        else {condition_a.id: code_a, condition_b.id: code_b}
                    )
                    for order_flip in (False, True):
                        first, second = (
                            (condition_b, condition_a)
                            if order_flip
                            else (condition_a, condition_b)
                        )
                        identity: dict[str, object] = {
                            "design_id": design_id,
                            "pair_id": pair_id,
                            "code_pair_id": code_pair_id,
                            "repetition": repetition,
                            "first_condition": first.id,
                            "second_condition": second.id,
                            "first_target": targets[first.id],
                            "second_target": targets[second.id],
                        }
                        trials.append(
                            Trial(
                                schema_version=SCHEMA_VERSION,
                                design_id=design_id,
                                trial_id=_trial_id(identity),
                                experiment=config.name,
                                pair_id=pair_id,
                                code_pair_id=code_pair_id,
                                repetition=repetition,
                                first_condition=first.id,
                                second_condition=second.id,
                                first_target=targets[first.id],
                                second_target=targets[second.id],
                                system_prompt=config.system_prompt,
                                user_prompt=_render_user_prompt(
                                    config.user_preamble,
                                    first,
                                    targets[first.id],
                                    second,
                                    targets[second.id],
                                ),
                            )
                        )

    random.Random(config.seed).shuffle(trials)
    if len(trials) != expected_trial_count(config):
        raise AssertionError("generated trial count does not match complete-design formula")
    if len({trial.trial_id for trial in trials}) != len(trials):
        raise AssertionError("trial identifiers are not unique")
    return trials
