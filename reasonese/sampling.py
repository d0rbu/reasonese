"""Reproducible connected subsamples of valid study pairings."""

from __future__ import annotations

from bisect import bisect_right
from random import Random

from beartype import beartype
from phantom.interval import Natural

from reasonese.axes import Assistant, Channel
from reasonese.planning import PromptSpec
from reasonese.study import PositiveInteger, Study, StudyInputs, make_study

DEFAULT_PAIRINGS_PER_ASSISTANT = 20_000


def _spec_key(spec: PromptSpec) -> tuple[str, str, str, str]:
    return (
        str(spec.instruction),
        str(spec.framing),
        str(spec.channel),
        str(spec.author),
    )


def _partition_specs(
    specs: tuple[PromptSpec, ...],
) -> tuple[tuple[PromptSpec, ...], tuple[PromptSpec, ...]]:
    if len(specs) < 2:
        raise ValueError("at least two prompt specifications are required")
    if len(specs) != len(set(specs)):
        raise ValueError("prompt specifications must be unique")
    ordered = tuple(sorted(specs, key=_spec_key))
    user_specs = tuple(spec for spec in ordered if spec.channel is Channel.USER)
    if not user_specs:
        raise ValueError("at least one prompt specification must use the user message channel")
    other_specs = tuple(spec for spec in ordered if spec.channel is not Channel.USER)
    return user_specs, other_specs


@beartype
def pairing_population_size(specs: tuple[PromptSpec, ...]) -> PositiveInteger:
    """Count distinct valid unordered pairs without materializing them."""
    user_specs, other_specs = _partition_specs(specs)
    user_count = len(user_specs)
    other_count = len(other_specs)
    return PositiveInteger.parse(
        user_count * (user_count - 1) // 2 + user_count * other_count
    )


@beartype
def minimum_connected_pairings(specs: tuple[PromptSpec, ...]) -> PositiveInteger:
    """Return the fewest edges that can cover and connect every specification."""
    _partition_specs(specs)
    return PositiveInteger.parse(len(specs) - 1)


@beartype
def default_pairing_count(specs: tuple[PromptSpec, ...]) -> PositiveInteger:
    """Choose 20,000 pairs, capped by the population and raised for connectivity."""
    population = int(pairing_population_size(specs))
    minimum = int(minimum_connected_pairings(specs))
    return PositiveInteger.parse(
        min(population, max(DEFAULT_PAIRINGS_PER_ASSISTANT, minimum))
    )


def _user_pair_rank(first: int, second: int, user_count: int) -> int:
    if first > second:
        first, second = second, first
    return first * (2 * user_count - first - 1) // 2 + second - first - 1


def _cross_pair_rank(user: int, other: int, user_count: int, other_count: int) -> int:
    return user_count * (user_count - 1) // 2 + user * other_count + other


def _user_pair_from_rank(rank: int, user_count: int) -> tuple[int, int]:
    low = 0
    high = user_count - 1
    while low < high:
        midpoint = (low + high) // 2
        pairs_through_midpoint = (midpoint + 1) * (2 * user_count - midpoint - 2) // 2
        if rank < pairs_through_midpoint:
            high = midpoint
        else:
            low = midpoint + 1
    first = low
    preceding = first * (2 * user_count - first - 1) // 2
    return first, first + 1 + rank - preceding


def _pair_from_rank(
    rank: int,
    user_specs: tuple[PromptSpec, ...],
    other_specs: tuple[PromptSpec, ...],
) -> StudyInputs:
    user_pair_count = len(user_specs) * (len(user_specs) - 1) // 2
    if rank < user_pair_count:
        first, second = _user_pair_from_rank(rank, len(user_specs))
        return StudyInputs.parse((user_specs[first], user_specs[second]))
    cross_rank = rank - user_pair_count
    user, other = divmod(cross_rank, len(other_specs))
    return StudyInputs.parse((user_specs[user], other_specs[other]))


def _expand_unblocked_rank(compact_rank: int, blocked: tuple[int, ...]) -> int:
    low = compact_rank
    high = compact_rank + len(blocked)
    target = compact_rank + 1
    while low < high:
        midpoint = (low + high) // 2
        unblocked_through_midpoint = midpoint + 1 - bisect_right(blocked, midpoint)
        if unblocked_through_midpoint >= target:
            high = midpoint
        else:
            low = midpoint + 1
    return low


def _connected_backbone_ranks(
    user_count: int,
    other_count: int,
    random: Random,
) -> set[int]:
    user_order = list(range(user_count))
    other_order = list(range(other_count))
    random.shuffle(user_order)
    random.shuffle(other_order)
    ranks = {
        _user_pair_rank(first, second, user_count)
        for first, second in zip(user_order, user_order[1:], strict=False)
    }
    ranks.update(
        _cross_pair_rank(
            user_order[index % user_count],
            other,
            user_count,
            other_count,
        )
        for index, other in enumerate(other_order)
    )
    return ranks


@beartype
def sample_study_inputs(
    specs: tuple[PromptSpec, ...],
    pairings: PositiveInteger,
    seed: Natural,
) -> tuple[StudyInputs, ...]:
    """Sample valid edges while guaranteeing full cell coverage and connectivity."""
    user_specs, other_specs = _partition_specs(specs)
    population = int(pairing_population_size(specs))
    minimum = int(minimum_connected_pairings(specs))
    requested = int(pairings)
    if requested < minimum:
        raise ValueError(
            f"pairings must be at least {minimum} to cover and connect all {len(specs)} cells"
        )
    if requested > population:
        raise ValueError(f"pairings cannot exceed the valid population of {population}")

    random = Random(int(seed))
    backbone = _connected_backbone_ranks(len(user_specs), len(other_specs), random)
    blocked = tuple(sorted(backbone))
    remaining = requested - len(backbone)
    available = population - len(backbone)
    sampled_compact_ranks = random.sample(range(available), remaining)
    selected_ranks = backbone | {
        _expand_unblocked_rank(rank, blocked) for rank in sampled_compact_ranks
    }
    return tuple(
        _pair_from_rank(rank, user_specs, other_specs) for rank in sorted(selected_ranks)
    )


@beartype
def build_sampled_studies(
    specs: tuple[PromptSpec, ...],
    assistants: tuple[Assistant, ...],
    pairings_per_assistant: PositiveInteger,
    rollouts_per_permutation: PositiveInteger,
    seed: Natural,
) -> tuple[Study, ...]:
    """Use one sampled comparison design for every requested assistant."""
    if not assistants:
        raise ValueError("at least one assistant is required")
    if len(assistants) != len(set(assistants)):
        raise ValueError("assistants must be unique")
    inputs = sample_study_inputs(specs, pairings_per_assistant, seed)
    return tuple(
        make_study(pair, assistant, int(rollouts_per_permutation))
        for assistant in assistants
        for pair in inputs
    )
