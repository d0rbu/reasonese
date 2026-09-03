"""Reproducible balanced subsamples of valid study pairings."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from random import Random

from beartype import beartype
from phantom.interval import Natural

from reasonese.axes import Assistant, Channel
from reasonese.planning import PromptSpec
from reasonese.study import PositiveInteger, Study, StudyInputs, make_study

DEFAULT_PAIRINGS_PER_ASSISTANT = 20_000
_CANDIDATE_MULTIPLIER = 3
_DEGREE_CHOICES = 8


@dataclass(frozen=True, slots=True)
class _PairingStratum:
    """A channel pairing and the exact set of axes changed across an edge."""

    channels: tuple[Channel, Channel]
    instruction_differs: bool
    framing_differs: bool
    author_differs: bool


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


def _pair_indices_from_rank(
    rank: int,
    user_count: int,
    other_count: int,
) -> tuple[int, int]:
    user_pair_count = user_count * (user_count - 1) // 2
    if rank < user_pair_count:
        return _user_pair_from_rank(rank, user_count)
    user, other = divmod(rank - user_pair_count, other_count)
    return user, user_count + other


def _rank_from_pair_indices(
    first: int,
    second: int,
    user_count: int,
    other_count: int,
) -> int:
    if first < user_count and second < user_count:
        return _user_pair_rank(first, second, user_count)
    if second < user_count:
        first, second = second, first
    if first >= user_count:
        raise ValueError("a valid pairing must contain a user-channel specification")
    return _cross_pair_rank(first, second - user_count, user_count, other_count)


def _pairing_stratum(first: PromptSpec, second: PromptSpec) -> _PairingStratum:
    if first.channel is not Channel.USER:
        first, second = second, first
    return _PairingStratum(
        (first.channel, second.channel),
        first.instruction != second.instruction,
        first.framing != second.framing,
        first.author != second.author,
    )


def _stratum_key(stratum: _PairingStratum) -> tuple[str, str, bool, bool, bool]:
    return (
        str(stratum.channels[0]),
        str(stratum.channels[1]),
        stratum.instruction_differs,
        stratum.framing_differs,
        stratum.author_differs,
    )


def _stratum_population_counts(
    population: int,
    ordered_specs: tuple[PromptSpec, ...],
    user_count: int,
    other_count: int,
) -> dict[_PairingStratum, int]:
    return dict(
        Counter(
            _stratum_for_rank(rank, ordered_specs, user_count, other_count)
            for rank in range(population)
        )
    )


def _proportional_quotas(
    population_counts: dict[_PairingStratum, int],
    requested: int,
    population: int,
) -> dict[_PairingStratum, int]:
    quotas = {
        stratum: requested * count // population
        for stratum, count in population_counts.items()
    }
    remaining = requested - sum(quotas.values())
    remainders = sorted(
        population_counts,
        key=lambda stratum: (
            -(requested * population_counts[stratum] % population),
            _stratum_key(stratum),
        ),
    )
    for stratum in remainders[:remaining]:
        quotas[stratum] += 1
    return quotas


def _stratum_for_rank(
    rank: int,
    ordered_specs: tuple[PromptSpec, ...],
    user_count: int,
    other_count: int,
) -> _PairingStratum:
    first, second = _pair_indices_from_rank(rank, user_count, other_count)
    return _pairing_stratum(ordered_specs[first], ordered_specs[second])


def _candidate_ranks(
    population_counts: dict[_PairingStratum, int],
    quotas: dict[_PairingStratum, int],
    population: int,
    ordered_specs: tuple[PromptSpec, ...],
    user_count: int,
    other_count: int,
    random: Random,
) -> dict[_PairingStratum, list[int]]:
    targets = {
        stratum: min(population_counts[stratum], _CANDIDATE_MULTIPLIER * quota)
        for stratum, quota in quotas.items()
        if quota
    }
    candidates = {stratum: [] for stratum in targets}
    seen: Counter[_PairingStratum] = Counter()
    for rank in range(population):
        stratum = _stratum_for_rank(rank, ordered_specs, user_count, other_count)
        if stratum not in targets:
            continue
        seen[stratum] += 1
        ranks = candidates[stratum]
        target = targets[stratum]
        if len(ranks) < target:
            ranks.append(rank)
        else:
            replacement = random.randrange(seen[stratum])
            if replacement < target:
                ranks[replacement] = rank
    for ranks in candidates.values():
        random.shuffle(ranks)
    return candidates


def _target_channel_degrees(
    quotas: dict[_PairingStratum, int],
    ordered_specs: tuple[PromptSpec, ...],
) -> dict[Channel, float]:
    totals = dict.fromkeys(Channel, 0)
    counts = Counter(spec.channel for spec in ordered_specs)
    for stratum, quota in quotas.items():
        for channel in stratum.channels:
            totals[channel] += quota
    return {
        channel: totals[channel] / counts[channel] if counts[channel] else 1.0
        for channel in Channel
    }


def _degree_score(
    rank: int,
    degrees: list[int],
    target_degrees: dict[Channel, float],
    ordered_specs: tuple[PromptSpec, ...],
    user_count: int,
    other_count: int,
) -> tuple[float, float]:
    first, second = _pair_indices_from_rank(rank, user_count, other_count)
    loads = tuple(
        (degrees[cell] + 1) / target_degrees[ordered_specs[cell].channel]
        for cell in (first, second)
    )
    return max(loads), sum(loads)


def _degree_aware_sample(
    quotas: dict[_PairingStratum, int],
    candidates: dict[_PairingStratum, list[int]],
    ordered_specs: tuple[PromptSpec, ...],
    user_count: int,
    other_count: int,
    random: Random,
) -> set[int]:
    target_degrees = _target_channel_degrees(quotas, ordered_specs)
    degrees = [0] * len(ordered_specs)
    schedule = [stratum for stratum, quota in quotas.items() for _ in range(quota)]
    random.shuffle(schedule)
    selected: set[int] = set()
    for stratum in schedule:
        pool = candidates[stratum]
        choices = random.sample(range(len(pool)), min(_DEGREE_CHOICES, len(pool)))
        selected_index = min(
            (
                _degree_score(
                    pool[index],
                    degrees,
                    target_degrees,
                    ordered_specs,
                    user_count,
                    other_count,
                ),
                index,
            )
            for index in choices
        )[1]
        rank = pool[selected_index]
        pool[selected_index] = pool[-1]
        pool.pop()
        first, second = _pair_indices_from_rank(rank, user_count, other_count)
        degrees[first] += 1
        degrees[second] += 1
        selected.add(rank)
    return selected


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))

    def find(self, item: int) -> int:
        while self.parents[item] != item:
            self.parents[item] = self.parents[self.parents[item]]
            item = self.parents[item]
        return item

    def union(self, first: int, second: int) -> bool:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return False
        self.parents[second_root] = first_root
        return True


def _repair_connectivity(
    selected: set[int],
    ordered_specs: tuple[PromptSpec, ...],
    user_count: int,
    other_count: int,
    random: Random,
) -> set[int]:
    forest = _DisjointSet(len(ordered_specs))
    shuffled = sorted(selected)
    random.shuffle(shuffled)
    redundant: list[int] = []
    for rank in shuffled:
        first, second = _pair_indices_from_rank(rank, user_count, other_count)
        if not forest.union(first, second):
            redundant.append(rank)

    components: dict[int, list[int]] = {}
    for cell in range(len(ordered_specs)):
        components.setdefault(forest.find(cell), []).append(cell)
    if len(components) == 1:
        return selected

    base_root = forest.find(0)
    base = components.pop(base_root)
    others = list(components.values())
    random.shuffle(others)
    bridges: list[int] = []
    for component in others:
        component_users = [cell for cell in component if cell < user_count]
        base_users = [cell for cell in base if cell < user_count]
        if component_users:
            first = random.choice(component_users)
            second = random.choice(base)
        else:
            first = random.choice(component)
            second = random.choice(base_users)
        bridges.append(_rank_from_pair_indices(first, second, user_count, other_count))
        base.extend(component)

    if len(redundant) < len(bridges):  # pragma: no cover - follows from m >= n - 1
        raise RuntimeError("not enough cycle edges to repair comparison connectivity")
    redundant_by_stratum: dict[_PairingStratum, list[int]] = {}
    for rank in redundant:
        stratum = _stratum_for_rank(rank, ordered_specs, user_count, other_count)
        redundant_by_stratum.setdefault(stratum, []).append(rank)
    available = set(redundant)
    fallback = list(redundant)
    removed: list[int] = []
    for bridge in bridges:
        bridge_stratum = _stratum_for_rank(bridge, ordered_specs, user_count, other_count)
        matching = redundant_by_stratum.get(bridge_stratum, [])
        while matching and matching[-1] not in available:
            matching.pop()
        while fallback and fallback[-1] not in available:
            fallback.pop()
        removal = matching.pop() if matching else fallback.pop()
        available.remove(removal)
        removed.append(removal)
    repaired = selected.difference(removed).union(bridges)
    if len(repaired) != len(selected):  # pragma: no cover - bridges join distinct components
        raise RuntimeError("connectivity repair changed the requested pairing count")
    check = _DisjointSet(len(ordered_specs))
    for rank in repaired:
        first, second = _pair_indices_from_rank(rank, user_count, other_count)
        check.union(first, second)
    if len({check.find(cell) for cell in range(len(ordered_specs))}) != 1:
        raise RuntimeError("comparison graph remains disconnected after repair")
    return repaired


@beartype
def sample_study_inputs(
    specs: tuple[PromptSpec, ...],
    pairings: PositiveInteger,
    seed: Natural,
) -> tuple[StudyInputs, ...]:
    """Sample stratified, degree-balanced edges and minimally repair connectivity."""
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

    if requested == population:
        return tuple(
            _pair_from_rank(rank, user_specs, other_specs) for rank in range(population)
        )

    random = Random(int(seed))
    ordered_specs = user_specs + other_specs
    population_counts = _stratum_population_counts(
        population,
        ordered_specs,
        len(user_specs),
        len(other_specs),
    )
    if sum(population_counts.values()) != population:
        raise RuntimeError("pairing strata do not partition the valid population")
    quotas = _proportional_quotas(population_counts, requested, population)
    candidates = _candidate_ranks(
        population_counts,
        quotas,
        population,
        ordered_specs,
        len(user_specs),
        len(other_specs),
        random,
    )
    selected_ranks = _degree_aware_sample(
        quotas,
        candidates,
        ordered_specs,
        len(user_specs),
        len(other_specs),
        random,
    )
    selected_ranks = _repair_connectivity(
        selected_ranks,
        ordered_specs,
        len(user_specs),
        len(other_specs),
        random,
    )
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
