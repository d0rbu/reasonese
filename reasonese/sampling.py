"""Reproducible balanced subsamples of within-pair study pairings.

A trial only carries signal when its two instructions come from the same
mutually exclusive pair, so the eligible population is enumerated one pair at a
time. Within a pair the two sides are disjoint sets of specifications, which
makes the population a rectangle minus a rectangle and the comparison graph
bipartite on ``len(first) + len(second)`` cells.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from random import Random

from beartype import beartype
from phantom.interval import Natural

from reasonese.axes import Assistant, Channel
from reasonese.instructions import PairId
from reasonese.planning import PairSpecs, PromptSpec
from reasonese.study import PositiveInteger, Study, StudyInputs, make_study

DEFAULT_PAIRINGS_PER_PAIR = 720
_DEGREE_CHOICES = 8


@dataclass(frozen=True, slots=True)
class _PairingStratum:
    """A channel pairing and which remaining axes differ across an edge."""

    channels: tuple[Channel, Channel]
    framing_differs: bool
    author_differs: bool


@dataclass(frozen=True, slots=True)
class _Sides:
    """One pair's two condition sets, indexed for constant-time rank arithmetic."""

    first: tuple[PromptSpec, ...]
    second: tuple[PromptSpec, ...]
    user_first: tuple[int, ...]
    other_first: tuple[int, ...]
    user_second: tuple[int, ...]
    user_first_position: dict[int, int]
    other_first_position: dict[int, int]
    user_second_position: dict[int, int]

    @property
    def node_count(self) -> int:
        """Return the number of cells in the pair's bipartite comparison graph."""
        return len(self.first) + len(self.second)

    @property
    def population(self) -> int:
        """Return the number of distinct valid unordered pairings."""
        return len(self.user_first) * len(self.second) + len(self.other_first) * len(
            self.user_second
        )

    @property
    def block(self) -> int:
        """Return the rank at which user-first pairings give way to user-second ones."""
        return len(self.user_first) * len(self.second)


def _channel_partition(specs: tuple[PromptSpec, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    user = tuple(index for index, spec in enumerate(specs) if spec.channel is Channel.USER)
    other = tuple(index for index, spec in enumerate(specs) if spec.channel is not Channel.USER)
    return user, other


def _sides(pair_specs: PairSpecs) -> _Sides:
    """Validate one pair's condition sets and index them for sampling."""
    first = pair_specs.first
    second = pair_specs.second
    if not first or not second:
        raise ValueError("both sides of an instruction pair need at least one specification")
    if len(set(first)) != len(first) or len(set(second)) != len(second):
        raise ValueError("prompt specifications must be unique within each side")
    if set(first) & set(second):
        raise ValueError("the two sides of an instruction pair must not share specifications")

    user_first, other_first = _channel_partition(first)
    user_second, other_second = _channel_partition(second)
    if not user_first and not user_second:
        raise ValueError("at least one specification must use the user message channel")
    # A pairing needs a user-channel endpoint, so a specification is only
    # reachable when its own channel is `user` or the opposite side offers one.
    if not user_second and other_first:
        raise ValueError(
            "every first-side specification must use the user message channel "
            "when the second side offers none"
        )
    if not user_first and other_second:
        raise ValueError(
            "every second-side specification must use the user message channel "
            "when the first side offers none"
        )

    return _Sides(
        first,
        second,
        user_first,
        other_first,
        user_second,
        {index: position for position, index in enumerate(user_first)},
        {index: position for position, index in enumerate(other_first)},
        {index: position for position, index in enumerate(user_second)},
    )


def _edge_from_rank(sides: _Sides, rank: int) -> tuple[int, int]:
    """Decode a rank into first-side and second-side specification indices."""
    if not 0 <= rank < sides.population:
        raise ValueError(f"pairing rank {rank} is outside the valid population")
    if rank < sides.block:
        row, column = divmod(rank, len(sides.second))
        return sides.user_first[row], column
    row, column = divmod(rank - sides.block, len(sides.user_second))
    return sides.other_first[row], sides.user_second[column]


def _rank_from_edge(sides: _Sides, first_index: int, second_index: int) -> int:
    """Encode a valid first-side and second-side index pair as its rank."""
    if first_index in sides.user_first_position:
        return sides.user_first_position[first_index] * len(sides.second) + second_index
    if second_index not in sides.user_second_position:
        raise ValueError("a valid pairing must contain a user-channel specification")
    return (
        sides.block
        + sides.other_first_position[first_index] * len(sides.user_second)
        + sides.user_second_position[second_index]
    )


def _edge_nodes(sides: _Sides, rank: int) -> tuple[int, int]:
    """Return the two graph node identifiers joined by one rank."""
    first_index, second_index = _edge_from_rank(sides, rank)
    return first_index, len(sides.first) + second_index


def _inputs_from_rank(sides: _Sides, rank: int) -> StudyInputs:
    first_index, second_index = _edge_from_rank(sides, rank)
    return StudyInputs.parse((sides.first[first_index], sides.second[second_index]))


def _stratum(first: PromptSpec, second: PromptSpec) -> _PairingStratum:
    return _PairingStratum(
        (first.channel, second.channel),
        first.framing != second.framing,
        first.author != second.author,
    )


def _stratum_for_rank(sides: _Sides, rank: int) -> _PairingStratum:
    first_index, second_index = _edge_from_rank(sides, rank)
    return _stratum(sides.first[first_index], sides.second[second_index])


def _stratum_key(stratum: _PairingStratum) -> tuple[str, str, bool, bool]:
    return (
        str(stratum.channels[0]),
        str(stratum.channels[1]),
        stratum.framing_differs,
        stratum.author_differs,
    )


@beartype
def pairing_population_size(pair_specs: PairSpecs) -> PositiveInteger:
    """Count the distinct valid within-pair pairings without materializing them."""
    return PositiveInteger.parse(_sides(pair_specs).population)


@beartype
def minimum_connected_pairings(pair_specs: PairSpecs) -> PositiveInteger:
    """Return the fewest edges that can cover and connect one pair's cells."""
    return PositiveInteger.parse(_sides(pair_specs).node_count - 1)


@beartype
def default_pairing_count(pair_specs: PairSpecs) -> PositiveInteger:
    """Choose 720 pairings, capped by the population and raised for connectivity."""
    sides = _sides(pair_specs)
    return PositiveInteger.parse(
        min(sides.population, max(DEFAULT_PAIRINGS_PER_PAIR, sides.node_count - 1))
    )


def _ranks_by_stratum(sides: _Sides) -> dict[_PairingStratum, list[int]]:
    grouped: dict[_PairingStratum, list[int]] = defaultdict(list)
    for rank in range(sides.population):
        grouped[_stratum_for_rank(sides, rank)].append(rank)
    return dict(grouped)


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


def _target_channel_degrees(
    sides: _Sides, quotas: dict[_PairingStratum, int]
) -> dict[Channel, float]:
    totals = dict.fromkeys(Channel, 0)
    for stratum, quota in quotas.items():
        for channel in stratum.channels:
            totals[channel] += quota
    counts = Counter(spec.channel for spec in sides.first + sides.second)
    targets: dict[Channel, float] = {}
    for channel in Channel:
        target = totals[channel] / counts[channel] if counts[channel] else 0.0
        targets[channel] = target if target > 0.0 else 1.0
    return targets


def _degree_score(
    sides: _Sides,
    rank: int,
    degrees: list[int],
    targets: dict[Channel, float],
) -> tuple[float, float]:
    first_index, second_index = _edge_from_rank(sides, rank)
    loads = (
        (degrees[first_index] + 1) / targets[sides.first[first_index].channel],
        (degrees[len(sides.first) + second_index] + 1)
        / targets[sides.second[second_index].channel],
    )
    return max(loads), sum(loads)


def _degree_aware_sample(
    sides: _Sides,
    ranks_by_stratum: dict[_PairingStratum, list[int]],
    quotas: dict[_PairingStratum, int],
    random: Random,
) -> set[int]:
    targets = _target_channel_degrees(sides, quotas)
    degrees = [0] * sides.node_count
    pools = {
        stratum: list(ranks_by_stratum[stratum])
        for stratum, quota in quotas.items()
        if quota
    }
    for pool in pools.values():
        random.shuffle(pool)
    schedule = [stratum for stratum, quota in quotas.items() for _ in range(quota)]
    random.shuffle(schedule)

    selected: set[int] = set()
    for stratum in schedule:
        pool = pools[stratum]
        choices = random.sample(range(len(pool)), min(_DEGREE_CHOICES, len(pool)))
        chosen = min(
            (_degree_score(sides, pool[index], degrees, targets), index) for index in choices
        )[1]
        rank = pool[chosen]
        pool[chosen] = pool[-1]
        pool.pop()
        first_node, second_node = _edge_nodes(sides, rank)
        degrees[first_node] += 1
        degrees[second_node] += 1
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


def _anchor_node(sides: _Sides) -> int:
    """Return a user-channel cell every other component can always be joined to."""
    if sides.user_first:
        return sides.user_first[0]
    return len(sides.first) + sides.user_second[0]


def _bridge_rank(sides: _Sides, anchor: int, component: list[int], base: list[int]) -> int:
    """Return a valid edge joining one component to the anchor's component.

    Components holding a cell on the side opposite the anchor attach straight to
    the anchor, which is on the user channel and therefore always a legal
    endpoint. Once every such component is merged the base holds every cell on
    that opposite side, so the remaining same-side components can reach it.
    """
    boundary = len(sides.first)
    anchor_is_first = anchor < boundary
    opposite = [node for node in component if (node < boundary) is not anchor_is_first]
    if opposite:
        partner = opposite[0]
        return (
            _rank_from_edge(sides, anchor, partner - boundary)
            if anchor_is_first
            else _rank_from_edge(sides, partner, anchor - boundary)
        )

    node = component[0]
    spec = sides.first[node] if anchor_is_first else sides.second[node - boundary]
    if anchor_is_first:
        candidates = [
            partner
            for partner in base
            if partner >= boundary
            and (
                spec.channel is Channel.USER
                or sides.second[partner - boundary].channel is Channel.USER
            )
        ]
        return _rank_from_edge(sides, node, candidates[0] - boundary)
    candidates = [
        partner
        for partner in base
        if partner < boundary
        and (spec.channel is Channel.USER or sides.first[partner].channel is Channel.USER)
    ]
    return _rank_from_edge(sides, candidates[0], node - boundary)


def _repair_connectivity(sides: _Sides, selected: set[int], random: Random) -> set[int]:
    """Swap the fewest redundant edges needed to connect the pair's cells."""
    forest = _DisjointSet(sides.node_count)
    shuffled = sorted(selected)
    random.shuffle(shuffled)
    redundant = [rank for rank in shuffled if not forest.union(*_edge_nodes(sides, rank))]

    components: dict[int, list[int]] = {}
    for node in range(sides.node_count):
        components.setdefault(forest.find(node), []).append(node)
    if len(components) == 1:
        return selected

    anchor = _anchor_node(sides)
    base = components.pop(forest.find(anchor))
    boundary = len(sides.first)
    anchor_is_first = anchor < boundary
    others = sorted(components.values(), key=lambda component: component[0])
    # Components holding a cell opposite the anchor first, so that the base owns
    # every opposite-side cell before the same-side leftovers need a partner.
    others.sort(
        key=lambda component: all((node < boundary) is anchor_is_first for node in component)
    )

    bridges: list[int] = []
    for component in others:
        bridges.append(_bridge_rank(sides, anchor, component, base))
        base.extend(component)

    if len(redundant) < len(bridges):  # pragma: no cover - follows from m >= n - 1
        raise RuntimeError("not enough cycle edges to repair comparison connectivity")

    redundant_by_stratum: dict[_PairingStratum, list[int]] = defaultdict(list)
    for rank in redundant:
        redundant_by_stratum[_stratum_for_rank(sides, rank)].append(rank)
    available = set(redundant)
    fallback = list(redundant)
    removed: list[int] = []
    for bridge in bridges:
        matching = redundant_by_stratum[_stratum_for_rank(sides, bridge)]
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
    check = _DisjointSet(sides.node_count)
    for rank in repaired:
        check.union(*_edge_nodes(sides, rank))
    if len({check.find(node) for node in range(sides.node_count)}) != 1:  # pragma: no cover
        raise RuntimeError("comparison graph remains disconnected after repair")
    return repaired


def _pair_seed(seed: Natural, pair_id: PairId) -> int:
    """Derive a per-pair seed that does not depend on the pair's position."""
    digest = hashlib.sha256(f"{int(seed)}:{pair_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


@beartype
def sample_pair_inputs(
    pair_specs: PairSpecs,
    pairings: PositiveInteger,
    seed: Natural,
) -> tuple[StudyInputs, ...]:
    """Sample stratified, degree-balanced edges within one instruction pair."""
    sides = _sides(pair_specs)
    population = sides.population
    minimum = sides.node_count - 1
    requested = int(pairings)
    if requested < minimum:
        raise ValueError(
            f"pairings must be at least {minimum} to cover and connect all "
            f"{sides.node_count} cells of pair {pair_specs.pair.pair_id}"
        )
    if requested > population:
        raise ValueError(
            f"pairings cannot exceed the valid population of {population} "
            f"for pair {pair_specs.pair.pair_id}"
        )

    if requested == population:
        return tuple(_inputs_from_rank(sides, rank) for rank in range(population))

    random = Random(_pair_seed(seed, pair_specs.pair.pair_id))
    ranks_by_stratum = _ranks_by_stratum(sides)
    if sum(len(ranks) for ranks in ranks_by_stratum.values()) != population:  # pragma: no cover
        raise RuntimeError("pairing strata do not partition the valid population")
    quotas = _proportional_quotas(
        {stratum: len(ranks) for stratum, ranks in ranks_by_stratum.items()},
        requested,
        population,
    )
    selected = _degree_aware_sample(sides, ranks_by_stratum, quotas, random)
    selected = _repair_connectivity(sides, selected, random)
    return tuple(_inputs_from_rank(sides, rank) for rank in sorted(selected))


@beartype
def build_sampled_studies(
    pair_specs: tuple[PairSpecs, ...],
    assistants: tuple[Assistant, ...],
    pairings_per_pair: PositiveInteger,
    rollouts_per_permutation: PositiveInteger,
    seed: Natural,
) -> tuple[Study, ...]:
    """Use one sampled within-pair design for every requested assistant."""
    if not pair_specs:
        raise ValueError("at least one instruction pair is required")
    if not assistants:
        raise ValueError("at least one assistant is required")
    if len(assistants) != len(set(assistants)):
        raise ValueError("assistants must be unique")
    if len({specs.pair.pair_id for specs in pair_specs}) != len(pair_specs):
        raise ValueError("instruction pair ids must be unique")

    sampled = tuple(
        sample_pair_inputs(specs, pairings_per_pair, seed) for specs in pair_specs
    )
    return tuple(
        make_study(inputs, assistant, int(rollouts_per_permutation))
        for assistant in assistants
        for pair_inputs in sampled
        for inputs in pair_inputs
    )
