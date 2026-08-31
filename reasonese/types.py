"""Small domain types used at file and configuration boundaries."""

from __future__ import annotations

from typing import Any

from hypothesis import strategies as st
from phantom import Phantom


def _is_probability(value: float) -> bool:
    return 0.0 <= value <= 1.0


class Probability(float, Phantom[float], predicate=_is_probability, bound=float):
    """A finite probability in the closed interval [0, 1]."""

    @classmethod
    def __register_strategy__(cls) -> Any:
        return st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def parse_probability(value: float | int | str) -> Probability:
    """Validate an untrusted scalar as a probability."""
    if isinstance(value, bool):
        raise TypeError("boolean values are not probabilities")
    raw = float(value) if isinstance(value, int | str) else value
    return Probability.parse(raw)
