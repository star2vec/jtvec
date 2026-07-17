"""LAW: no number derived from a stochastic estimator is reported from a
single draw. Minimum 3 independent draws; report median and IQR.

`DrawSet` is the only representation of a stochastic scalar estimate in this
codebase. It cannot be constructed with fewer than MIN_DRAWS draws, each draw
must carry a distinct seed (the independence witness), and the only summary
statistics it exposes are median and IQR. There is no single-scalar code path.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

MIN_DRAWS = 3


class DrawCountError(ValueError):
    """Raised when a stochastic estimate is built from too few draws."""


class DrawIndependenceError(ValueError):
    """Raised when draws do not carry distinct seeds."""


@dataclass(frozen=True)
class DrawSet:
    """Independent draws of one scalar statistic from a stochastic estimator.

    values: the statistic, one entry per independent draw
    seeds:  the RNG seed of each draw; must be distinct, one per value
    """

    values: tuple[float, ...]
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.values) < MIN_DRAWS:
            raise DrawCountError(
                f"LAW violation: {len(self.values)} draw(s) provided, "
                f"minimum is {MIN_DRAWS} independent draws"
            )
        if len(self.seeds) != len(self.values):
            raise DrawIndependenceError(
                f"{len(self.values)} values but {len(self.seeds)} seeds; "
                "every draw must carry its own seed"
            )
        if len(set(self.seeds)) != len(self.seeds):
            raise DrawIndependenceError(
                f"seeds {self.seeds} are not distinct; draws must be independent"
            )

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def median(self) -> float:
        return statistics.median(self.values)

    @property
    def iqr(self) -> float:
        q1, _, q3 = statistics.quantiles(self.values, n=4, method="inclusive")
        return q3 - q1

    def summary(self) -> str:
        return f"median={self.median:.4g}, IQR={self.iqr:.4g}, n_draws={self.n}"
