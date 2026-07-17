"""LAW: every intervention runs with an auto-generated sham twin (matched
norm, count, layers, positions). No intervention effect is quoted without
its sham in the same table.

`InterventionResult.sham` is a non-optional field: constructing a result
without a sham is a TypeError, and construction verifies the sham twin is
matched on layers, positions, direction count, and norm. The only rendering
method emits effect and sham in the same row. Effects are `DrawSet`s, so the
minimum-draws LAW applies to intervention numbers too.
"""

from __future__ import annotations

from dataclasses import dataclass

from jtvec.core.draws import DrawSet

NORM_MATCH_RTOL = 0.01


class ShamMismatchError(ValueError):
    """Raised when a sham twin is not matched to its intervention."""


@dataclass(frozen=True)
class InterventionSpec:
    """What was done to the model."""

    kind: str  # "ablate" | "inject" | "swap"
    layers: tuple[int, ...]
    positions: tuple[int, ...]  # token positions edited
    n_directions: int
    norm: float


@dataclass(frozen=True)
class ShamResult:
    """The matched-random twin of an intervention."""

    spec: InterventionSpec
    effect: DrawSet


@dataclass(frozen=True)
class InterventionResult:
    """An intervention effect, inseparable from its sham twin."""

    spec: InterventionSpec
    effect: DrawSet
    sham: ShamResult  # non-optional: omitting it fails at construction

    def __post_init__(self) -> None:
        real, twin = self.spec, self.sham.spec
        if twin.layers != real.layers:
            raise ShamMismatchError(f"sham layers {twin.layers} != {real.layers}")
        if twin.positions != real.positions:
            raise ShamMismatchError(f"sham positions {twin.positions} != {real.positions}")
        if twin.n_directions != real.n_directions:
            raise ShamMismatchError(
                f"sham n_directions {twin.n_directions} != {real.n_directions}"
            )
        if real.norm == 0 or abs(twin.norm - real.norm) / abs(real.norm) > NORM_MATCH_RTOL:
            raise ShamMismatchError(
                f"sham norm {twin.norm} not within {NORM_MATCH_RTOL:.0%} of {real.norm}"
            )

    def table_row(self) -> str:
        """The only rendering: effect and sham in the same row."""
        return (
            f"| {self.spec.kind} | layers={list(self.spec.layers)} | "
            f"{self.effect.summary()} | sham: {self.sham.effect.summary()} |"
        )
