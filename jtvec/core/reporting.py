"""LAW: language discipline in all reports and notes: state observations with
their scope ("on Pythia-410M, config X, N=60"), never as general facts.

This is the hard-block half of the enforcement (the lint half lives in
jtvec/validators/language.py): report generators may only emit numbers
through these formatters, whose scope arguments are required keyword-only
parameters. Intervention numbers additionally require the sham in the same
string, per the sham LAW.
"""

from __future__ import annotations

from jtvec.core.draws import DrawSet


def scoped(name: str, value: DrawSet | float, *, model: str, config: str, n: int) -> str:
    """Render one observation with its full scope. No scope, no string."""
    shown = value.summary() if isinstance(value, DrawSet) else f"{value:.4g}"
    return f"on {model}, {config}, N={n}: {name} = {shown}"


def scoped_intervention(
    name: str,
    effect: DrawSet,
    sham: DrawSet,
    *,
    model: str,
    config: str,
    n: int,
) -> str:
    """Intervention numbers never appear without their sham."""
    return (
        f"on {model}, {config}, N={n}: {name} = {effect.summary()} "
        f"(sham: {sham.summary()})"
    )
