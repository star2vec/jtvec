"""LAW: every instrument passes a positive control AND a negative control
before its readings count. An instrument that cannot separate known-signal
from known-noise is withdrawn (precedent: v1 Exp-2).

Every eval runner must call `require_controlled(instrument)` before taking a
scientific reading. Instruments named in BANNED_INSTRUMENTS are rejected
regardless of any control record, per the CONSTRAINTS.md VERIFIED (negative)
entry that withdrew them.
"""

from __future__ import annotations

from dataclasses import dataclass


class UncontrolledInstrumentError(RuntimeError):
    """Raised when an instrument without both passing controls is used."""


class BannedInstrumentError(RuntimeError):
    """Raised when a withdrawn instrument is used."""


# Withdrawn instruments; banned unless rebuilt and re-controlled under a new
# name with a fresh ControlRecord pair. First entry per the CONSTRAINTS.md
# VERIFIED (negative) entry (v1); the @task-scoped entry per D-013 (M3 run 3:
# results/m3/20260718-174954-instrument-gate). The @task naming follows the
# M2 certificate convention (fv_todd@<task>).
BANNED_INSTRUMENTS = {
    "jspace-fraction-k25-gradient-pursuit": (
        "v1 positive control failed: cannot separate lens-readable residuals "
        "from arbitrary directions on Pythia-410M"
    ),
    "report-probe-forced-choice@singular-plural": (
        "M3 negative control failed (D-013): random-word null with zero "
        "plural outputs still reads the singular-plural label 26/36 on "
        "Pythia-410M; the probe cannot separate task-present from "
        "task-absent on this task"
    ),
}


@dataclass(frozen=True)
class ControlRecord:
    """One control run: where it lives and whether it passed."""

    run: str  # results directory of the control run
    passed: bool
    date: str  # ISO date


@dataclass(frozen=True)
class Instrument:
    """A reader whose outputs are used as scientific evidence.

    Examples: J-lens readout, logit lens, forced-choice report probe.
    """

    name: str
    positive_control: ControlRecord | None = None
    negative_control: ControlRecord | None = None

    def is_controlled(self) -> bool:
        return (
            self.positive_control is not None
            and self.positive_control.passed
            and self.negative_control is not None
            and self.negative_control.passed
        )


def require_controlled(instrument: Instrument) -> None:
    """Gatekeeper called by every eval runner before a scientific reading."""
    if instrument.name in BANNED_INSTRUMENTS:
        raise BannedInstrumentError(
            f"instrument '{instrument.name}' is withdrawn: "
            f"{BANNED_INSTRUMENTS[instrument.name]}"
        )
    if not instrument.is_controlled():
        raise UncontrolledInstrumentError(
            f"LAW violation: instrument '{instrument.name}' lacks a passing "
            "positive AND negative control; its readings do not count"
        )
