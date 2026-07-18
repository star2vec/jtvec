import pytest

from jtvec.core.instruments import (
    BannedInstrumentError,
    ControlRecord,
    Instrument,
    UncontrolledInstrumentError,
    require_controlled,
)

PASSING = ControlRecord(run="results/ctrl", passed=True, date="2026-07-17")
FAILING = ControlRecord(run="results/ctrl", passed=False, date="2026-07-17")


def test_uncontrolled_instrument_rejected():
    with pytest.raises(UncontrolledInstrumentError):
        require_controlled(Instrument(name="jlens-readout"))


def test_positive_control_alone_insufficient():
    with pytest.raises(UncontrolledInstrumentError):
        require_controlled(Instrument(name="jlens-readout", positive_control=PASSING))


def test_failed_negative_control_rejected():
    with pytest.raises(UncontrolledInstrumentError):
        require_controlled(
            Instrument(
                name="jlens-readout",
                positive_control=PASSING,
                negative_control=FAILING,
            )
        )


def test_both_passing_controls_admit_instrument():
    require_controlled(
        Instrument(
            name="jlens-readout",
            positive_control=PASSING,
            negative_control=PASSING,
        )
    )


def test_withdrawn_instrument_banned_even_with_controls():
    with pytest.raises(BannedInstrumentError):
        require_controlled(
            Instrument(
                name="jspace-fraction-k25-gradient-pursuit",
                positive_control=PASSING,
                negative_control=PASSING,
            )
        )


def test_report_probe_banned_on_singular_plural_only():
    # D-013: withdrawn per-task; the same probe on the gated tasks admits.
    with pytest.raises(BannedInstrumentError):
        require_controlled(
            Instrument(
                name="report-probe-forced-choice@singular-plural",
                positive_control=PASSING,
                negative_control=PASSING,
            )
        )
    require_controlled(
        Instrument(
            name="report-probe-forced-choice@capitalize",
            positive_control=PASSING,
            negative_control=PASSING,
        )
    )
