"""D-008: the win32 platform guard in vendored jvec/utils.py.

peak_rss_gb must return a real, positive number on every platform the
project runs on (its value feeds resource-estimate checks for long runs).
On POSIX it reads ru_maxrss; on win32 it falls back to the psapi peak
working set. This test exercises whichever branch the host provides.
"""

from jvec.utils import peak_rss_gb


def test_peak_rss_positive_and_finite():
    rss = peak_rss_gb()
    assert rss > 0.0
    assert rss < 1e3  # sanity: a test process is not using a terabyte


def test_peak_rss_monotone_nondecreasing():
    first = peak_rss_gb()
    _ballast = [0] * 5_000_000  # ~40 MB, forces the peak upward or flat
    second = peak_rss_gb()
    del _ballast
    assert second >= first
