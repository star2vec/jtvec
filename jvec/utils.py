"""Shared plumbing: seeding, run directories, provenance."""

from __future__ import annotations

import datetime
import random
import subprocess
import sys
from pathlib import Path

try:
    import resource  # POSIX-only (macOS/Linux)
except ImportError:  # win32 (D-008): stdlib has no `resource`; see peak_rss_gb
    resource = None

import numpy as np
import torch

from jvec.config import Config

REPO_ROOT = Path(__file__).resolve().parent.parent
JLENS_DIR = REPO_ROOT / "third_party" / "jacobian-lens"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds CPU, CUDA, and MPS generators


def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def make_run_dir(cfg: Config, experiment: str) -> Path:
    """Create ``results/<experiment>/<timestamp>/`` with the resolved config in it."""
    run_dir = REPO_ROOT / cfg.results_dir / experiment / timestamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg.save(run_dir / "config.yaml")
    return run_dir


def jlens_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(JLENS_DIR), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def peak_rss_gb() -> float:
    """Peak resident set size of this process, in GB.

    ``ru_maxrss`` is bytes on macOS but kilobytes on Linux (A100 box).
    On win32 (D-008) there is no ``resource``; the psapi peak working set
    is the ru_maxrss analogue.
    """
    if resource is None:  # win32 (D-008)
        return _win32_peak_working_set_bytes() / 1e9
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024
    return rss / 1e9


def _win32_peak_working_set_bytes() -> int:
    """GetProcessMemoryInfo -> PeakWorkingSetSize, via ctypes (D-008)."""
    import ctypes
    from ctypes import wintypes

    class _PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return pmc.PeakWorkingSetSize
