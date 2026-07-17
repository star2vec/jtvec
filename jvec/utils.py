"""Shared plumbing: seeding, run directories, provenance."""

from __future__ import annotations

import datetime
import random
import resource
import subprocess
import sys
from pathlib import Path

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
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024
    return rss / 1e9
