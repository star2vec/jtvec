"""Single source of truth for experiment configuration.

Every script takes ``--config <yaml>``; the resolved config is copied into the
run's results directory so results are always reproducible from their folder.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str = "gpt2"
    # None = let HF resolve; the resolved commit sha is recorded in the manifest.
    revision: str | None = None
    dtype: str = "float32"


@dataclass(frozen=True)
class CalibrationConfig:
    corpus: str = "NeelNanda/pile-10k"
    split: str = "train"
    n_prompts: int = 5
    n_heldout: int = 3


@dataclass(frozen=True)
class FitConfig:
    max_seq_len: int = 128
    dim_batch: int = 8
    skip_first_variants: tuple[int, ...] = (4, 16)
    # None = jlens defaults (all layers below the final layer / final layer).
    source_layers: tuple[int, ...] | None = None
    target_layer: int | None = None


@dataclass(frozen=True)
class EvalConfig:
    pass_k: int = 10
    # Inclusive layer band where the J-lens is expected to beat the logit lens
    # (FVs act early/mid; gate comparisons happen here).
    band: tuple[int, int] = (2, 8)
    baseline_threshold: float = 0.8
    topk_report: int = 10
    n_random_seeds: int = 3
    # Swap-eval knobs: alpha scales the moved component; rcond truncates the
    # pseudoinverse (load-bearing — see jvec/evals/swap.py docstring).
    swap_alpha: float = 1.0
    swap_rcond: float = 0.05


@dataclass(frozen=True)
class FVConfig:
    tasks: tuple[str, ...] = ()
    n_shots: int = 10
    n_trials_mean: int = 100
    n_trials_aie: int = 25
    n_top_heads: int = 10
    # Layer at which the FV is added for steering checks (~1/3 depth per Todd).
    edit_layer: int = 8
    # Extraction floors (LABNOTES 2026-07-14): a task is extracted if 10-shot
    # test accuracy >= min_accuracy AND the model answers >= min_correct_valid
    # items of the valid split correctly (Todd's filter_set is built from
    # those). Headline membership for Experiments 1-2 is decided by zero-shot
    # FV induction strength afterwards, not by these floors.
    min_accuracy: float = 0.2
    min_correct_valid: int = 50
    # Reporting stratum only (tasks at/above this are the high-accuracy set).
    baseline_threshold: float = 0.8


@dataclass(frozen=True)
class Config:
    experiment: str = "gpt2_phase1"
    seed: int = 0
    device: str = "mps"  # mps | cuda | cpu — no silent fallback
    model: ModelConfig = field(default_factory=ModelConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    evals: EvalConfig = field(default_factory=EvalConfig)
    fv: FVConfig = field(default_factory=FVConfig)
    cache_dir: str = "cache"
    results_dir: str = "results"

    @classmethod
    def load(cls, path: str | Path) -> Config:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return _from_dict(cls, raw)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def torch_device(self) -> torch.device:
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("config requests device=mps but MPS is unavailable")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("config requests device=cuda but CUDA is unavailable")
        if self.device not in ("mps", "cuda", "cpu"):
            raise ValueError(f"unknown device {self.device!r}")
        return torch.device(self.device)

    def torch_dtype(self) -> torch.dtype:
        dtype = getattr(torch, self.model.dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"unknown dtype {self.model.dtype!r}")
        return dtype


def _from_dict(cls, raw: dict):
    """Build a (nested) dataclass from a dict, rejecting unknown keys."""
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(fields)
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in raw.items():
        f = fields[name]
        if dataclasses.is_dataclass(f.type) or (
            isinstance(f.type, str) and f.type[0].isupper()
        ):
            sub_cls = {
                "model": ModelConfig,
                "calibration": CalibrationConfig,
                "fit": FitConfig,
                "evals": EvalConfig,
                "fv": FVConfig,
            }[name]
            kwargs[name] = _from_dict(sub_cls, value)
        elif isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)
