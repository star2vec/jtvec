"""Fit-and-cache wrapper around ``jlens.fit`` with a manifest.

The manifest pins everything that determines the lens's identity (model +
revision, exact prompt hashes, fit hyperparameters, seed, jlens commit).
``load_lens`` refuses to hand back a lens whose manifest does not match the
requested config — and never recomputes anything itself. Refitting is an
explicit act (``fit_lens(..., refit=True)`` from script 01).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jlens
import torch
import transformers

from jvec.calibration import PromptSet
from jvec.config import Config
from jvec.utils import REPO_ROOT, jlens_commit, peak_rss_gb

#: Manifest keys that define the lens's scientific identity. The rest
#: (versions, device, wall-clock, timestamp) are provenance only.
IDENTITY_KEYS = (
    "model_name",
    "model_revision",
    "calibration_sha256",
    "n_prompts",
    "max_seq_len",
    "dim_batch",
    "skip_first",
    "source_layers",
    "target_layer",
    "seed",
    "jlens_commit",
)


class ManifestMismatch(RuntimeError):
    pass


def lens_dir(cfg: Config, skip_first: int) -> Path:
    variant = f"skip{skip_first}_n{cfg.calibration.n_prompts}"
    model_slug = cfg.model.name
    if cfg.model.revision:  # checkpoint sweeps: one cache per revision
        model_slug = f"{cfg.model.name}@{cfg.model.revision}"
    return REPO_ROOT / cfg.cache_dir / "lenses" / model_slug / variant


def expected_identity(
    cfg: Config, skip_first: int, prompts: PromptSet, model_revision: str
) -> dict:
    return {
        "model_name": cfg.model.name,
        "model_revision": model_revision,
        "calibration_sha256": prompts.calibration_sha256,
        "n_prompts": cfg.calibration.n_prompts,
        "max_seq_len": cfg.fit.max_seq_len,
        "dim_batch": cfg.fit.dim_batch,
        "skip_first": skip_first,
        "source_layers": list(cfg.fit.source_layers) if cfg.fit.source_layers else None,
        "target_layer": cfg.fit.target_layer,
        "seed": cfg.seed,
        "jlens_commit": jlens_commit(),
    }


def load_lens(cfg: Config, skip_first: int, prompts: PromptSet, model_revision: str) -> jlens.JacobianLens:
    """Load a cached lens, verifying its manifest. Never fits."""
    directory = lens_dir(cfg, skip_first)
    lens_path = directory / "lens.pt"
    manifest_path = directory / "manifest.json"
    if not lens_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"no cached lens at {directory}; run scripts/01_fit_lens.py first"
        )
    manifest = json.loads(manifest_path.read_text())
    expected = expected_identity(cfg, skip_first, prompts, model_revision)
    mismatches = {
        k: (manifest.get(k), expected[k])
        for k in IDENTITY_KEYS
        if manifest.get(k) != expected[k]
    }
    if mismatches:
        lines = "\n".join(
            f"  {k}: cached={cached!r} requested={want!r}"
            for k, (cached, want) in mismatches.items()
        )
        raise ManifestMismatch(
            f"cached lens at {directory} does not match the requested config:\n"
            f"{lines}\n"
            f"Refusing to recompute silently — rerun scripts/01_fit_lens.py "
            f"--refit to refit under the new settings."
        )
    return jlens.JacobianLens.load(str(lens_path))


def fit_lens(
    cfg: Config,
    skip_first: int,
    prompts: PromptSet,
    model: jlens.HFLensModel,
    model_revision: str,
    *,
    refit: bool = False,
) -> jlens.JacobianLens:
    """Fit (or load, if already cached and matching) one lens variant."""
    directory = lens_dir(cfg, skip_first)
    directory.mkdir(parents=True, exist_ok=True)
    lens_path = directory / "lens.pt"

    if lens_path.exists() and not refit:
        try:
            lens = load_lens(cfg, skip_first, prompts, model_revision)
            print(f"[cache hit] {lens_path}")
            return lens
        except ManifestMismatch:
            raise

    checkpoint = directory / "fit_checkpoint.pt"
    if refit and checkpoint.exists():
        checkpoint.unlink()

    start = time.perf_counter()
    lens = jlens.fit(
        model,
        prompts.calibration,
        source_layers=cfg.fit.source_layers,
        target_layer=cfg.fit.target_layer,
        dim_batch=cfg.fit.dim_batch,
        max_seq_len=cfg.fit.max_seq_len,
        skip_first=skip_first,
        checkpoint_path=str(checkpoint),
        resume=not refit,
    )
    wall = time.perf_counter() - start

    # fp32 on disk: the default fp16 save would quantize the lens we validated.
    lens.save(str(lens_path), dtype=torch.float32)
    manifest = expected_identity(cfg, skip_first, prompts, model_revision) | {
        "device": cfg.device,
        "model_dtype": cfg.model.dtype,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "wall_clock_s": round(wall, 1),
        "peak_rss_gb": round(peak_rss_gb(), 2),
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fitted_layers": lens.source_layers,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (directory / "calibration_prompts.json").write_text(
        json.dumps(
            {
                "corpus": prompts.corpus,
                "seed": prompts.seed,
                "calibration": prompts.calibration,
                "heldout": prompts.heldout,
            },
            indent=2,
        )
    )
    print(f"[fitted] {lens_path} in {wall:.0f}s")
    return lens
