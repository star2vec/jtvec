"""Fit the J-lens variants (one per skip_first value) on GPT-2-small and cache them.

Runs a one-prompt timing probe first and prints a wall-clock + memory estimate
before committing to the full fits. Aborts (flagging the A100) if the
projection exceeds 12h.

Usage: uv run python scripts/01_fit_lens.py --config configs/gpt2_phase1.yaml [--refit]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jlens

from jvec.calibration import select_prompts
from jvec.config import Config
from jvec.lens_cache import fit_lens, lens_dir
from jvec.modeling import load_model
from jvec.utils import make_run_dir, peak_rss_gb, set_seed

MAX_LOCAL_HOURS = 12.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--refit", action="store_true", help="refit even if cached")
    parser.add_argument(
        "--skip-probe", action="store_true", help="skip the timing probe (cache hits)"
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed)
    jlens.configure_logging()

    print(f"loading {cfg.model.name} on {cfg.device} ({cfg.model.dtype})")
    model, tok, revision = load_model(cfg)
    prompts = select_prompts(cfg, tok)
    print(
        f"selected {len(prompts.calibration)} calibration + {len(prompts.heldout)} "
        f"held-out prompts from {prompts.corpus} (seed={cfg.seed})"
    )

    variants = list(cfg.fit.skip_first_variants)
    all_cached = all(
        (lens_dir(cfg, s) / "lens.pt").exists() for s in variants
    ) and not args.refit

    probe_s = None
    if not args.skip_probe and not all_cached:
        print("timing probe: one full prompt (all dim-batches) ...")
        t0 = time.perf_counter()
        jlens.jacobian_for_prompt(
            model,
            prompts.calibration[0],
            source_layers=list(range(model.n_layers - 1)),
            dim_batch=cfg.fit.dim_batch,
            max_seq_len=cfg.fit.max_seq_len,
            skip_first=variants[0],
        )
        probe_s = time.perf_counter() - t0
        total_est_s = probe_s * cfg.calibration.n_prompts * len(variants) * 1.2
        print(
            f"probe: {probe_s:.1f}s/prompt, peak RSS {peak_rss_gb():.2f} GB\n"
            f"estimate for {len(variants)} lens(es) x {cfg.calibration.n_prompts} "
            f"prompts: {total_est_s / 60:.1f} min"
        )
        if total_est_s > MAX_LOCAL_HOURS * 3600:
            sys.exit(
                f"projected {total_est_s / 3600:.1f}h exceeds the {MAX_LOCAL_HOURS:.0f}h "
                f"local budget — flag this run for the A100 instead."
            )

    summary = {}
    for skip_first in variants:
        print(f"\n=== lens variant skip_first={skip_first} ===")
        t0 = time.perf_counter()
        lens = fit_lens(
            cfg, skip_first, prompts, model, revision, refit=args.refit
        )
        summary[f"skip{skip_first}"] = {
            "lens_dir": str(lens_dir(cfg, skip_first)),
            "source_layers": lens.source_layers,
            "n_prompts": lens.n_prompts,
            "wall_clock_s": round(time.perf_counter() - t0, 1),
        }

    run_dir = make_run_dir(cfg, "lens_fit")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "model_revision": revision,
                "probe_s_per_prompt": probe_s,
                "peak_rss_gb": round(peak_rss_gb(), 2),
                "variants": summary,
            },
            indent=2,
        )
    )
    print(f"\nrun record: {run_dir}")


if __name__ == "__main__":
    main()
