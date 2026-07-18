"""Experiment 2: gradient-pursuit decomposition of FVs against the J-lens
dictionary (rows of W_U @ J_l), per the workspace paper (k <= 25 atoms,
non-negative coefficients).

For each cached task FV we decompose, at every band layer:
- the Todd FV (one vector, decomposed against each layer's dictionary),
- the Hendel vector at its own layer,
- ICL residuals at the last prompt token (per-layer, a few prompts),
and as controls: norm-matched random vectors and residuals from held-out pile
prompts (matched non-task directions). Reported metric: J-space fraction
(share of squared norm reconstructed) and the top atoms (tokens).

Runs on CPU so it can coexist with extraction jobs.

Usage: uv run python scripts/07_decompose_fvs.py --config configs/pythia410m_phase2.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from jvec.calibration import select_prompts
from jvec.config import Config
from jvec.decompose import Decomposition, gradient_pursuit, jlens_dictionary
from jvec.evals.fvprobe import random_like
from jvec.fv import load_cached_fv, load_fv_model
from jvec.lens_cache import load_lens
from jvec.utils import make_run_dir, set_seed

K_ATOMS = 25
N_CONTROL_SEEDS = 3


def summarize(dec: Decomposition, tokenizer, n_top: int = 8) -> dict:
    order = sorted(
        zip(dec.indices, dec.coefficients), key=lambda ic: -ic[1]
    )[:n_top]
    return {
        "fraction": round(dec.fraction, 4),
        "n_atoms": len(dec.indices),
        "top_atoms": [(tokenizer.decode([i]), round(c, 3)) for i, c in order],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu", help="decomposition device (default cpu)")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed)

    import dataclasses
    cpu_cfg = dataclasses.replace(cfg, device=args.device)
    hf_model, tokenizer, model_config, revision = load_fv_model(cpu_cfg)
    W_U = hf_model.embed_out.weight.detach().float().cpu()

    prompts = select_prompts(cfg, tokenizer)
    lens = load_lens(cfg, cfg.fit.skip_first_variants[0], prompts, revision)
    lo, hi = cfg.evals.band
    band_layers = [l for l in lens.source_layers if lo <= l <= hi]
    dictionaries = {l: jlens_dictionary(lens, W_U, l) for l in band_layers}
    print(f"dictionaries built for layers {band_layers} ({W_U.shape[0]} atoms, d={W_U.shape[1]})")

    # Matched non-task control: held-out pile residuals per layer.
    import jlens as jlens_pkg
    model_j = jlens_pkg.from_hf(hf_model, tokenizer)
    from jlens import ActivationRecorder
    pile_residuals: dict[int, list[torch.Tensor]] = {l: [] for l in band_layers}
    with torch.no_grad():
        for text in prompts.heldout:
            ids = model_j.encode(text, max_length=cfg.fit.max_seq_len)
            with ActivationRecorder(model_j.layers, at=band_layers) as rec:
                model_j.forward(ids)
                for l in band_layers:
                    pile_residuals[l].append(rec.activations[l][0, -1].detach().float().cpu())

    results: dict = {"pile_controls": {}, "tasks": {}}
    for l in band_layers:
        fracs = [gradient_pursuit(h, dictionaries[l], k=K_ATOMS).fraction for h in pile_residuals[l]]
        results["pile_controls"][l] = [round(f, 4) for f in fracs]

    for task in cfg.fv.tasks:
        artifacts = load_cached_fv(cfg, task, revision)
        if artifacts is None:
            continue
        fv_todd = artifacts["fv_todd"].float().cpu()
        entry = {"todd": {}, "hendel": {}, "random": {}}
        for l in band_layers:
            entry["todd"][l] = summarize(
                gradient_pursuit(fv_todd, dictionaries[l], k=K_ATOMS), tokenizer
            )
            entry["hendel"][l] = summarize(
                gradient_pursuit(artifacts["fv_hendel"][l], dictionaries[l], k=K_ATOMS),
                tokenizer,
            )
        for seed in range(N_CONTROL_SEEDS):
            rv = random_like(fv_todd, seed)
            entry["random"][seed] = {
                l: round(gradient_pursuit(rv, dictionaries[l], k=K_ATOMS).fraction, 4)
                for l in band_layers
            }
        results["tasks"][task] = entry
        todd_fracs = {l: entry["todd"][l]["fraction"] for l in band_layers}
        best_l = max(todd_fracs, key=todd_fracs.get)
        rand_mean = sum(
            entry["random"][s][best_l] for s in range(N_CONTROL_SEEDS)
        ) / N_CONTROL_SEEDS
        pile_mean = sum(results["pile_controls"][best_l]) / len(results["pile_controls"][best_l])
        print(
            f"{task:18s} todd J-frac best L{best_l}={todd_fracs[best_l]:.3f} "
            f"(random={rand_mean:.3f}, pile={pile_mean:.3f}) "
            f"hendel@L{best_l}={entry['hendel'][best_l]['fraction']:.3f} "
            f"top atoms: {[a for a, _ in entry['todd'][best_l]['top_atoms'][:5]]}"
        )

    run_dir = make_run_dir(cfg, "fv_decomposition")
    (run_dir / "decomposition.json").write_text(
        json.dumps(results, indent=1, default=str)
    )
    print(f"\nwrote {run_dir / 'decomposition.json'}")


if __name__ == "__main__":
    main()
