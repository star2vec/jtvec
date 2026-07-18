"""Workstream B: Exp-1 hardening per PREREGISTRATION §B.

For every available lens variant × cached FV (Todd + per-layer Hendel) ×
label set (Set-1/2/3, registered before scoring), computes:
- min-over-layers full-vocab label rank (J-lens vs logit lens),
- random-vector control on the same statistic, n_random seeds
  ("above chance" ⇔ J-lens beats >= 95% of seeds),
- output-vocab mean rank with 95% bootstrap CI over output tokens.

"Robustly decodable" (pre-registered): J-lens < logit ordering on the label
statistic preserved in EVERY grid cell for that task, and >=95/100 random
seeds beaten under the primary variant/Set-1.

Everything is deterministic given the caches; the full per-cell rank grid is
dumped alongside the summary.

Usage: uv run python scripts/14_harden_exp1.py --config configs/pythia410m_phase2.yaml
       [--variants skip4_n10,skip16_n10,skip4_n5] [--n-random 100]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from jvec.calibration import select_prompts
from jvec.config import Config
from jvec.evals.fvprobe import TASK_LABEL_WORDS, output_token_ids, random_like
from jvec.evals.tasks import surface_token_ids
from jvec.fv import FV_REPO, load_cached_fv, load_fv_model
from jvec.lens_cache import load_lens
from jvec.utils import make_run_dir, set_seed

# PREREGISTRATION §B label sets (fixed before scoring).
LABEL_SETS = {
    "set1": TASK_LABEL_WORDS,
    "set2": {
        "antonym": ["opposite"], "english-french": ["French"],
        "english-spanish": ["Spanish"], "present-past": ["past"],
        "singular-plural": ["plural"], "capitalize": ["uppercase"],
        "landmark-country": ["country"], "person-sport": ["sport"],
    },
    "set3": {
        task: sorted(set(words) | set(extra))
        for task, words, extra in [
            ("antonym", TASK_LABEL_WORDS["antonym"], ["inverse", "negation"]),
            ("english-french", TASK_LABEL_WORDS["english-french"], ["language", "foreign"]),
            ("english-spanish", TASK_LABEL_WORDS["english-spanish"], ["language", "foreign"]),
            ("present-past", TASK_LABEL_WORDS["present-past"], ["preterite", "yesterday"]),
            ("singular-plural", TASK_LABEL_WORDS["singular-plural"], ["many", "pluralize"]),
            ("capitalize", TASK_LABEL_WORDS["capitalize"], ["capitalized", "caps"]),
            ("landmark-country", TASK_LABEL_WORDS["landmark-country"], ["nation", "located"]),
            ("person-sport", TASK_LABEL_WORDS["person-sport"], ["athlete", "game"]),
        ]
    },
}


def variant_cfg(cfg: Config, variant: str) -> tuple[Config, int]:
    skip, n = variant.replace("skip", "").split("_n")
    new = dataclasses.replace(
        cfg,
        calibration=dataclasses.replace(cfg.calibration, n_prompts=int(n)),
        fit=dataclasses.replace(cfg.fit, skip_first_variants=(int(skip),)),
    )
    return new, int(skip)


def label_stat(logits_per_layer: dict[int, torch.Tensor], tokenizer, words) -> int:
    """min over layers of (min over label-set surface tokens) full-vocab rank."""
    best = None
    for logits in logits_per_layer.values():
        for w in words:
            for tid in surface_token_ids(tokenizer, w):
                r = int(1 + (logits > logits[tid]).sum())
                best = r if best is None or r < best else best
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variants", default="skip4_n10,skip16_n10,skip4_n5")
    parser.add_argument("--n-random", type=int, default=100)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed)
    hf_model, tokenizer, model_config, revision = load_fv_model(cfg)
    import jlens as jlens_pkg
    model_j = jlens_pkg.from_hf(hf_model, tokenizer)

    from utils.prompt_utils import load_dataset  # noqa: PLC0415

    lenses = {}
    for variant in args.variants.split(","):
        vcfg, skip = variant_cfg(cfg, variant)
        prompts = select_prompts(vcfg, tokenizer)
        try:
            lenses[variant] = load_lens(vcfg, skip, prompts, revision)
        except FileNotFoundError:
            print(f"[missing] lens {variant} not cached — fit it first; skipping")
    if "skip4_n10" not in lenses:
        sys.exit("primary lens skip4_n10 required")

    tasks = [t for t in cfg.fv.tasks if load_cached_fv(cfg, t, revision) is not None]
    device = model_j.input_device

    @torch.no_grad()
    def decode(vector: torch.Tensor, lens, layers) -> dict[int, torch.Tensor]:
        v = vector.float().to(device)
        return {
            l: model_j.unembed(lens.transport(v, l)).float().cpu() for l in layers
        }

    @torch.no_grad()
    def decode_logit(vector: torch.Tensor) -> dict[int, torch.Tensor]:
        v = vector.float().to(device)
        return {-1: model_j.unembed(v).float().cpu()}

    grid = {}
    rng = np.random.default_rng(cfg.seed)
    run_dir = make_run_dir(cfg, "fv_probe_hardened")
    for task in tasks:
        artifacts = load_cached_fv(cfg, task, revision)
        dataset = load_dataset(task, root_data_dir=str(FV_REPO / "dataset_files"), seed=cfg.seed)
        out_ids = output_token_ids(dataset, tokenizer)
        fv_todd = artifacts["fv_todd"]
        entry = {"variants": {}, "output_vocab": {}, "random_control": {}}

        for variant, lens in lenses.items():
            layers = lens.source_layers
            jl = decode(fv_todd, lens, layers)
            lg = decode_logit(fv_todd)
            hend = {
                l: model_j.unembed(lens.transport(artifacts["fv_hendel"][l].float().to(device), l)).float().cpu()
                for l in layers
            }
            hend_logit = {
                l: model_j.unembed(artifacts["fv_hendel"][l].float().to(device)).float().cpu()
                for l in layers
            }
            entry["variants"][variant] = {
                lset: {
                    "todd_jlens": label_stat(jl, tokenizer, words[task]),
                    "todd_logit": label_stat(lg, tokenizer, words[task]),
                    "hendel_jlens": label_stat(hend, tokenizer, words[task]),
                    "hendel_logit": label_stat(hend_logit, tokenizer, words[task]),
                }
                for lset, words in LABEL_SETS.items()
            }

        # output-vocab mean rank + bootstrap CI over output tokens (primary variant)
        lens = lenses["skip4_n10"]
        jl = decode(fv_todd, lens, lens.source_layers)
        best_layer = min(
            jl, key=lambda l: float(np.mean([1 + (jl[l] > jl[l][t]).sum() for t in out_ids]))
        )
        ranks = np.array([int(1 + (jl[best_layer] > jl[best_layer][t]).sum()) for t in out_ids])
        boots = rng.choice(ranks, size=(10_000, len(ranks)), replace=True).mean(axis=1)
        lgits = decode_logit(fv_todd)[-1]
        lg_ranks = np.array([int(1 + (lgits > lgits[t]).sum()) for t in out_ids])
        entry["output_vocab"] = {
            "jlens_mean": float(ranks.mean()),
            "jlens_ci": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
            "logit_mean": float(lg_ranks.mean()),
            "best_layer": int(best_layer),
        }

        # random-vector control, primary variant, Set-1 statistic
        fv_stat = entry["variants"]["skip4_n10"]["set1"]["todd_jlens"]
        beaten = 0
        for seed in range(args.n_random):
            rv = random_like(fv_todd, seed + 1000)
            r = label_stat(decode(rv, lens, lens.source_layers), tokenizer, LABEL_SETS["set1"][task])
            beaten += fv_stat < r
        entry["random_control"] = {"beaten": beaten, "n": args.n_random}

        ordering_ok = all(
            cells["todd_jlens"] < cells["todd_logit"]
            and cells["hendel_jlens"] < cells["hendel_logit"]
            for v in entry["variants"].values() for cells in v.values()
        )
        entry["robustly_decodable"] = bool(ordering_ok and beaten >= 0.95 * args.n_random)
        grid[task] = entry
        s1 = entry["variants"]["skip4_n10"]["set1"]
        print(f"{task:18s} set1 jlens={s1['todd_jlens']:>5} logit={s1['todd_logit']:>6} "
              f"hendel={s1['hendel_jlens']:>5}  rand beaten {beaten}/{args.n_random}  "
              f"outvocab {entry['output_vocab']['jlens_mean']:.0f} "
              f"[{entry['output_vocab']['jlens_ci'][0]:.0f},{entry['output_vocab']['jlens_ci'][1]:.0f}] "
              f"vs logit {entry['output_vocab']['logit_mean']:.0f}  "
              f"-> {'ROBUST' if entry['robustly_decodable'] else 'not robust'}")

    (run_dir / "hardened.json").write_text(json.dumps(grid, indent=1))
    print(f"\nwrote {run_dir / 'hardened.json'}")


if __name__ == "__main__":
    main()
