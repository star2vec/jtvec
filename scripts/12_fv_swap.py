"""Experiment 4 driver: cross-basis FV swap on shared-query task pairs.

For each ordered pair (A -> B): queries = inputs present in BOTH datasets
with defined outputs; task-A 10-shot context + query; conditions
{none, lens_swap, direct_swap, random_target}; per-trial JSONL. Verdict
(PREREGISTRATION §C): lens_swap B-answer rate >> random_target with
non-overlapping 95% bootstrap CIs on >=2 pairs; lens_swap vs direct_swap
decides J-specificity.

Usage: uv run python scripts/12_fv_swap.py --config configs/pythia410m_phase2.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from jvec.calibration import select_prompts
from jvec.config import Config
from jvec.evals.fvswap import final_logits, make_swap_hooks
from jvec.evals.swap import pinv_jacobians
from jvec.evals.tasks import surface_token_ids
from jvec.fv import FV_REPO, load_cached_fv, load_fv_model
from jvec.lens_cache import load_lens
from jvec.utils import make_run_dir, set_seed

PAIRS = [
    ("english-french", "english-spanish"),
    ("english-spanish", "english-french"),
    ("capitalize", "singular-plural"),
    ("singular-plural", "capitalize"),
]
CONDITIONS = ("none", "lens_swap", "direct_swap", "random_target")
N_QUERIES = 30


def io_map(dataset) -> dict[str, str]:
    out = {}
    for split in ("train", "valid", "test"):
        ds = dataset[split]
        for x, y in zip(ds["input"], ds["output"]):
            out[str(x)] = str(y if not isinstance(y, list) else y[0])
    return out


def bootstrap_ci(hits, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(hits, dtype=float)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed + 3)
    hf_model, tokenizer, model_config, revision = load_fv_model(cfg)
    import jlens as jlens_pkg
    model_j = jlens_pkg.from_hf(hf_model, tokenizer)
    prompts = select_prompts(cfg, tokenizer)
    lens = load_lens(cfg, cfg.fit.skip_first_variants[0], prompts, revision)
    lo, hi = cfg.evals.band
    band_layers = [l for l in lens.source_layers if lo <= l <= hi]
    pinvs = pinv_jacobians(lens, band_layers, rcond=cfg.evals.swap_rcond)
    bos = tokenizer.bos_token or ""
    rng = np.random.default_rng(cfg.seed + 3)

    from utils.prompt_utils import load_dataset  # noqa: PLC0415

    run_dir = make_run_dir(cfg, "fv_swap")
    out = open(run_dir / "trials.jsonl", "w")
    summary = {}
    for task_a, task_b in PAIRS:
        fv_a = load_cached_fv(cfg, task_a, revision)["fv_todd"]
        fv_b = load_cached_fv(cfg, task_b, revision)["fv_todd"]
        ds_a = load_dataset(task_a, root_data_dir=str(FV_REPO / "dataset_files"), seed=cfg.seed)
        ds_b = load_dataset(task_b, root_data_dir=str(FV_REPO / "dataset_files"), seed=cfg.seed)
        map_a, map_b = io_map(ds_a), io_map(ds_b)
        common = set(map_a) & set(map_b)
        shared = sorted(q for q in common if map_a[q] != map_b[q])
        if len(shared) < 10:
            print(f"[skip] {task_a}->{task_b}: only {len(shared)} shared queries")
            continue
        queries = list(rng.choice(shared, min(N_QUERIES, len(shared)), replace=False))

        pair_key = f"{task_a}->{task_b}"
        summary[pair_key] = {}
        train = ds_a["train"]
        for condition in CONDITIONS:
            hooks = make_swap_hooks(
                condition, band_layers, lens, fv_a, fv_b, pinvs, seed=cfg.seed
            )
            hits_b, hits_a = [], []
            for q in queries:
                idx = rng.choice(len(train), cfg.fv.n_shots, replace=False)
                chosen = train[idx]
                ctx = bos + "".join(
                    f"Q: {x}\nA: {y}\n\n" for x, y in zip(chosen["input"], chosen["output"])
                )
                logits = final_logits(model_j, ctx + f"Q: {q}\nA:", hooks)
                top1 = int(logits.argmax())
                b_hit = top1 in surface_token_ids(tokenizer, map_b[q])
                a_hit = top1 in surface_token_ids(tokenizer, map_a[q])
                hits_b.append(b_hit)
                hits_a.append(a_hit)
                out.write(json.dumps({
                    "pair": pair_key, "condition": condition, "query": q,
                    "target_a": map_a[q], "target_b": map_b[q],
                    "top5": [tokenizer.decode([t]) for t in logits.topk(5).indices],
                    "a_hit": bool(a_hit), "b_hit": bool(b_hit),
                }) + "\n")
            mb, lb, hb = bootstrap_ci(hits_b)
            summary[pair_key][condition] = {
                "b_rate": round(mb, 4), "b_ci": [round(lb, 4), round(hb, 4)],
                "a_rate": round(float(np.mean(hits_a)), 4), "n": len(queries),
            }
        s = summary[pair_key]
        print(f"{pair_key:38s} " + "  ".join(
            f"{c}: B={s[c]['b_rate']:.0%} A={s[c]['a_rate']:.0%}" for c in CONDITIONS
        ))

    out.close()
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
