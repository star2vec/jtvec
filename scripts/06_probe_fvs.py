"""Experiment 1: J-lens probe of function vectors vs logit-lens baseline.

Decodes each cached FV (Todd + Hendel per-layer) through the validated
Phase-1 J-lens and the logit lens, with norm-matched random-vector controls,
scoring task-label-word rank and output-token-cloud rank.

Usage: uv run python scripts/06_probe_fvs.py --config configs/pythia410m_phase2.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jlens
import torch

from jvec.calibration import select_prompts
from jvec.config import Config
from jvec.evals.fvprobe import (
    TASK_LABEL_WORDS,
    decode_vector,
    output_token_ids,
    random_like,
)
from jvec.fv import FV_REPO, load_cached_fv, load_fv_model
from jvec.lens_cache import load_lens
from jvec.utils import make_run_dir, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed)
    hf_model, tokenizer, model_config, revision = load_fv_model(cfg)
    model = jlens.from_hf(hf_model, tokenizer)

    prompts = select_prompts(cfg, tokenizer)
    skip_first = cfg.fit.skip_first_variants[0]
    lens = load_lens(cfg, skip_first, prompts, revision)
    layers = lens.source_layers

    from utils.prompt_utils import load_dataset  # noqa: PLC0415

    results = {}
    for task in cfg.fv.tasks:
        if task not in TASK_LABEL_WORDS:
            sys.exit(f"no TASK_LABEL_WORDS entry for {task}")
        artifacts = load_cached_fv(cfg, task, revision)
        if artifacts is None:
            print(f"[skip] {task}: no cached FV")
            continue
        dataset = load_dataset(task, root_data_dir=str(FV_REPO / "dataset_files"), seed=cfg.seed)
        out_ids = output_token_ids(dataset, tokenizer)

        entry = {"fv_todd_norm": float(artifacts["fv_todd"].norm())}
        # Todd FV: one vector, decoded through every layer's J_l.
        entry["todd"] = decode_vector(
            model, tokenizer, lens, artifacts["fv_todd"], task, out_ids,
            layers=layers, topk=cfg.evals.topk_report,
        )
        # Hendel: one vector per layer, each decoded at its own layer.
        entry["hendel"] = {
            l: decode_vector(
                model, tokenizer, lens, artifacts["fv_hendel"][l], task, out_ids,
                layers=[l], topk=cfg.evals.topk_report,
            )[l]
            for l in layers
        }
        # Norm-matched random controls for the Todd FV.
        entry["random"] = {}
        for seed in range(cfg.evals.n_random_seeds):
            rv = random_like(artifacts["fv_todd"], seed)
            entry["random"][seed] = decode_vector(
                model, tokenizer, lens, rv, task, out_ids,
                layers=layers, topk=0,
            )
        results[task] = entry

        best = {
            arm: min(entry["todd"][l][arm]["label_rank"] for l in layers)
            for arm in ("jlens", "logit")
        }
        best_out = {
            arm: min(entry["todd"][l][arm]["output_mean_rank"] for l in layers)
            for arm in ("jlens", "logit")
        }
        rand_best = min(
            entry["random"][s][l]["jlens"]["label_rank"]
            for s in entry["random"] for l in layers
        )
        print(
            f"{task:18s} todd label-rank best: jlens={best['jlens']:>6} logit={best['logit']:>6} "
            f"rand-jlens={rand_best:>6} | output-cloud mean rank best: "
            f"jlens={best_out['jlens']:>8.1f} logit={best_out['logit']:>8.1f}"
        )

    run_dir = make_run_dir(cfg, "fv_probe")
    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, torch.Tensor):
            return o.tolist()
        return o
    (run_dir / "fv_probe.json").write_text(json.dumps(_clean(results), indent=1))
    print(f"\nwrote {run_dir / 'fv_probe.json'}")


if __name__ == "__main__":
    main()
