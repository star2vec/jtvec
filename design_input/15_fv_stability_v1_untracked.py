"""FV extraction-stability experiment (blocking; discovered 2026-07-17).

The checkpoint sweep's endpoint re-extraction produced FVs at cosine 0.43–0.61
to their Phase-2 twins on identical weights (same seed value, different RNG
stream position), with induction gains collapsing from +38.8 to +1.8 (capitalize).
This script bounds the draw variance: for each task, extract ``--n-draws``
fresh FVs (only the extraction RNG stream varies; datasets, weights, and —
crucially — the evaluation contexts are held FIXED across draws), then
measure per-draw induction gain, label/output-vocab decodability, and the
pairwise cosine matrix including the two existing draws (phase2, sweep-end).

Draw caches live under cache/fv_draws/draw{k}/ so nothing collides with the
canonical caches. Per-draw records go to results/fv_stability/<ts>/draws.jsonl.

Usage: uv run python scripts/15_fv_stability.py --config configs/pythia410m_phase2.yaml
       [--tasks capitalize,singular-plural,english-french] [--n-draws 4]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from jvec.calibration import select_prompts
from jvec.config import Config
from jvec.evals.fvprobe import TASK_LABEL_WORDS, output_token_ids
from jvec.evals.tasks import surface_token_ids
from jvec.fv import FV_REPO, extract_task_fvs, load_cached_fv, load_fv_model
from jvec.lens_cache import load_lens
from jvec.utils import REPO_ROOT, make_run_dir, set_seed

EVAL_SEED = 999  # evaluation contexts held fixed across draws


def label_stat(per_layer, tokenizer, words) -> int:
    best = None
    for logits in per_layer.values():
        for w in words:
            for tid in surface_token_ids(tokenizer, w):
                r = int(1 + (logits > logits[tid]).sum())
                best = r if best is None or r < best else best
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", default="capitalize,singular-plural,english-french")
    parser.add_argument("--n-draws", type=int, default=4)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    hf_model, tokenizer, model_config, revision = load_fv_model(cfg)
    import jlens as jlens_pkg
    model_j = jlens_pkg.from_hf(hf_model, tokenizer)
    prompts = select_prompts(cfg, tokenizer)
    lens = load_lens(cfg, cfg.fit.skip_first_variants[0], prompts, revision)
    device = model_j.input_device

    from utils.eval_utils import n_shot_eval, n_shot_eval_no_intervention  # noqa: PLC0415
    from utils.prompt_utils import load_dataset  # noqa: PLC0415

    run_dir = make_run_dir(cfg, "fv_stability")
    out = open(run_dir / "draws.jsonl", "a")

    tasks = args.tasks.split(",")
    for task in tasks:
        dataset = load_dataset(task, root_data_dir=str(FV_REPO / "dataset_files"), seed=cfg.seed)
        out_ids = output_token_ids(dataset, tokenizer)

        # Existing draws: phase2 canonical + sweep endpoint (if present).
        draws: dict[str, torch.Tensor] = {}
        draws["phase2"] = load_cached_fv(cfg, task, revision)["fv_todd"].float()
        sweep_path = (REPO_ROOT / cfg.cache_dir / "fvs"
                      / f"{cfg.model.name}@step143000" / task / "fv_todd.pt")
        if sweep_path.exists():
            draws["sweep_end"] = torch.load(sweep_path, weights_only=True).float()

        for k in range(1, args.n_draws + 1):
            name = f"draw{k}"
            dcfg = dataclasses.replace(cfg, cache_dir=f"cache/fv_draws/{name}")
            cached = None
            try:
                cached = load_cached_fv(dcfg, task, revision)
            except Exception:
                pass
            if cached is not None:
                draws[name] = cached["fv_todd"].float()
                continue
            t0 = time.perf_counter()
            set_seed(cfg.seed * 1000 + k)  # vary ONLY the extraction stream
            artifacts = extract_task_fvs(dcfg, task, hf_model, tokenizer, model_config, revision)
            draws[name] = artifacts["fv_todd"].float()
            print(f"  extracted {task}/{name} in {(time.perf_counter()-t0)/60:.1f} min")

        # Fixed-context evaluation for every draw.
        set_seed(EVAL_SEED)
        zs = n_shot_eval_no_intervention(dataset, 0, hf_model, model_config,
                                         tokenizer, compute_ppl=False, test_split="test")
        zs_top1 = dict(zs["clean_topk"])[1]
        for name, fv in draws.items():
            set_seed(EVAL_SEED)  # identical eval contexts for every draw
            steered = n_shot_eval(dataset, fv.reshape(1, -1).to(hf_model.device),
                                  cfg.fv.edit_layer, 0, hf_model, model_config, tokenizer)
            ind = dict(steered["intervention_topk"])[1]
            v = fv.to(device)
            per_layer = {l: model_j.unembed(lens.transport(v, l)).float().cpu()
                         for l in lens.source_layers}
            best_layer = min(per_layer, key=lambda l: float(np.mean(
                [1 + (per_layer[l] > per_layer[l][t]).sum() for t in out_ids])))
            rec = {
                "task": task, "draw": name, "norm": float(fv.norm()),
                "zero_shot": zs_top1, "induction": ind,
                "gain": round(ind - zs_top1, 4),
                "label_rank": label_stat(per_layer, tokenizer, TASK_LABEL_WORDS[task]),
                "outvocab": float(np.mean(
                    [1 + (per_layer[best_layer] > per_layer[best_layer][t]).sum() for t in out_ids])),
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"{task:18s} {name:10s} gain {rec['gain']:+.3f}  label {rec['label_rank']:>5}  "
                  f"outvocab {rec['outvocab']:>8.0f}  norm {rec['norm']:.1f}")

        names = list(draws)
        cos = {
            f"{a}|{b}": round(float(torch.nn.functional.cosine_similarity(
                draws[a], draws[b], dim=0)), 3)
            for i, a in enumerate(names) for b in names[i + 1:]
        }
        out.write(json.dumps({"task": task, "kind": "cosines", "cosines": cos}) + "\n")
        out.flush()
        print(f"  cosines: {cos}")

    out.close()
    print(f"\nwrote {run_dir / 'draws.jsonl'}")


if __name__ == "__main__":
    main()
