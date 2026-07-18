"""Timing probe for the M2 ladder — NOT a scientific run.

Measures per-stage model cost on the current machine (filter-set eval,
mean-activation trial, AIE trial, 0-shot/10-shot evals) with tiny trial
counts, then extrapolates the full M2 ladder (3 draws x n_trials_aie=200 x 3
tasks + the m2_gate eval grid). Writes nothing under results/ (no prereg, no
start_run — this is the resource-estimate step the >10-min rule requires
before the real run) and nothing under the canonical cache/m2 (it calls the
stage functions directly, so no FV artifacts or manifests are produced).

The probe respects a wall-clock budget: it measures tasks in order and stops
measuring (falling back to v1 cost ratios) when the budget would be exceeded.
v1 MacBook reference: 36/38/77 s per AIE trial for capitalize /
english-french / singular-plural at 25 trials (LABNOTES, D-007 note).

Usage: uv run python scripts/m2_probe.py [--config configs/m2_pythia410m.yaml]
       [--aie-trials 2] [--mean-trials 5] [--budget-s 540] [--out probe.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from jvec.config import Config
from jvec.fv import FV_REPO, correct_valid_filter_set, load_fv_model
from jvec.utils import peak_rss_gb, set_seed
from jtvec.fv_stability import RUNGS

# v1 MacBook AIE s/trial (LABNOTES): task -> ratio vs capitalize.
V1_RATIOS = {"capitalize": 1.0, "english-french": 38 / 36, "singular-plural": 77 / 36}
N_EVALS_PER_TASK = len(RUNGS) * 3 * 2 + 1  # rungs x draws x (fv, sham) + zero-shot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/m2_pythia410m.yaml"))
    parser.add_argument("--aie-trials", type=int, default=2)
    parser.add_argument("--mean-trials", type=int, default=5)
    parser.add_argument("--budget-s", type=float, default=540.0)
    parser.add_argument("--out", default=None, help="JSON output path (default: stdout only)")
    args = parser.parse_args()
    t_start = time.perf_counter()

    def remaining() -> float:
        return args.budget_s - (time.perf_counter() - t_start)

    cfg = Config.load(args.config)
    set_seed(cfg.seed)
    model, tokenizer, model_config, revision = load_fv_model(cfg)

    from utils.eval_utils import n_shot_eval, n_shot_eval_no_intervention  # noqa: PLC0415
    from utils.extract_utils import get_mean_head_activations  # noqa: PLC0415
    from utils.prompt_utils import load_dataset  # noqa: PLC0415
    from compute_indirect_effect import compute_indirect_effect  # noqa: PLC0415

    measured: dict[str, dict] = {}
    eval0_s = eval10_s = None

    for task in cfg.fv.tasks:
        # A later task costs at least what capitalize cost; stop if that
        # would blow the budget and extrapolate from ratios instead.
        if measured and remaining() < 1.5 * measured[cfg.fv.tasks[0]]["task_probe_s"]:
            print(f"[budget] skipping direct probe of {task}; using v1 ratios", flush=True)
            continue
        t_task = time.perf_counter()
        dataset = load_dataset(task, root_data_dir=str(FV_REPO / "dataset_files"), seed=cfg.seed)

        t0 = time.perf_counter()
        filter_set = correct_valid_filter_set(cfg, dataset, model, tokenizer, model_config)
        filter_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        mean_acts = get_mean_head_activations(
            dataset, model, model_config, tokenizer,
            n_icl_examples=cfg.fv.n_shots, N_TRIALS=args.mean_trials, filter_set=filter_set,
        )
        mean_s = (time.perf_counter() - t0) / args.mean_trials

        t0 = time.perf_counter()
        compute_indirect_effect(
            dataset, mean_acts, model, model_config, tokenizer,
            n_shots=cfg.fv.n_shots, n_trials=args.aie_trials,
            last_token_only=True, filter_set=filter_set,
        )
        aie_s = (time.perf_counter() - t0) / args.aie_trials

        entry = {
            "filter_s": round(filter_s, 1),
            "mean_s_per_trial": round(mean_s, 2),
            "aie_s_per_trial": round(aie_s, 2),
            "n_valid": len(dataset["valid"]),
            "n_test": len(dataset["test"]),
            "n_correct_valid": int(len(filter_set)),
        }

        if eval0_s is None:  # eval costs: measured once, on the first task
            probe_vec = torch.randn(model_config["resid_dim"]) * 5.0
            t0 = time.perf_counter()
            n_shot_eval(
                dataset, probe_vec.reshape(1, -1).to(model.device),
                cfg.fv.edit_layer, 0, model, model_config, tokenizer,
            )
            eval0_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            n_shot_eval_no_intervention(
                dataset, cfg.fv.n_shots, model, model_config, tokenizer,
                compute_ppl=False, test_split="test",
            )
            eval10_s = time.perf_counter() - t0
            entry["eval0_s"] = round(eval0_s, 1)
            entry["eval10_s"] = round(eval10_s, 1)

        entry["task_probe_s"] = round(time.perf_counter() - t_task, 1)
        measured[task] = entry
        print(f"{task:16s} {entry}", flush=True)

    base = cfg.fv.tasks[0]
    if base not in measured:
        sys.exit("probe measured nothing inside the budget; raise --budget-s")

    # --- extrapolate the full M2 ladder per m2_gate's cost structure ----------
    def task_scale(task: str) -> float:
        if task in measured:
            return measured[task]["aie_s_per_trial"] / measured[base]["aie_s_per_trial"]
        return V1_RATIOS.get(task, 1.0) / V1_RATIOS[base]

    per_task, total_s = {}, 0.0
    for task in cfg.fv.tasks:
        s = task_scale(task)
        m = measured.get(task, {})
        filter_s = m.get("filter_s", measured[base]["filter_s"] * s)
        mean_s = m.get("mean_s_per_trial", measured[base]["mean_s_per_trial"] * s)
        aie_s = m.get("aie_s_per_trial", measured[base]["aie_s_per_trial"] * s)
        extraction_s = filter_s + mean_s * cfg.fv.n_trials_mean * 2 + aie_s * cfg.fv.n_trials_aie
        evals_s = N_EVALS_PER_TASK * eval0_s * s + eval10_s * s
        t_total = 3 * extraction_s + evals_s  # 3 draws
        per_task[task] = {
            "measured_directly": task in measured,
            "aie_s_per_trial": round(aie_s, 2),
            "extraction_s_per_draw": round(extraction_s, 1),
            "evals_s": round(evals_s, 1),
            "total_s": round(t_total, 1),
        }
        total_s += t_total

    total_s *= 1.15  # slack factor, M1-prereg convention
    out = {
        "machine": {
            "device": cfg.device,
            "torch": torch.__version__,
            "cuda_name": torch.cuda.get_device_name(0) if cfg.device == "cuda" else None,
        },
        "model_revision": revision,
        "probe_args": {"aie_trials": args.aie_trials, "mean_trials": args.mean_trials},
        "measured": measured,
        "ladder": {
            "n_draws": 3,
            "n_trials_aie": cfg.fv.n_trials_aie,
            "rungs": list(RUNGS),
            "per_task": per_task,
            "total_s_with_slack": round(total_s, 0),
            "total_h_with_slack": round(total_s / 3600, 2),
        },
        "probe_wall_s": round(time.perf_counter() - t_start, 1),
        "peak_rss_gb": round(peak_rss_gb(), 2),
    }
    print(json.dumps(out["ladder"], indent=2))
    print(
        f"\nfull ladder estimate on this machine: "
        f"{out['ladder']['total_h_with_slack']:.1f} h "
        f"(probe itself: {out['probe_wall_s']:.0f} s, peak RSS {out['peak_rss_gb']} GB)"
    )
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
