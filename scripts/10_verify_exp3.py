"""Step 0 verification of Experiment 3 (LABNOTES 2026-07-15 roadmap).

Independent replay of the headline cells with FULL per-trial logging — the
original script saved only aggregates, which is a flaw this run corrects.
Prompt construction and scoring are deliberately re-implemented from the
protocol spec (not imported from scripts/09) so implementation bugs would be
caught rather than replicated; only the ablation hooks are shared, since they
ARE the intervention under test, and a residual-delta diagnostic proves they
fire.

Logged per trial (results/exp3_verification/<ts>/trials.jsonl):
  task, mode, condition, trial, the exact prompt tail, model top-10 tokens
  with logits, all 8 candidate-label logits and FULL-VOCAB ranks, the chosen
  label, correctness.

Extra artifact checks:
  - grammar-prior: clean report trials on the other six tasks' contexts —
    if " plural"/" country" win there too, the 100% report is a probe artifact;
  - shuffled-context report prior for the two headline tasks;
  - residual-delta norms per band layer for each ablation condition.

Usage: uv run python scripts/10_verify_exp3.py --config configs/pythia410m_phase2.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from jlens import ActivationRecorder

from jvec.calibration import select_prompts
from jvec.config import Config
from jvec.evals.exp3 import REPORT_LABELS, REPORT_PROBES, make_hooks
from jvec.fv import FV_REPO, load_cached_fv, load_fv_model
from jvec.lens_cache import load_lens
from jvec.utils import make_run_dir, set_seed

HEADLINE_TASKS = ["singular-plural", "landmark-country"]
CONDITIONS = ("none", "jspace", "sham_jspace", "fv", "sham_fv")
N_TRIALS = 20
N_OTHER = 10
PROBE = REPORT_PROBES["P3"]
M_TOP = 10


@torch.no_grad()
def logits_and_residuals(model_j, prompt, hooks, band_layers):
    final = model_j.n_layers - 1
    handles = [model_j.layers[l].register_forward_hook(h) for l, h in hooks.items()]
    try:
        ids = model_j.encode(prompt, max_length=1024)
        with ActivationRecorder(model_j.layers, at=[*band_layers, final]) as rec:
            model_j.forward(ids)
            residuals = {
                l: rec.activations[l][0, -1].detach().float().cpu() for l in band_layers
            }
            final_res = rec.activations[final][0, -1].detach()
        return model_j.unembed(final_res.float()).float().cpu(), residuals
    finally:
        for h in handles:
            h.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed + 1)  # independent draws from the original run
    hf_model, tokenizer, model_config, revision = load_fv_model(cfg)
    import jlens as jlens_pkg
    model_j = jlens_pkg.from_hf(hf_model, tokenizer)
    W_U = hf_model.embed_out.weight.detach().float().cpu()
    prompts = select_prompts(cfg, tokenizer)
    lens = load_lens(cfg, cfg.fit.skip_first_variants[0], prompts, revision)
    lo, hi = cfg.evals.band
    band_layers = [l for l in lens.source_layers if lo <= l <= hi]
    bos = tokenizer.bos_token or ""
    rng = np.random.default_rng(cfg.seed + 1)

    from utils.prompt_utils import load_dataset  # noqa: PLC0415

    all_tasks = [t for t in cfg.fv.tasks if t in REPORT_LABELS]
    datasets = {
        t: load_dataset(t, root_data_dir=str(FV_REPO / "dataset_files"), seed=cfg.seed)
        for t in all_tasks
    }
    # Independent scoring path: candidate ids re-derived from the label spec.
    cand = {
        t: tokenizer(" " + REPORT_LABELS[t], add_special_tokens=False).input_ids[0]
        for t in all_tasks
    }

    def pairs_of(task, split, n):
        ds = datasets[task][split]
        idx = rng.choice(len(ds), n, replace=False)
        chosen = ds[idx]
        return list(zip(chosen["input"], chosen["output"]))

    def context_of(pairs):
        return bos + "".join(f"Q: {x}\nA: {y}\n\n" for x, y in pairs)

    def record(logits, task):
        top = logits.topk(10)
        cand_info = {}
        for t, tid in cand.items():
            cand_info[REPORT_LABELS[t]] = {
                "logit": round(float(logits[tid]), 3),
                "vocab_rank": int(1 + (logits > logits[tid]).sum()),
            }
        chosen_task = max(all_tasks, key=lambda t: float(logits[cand[t]]))
        return {
            "top10": [
                (tokenizer.decode([i]), round(float(v), 3))
                for v, i in zip(top.values, top.indices)
            ],
            "candidates": cand_info,
            "chosen_label": REPORT_LABELS[chosen_task],
            "report_correct": chosen_task == task,
        }

    run_dir = make_run_dir(cfg, "exp3_verification")
    out = open(run_dir / "trials.jsonl", "w")

    def emit(row):
        out.write(json.dumps(row) + "\n")

    # --- headline cells ---------------------------------------------------------
    for task in HEADLINE_TASKS:
        fv = load_cached_fv(cfg, task, revision)["fv_todd"]
        for condition in CONDITIONS:
            hooks = make_hooks(condition, band_layers, lens, W_U, fv, m_top=M_TOP, seed=cfg.seed)
            # residual-delta diagnostic on one fixed prompt
            fixed = context_of(pairs_of(task, "train", cfg.fv.n_shots)) + PROBE
            _, res_clean = logits_and_residuals(model_j, fixed, {}, band_layers)
            _, res_abl = logits_and_residuals(model_j, fixed, hooks, band_layers)
            deltas = {
                l: round(float((res_abl[l] - res_clean[l]).norm() / res_clean[l].norm()), 4)
                for l in band_layers
            }
            emit({"kind": "residual_delta", "task": task, "condition": condition, "rel_delta": deltas})

            for i in range(N_TRIALS):
                pairs = pairs_of(task, "train", cfg.fv.n_shots)
                rep_prompt = context_of(pairs) + PROBE
                logits, _ = logits_and_residuals(model_j, rep_prompt, hooks, band_layers)
                emit({"kind": "trial", "mode": "report", "task": task,
                      "condition": condition, "trial": i,
                      "prompt_tail": rep_prompt[-160:], **record(logits, task)})

                qx, qy = pairs_of(task, "test", 1)[0]
                ex_prompt = context_of(pairs) + f"Q: {qx}\nA:"
                logits, _ = logits_and_residuals(model_j, ex_prompt, hooks, band_layers)
                target_ids = [
                    tokenizer(f" {qy}", add_special_tokens=False).input_ids[0],
                    tokenizer(str(qy), add_special_tokens=False).input_ids[0],
                ]
                emit({"kind": "trial", "mode": "exec", "task": task,
                      "condition": condition, "trial": i,
                      "query": str(qx), "target": str(qy),
                      "top10": [(tokenizer.decode([t]), round(float(v), 3))
                                for v, t in zip(*[x for x in logits.topk(10)])],
                      "exec_correct": int(logits.argmax()) in target_ids})

    # --- grammar-prior check: clean report probe on the other tasks' contexts ---
    for task in all_tasks:
        if task in HEADLINE_TASKS:
            continue
        for i in range(N_OTHER):
            pairs = pairs_of(task, "train", cfg.fv.n_shots)
            logits, _ = logits_and_residuals(model_j, context_of(pairs) + PROBE, {}, band_layers)
            emit({"kind": "trial", "mode": "report_other", "task": task,
                  "condition": "none", "trial": i, **record(logits, task)})

    # --- shuffled-context prior for headline tasks -------------------------------
    for task in HEADLINE_TASKS:
        for i in range(N_TRIALS):
            pairs = pairs_of(task, "train", cfg.fv.n_shots)
            ys = [y for _, y in pairs]
            rng.shuffle(ys)
            shuffled = [(x, y) for (x, _), y in zip(pairs, ys)]
            logits, _ = logits_and_residuals(model_j, context_of(shuffled) + PROBE, {}, band_layers)
            emit({"kind": "trial", "mode": "report_shuffled", "task": task,
                  "condition": "none", "trial": i, **record(logits, task)})

    out.close()
    print(f"wrote {run_dir / 'trials.jsonl'}")


if __name__ == "__main__":
    main()
