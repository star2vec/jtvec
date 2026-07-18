"""Workstream D: emergence sweep over Pythia-410m training checkpoints.

Per revision (PREREGISTRATION §D): fit the J-lens (n=10, skip4), run the
capital-recall mini-gate (task accuracy + J-lens-vs-logit HMR), extract FVs
for capitalize / english-french / singular-plural, and measure (a) 10-shot
execution accuracy, (b) zero-shot FV induction gain, (c) label + output-vocab
decodability through that checkpoint's own lens.

Per-revision results are appended incrementally to sweep.jsonl so a crash
preserves progress; lens and FV caches are revision-keyed, so reruns resume.
Early checkpoints that cannot produce a filter_set are recorded as
"too weak" and skipped.

Competing predictions P-D1/2/3 are registered in PREREGISTRATION §D; any
surprising point gets a Step-0-style raw replay before being trusted.

Usage: uv run python scripts/13_checkpoint_sweep.py --config configs/pythia410m_phase2.yaml
       [--revisions step4000,step16000,...]
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
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
from jvec.evals.probe import probe_task
from jvec.evals.baseline import score_task
from jvec.evals.tasks import load_tasks, surface_token_ids
from jvec.fv import FV_REPO, extract_task_fvs, load_cached_fv, load_fv_model
from jvec.lens_cache import fit_lens, load_lens
from jvec.utils import make_run_dir, set_seed

REVISIONS = ["step4000", "step16000", "step36000", "step72000", "step107000", "step143000"]
SWEEP_TASKS = ["capitalize", "english-french", "singular-plural"]


def label_stat(logits_per_layer, tokenizer, words) -> int:
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
    parser.add_argument("--revisions", default=",".join(REVISIONS))
    args = parser.parse_args()

    base_cfg = Config.load(args.config)
    run_dir = make_run_dir(base_cfg, "checkpoint_sweep")
    out = open(run_dir / "sweep.jsonl", "a")
    capital_task = [t for t in load_tasks() if t.name == "capital-recall"][0]

    for revision in args.revisions.split(","):
        t_rev = time.perf_counter()
        cfg = dataclasses.replace(
            base_cfg,
            model=dataclasses.replace(base_cfg.model, revision=revision),
            fv=dataclasses.replace(base_cfg.fv, tasks=tuple(SWEEP_TASKS)),
        )
        set_seed(cfg.seed)
        print(f"\n=== {revision} ===")
        hf_model, tokenizer, model_config, resolved = load_fv_model(cfg)
        import jlens as jlens_pkg
        model_j = jlens_pkg.from_hf(hf_model, tokenizer)
        prompts = select_prompts(cfg, tokenizer)
        skip = cfg.fit.skip_first_variants[0]

        lens = fit_lens(cfg, skip, prompts, model_j, resolved)
        record = {"revision": revision, "resolved": resolved, "tasks": {}}

        # mini-gate: capital-recall accuracy + lens-vs-logit best band HMR
        gate = score_task(model_j, tokenizer, capital_task)
        probe = probe_task(model_j, tokenizer, lens, capital_task,
                           pass_k=cfg.evals.pass_k, n_random_seeds=2)
        lo, hi = cfg.evals.band
        band = [l for l in probe["layers"] if lo <= l <= hi]
        record["mini_gate"] = {
            "capital_recall_acc": gate["accuracy"],
            "jlens_best_hmr": min(probe["metrics"]["jlens"]["per_layer"][l]["hmr"] for l in band),
            "logit_best_hmr": min(probe["metrics"]["logit"]["per_layer"][l]["hmr"] for l in band),
        }
        print(f"  mini-gate: capital acc {gate['accuracy']:.1%}, "
              f"band HMR jlens {record['mini_gate']['jlens_best_hmr']:.1f} "
              f"vs logit {record['mini_gate']['logit_best_hmr']:.1f}")

        from utils.eval_utils import n_shot_eval, n_shot_eval_no_intervention  # noqa: PLC0415
        from utils.prompt_utils import load_dataset  # noqa: PLC0415

        device = model_j.input_device
        for task in SWEEP_TASKS:
            entry = {}
            dataset = load_dataset(task, root_data_dir=str(FV_REPO / "dataset_files"), seed=cfg.seed)
            ten_shot = n_shot_eval_no_intervention(
                dataset, cfg.fv.n_shots, hf_model, model_config, tokenizer,
                compute_ppl=False, test_split="test")
            entry["exec_10shot"] = dict(ten_shot["clean_topk"])[1]
            try:
                artifacts = load_cached_fv(cfg, task, resolved) or extract_task_fvs(
                    cfg, task, hf_model, tokenizer, model_config, resolved)
            except ValueError as e:
                entry["fv"] = f"too weak: {e}"
                record["tasks"][task] = entry
                print(f"  {task}: exec {entry['exec_10shot']:.1%}, FV skipped ({e})")
                continue
            fv = artifacts["fv_todd"].to(hf_model.device)
            zs = n_shot_eval_no_intervention(dataset, 0, hf_model, model_config,
                                             tokenizer, compute_ppl=False, test_split="test")
            steered = n_shot_eval(dataset, fv.reshape(1, -1), cfg.fv.edit_layer, 0,
                                  hf_model, model_config, tokenizer)
            entry["zero_shot"] = dict(zs["clean_topk"])[1]
            entry["induction"] = dict(steered["intervention_topk"])[1]
            # decodability through this checkpoint's lens
            v = artifacts["fv_todd"].float().to(device)
            per_layer = {
                l: model_j.unembed(lens.transport(v, l)).float().cpu()
                for l in lens.source_layers
            }
            logit_only = {-1: model_j.unembed(v).float().cpu()}
            out_ids = output_token_ids(dataset, tokenizer)
            best_layer = min(per_layer, key=lambda l: float(np.mean(
                [1 + (per_layer[l] > per_layer[l][t]).sum() for t in out_ids])))
            entry["label_rank_jlens"] = label_stat(per_layer, tokenizer, TASK_LABEL_WORDS[task])
            entry["label_rank_logit"] = label_stat(logit_only, tokenizer, TASK_LABEL_WORDS[task])
            entry["outvocab_jlens"] = float(np.mean(
                [1 + (per_layer[best_layer] > per_layer[best_layer][t]).sum() for t in out_ids]))
            entry["outvocab_logit"] = float(np.mean(
                [1 + (logit_only[-1] > logit_only[-1][t]).sum() for t in out_ids]))
            record["tasks"][task] = entry
            print(f"  {task}: exec {entry['exec_10shot']:.1%}, induction "
                  f"{entry['zero_shot']:.1%}->{entry['induction']:.1%}, label "
                  f"{entry['label_rank_jlens']} (logit {entry['label_rank_logit']}), "
                  f"outvocab {entry['outvocab_jlens']:.0f} (logit {entry['outvocab_logit']:.0f})")

        record["wall_clock_s"] = round(time.perf_counter() - t_rev, 1)
        out.write(json.dumps(record) + "\n")
        out.flush()

        del hf_model, model_j, lens
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    out.close()
    print(f"\nwrote {run_dir / 'sweep.jsonl'}")


if __name__ == "__main__":
    main()
