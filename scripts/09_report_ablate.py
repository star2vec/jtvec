"""Experiment 3: metacognitive report + double ablation.

Phase A picks the best report-probe phrasing (clean runs + shuffled-context
prior baseline). Phase B measures report accuracy and execution accuracy under
{none, jspace, sham_jspace, fv, sham_fv} ablations at the final position of
each band layer.

Prompts use the Todd Q/A template built explicitly here so the execution and
report runs share the identical ICL context (only the tail differs), with the
BOS token prepended as in Todd's neox path.

Usage: uv run python scripts/09_report_ablate.py --config configs/pythia410m_phase2.yaml
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
from jvec.evals.exp3 import (
    REPORT_LABELS,
    REPORT_PROBES,
    final_logits_under,
    label_token_ids,
    make_hooks,
)
from jvec.evals.tasks import surface_token_ids
from jvec.fv import FV_REPO, load_cached_fv, load_fv_model
from jvec.lens_cache import load_lens
from jvec.utils import make_run_dir, set_seed

N_PILOT = 12
N_REPORT = 15
N_EXEC = 30
M_TOP = 10
CONDITIONS = ("none", "jspace", "sham_jspace", "fv", "sham_fv")


def build_context(pairs, bos: str) -> str:
    return bos + "".join(f"Q: {x}\nA: {y}\n\n" for x, y in pairs)


def sample_pairs(dataset, split, rng, n):
    idx = rng.choice(len(dataset[split]), n, replace=False)
    chosen = dataset[split][idx]
    return list(zip(chosen["input"], chosen["output"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", default=None, help="comma-separated subset")
    parser.add_argument("--n-report", type=int, default=N_REPORT)
    parser.add_argument("--n-exec", type=int, default=N_EXEC)
    parser.add_argument("--probe", default=None, help="skip pilot, use this probe id")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed)
    hf_model, tokenizer, model_config, revision = load_fv_model(cfg)
    import jlens as jlens_pkg
    model_j = jlens_pkg.from_hf(hf_model, tokenizer)
    W_U = hf_model.embed_out.weight.detach().float().cpu()
    prompts = select_prompts(cfg, tokenizer)
    lens = load_lens(cfg, cfg.fit.skip_first_variants[0], prompts, revision)
    lo, hi = cfg.evals.band
    band_layers = [l for l in lens.source_layers if lo <= l <= hi]
    bos = tokenizer.bos_token or ""

    from utils.prompt_utils import load_dataset  # noqa: PLC0415

    tasks = [t for t in cfg.fv.tasks if load_cached_fv(cfg, t, revision) is not None]
    labels = label_token_ids(tokenizer)
    # Candidate set stays the FULL task set (fixed 8-way forced choice) even
    # when only a subset is being (re)measured.
    candidate_ids = [labels[t] for t in tasks]
    if args.tasks:
        subset = args.tasks.split(",")
        unknown = set(subset) - set(tasks)
        if unknown:
            sys.exit(f"unknown/uncached tasks: {sorted(unknown)}")
        tasks = subset
    n_report, n_exec = args.n_report, args.n_exec
    datasets = {
        t: load_dataset(t, root_data_dir=str(FV_REPO / "dataset_files"), seed=cfg.seed)
        for t in tasks
    }

    def report_correct(logits: torch.Tensor, task: str) -> bool:
        cand = torch.tensor([logits[i] for i in candidate_ids])
        return candidate_ids[int(cand.argmax())] == labels[task]

    # --- Phase A: pick the report probe ---------------------------------------
    rng = np.random.default_rng(cfg.seed)
    probe_scores = {}
    for name, probe in {} if args.probe else REPORT_PROBES.items():
        correct = shuffled_correct = 0
        for task in tasks:
            for _ in range(N_PILOT):
                pairs = sample_pairs(datasets[task], "train", rng, cfg.fv.n_shots)
                logits = final_logits_under(model_j, build_context(pairs, bos) + probe, {})
                correct += report_correct(logits, task)
                # shuffled outputs -> no coherent task: label-prior baseline
                ys = [y for _, y in pairs]
                rng.shuffle(ys)
                shuffled = [(x, y) for (x, _), y in zip(pairs, ys)]
                logits_s = final_logits_under(model_j, build_context(shuffled, bos) + probe, {})
                shuffled_correct += report_correct(logits_s, task)
        n = len(tasks) * N_PILOT
        probe_scores[name] = {"acc": correct / n, "shuffled_acc": shuffled_correct / n}
        print(f"probe {name}: report acc {correct / n:.1%} (shuffled-context {shuffled_correct / n:.1%}, chance {1 / len(tasks):.1%})")

    best_probe = args.probe or max(probe_scores, key=lambda p: probe_scores[p]["acc"])
    probe = REPORT_PROBES[best_probe]
    print(f"using probe {best_probe}")

    # --- Phase B: report + execution under ablations ---------------------------
    results = {"probe": best_probe, "probe_scores": probe_scores, "m_top": M_TOP,
               "band_layers": band_layers, "tasks": {}}
    for task in tasks:
        fv = load_cached_fv(cfg, task, revision)["fv_todd"]
        entry = {"report": {}, "execution": {}}
        for condition in CONDITIONS:
            hooks = make_hooks(condition, band_layers, lens, W_U, fv,
                               m_top=M_TOP, seed=cfg.seed)
            rep = 0
            for _ in range(n_report):
                pairs = sample_pairs(datasets[task], "train", rng, cfg.fv.n_shots)
                logits = final_logits_under(model_j, build_context(pairs, bos) + probe, hooks)
                rep += report_correct(logits, task)
            ex = 0
            for _ in range(n_exec):
                pairs = sample_pairs(datasets[task], "train", rng, cfg.fv.n_shots)
                qx, qy = sample_pairs(datasets[task], "test", rng, 1)[0]
                prompt = build_context(pairs, bos) + f"Q: {qx}\nA:"
                logits = final_logits_under(model_j, prompt, hooks)
                ex += int(logits.argmax()) in surface_token_ids(tokenizer, str(qy))
            entry["report"][condition] = rep / n_report
            entry["execution"][condition] = ex / n_exec
        results["tasks"][task] = entry
        r, e = entry["report"], entry["execution"]
        print(f"{task:18s} report: none {r['none']:.0%} jspace {r['jspace']:.0%} "
              f"(sham {r['sham_jspace']:.0%}) fv {r['fv']:.0%} (sham {r['sham_fv']:.0%}) | "
              f"exec: none {e['none']:.0%} jspace {e['jspace']:.0%} "
              f"(sham {e['sham_jspace']:.0%}) fv {e['fv']:.0%} (sham {e['sham_fv']:.0%})")

    run_dir = make_run_dir(cfg, "exp3_report_ablate")
    (run_dir / "exp3.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {run_dir / 'exp3.json'}")


if __name__ == "__main__":
    main()
