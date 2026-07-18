"""Workstream A1: prior-corrected report protocol at Pythia-410M.

Scorer (PREREGISTRATION §A1): report_score(label, ctx) =
    log p(label | ctx + probe) − mean_neutral log p(label | neutral + probe)
with neutral contexts assembled from mixed other-task pairs, so per-label
grammatical priors cancel by construction.

Cells: 8 tasks × {coherent, shuffled-mapping, other-task} × 2 probe phrasings
× N=40 trials, plus a shared pool of 40 neutral contexts per phrasing. Every
trial logs a raw record (prompt tail, top-10 free-running tokens, label
logprobs). Pass/fail is decided by the pre-registered criteria, evaluated by
scripts/11b_report_verdict.py — this script only collects data.

Usage: uv run python scripts/11_report_protocol.py --config configs/pythia410m_phase2.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from jvec.config import Config
from jvec.evals.exp3 import REPORT_LABELS, label_token_ids
from jvec.fv import FV_REPO, load_fv_model
from jvec.utils import make_run_dir, set_seed

N_TRIALS = 40
N_NEUTRAL = 40
PROBES = {
    "P1": "\nEach answer above is the question word's",
    "P3": "\nThe rule of the list above: the answer is always the question word's",
}
CTX_TYPES = ("coherent", "shuffled", "other")


@torch.no_grad()
def final_logprobs(model_j, prompt: str) -> torch.Tensor:
    from jlens import ActivationRecorder  # noqa: PLC0415

    final = model_j.n_layers - 1
    ids = model_j.encode(prompt, max_length=1024)
    with ActivationRecorder(model_j.layers, at=[final]) as rec:
        model_j.forward(ids)
        residual = rec.activations[final][0, -1].detach()
    logits = model_j.unembed(residual.float()).float().cpu()
    return torch.log_softmax(logits, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed + 2)
    hf_model, tokenizer, model_config, revision = load_fv_model(cfg)
    import jlens as jlens_pkg
    model_j = jlens_pkg.from_hf(hf_model, tokenizer)
    bos = tokenizer.bos_token or ""
    rng = np.random.default_rng(cfg.seed + 2)

    from utils.prompt_utils import load_dataset  # noqa: PLC0415

    tasks = [t for t in cfg.fv.tasks if t in REPORT_LABELS]
    labels = label_token_ids(tokenizer)
    datasets = {
        t: load_dataset(t, root_data_dir=str(FV_REPO / "dataset_files"), seed=cfg.seed)
        for t in tasks
    }

    def pairs_of(task, n):
        ds = datasets[task]["train"]
        idx = rng.choice(len(ds), n, replace=False)
        chosen = ds[idx]
        return list(zip(chosen["input"], chosen["output"]))

    def context(pairs):
        return bos + "".join(f"Q: {x}\nA: {y}\n\n" for x, y in pairs)

    def neutral_pairs(n):
        """One pair from n distinct randomly-chosen tasks — no coherent rule."""
        chosen_tasks = rng.choice(tasks, n, replace=True)
        return [pairs_of(t, 1)[0] for t in chosen_tasks]

    run_dir = make_run_dir(cfg, "report_protocol_a1")
    out = open(run_dir / "trials.jsonl", "w")

    # --- neutral baseline pool (shared across tasks, per phrasing) --------------
    neutral_logprobs: dict[str, dict[str, list[float]]] = {}
    for pname, probe in PROBES.items():
        neutral_logprobs[pname] = {t: [] for t in tasks}
        for i in range(N_NEUTRAL):
            lp = final_logprobs(model_j, context(neutral_pairs(cfg.fv.n_shots)) + probe)
            for t in tasks:
                neutral_logprobs[pname][t].append(float(lp[labels[t]]))
            out.write(json.dumps({"kind": "neutral", "probe": pname, "trial": i,
                                  "label_logprobs": {REPORT_LABELS[t]: float(lp[labels[t]]) for t in tasks}}) + "\n")
    baselines = {
        pname: {t: sum(v) / len(v) for t, v in per_task.items()}
        for pname, per_task in neutral_logprobs.items()
    }

    # --- main cells --------------------------------------------------------------
    for pname, probe in PROBES.items():
        for task in tasks:
            for ctx_type in CTX_TYPES:
                for i in range(N_TRIALS):
                    if ctx_type == "coherent":
                        pairs = pairs_of(task, cfg.fv.n_shots)
                    elif ctx_type == "shuffled":
                        pairs = pairs_of(task, cfg.fv.n_shots)
                        ys = [y for _, y in pairs]
                        rng.shuffle(ys)
                        pairs = [(x, y) for (x, _), y in zip(pairs, ys)]
                    else:  # other-task context, scored for THIS task's label
                        offset = 1 + int(rng.integers(len(tasks) - 1))
                        other = tasks[(tasks.index(task) + offset) % len(tasks)]
                        pairs = pairs_of(other, cfg.fv.n_shots)
                    lp = final_logprobs(model_j, context(pairs) + probe)
                    top = lp.topk(10)
                    score = float(lp[labels[task]]) - baselines[pname][task]
                    out.write(json.dumps({
                        "kind": "trial", "probe": pname, "task": task,
                        "ctx_type": ctx_type, "trial": i,
                        "label": REPORT_LABELS[task],
                        "label_logprob": float(lp[labels[task]]),
                        "report_score": round(score, 4),
                        "label_vocab_rank": int(1 + (lp > lp[labels[task]]).sum()),
                        "top10": [tokenizer.decode([t]) for t in top.indices],
                    }) + "\n")

    out.close()
    (run_dir / "baselines.json").write_text(json.dumps(
        {p: {REPORT_LABELS[t]: round(v, 4) for t, v in b.items()} for p, b in baselines.items()},
        indent=1))
    print(f"wrote {run_dir / 'trials.jsonl'}")


if __name__ == "__main__":
    main()
