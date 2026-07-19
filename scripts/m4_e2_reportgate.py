"""M4-E2 prerequisite gate: re-control a report instrument on singular-plural.

Rebuilds the singular-plural report measure withdrawn at D-013 as v1's
prior-corrected report SCORE and gates it with a positive control (coherent
context elevated above the neutral prior) and a negative control (the D-013
discriminator: coherent >> shuffled, where shuffled keeps the singular-noun
inputs but scrambles the mapping; and coherent >> other). Model-only forward
passes — no lens, no FV. If the instrument gates, E2's verbalization measure
on singular-plural exists; if not, E2's dissociation there is blocked and
that is the reported result (instruments LAW).

Prereg: harness/preregs/EXP-M4-E2-reportgate.md (committed before first run;
start_run enforces).

Usage: uv run python scripts/m4_e2_reportgate.py [--config configs/m4_e2_reportgate_pythia410m.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from jlens import ActivationRecorder

from jvec.config import Config
from jvec.evals.exp3 import REPORT_LABELS, REPORT_PROBES
from jvec.fv import FV_REPO  # noqa: F401  (sys.path side-effect for the Todd repo)
from jvec.modeling import load_model
from jvec.utils import peak_rss_gb, set_seed
from jtvec.core.instruments import ControlRecord, Instrument, require_controlled
from jtvec.core.reporting import scoped
from jtvec.core.runctx import start_run
from jtvec.report_instruments import (
    ReportScoreControlRule,
    instrument_name,
)

PREREG = REPO_ROOT / "harness/preregs/EXP-M4-E2-reportgate.md"

TASK = "singular-plural"
N_TRIALS = 40
N_NEUTRAL = 40
CTX_RNG_SEED = 7070  # all context sampling (preregistered)
RULE = ReportScoreControlRule(n_boot=10_000, boot_seed=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs/m4_e2_reportgate_pythia410m.yaml")
    )
    args = parser.parse_args()
    t_start = time.perf_counter()

    cfg = Config.load(args.config)
    ctx = start_run(
        repo_root=REPO_ROOT,
        config_path=Path(args.config),
        results_root=REPO_ROOT / cfg.results_dir,
        run_name="e2-reportgate",
        prereg_path=PREREG,
    )
    print(f"E2 report-gate run dir: {ctx.results_dir}", flush=True)

    set_seed(cfg.seed)
    model_j, tokenizer, revision = load_model(cfg)
    model_scope = f"{cfg.model.name}@{revision[:7]}"
    config_scope = f"EXP-M4-E2-reportgate ({Path(args.config).name})"
    bos = tokenizer.bos_token or ""
    rng = np.random.default_rng(CTX_RNG_SEED)
    label_id = tokenizer(" " + REPORT_LABELS[TASK], add_special_tokens=False).input_ids[0]

    other_tasks = [t for t in cfg.fv.tasks if t != TASK]
    if not other_tasks:
        sys.exit("no other certified tasks for the neutral/other pools")

    from utils.prompt_utils import load_dataset  # noqa: PLC0415

    datasets = {
        t: load_dataset(t, root_data_dir=str(FV_REPO / "dataset_files"), seed=cfg.seed)
        for t in cfg.fv.tasks
    }

    final = model_j.n_layers - 1

    @torch.no_grad()
    def label_logprob(prompt: str) -> tuple[float, list[str], int]:
        ids = model_j.encode(prompt, max_length=1024)
        with ActivationRecorder(model_j.layers, at=[final]) as rec:
            model_j.forward(ids)
            residual = rec.activations[final][0, -1].detach()
        logits = model_j.unembed(residual.float()).float().cpu()
        lp = torch.log_softmax(logits, dim=-1)
        top = [tokenizer.decode([t]) for t in logits.topk(10).indices]
        rank = int(1 + (lp > lp[label_id]).sum())
        return float(lp[label_id]), top, rank

    def pairs_of(task: str, n: int):
        ds = datasets[task]["train"]
        idx = rng.choice(len(ds), n, replace=False)
        chosen = ds[idx]
        return list(zip(chosen["input"], chosen["output"]))

    def context(pairs) -> str:
        return bos + "".join(f"Q: {x}\nA: {y}\n\n" for x, y in pairs)

    def neutral_pairs(n: int):
        """One pair from each of n randomly-chosen other tasks — no coherent rule."""
        chosen = rng.choice(other_tasks, n, replace=True)
        return [pairs_of(t, 1)[0] for t in chosen]

    n_shots = cfg.fv.n_shots

    # --- neutral baseline pool (per phrasing): the "plural" prior under a
    #     no-coherent-rule ICL context; subtracted from every scored cell. ---
    baselines: dict[str, float] = {}
    for pname, probe in REPORT_PROBES.items():
        vals, rows = [], []
        for i in range(N_NEUTRAL):
            lp, top, rank = label_logprob(context(neutral_pairs(n_shots)) + probe)
            vals.append(lp)
            rows.append({"probe": pname, "trial": i, "label_logprob": round(lp, 4),
                         "label_rank": rank, "top10": top})
        ctx.save_raw_completions(f"neutral_{pname}", rows)
        baselines[pname] = float(np.mean(vals))
    print(f"[neutral] baselines log p(' plural'): "
          f"{ {p: round(b, 3) for p, b in baselines.items()} }", flush=True)

    # --- scored cells: coherent / shuffled / other, per phrasing ---------------
    def scored_cell(ctx_type: str, pname: str, probe: str) -> list[float]:
        scores, rows = [], []
        for i in range(N_TRIALS):
            if ctx_type == "coherent":
                pairs = pairs_of(TASK, n_shots)
            elif ctx_type == "shuffled":
                pairs = pairs_of(TASK, n_shots)
                ys = [y for _, y in pairs]
                rng.shuffle(ys)
                pairs = [(x, y) for (x, _), y in zip(pairs, ys)]
            else:  # other: a different task's context, scored for "plural"
                other = other_tasks[int(rng.integers(len(other_tasks)))]
                pairs = pairs_of(other, n_shots)
            lp, top, rank = label_logprob(context(pairs) + probe)
            score = lp - baselines[pname]
            scores.append(score)
            rows.append({"probe": pname, "ctx_type": ctx_type, "trial": i,
                         "label_logprob": round(lp, 4), "report_score": round(score, 4),
                         "label_rank": rank, "top10": top})
        ctx.save_raw_completions(f"report_{ctx_type}_{pname}", rows)
        return scores

    per_phrasing = {}
    coherent_ranks: dict[str, list[int]] = {}
    for pname, probe in REPORT_PROBES.items():
        coherent = scored_cell("coherent", pname, probe)
        shuffled = scored_cell("shuffled", pname, probe)
        other = scored_cell("other", pname, probe)
        per_phrasing[pname] = RULE.evaluate_phrasing(
            coherent=coherent, shuffled=shuffled, other=other
        )
        r = per_phrasing[pname]
        print(f"[{pname}] coherent {r['coherent_ci'][0]:+.2f} "
              f"[{r['coherent_ci'][1]:+.2f},{r['coherent_ci'][2]:+.2f}] | "
              f"shuffled {r['shuffled_ci'][0]:+.2f} "
              f"[{r['shuffled_ci'][1]:+.2f},{r['shuffled_ci'][2]:+.2f}] | "
              f"other {r['other_ci'][0]:+.2f} "
              f"[{r['other_ci'][1]:+.2f},{r['other_ci'][2]:+.2f}] -> "
              f"pos {r['positive_pass']} neg {r['negative_pass']}", flush=True)

    verdict = RULE.verdict(per_phrasing)

    # --- ControlRecord + instrument (instruments LAW) --------------------------
    today = time.strftime("%Y-%m-%d")
    run = str(ctx.results_dir)
    name = instrument_name(TASK)
    inst = Instrument(
        name=name,
        positive_control=ControlRecord(run=run, passed=verdict["positive"], date=today),
        negative_control=ControlRecord(run=run, passed=verdict["negative"], date=today),
    )
    gated = inst.is_controlled() and verdict["gated"]
    if gated:
        require_controlled(inst)  # sanity: constructible AND passing

    (ctx.results_dir / "controls.json").write_text(
        json.dumps(
            {name: {"positive": verdict["positive"], "negative": verdict["negative"],
                    "gated": gated, "best_phrasing": verdict["best_phrasing"],
                    "run": run, "date": today, "per_phrasing": verdict["per_phrasing"]}},
            indent=2, default=str,
        ),
        encoding="utf-8",
    )

    # --- report ----------------------------------------------------------------
    lines = [
        "# EXP-M4-E2 report-gate: prior-corrected report instrument on singular-plural",
        "",
        f"- model: {model_scope} (full sha in run.json/config)",
        "- prereg: harness/preregs/EXP-M4-E2-reportgate.md",
        f"- instrument: {name} (rebuild of the D-013-withdrawn "
        "report-probe-forced-choice@singular-plural, under a new name)",
        f"- report_score(ctx) = log p(' plural' | ctx+probe) - neutral baseline; "
        f"context rng {CTX_RNG_SEED}; N={N_TRIALS}/cell; neutral pool N={N_NEUTRAL}",
        "",
        "## Control arms (bootstrap 95% CIs; gated iff one phrasing does both)",
        "",
        "| phrasing | coherent | shuffled | other | positive | negative |",
        "|---|---|---|---|---|---|",
    ]
    for pname, r in per_phrasing.items():
        fmt = lambda c: f"{c[0]:+.2f} [{c[1]:+.2f},{c[2]:+.2f}]"
        lines.append(
            f"| {pname} | {fmt(r['coherent_ci'])} | {fmt(r['shuffled_ci'])} | "
            f"{fmt(r['other_ci'])} | {'PASS' if r['positive_pass'] else 'fail'} | "
            f"{'PASS' if r['negative_pass'] else 'fail'} |"
        )
    lines += ["", "## Verdict", ""]
    best = verdict["best_phrasing"]
    if gated:
        r = per_phrasing[best]
        lines.append("- " + scoped(
            f"report instrument GATED on {TASK} under {best}: coherent report_score "
            f"{r['coherent_ci'][0]:+.2f} [{r['coherent_ci'][1]:+.2f},"
            f"{r['coherent_ci'][2]:+.2f}] vs shuffled CI-high {r['shuffled_ci'][2]:+.2f} "
            f"and other CI-high {r['other_ci'][2]:+.2f} (coherent CI-low "
            f"{r['coherent_ci'][1]:+.2f} clears both)",
            float(r["coherent_ci"][0]), model=model_scope, config=config_scope,
            n=N_TRIALS,
        ))
    else:
        lines.append(f"- report instrument NOT gated on {TASK}: no phrasing shows the "
                     "coherent reading elevated above prior AND specific to the mapping "
                     "vs shuffled+other. Under Path A this blocks E2's verbalization "
                     "measure on this task (instruments LAW); disposition to Ecaterina.")
    wall_s = round(time.perf_counter() - t_start, 1)
    lines += [
        "",
        f"**report-gate verdict: {name} {'GATED' if gated else 'NOT gated'}**",
        "",
        f"wall-clock {wall_s} s; peak RSS {peak_rss_gb():.2f} GB; device {cfg.device}; "
        "ControlRecord in controls.json; raw per-item cells under raw_completions/.",
        "",
    ]
    (ctx.results_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    ctx.finalize(
        instrument=name,
        gated=gated,
        best_phrasing=best,
        model_revision=revision,
        wall_clock_s=wall_s,
        peak_rss_gb=round(peak_rss_gb(), 2),
    )
    print(f"\nreport-gate verdict: {name} {'GATED' if gated else 'NOT gated'} "
          f"(best phrasing {best})")
    print(f"report: {ctx.results_dir / 'report.md'}")


if __name__ == "__main__":
    main()
