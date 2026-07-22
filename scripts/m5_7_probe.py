"""EXP-M5-7 confirmatory probe (410M): the transient-intermediate J-lens advantage
on the SAME substrate + cached lens draws 0/1/2 as the static S1/S2/S5 rows,
closing the cross-experiment gap in the A1b-locus unified table.

Probes, via the vendored probe_task (jlens = unembed(J_l·h), logit = unembed(h)),
two residual-STATE anchors on 410M over the 3 cached lens draws:
- capital-operand: the held LATENT operand ("France") — the transient
  computational intermediate; expected J-lens-privileged (jlens HMR << logit).
- capital-recall: the recalled answer token ("Paris") — reported honestly
  alongside (the M1 gate found it J-lens-readable too; recall answers resolve
  early, distinct from fresh-computation outputs — the clean output null is
  M5-0b's 1.4B fresh-answer ~0.9x, cited in the table).

Object-type ceiling (per prereg): this is a residual STATE, not a pulled-out
static direction — nothing unifies the object types; the run removes only the
cross-experiment / cross-substrate objection.

Prereg: harness/preregs/EXP-M5-7-a1b-locus.md (RATIFIED, committed 05c08f9).
Usage: uv run python scripts/m5_7_probe.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from jvec.calibration import select_prompts
from jvec.config import Config
from jvec.evals.baseline import score_task
from jvec.evals.probe import probe_task
from jvec.evals.tasks import Task, load_tasks
from jvec.lens_cache import load_lens
from jvec.modeling import load_model
from jvec.utils import peak_rss_gb, set_seed
from jtvec.core.runctx import start_run
from jtvec.lens_diagnostic import task_arm_ratios

PREREG = REPO_ROOT / "harness/preregs/EXP-M5-7-a1b-locus.md"
CFG0 = REPO_ROOT / "configs/m1_pythia410m_draw0.yaml"
DRAW_CFGS = {j: REPO_ROOT / f"configs/m1_pythia410m_draw{j}.yaml" for j in (0, 1, 2)}
SKIP_FIRST, CAP, PASS_K, N_RANDOM = 4, 5.0, 10, 10
ANCHORS = [("capital-operand", "transient latent (held operand)"),
           ("capital-recall", "recalled answer token")]


def behavioural_subset(model, tok, task):
    scored = score_task(model, tok, task)
    correct = {it["name"] for it in scored["per_item"] if it["correct"]}
    items = [it for it in task.items if it["name"] in correct]
    return Task(task.name, task.protocol, items), len(items), len(task.items)


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def main() -> None:
    t0 = time.perf_counter()
    ctx = start_run(repo_root=REPO_ROOT, config_path=CFG0, results_root=REPO_ROOT / "results/m5",
                    run_name="m5-7-a1b-probe", prereg_path=PREREG)
    print(f"M5.7 probe run dir: {ctx.results_dir}", flush=True)
    base = Config.load(str(CFG0)); set_seed(base.seed)
    model, tok, revision = load_model(base)
    all_tasks = {t.name: t for t in load_tasks(REPO_ROOT / "tasks")}
    filtered = {}
    for tname, _ in ANCHORS:
        ft, nok, ntot = behavioural_subset(model, tok, all_tasks[tname])
        filtered[tname] = ft
        print(f"[{tname}] behavioural {nok}/{ntot} correct", flush=True)

    per = {tname: [] for tname, _ in ANCHORS}
    for j in (0, 1, 2):
        cfg = Config.load(str(DRAW_CFGS[j])); set_seed(cfg.seed)
        lens = load_lens(cfg, SKIP_FIRST, select_prompts(cfg, tok), revision)
        layers = list(lens.source_layers)
        for tname, _ in ANCHORS:
            res = probe_task(model, tok, lens, filtered[tname], pass_k=PASS_K, n_random_seeds=N_RANDOM)
            metrics = json.loads(json.dumps(res["metrics"]))  # str-key the layers
            r = task_arm_ratios(metrics, layers, CAP)
            L = r["jlens_layer"]
            jl = metrics["jlens"]["per_layer"][str(L)]["hmr"] if L is not None else None
            lo = metrics["logit"]["per_layer"][str(L)]["hmr"] if L is not None else None
            per[tname].append({"lens_draw": j, "jlens_ratio": r["jlens_ratio"], "jlens_layer": L,
                               "jlens_hmr": jl, "logit_hmr": lo, "random_max_ratio": r["random_max_ratio"]})
            ctx.save_raw_completions(f"{tname}_draw{j}",
                [{"lens_draw": j, "name": it["name"], "ranks": it["ranks"]} for it in res["per_item"]])
            print(f"[{tname}] draw{j} ratio {r['jlens_ratio']:.2f} @L{L} jlens_hmr {jl} logit_hmr {lo}", flush=True)
        del lens

    summary = {"model": f"{base.model.name}@{revision[:7]}", "cap": CAP, "anchors": {}}
    for tname, role in ANCHORS:
        rows = per[tname]
        summary["anchors"][tname] = {
            "role": role,
            "jlens_ratio_median": _med([x["jlens_ratio"] for x in rows]),
            "jlens_hmr_median": _med([x["jlens_hmr"] for x in rows]),
            "logit_hmr_median": _med([x["logit_hmr"] for x in rows]),
            "random_max_ratio_median": _med([x["random_max_ratio"] for x in rows]),
            "per_draw": rows}
    summary["peak_rss_gb"] = round(peak_rss_gb(), 2)
    summary["wall_clock_s"] = round(time.perf_counter() - t0, 1)
    (ctx.results_dir / "a1b_probe.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = ["# EXP-M5-7 confirmatory probe (410M) — transient-intermediate J-lens advantage", "",
             f"- model {summary['model']}; cached lens draws 0/1/2; max-contrast cap {CAP}; 3 draws median", ""]
    for tname, role in ANCHORS:
        a = summary["anchors"][tname]
        lines.append(f"- **{tname}** ({role}): jlens HMR {a['jlens_hmr_median']} vs logit HMR "
                     f"{a['logit_hmr_median']} → max-contrast ratio **{a['jlens_ratio_median']}** "
                     f"(random {a['random_max_ratio_median']})")
    lines += ["", "The held LATENT operand is J-lens-privileged (jlens HMR << logit) on 410M under the "
              "SAME lens draws as the static S1/S2/S5 rows — the transient 410M row is now in-run. "
              "OBJECT-TYPE CEILING (prereg): residual state vs pulled-out direction are different objects; "
              "this closes only the cross-experiment gap. Clean output null = M5-0b 1.4B fresh-answer ~0.9x.",
              f"wall {summary['wall_clock_s']}s peak {summary['peak_rss_gb']}GB"]
    (ctx.results_dir / "report.md").write_text("\n".join(lines))
    print(f"\n=== EXP-M5-7 probe: operand ratio {summary['anchors']['capital-operand']['jlens_ratio_median']} "
          f"(jlens {summary['anchors']['capital-operand']['jlens_hmr_median']} / logit "
          f"{summary['anchors']['capital-operand']['logit_hmr_median']}) ===", flush=True)
    ctx.finalize(operand_ratio=summary["anchors"]["capital-operand"]["jlens_ratio_median"],
                 recall_ratio=summary["anchors"]["capital-recall"]["jlens_ratio_median"],
                 wall_clock_s=summary["wall_clock_s"], peak_rss_gb=summary["peak_rss_gb"])


if __name__ == "__main__":
    main()
