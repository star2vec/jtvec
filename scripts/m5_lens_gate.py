"""EXP-M5-0 rule 5: M1-style lens gate for a NEW substrate (Pythia-1.4B).

Generalizes scripts/m1_gate.py off the 410M/v1-reproduction anchors. The M1
gate checked six rules, three of which only make sense as a bit-for-bit v1
reproduction (R3 capital-recall exact contrast, R4 calibration-hash identity,
R5 the v1 baseline table). A fresh substrate has no v1 reference, so this gate
keeps only the model-agnostic content as the preregistered Q1-Q6:

  Q1  every draw passes the vendored 9-check sanity gate (report verdict)
  Q2  positive control: median dp(swap_answer) >= 0.30, median flip >= 0.75
  Q3  sham: median |dp(swap_answer_random_ctrl)| <= max(0.03, 1/N)
  Q4  negative control: the 10-seed random-matrix lens arm never beats the
      jlens band-min HMR on any anchor task, AND the random-direction swap
      arm clears the Q3 bound (the two matched-noise arms)
  Q5  probing contrast: on >= 2 anchor tasks, band-min jlens HMR <= 5.0 with
      logit HMR >= 5x the jlens HMR at that layer
  Q6  draw stability: IQR over draws of dp(swap_answer) <= 0.05 AND of the
      band-min jlens HMR (per anchor task) <= 0.5

skip4-only, 3 draws (D-020... no — compute ruling 2026-07-20). The pipeline
(vendored scripts 01-04) is unchanged and already validated on this model's
one-prompt fit probe; only the orchestration + verdict are new. evaluate_gate
is a pure function so the verdict logic has a model-free landing test
(tests/test_m5_lens_gate.py).

Prereg: harness/preregs/EXP-M5-0-qualification.md (committed 113d04f).
Usage: uv run python scripts/m5_lens_gate.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from jvec.config import Config
from jtvec.core.runctx import start_run

DRAWS = (0, 1, 2)
CONFIGS = {k: REPO_ROOT / f"configs/m5_lens_pythia1p4b_draw{k}.yaml" for k in DRAWS}
PREREG = REPO_ROOT / "harness/preregs/EXP-M5-0-qualification.md"
VARIANT = "skip4"  # skip4-only (compute ruling 2026-07-20)

# Preregistered Q-rule constants (EXP-M5-0 rule 5).
Q2_DP_MIN = 0.30
Q2_FLIP_MIN = 0.75
Q3_ABS_BASE = 0.03
Q5_JLENS_HMR_MAX = 5.0
Q5_LOGIT_RATIO = 5.0
Q5_MIN_TASKS = 2
Q6_DP_IQR_MAX = 0.05
Q6_HMR_IQR_MAX = 0.5


def run_stage(script: str, config: Path, *extra: str) -> None:
    cmd = [sys.executable, f"scripts/{script}", "--config", str(config), *extra]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def latest(results_dir: str, experiment: str, filename: str) -> Path:
    hits = sorted((REPO_ROOT / results_dir / experiment).glob(f"*/{filename}"))
    if not hits:
        raise FileNotFoundError(f"no {experiment}/{filename} under {results_dir}")
    return hits[-1]


def iqr(values: list[float]) -> float:
    a = np.asarray(values, dtype=float)
    return float(np.percentile(a, 75) - np.percentile(a, 25))


def band_min_jlens(probe_task: dict, band: range) -> tuple[int, float]:
    """(layer, hmr) of the jlens band-minimum HMR for one task."""
    per_layer = probe_task["metrics"]["jlens"]["per_layer"]
    layer = min(band, key=lambda l: per_layer[str(l)]["hmr"])
    return layer, per_layer[str(layer)]["hmr"]


def random_arms_beat_bandmin(probe_task: dict, band: range, bandmin_hmr: float) -> bool:
    """True if ANY 10-seed random-matrix arm reaches a band HMR <= the jlens
    band-min on this task (a negative-control breach)."""
    metrics = probe_task["metrics"]
    for arm, per in metrics.items():
        if not arm.startswith("random-"):
            continue
        arm_min = min(per["per_layer"][str(l)]["hmr"] for l in band)
        if arm_min <= bandmin_hmr:
            return True
    return False


def evaluate_gate(draws: dict, band: range) -> tuple[dict, bool]:
    """Pure Q1-Q6 verdict over the per-draw eval payloads. `draws[k]` has keys
    'verdict_pass' (bool), 'swap' (swap.json[variant][swap-task]), and 'probe'
    (probe.json[variant]: {task: {...}}). Model-free; landing-tested."""
    ks = sorted(draws)
    swaps = [draws[k]["swap"]["metrics"] for k in ks]
    n_swap = min(int(s["n_scored"]) for s in swaps)
    sham_bound = max(Q3_ABS_BASE, 1.0 / n_swap)

    dp = [s["mean_dp_swap_answer"] for s in swaps]
    flip = [s["swap_top1_rate"] for s in swaps]
    sham = [abs(s["mean_dp_swap_answer_random_ctrl"]) for s in swaps]

    # anchor tasks present in every draw's probe payload
    anchor_tasks = sorted(set.intersection(*[set(draws[k]["probe"]) for k in ks]))

    # Q5: per task (using draw 0), band-min jlens HMR with the logit contrast
    q5_tasks = []
    for task in anchor_tasks:
        layer, hmr = band_min_jlens(draws[ks[0]]["probe"][task], band)
        logit_hmr = draws[ks[0]]["probe"][task]["metrics"]["logit"]["per_layer"][str(layer)]["hmr"]
        if hmr <= Q5_JLENS_HMR_MAX and logit_hmr >= Q5_LOGIT_RATIO * hmr:
            q5_tasks.append(task)

    # Q4: no random-matrix arm beats the jlens band-min on any anchor task
    q4_random_clean = True
    for task in anchor_tasks:
        _, hmr = band_min_jlens(draws[ks[0]]["probe"][task], band)
        if random_arms_beat_bandmin(draws[ks[0]]["probe"][task], band, hmr):
            q4_random_clean = False

    # Q6: per-anchor-task band-min jlens HMR IQR over draws
    hmr_iqrs = {}
    for task in anchor_tasks:
        per_draw = [band_min_jlens(draws[k]["probe"][task], band)[1] for k in ks]
        hmr_iqrs[task] = iqr(per_draw)

    rules = {
        "Q1_all_draws_gate_pass": all(draws[k]["verdict_pass"] for k in ks),
        "Q2_positive_control": (
            float(np.median(dp)) >= Q2_DP_MIN and float(np.median(flip)) >= Q2_FLIP_MIN
        ),
        "Q3_sham": float(np.median(sham)) <= sham_bound,
        "Q4_negative_control": q4_random_clean and float(np.median(sham)) <= sham_bound,
        "Q5_probing_contrast": len(q5_tasks) >= Q5_MIN_TASKS,
        "Q6_draw_stability": (
            iqr(dp) <= Q6_DP_IQR_MAX
            and (max(hmr_iqrs.values()) <= Q6_HMR_IQR_MAX if hmr_iqrs else False)
        ),
    }
    diagnostics = {
        "n_swap": n_swap,
        "sham_bound": round(sham_bound, 4),
        "dp_median": round(float(np.median(dp)), 4),
        "dp_iqr": round(iqr(dp), 4),
        "flip_median": round(float(np.median(flip)), 4),
        "sham_median": round(float(np.median(sham)), 4),
        "anchor_tasks": anchor_tasks,
        "q5_passing_tasks": q5_tasks,
        "hmr_iqr_by_task": {t: round(v, 4) for t, v in hmr_iqrs.items()},
    }
    return {"rules": rules, "diagnostics": diagnostics}, all(rules.values())


def main() -> None:
    cfgs = {k: Config.load(str(CONFIGS[k])) for k in DRAWS}
    band = range(cfgs[0].evals.band[0], cfgs[0].evals.band[1] + 1)

    ctx = start_run(
        repo_root=REPO_ROOT,
        config_path=CONFIGS[0],
        results_root=REPO_ROOT / "results/m5",
        run_name="p14b-lens-gate",
        prereg_path=PREREG,
    )
    print(f"M5.0 1.4B lens-gate run dir: {ctx.results_dir}", flush=True)
    for k in (1, 2):
        shutil.copy2(CONFIGS[k], ctx.results_dir / CONFIGS[k].name)

    # --- pipeline: baselines once (raw model, no lens), then fit -> evals ->
    # report PER DRAW so a draw-0 eval failure surfaces at ~hour 2 of an
    # unattended run rather than after all three fits (~hour 5). ---
    run_stage("02_task_baselines.py", CONFIGS[0])
    baselines_path = latest(cfgs[0].results_dir, "task_baselines", "baselines.json")
    for k in DRAWS:
        run_stage("01_fit_lens.py", CONFIGS[k])
        run_stage("03_run_evals.py", CONFIGS[k], "--baselines", str(baselines_path))
        evals_dir = latest(cfgs[k].results_dir, "lens_evals", "probe.json").parent
        run_stage(
            "04_report.py", CONFIGS[k],
            "--evals", str(evals_dir), "--baselines", str(baselines_path),
        )

    # --- collect per-draw outputs ---
    draws = {}
    for k in DRAWS:
        evals_dir = latest(cfgs[k].results_dir, "lens_evals", "probe.json").parent
        report_path = latest(cfgs[k].results_dir, "phase1_report", "report.md")
        probe = json.loads((evals_dir / "probe.json").read_text())[VARIANT]
        swap_all = json.loads((evals_dir / "swap.json").read_text())[VARIANT]
        swap_task = next(iter(swap_all))  # the swap task (swap-capitals if included)
        draws[k] = {
            "verdict_pass": f"### {VARIANT}: **PASS**" in report_path.read_text(),
            "probe": probe,
            "swap": swap_all[swap_task],
            "swap_task": swap_task,
            "report_path": report_path,
            "evals_dir": evals_dir,
        }
        dest = ctx.results_dir / "draws" / f"draw{k}"
        dest.mkdir(parents=True)
        shutil.copy2(report_path, dest / "report.md")
        shutil.copy2(evals_dir / "probe.json", dest / "probe.json")
        shutil.copy2(evals_dir / "swap.json", dest / "swap.json")
    baselines = json.loads(baselines_path.read_text())
    shutil.copy2(baselines_path, ctx.results_dir / "draws" / "baselines.json")

    # --- raw per-item records (LAW: retained for every reported number) ---
    for k in DRAWS:
        ctx.save_raw_completions(
            f"{draws[k]['swap_task']}_dp",
            [{"draw": k, "seed": k, **item} for item in draws[k]["swap"]["per_item"]],
        )
        for task, result in draws[k]["probe"].items():
            ctx.save_raw_completions(
                f"{task}_bandmin_rank",
                [
                    {
                        "draw": k, "seed": k, "name": item["name"],
                        "bandmin_hmr_jlens": min(
                            item["ranks"]["jlens"][str(l)] for l in band
                        ) if "ranks" in item else None,
                        "ranks": item.get("ranks"),
                    }
                    for item in result.get("per_item", [])
                ],
            )
    ctx.save_raw_completions(
        "task-baselines",
        [{"task": name, **item} for name, b in baselines.items() for item in b["per_item"]],
    )

    # --- Q1-Q6 verdict (prereg EXP-M5-0 rule 5) ---
    verdict, all_pass = evaluate_gate(draws, band)
    verdict["variant"] = VARIANT
    verdict["model"] = f"{cfgs[0].model.name}@{cfgs[0].model.revision[:7]}"
    verdict["band"] = [band.start, band.stop - 1]
    verdict["m5_0_lens_verdict"] = "PASS" if all_pass else "FAIL"
    (ctx.results_dir / "lens_gate_verdict.json").write_text(json.dumps(verdict, indent=2))

    print("\n=== EXP-M5-0 1.4B lens gate ===", flush=True)
    for name, ok in verdict["rules"].items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"  diagnostics: {json.dumps(verdict['diagnostics'])}")
    print(f"\n  m5_0_lens_verdict: {verdict['m5_0_lens_verdict']}")
    print(f"  run dir: {ctx.results_dir}")
    ctx.finalize(m5_0_lens_verdict=verdict["m5_0_lens_verdict"], rules=verdict["rules"])


if __name__ == "__main__":
    main()
