"""M1 orchestrator: reproduce the v1 phase-1 lens gate on Pythia-410M.

Runs the vendored v1 pipeline (scripts 01-04) for three independent lens
draws (seeds 0/1/2 -> re-sampled calibration prompts), then evaluates the
preregistered decision rules R1-R6 against the v1 reference numbers and
packages a LAW-conformant M1 results directory (prereg-gated via start_run,
config copies, raw per-item records, median/IQR over draws).

Prereg: harness/preregs/EXP-M1-lens-gate.md (committed before first run).
v1 reference: ~/Developer/jvec-outdated results/phase1_report/20260714-051555
(numbers inlined below so the check does not depend on the v1 checkout).

Usage: uv run python scripts/m1_gate.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from jtvec.core.draws import DrawSet
from jtvec.core.reporting import scoped, scoped_intervention
from jtvec.core.runctx import start_run

DRAWS = (0, 1, 2)
CONFIGS = {k: REPO_ROOT / f"configs/m1_pythia410m_draw{k}.yaml" for k in DRAWS}
PREREG = REPO_ROOT / "harness/preregs/EXP-M1-lens-gate.md"
MODEL_REV = "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"
MODEL_SCOPE = f"EleutherAI/pythia-410m@{MODEL_REV[:7]}"
BAND = range(4, 17)

# v1 reference: results/phase1_report/20260714-051555 (report.md) and the
# skip4_n10 lens manifest. Tolerances live in the prereg, not here.
V1 = {
    "swap_dp": 0.6046,
    "swap_random": 0.0086,
    "swap_flip_rate": 0.875,
    "baselines": {
        "capital-operand": (0.861, True),
        "capital-recall": (0.861, True),
        "context-binding": (0.533, False),
        "multihop-scaled": (0.500, False),
        "opposites": (1.000, True),
        "swap-capitals": (0.938, True),
        "typo-robustness": (0.700, False),
        "word-pairs": (0.917, True),
    },
    "bandmin_jlens_hmr": {
        "capital-operand": 2.6,
        "capital-recall": 2.5,
        "opposites": 1.3,
        "word-pairs": 2.7,
    },
    "calibration_sha256": [
        "08762a81a3f9f103ee1e7306781d2cfb7c01d962f5cb3ba361ee09386f5b83a3",
        "4ca7482ef2e5115380d84c11997de40e2677a418a8e8ebe30cecbcf3631c87d6",
        "c9431a8e3ddeb746a33db4b7105ba9d46e424e9ecfe6238710cfb1b1e231bf31",
        "be83dba02f30ad74c38f72f479560e0b0419ceaa2c0421e09fb19fcf51ecfbc3",
        "f845b3b9380617957760792b8a26ae8218c1fe09782377f878797efdbeb1d358",
        "51db83146aaa3e75c985167f8b52a255ff43a40b03e9fdc14d18a085f965cb91",
        "ecc78a4119870fe7781027c999c1c4969bb7941ef2ff39c594eb6770dd1601d7",
        "f6714e1b6310d821eeed89ca90db20e7fd57e4ac66e0b6eb3e5be59c3439d08b",
        "6602ba5747dfa15afd75111da48e19f2a99b965b3ce4f53c4cdf9543e2ba3d0b",
        "4bfd4f3a447c026842f7800e913f207ac08d8c1d97ee21988630fa08e703a15b",
    ],
}


def run_stage(script: str, config: Path, *extra: str) -> None:
    cmd = [sys.executable, f"scripts/{script}", "--config", str(config), *extra]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def latest(results_dir: str, experiment: str, filename: str) -> Path:
    hits = sorted((REPO_ROOT / results_dir / experiment).glob(f"*/{filename}"))
    if not hits:
        raise FileNotFoundError(f"no {experiment}/{filename} under {results_dir}")
    return hits[-1]


def draw_results_dir(k: int) -> str:
    return f"results/m1/draw{k}"


def manifest_path(k: int) -> Path:
    return (
        REPO_ROOT
        / f"cache/draw{k}/lenses/EleutherAI/pythia-410m@{MODEL_REV}/skip4_n10/manifest.json"
    )


def bandmin(per_layer: dict) -> tuple[int, float]:
    """(layer, hmr) of the band minimum; JSON layer keys are strings."""
    layer = min(BAND, key=lambda l: per_layer[str(l)]["hmr"])
    return layer, per_layer[str(layer)]["hmr"]


def main() -> None:
    ctx = start_run(
        repo_root=REPO_ROOT,
        config_path=CONFIGS[0],
        results_root=REPO_ROOT / "results/m1",
        run_name="lens-gate",
        prereg_path=PREREG,
    )
    print(f"M1 run dir: {ctx.results_dir}", flush=True)
    for k in (1, 2):
        shutil.copy2(CONFIGS[k], ctx.results_dir / CONFIGS[k].name)

    # --- pipeline: fit -> (baselines once) -> evals -> report, per draw ---
    for k in DRAWS:
        run_stage("01_fit_lens.py", CONFIGS[k])
    run_stage("02_task_baselines.py", CONFIGS[0])
    baselines_path = latest(draw_results_dir(0), "task_baselines", "baselines.json")
    for k in DRAWS:
        run_stage("03_run_evals.py", CONFIGS[k], "--baselines", str(baselines_path))
        evals_dir = latest(draw_results_dir(k), "lens_evals", "probe.json").parent
        run_stage(
            "04_report.py", CONFIGS[k],
            "--evals", str(evals_dir), "--baselines", str(baselines_path),
        )

    # --- collect per-draw outputs ---
    draws = {}
    for k in DRAWS:
        evals_dir = latest(draw_results_dir(k), "lens_evals", "probe.json").parent
        report_path = latest(draw_results_dir(k), "phase1_report", "report.md")
        probe = json.loads((evals_dir / "probe.json").read_text())["skip4"]
        swap = json.loads((evals_dir / "swap.json").read_text())["skip4"]["swap-capitals"]
        manifest = json.loads(manifest_path(k).read_text())
        draws[k] = {
            "verdict_pass": "### skip4: **PASS**" in report_path.read_text(),
            "probe": probe,
            "swap": swap,
            "manifest": manifest,
            "report_path": report_path,
            "evals_dir": evals_dir,
        }
        dest = ctx.results_dir / "draws" / f"draw{k}"
        dest.mkdir(parents=True)
        shutil.copy2(report_path, dest / "report.md")
        shutil.copy2(evals_dir / "probe.json", dest / "probe.json")
        shutil.copy2(evals_dir / "swap.json", dest / "swap.json")
        shutil.copy2(manifest_path(k), dest / "manifest.json")
    baselines = json.loads(baselines_path.read_text())
    shutil.copy2(baselines_path, ctx.results_dir / "draws" / "baselines.json")

    # --- raw per-item records (LAW: retained for every reported number) ---
    for k in DRAWS:
        ctx.save_raw_completions(
            "swap-capitals_dp",
            [
                {"draw": k, "seed": k, **item}
                for item in draws[k]["swap"]["per_item"]
            ],
        )
        for task, result in draws[k]["probe"].items():
            ctx.save_raw_completions(
                f"{task}_bandmin_rank",
                [
                    {
                        "draw": k,
                        "seed": k,
                        "name": item["name"],
                        "bandmin_rank_jlens": min(
                            item["ranks"]["jlens"][str(l)] for l in BAND
                        ),
                        "bandmin_rank_logit": min(
                            item["ranks"]["logit"][str(l)] for l in BAND
                        ),
                        "ranks": item["ranks"],
                    }
                    for item in result["per_item"]
                ],
            )
    ctx.save_raw_completions(
        "task-baselines",
        [
            {"task": name, **item}
            for name, b in baselines.items()
            for item in b["per_item"]
        ],
    )

    # --- decision rules (tolerances: prereg EXP-M1, Decision rule section) ---
    d0 = draws[0]
    m0 = d0["swap"]["metrics"]
    cr_layer, cr_hmr = bandmin(d0["probe"]["capital-recall"]["metrics"]["jlens"]["per_layer"])
    cr_logit_hmr = d0["probe"]["capital-recall"]["metrics"]["logit"]["per_layer"][str(cr_layer)]["hmr"]

    rules = {
        "R1_draw0_gate_pass": d0["verdict_pass"],
        "R2_swap": (
            0.55 <= m0["mean_dp_swap_answer"] <= 0.66
            and abs(m0["mean_dp_swap_answer_random_ctrl"]) <= 0.03
            and m0["swap_top1_rate"] >= 0.75
        ),
        "R3_capital_recall_contrast": (
            cr_hmr <= 5.0 and cr_logit_hmr >= 5 * cr_hmr
        ),
        "R4_calibration_hashes_exact": (
            d0["manifest"]["calibration_sha256"] == V1["calibration_sha256"]
        ),
        "R5_baselines": all(
            abs(baselines[t]["accuracy"] - acc) <= 0.03
            and baselines[t]["included"] == inc
            for t, (acc, inc) in V1["baselines"].items()
        ),
        "R6_draws_stable": all(draws[k]["verdict_pass"] for k in (1, 2)),
    }
    all_pass = all(rules.values())

    # --- median/IQR over draws for every headline number ---
    seeds = tuple(DRAWS)
    swap_ds = DrawSet(
        values=tuple(draws[k]["swap"]["metrics"]["mean_dp_swap_answer"] for k in DRAWS),
        seeds=seeds,
    )
    sham_ds = DrawSet(
        values=tuple(
            draws[k]["swap"]["metrics"]["mean_dp_swap_answer_random_ctrl"] for k in DRAWS
        ),
        seeds=seeds,
    )
    flip_ds = DrawSet(
        values=tuple(draws[k]["swap"]["metrics"]["swap_top1_rate"] for k in DRAWS),
        seeds=seeds,
    )
    hmr_ds = {
        task: DrawSet(
            values=tuple(
                bandmin(draws[k]["probe"][task]["metrics"]["jlens"]["per_layer"])[1]
                for k in DRAWS
            ),
            seeds=seeds,
        )
        for task in V1["bandmin_jlens_hmr"]
    }

    # --- M1 report ---
    n_swap = draws[0]["swap"]["metrics"]["n_scored"]
    config_scope = "skip4_n10 (configs/m1_pythia410m_draw*.yaml)"
    lines = [
        "# M1 report: lens port + 9-check gate reproduction",
        "",
        f"- model: {MODEL_SCOPE} (full sha in run.json/configs)",
        "- prereg: harness/preregs/EXP-M1-lens-gate.md",
        "- v1 reference: jvec-outdated results/phase1_report/20260714-051555",
        f"- draws: lens fits at seeds {list(DRAWS)} (independently re-sampled "
        "calibration prompts); task baselines are deterministic and shared",
        "- controls: per-layer 10-seed Frobenius-matched random-matrix arm "
        "(probing), random-unit-direction swap arm with matched edit energy "
        "(10 seeds/item), logit-lens comparator",
        "",
        "## Decision rules (tolerances preregistered)",
        "",
        "| rule | outcome |",
        "|---|---|",
    ]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in rules.items()]
    lines += [
        "",
        f"**M1 verdict: {'PASS (R1-R6)' if all_pass else 'FAIL — see raw replay'}**",
        "",
        "## Headline numbers, median/IQR over the 3 draws",
        "",
        f"- {scoped_intervention('dp(swap_answer), swap-capitals', swap_ds, sham_ds, model=MODEL_SCOPE, config=config_scope, n=n_swap)}",
        f"- {scoped('swap top-1 flip rate', flip_ds, model=MODEL_SCOPE, config=config_scope, n=n_swap)}",
    ]
    for task, ds in hmr_ds.items():
        n_items = len(draws[0]["probe"][task]["per_item"])
        lines.append(
            f"- {scoped(f'band-min J-lens HMR, {task}', ds, model=MODEL_SCOPE, config=config_scope, n=n_items)}"
        )
    lines += [
        "",
        "## Draw-0 vs v1 reference",
        "",
        "| quantity | v1 | draw 0 |",
        "|---|---|---|",
        f"| dp(swap_answer) | +{V1['swap_dp']:.4f} | {m0['mean_dp_swap_answer']:+.4f} |",
        f"| dp random ctrl | +{V1['swap_random']:.4f} | {m0['mean_dp_swap_answer_random_ctrl']:+.4f} |",
        f"| top-1 flip rate | {V1['swap_flip_rate']:.1%} | {m0['swap_top1_rate']:.1%} |",
        f"| capital-recall band-min J-lens HMR | {V1['bandmin_jlens_hmr']['capital-recall']} | {cr_hmr} (L{cr_layer}) |",
        f"| logit HMR at that layer | 61.5 (L16) | {cr_logit_hmr} |",
        f"| calibration sha256 (10 prompts) | — | {'identical to v1' if rules['R4_calibration_hashes_exact'] else 'MISMATCH'} |",
        "",
        "| task | v1 baseline | draw 0 baseline | included |",
        "|---|---|---|---|",
    ]
    for t, (acc, inc) in V1["baselines"].items():
        lines.append(
            f"| {t} | {acc:.1%} ({'in' if inc else 'out'}) | "
            f"{baselines[t]['accuracy']:.1%} | {baselines[t]['included']} |"
        )
    lines += [
        "",
        "| task | v1 band-min J-lens HMR | draws 0/1/2 |",
        "|---|---|---|",
    ]
    for task, ds in hmr_ds.items():
        lines.append(
            f"| {task} | {V1['bandmin_jlens_hmr'][task]} | "
            + "/".join(f"{v}" for v in ds.values)
            + " |"
        )
    lines += [
        "",
        "## Provenance",
        "",
        "| draw | seed | fit wall-clock (s) | peak RSS (GB) | gate |",
        "|---|---|---|---|---|",
    ]
    for k in DRAWS:
        man = draws[k]["manifest"]
        lines.append(
            f"| {k} | {man['seed']} | {man['wall_clock_s']} | {man['peak_rss_gb']} | "
            f"{'PASS' if draws[k]['verdict_pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        f"jlens commit: {draws[0]['manifest']['jlens_commit']} (submodule); "
        "per-draw reports, eval JSONs (per-item records), and lens manifests "
        "are under draws/; raw per-item cells under raw_completions/.",
        "",
    ]
    (ctx.results_dir / "report.md").write_text("\n".join(lines))

    ctx.finalize(
        seeds=list(DRAWS),
        rules=rules,
        m1_verdict="PASS" if all_pass else "FAIL",
        draw_gate_verdicts={k: draws[k]["verdict_pass"] for k in DRAWS},
        baselines_path=str(baselines_path),
        model_revision=MODEL_REV,
    )
    print(f"\nM1 verdict: {'PASS' if all_pass else 'FAIL'}")
    print(f"report: {ctx.results_dir / 'report.md'}")
    for name, ok in rules.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
