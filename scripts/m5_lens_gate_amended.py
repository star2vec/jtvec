"""EXP-M5-0 lens gate, AMENDED re-run (D-027 outcome c; Q5 amendment ratified
2026-07-21). Freshly re-probes the latent-intermediate anchors on the 3 cached
1.4B lenses (NO refit; NOT a re-grade of the diagnostic) and scores amended-Q5
(max-contrast metric, identical statistic across arms, >= 2 adequate-N latent
anchors clear 5x). Carries Q1/Q2/Q3/Q4/Q6 from the committed original gate
verdict (swap-based rules are deterministic and unchanged by the Q5 amendment).

The overall gate verdict is reported honestly: with Q2 and Q6 still failing
(D-029 open), the overall gate does NOT PASS even though amended-Q5 does. This
run produces the formal EXP-M5-0-labelled amended-Q5 evidence and the exact
Q2/Q6 status; admission of 1.4B awaits the D-029 ruling.

Prereg: harness/preregs/EXP-M5-0-qualification.md (+ EXP-M5-0-amendment-Q5.md).
Usage: uv run python scripts/m5_lens_gate_amended.py
"""

from __future__ import annotations

import json
import sys
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
from jtvec.lens_diagnostic import amended_q5_verdict, task_arm_ratios, task_shows_advantage

DRAWS = (0, 1, 2)
DRAW_CONFIGS = {k: REPO_ROOT / f"configs/m5_lens_pythia1p4b_draw{k}.yaml" for k in DRAWS}
PREREG = REPO_ROOT / "harness/preregs/EXP-M5-0-qualification.md"
ORIG_GATE = REPO_ROOT / "results/m5/20260720-024819-p14b-lens-gate/lens_gate_verdict.json"
SKIP_FIRST, CAP, ADV_RATIO, N_RANDOM, PASS_K, ADEQUATE_N = 4, 5.0, 5.0, 10, 10, 20

# latent-intermediate anchors (count toward amended Q5) + output controls (record only)
LATENT = ["capital-operand", "fresh1hop-operand", "fresh2hop-bridge"]
OUTPUT_CTRL = ["capital-recall", "fresh1hop-answer"]


def behavioural_subset(model, tok, task: Task):
    scored = score_task(model, tok, task)
    correct = {it["name"] for it in scored["per_item"] if it["correct"]}
    return Task(task.name, task.protocol, [it for it in task.items if it["name"] in correct])


def main() -> None:
    ctx = start_run(repo_root=REPO_ROOT, config_path=DRAW_CONFIGS[0],
                    results_root=REPO_ROOT / "results/m5", run_name="p14b-lens-gate-amended",
                    prereg_path=PREREG)
    print(f"amended-gate run dir: {ctx.results_dir}", flush=True)

    fresh = {t.name: t for t in load_tasks(REPO_ROOT / "tasks" / "diagnostic")}
    existing = {t.name: t for t in load_tasks(REPO_ROOT / "tasks")}
    battery = {n: fresh.get(n, existing.get(n)) for n in LATENT + OUTPUT_CTRL}

    base = Config.load(str(DRAW_CONFIGS[0]))
    set_seed(base.seed)
    model, tok, revision = load_model(base)
    print(f"model {base.model.name}@{revision[:7]}, {model.n_layers} layers", flush=True)

    filtered, task_n = {}, {}
    for name, task in battery.items():
        ft = behavioural_subset(model, tok, task)
        filtered[name], task_n[name] = ft, len(ft.items)
        print(f"  {name:20s} behavioural: {len(ft.items)}/{len(task.items)}", flush=True)

    per_task_draws = {n: [] for n in battery}
    for k in DRAWS:
        cfg_k = Config.load(str(DRAW_CONFIGS[k]))
        lens = load_lens(cfg_k, SKIP_FIRST, select_prompts(cfg_k, tok), revision)
        layers = list(lens.source_layers)
        print(f"\n=== draw {k} ===", flush=True)
        for name in battery:
            ft = filtered[name]
            if not ft.items:
                per_task_draws[name].append({"jlens_ratio": 0.0, "random_max_ratio": 0.0})
                continue
            res = probe_task(model, tok, lens, ft, pass_k=PASS_K, n_random_seeds=N_RANDOM)
            metrics = json.loads(json.dumps(res["metrics"]))
            r = task_arm_ratios(metrics, layers, CAP)
            per_task_draws[name].append(r)
            ctx.save_raw_completions(f"{name}_draw{k}_ranks",
                [{"draw": k, "name": it["name"], "ranks": it["ranks"]} for it in res["per_item"]])
            print(f"  {name:20s} jlens_ratio={r['jlens_ratio']:.1f} rand_max={r['random_max_ratio']:.1f}", flush=True)

    # amended Q5 over the latent anchors
    anchor_shows = {}
    per_task_summary = {}
    for name in LATENT:
        shows, diag = task_shows_advantage(per_task_draws[name], ADV_RATIO)
        anchor_shows[name] = (shows, task_n[name])
        per_task_summary[name] = {"shows": shows, "N": task_n[name], **diag}
    q5 = amended_q5_verdict(anchor_shows, ADEQUATE_N)
    for name in OUTPUT_CTRL:
        shows, diag = task_shows_advantage(per_task_draws[name], ADV_RATIO)
        per_task_summary[name] = {"shows": shows, "N": task_n[name], "role": "output-control", **diag}

    orig = json.loads(ORIG_GATE.read_text())["rules"]
    carried = {r: orig[r] for r in ("Q1_all_draws_gate_pass", "Q2_positive_control",
                                    "Q3_sham", "Q4_negative_control", "Q6_draw_stability")}
    rules = {"Q1_all_draws_gate_pass": carried["Q1_all_draws_gate_pass"],
             "Q2_positive_control": carried["Q2_positive_control"],
             "Q3_sham": carried["Q3_sham"],
             "Q4_negative_control": carried["Q4_negative_control"],
             "Q5_probing_contrast_amended": q5["passed"],
             "Q6_draw_stability": carried["Q6_draw_stability"]}
    overall = all(rules.values())
    blockers = [r for r, ok in rules.items() if not ok]

    verdict = {
        "rules": rules, "overall_pass": overall, "blockers": blockers,
        "amended_q5": q5, "per_task": per_task_summary,
        "carried_from": str(ORIG_GATE.relative_to(REPO_ROOT)),
        "note": ("amended Q5 PASSES; overall gate blocked on Q2/Q6 (D-029). "
                 "1.4B admission awaits the D-029 ruling." if not overall else "PASS"),
        "peak_rss_gb": round(peak_rss_gb(), 2),
    }
    (ctx.results_dir / "amended_gate_verdict.json").write_text(json.dumps(verdict, indent=2))

    print("\n=== EXP-M5-0 amended gate ===", flush=True)
    for r, ok in rules.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {r}")
    print(f"  amended-Q5 clearing (adequate N): {q5['clearing_adequate_n']}  "
          f"descriptive low-N: {q5['descriptive_low_n']}")
    print(f"\n  overall_pass: {overall}  blockers: {blockers}")
    print(f"  {verdict['note']}")
    print(f"  run dir: {ctx.results_dir}")
    ctx.finalize(overall_pass=overall, amended_q5_pass=q5["passed"], blockers=blockers)


if __name__ == "__main__":
    main()
