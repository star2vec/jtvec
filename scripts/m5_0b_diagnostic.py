"""EXP-M5-0b: 1.4B lens diagnostic (latent-vs-output probe), evals-only.

Probes the fresh matched battery (tasks/diagnostic/) + the post-hoc existing
anchors on the 3 already-fitted M5.0 lenses (cache/m5/p14b_draw{0,1,2}) — NO
refit (D-027). Per (task, draw): behavioural-filter to the items the model
gets right, probe per-layer J-lens/logit/random HMR (vendored probe_task),
then the max-contrast statistic (jtvec.lens_diagnostic) applied identically to
every arm. Decision: >= 2 fresh latent tasks show the J-lens advantage while
their matched output tasks do not (jtvec.lens_diagnostic.diagnostic_verdict).

Prereg: harness/preregs/EXP-M5-0b-lens-diagnostic.md (committed c9f2acd).
Usage: uv run python scripts/m5_0b_diagnostic.py
"""

from __future__ import annotations

import dataclasses
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
from jtvec.lens_diagnostic import MATCHED, diagnostic_verdict, task_arm_ratios

DRAWS = (0, 1, 2)
DRAW_CONFIGS = {k: REPO_ROOT / f"configs/m5_lens_pythia1p4b_draw{k}.yaml" for k in DRAWS}
DIAG_CONFIG = REPO_ROOT / "configs/m5_0b_diagnostic_pythia1p4b.yaml"
PREREG = REPO_ROOT / "harness/preregs/EXP-M5-0b-lens-diagnostic.md"
SKIP_FIRST = 4
CAP = 5.0
ADV_RATIO = 5.0
N_RANDOM = 10
PASS_K = 10

# Fresh matched battery (decision-bearing) + post-hoc existing anchors (context).
FRESH = ["fresh1hop-operand", "fresh1hop-answer", "fresh2hop-bridge", "fresh2hop-answer"]
POSTHOC = ["capital-operand", "capital-recall", "multihop-scaled", "opposites", "word-pairs"]


def behavioural_subset(model, tok, task: Task) -> Task:
    """Keep only items the model gets right (greedy top-1 == target)."""
    scored = score_task(model, tok, task)
    correct = {it["name"] for it in scored["per_item"] if it["correct"]}
    items = [it for it in task.items if it["name"] in correct]
    return Task(name=task.name, protocol=task.protocol, items=items), len(items), len(task.items)


def main() -> None:
    ctx = start_run(
        repo_root=REPO_ROOT, config_path=DIAG_CONFIG,
        results_root=REPO_ROOT / "results/m5", run_name="p14b-lens-diagnostic",
        prereg_path=PREREG,
    )
    print(f"diagnostic run dir: {ctx.results_dir}", flush=True)

    fresh_tasks = {t.name: t for t in load_tasks(REPO_ROOT / "tasks" / "diagnostic")}
    all_tasks = {t.name: t for t in load_tasks(REPO_ROOT / "tasks")}
    battery = {name: fresh_tasks.get(name, all_tasks.get(name)) for name in FRESH + POSTHOC}
    missing = [n for n, t in battery.items() if t is None]
    if missing:
        sys.exit(f"battery tasks not found: {missing}")

    base_cfg = Config.load(str(DRAW_CONFIGS[0]))
    set_seed(base_cfg.seed)
    model, tok, revision = load_model(base_cfg)
    print(f"model {base_cfg.model.name}@{revision[:7]} on {base_cfg.device}, "
          f"{model.n_layers} layers", flush=True)

    # behavioural filter is model-only (lens-independent); do it once per task
    filtered, task_n = {}, {}
    for name, task in battery.items():
        ftask, n_ok, n_tot = behavioural_subset(model, tok, task)
        filtered[name], task_n[name] = ftask, {"n_correct": n_ok, "n_total": n_tot}
        print(f"  {name:20s} behavioural: {n_ok}/{n_tot} correct", flush=True)

    per_task_draws: dict[str, list[dict]] = {name: [] for name in battery}
    for k in DRAWS:
        cfg_k = Config.load(str(DRAW_CONFIGS[k]))
        prompts_k = select_prompts(cfg_k, tok)
        lens = load_lens(cfg_k, SKIP_FIRST, prompts_k, revision)
        layers = list(lens.source_layers)  # candidate set S = the probed layers
        print(f"\n=== draw {k}: {lens} ===", flush=True)
        for name in battery:
            ftask = filtered[name]
            if not ftask.items:
                per_task_draws[name].append({"jlens_ratio": 0.0, "jlens_layer": None,
                                             "random_max_ratio": 0.0, "n": 0})
                continue
            res = probe_task(model, tok, lens, ftask, pass_k=PASS_K, n_random_seeds=N_RANDOM)
            metrics = json.loads(json.dumps(res["metrics"]))  # normalise layer keys to str
            ratios = task_arm_ratios(metrics, layers, CAP)
            ratios["n"] = len(ftask.items)
            per_task_draws[name].append(ratios)
            ctx.save_raw_completions(
                f"{name}_draw{k}_ranks",
                [{"draw": k, "name": it["name"], "ranks": it["ranks"]}
                 for it in res["per_item"]],
            )
            print(f"  {name:20s} jlens_ratio={ratios['jlens_ratio']:.1f} "
                  f"@L{ratios['jlens_layer']}  rand_max={ratios['random_max_ratio']:.1f}",
                  flush=True)

    verdict = diagnostic_verdict({t: per_task_draws[t] for t in FRESH}, ADV_RATIO, MATCHED)
    verdict["posthoc_context"] = {
        t: {"jlens_ratio_median": round(_med([d["jlens_ratio"] for d in per_task_draws[t]]), 3)}
        for t in POSTHOC
    }
    verdict["task_n"] = task_n
    verdict["per_task_draws"] = per_task_draws
    verdict["peak_rss_gb"] = round(peak_rss_gb(), 2)
    (ctx.results_dir / "diagnostic_verdict.json").write_text(json.dumps(verdict, indent=2))

    print("\n=== EXP-M5-0b diagnostic ===", flush=True)
    for t in FRESH:
        s = verdict["per_task"][t]
        print(f"  {t:20s} shows_advantage={s['shows_advantage']}  "
              f"jlens_med={s['jlens_ratio_median']}  rand_med={s['random_ratio_median']}  "
              f"N={task_n[t]['n_correct']}")
    print(f"\n  verdict: {verdict['verdict']}  ({verdict['d027_outcome']})")
    print(f"  dissociating pairs: {verdict['n_dissociating_pairs']}")
    print(f"  run dir: {ctx.results_dir}")
    ctx.finalize(diagnostic_verdict=verdict["verdict"],
                 n_dissociating_pairs=verdict["n_dissociating_pairs"])


def _med(xs):
    import statistics
    return float(statistics.median(xs))


if __name__ == "__main__":
    main()
