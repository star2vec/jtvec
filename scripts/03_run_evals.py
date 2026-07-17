"""Run the probing and causal-swap sanity evals for every cached lens variant.

Requires a prior baselines run (scripts/02_task_baselines.py); tasks that
failed the baseline gate are skipped here and reported as dropped.

Usage: uv run python scripts/03_run_evals.py --config configs/gpt2_phase1.yaml
       [--baselines results/task_baselines/<ts>/baselines.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jvec.calibration import select_prompts
from jvec.config import Config
from jvec.evals.probe import probe_task
from jvec.evals.swap import swap_task
from jvec.evals.tasks import load_tasks
from jvec.lens_cache import load_lens
from jvec.modeling import load_model
from jvec.utils import REPO_ROOT, make_run_dir, set_seed


def latest_baselines(cfg: Config) -> Path:
    candidates = sorted(
        (REPO_ROOT / cfg.results_dir / "task_baselines").glob("*/baselines.json")
    )
    if not candidates:
        sys.exit("no baselines found; run scripts/02_task_baselines.py first")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--baselines", default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed)
    baselines_path = Path(args.baselines) if args.baselines else latest_baselines(cfg)
    baselines = json.loads(baselines_path.read_text())
    print(f"baselines: {baselines_path}")

    model, tok, revision = load_model(cfg)
    prompts = select_prompts(cfg, tok)

    probe_results: dict[str, dict] = {}
    swap_results: dict[str, dict] = {}
    for skip_first in cfg.fit.skip_first_variants:
        variant = f"skip{skip_first}"
        lens = load_lens(cfg, skip_first, prompts, revision)
        print(f"\n=== {variant}: {lens} ===")
        probe_results[variant] = {}
        swap_results[variant] = {}
        for task in load_tasks():
            gate = baselines.get(task.name)
            if gate is None:
                sys.exit(f"task {task.name} missing from {baselines_path}; rerun script 02")
            if not gate["included"]:
                print(f"  {task.name}: dropped at baseline gate ({gate['accuracy']:.1%})")
                continue
            if task.protocol in ("completion", "typo"):
                result = probe_task(
                    model, tok, lens, task,
                    pass_k=cfg.evals.pass_k,
                    n_random_seeds=cfg.evals.n_random_seeds,
                )
                probe_results[variant][task.name] = result
                band = [
                    l for l in result["layers"]
                    if cfg.evals.band[0] <= l <= cfg.evals.band[1]
                ]
                best = {
                    arm: min(result["metrics"][arm]["per_layer"][l]["hmr"] for l in band)
                    for arm in result["arms"]
                }
                print(f"  {task.name}: best band HMR  " + "  ".join(
                    f"{arm}={hmr:.1f}" for arm, hmr in best.items()
                ))
            elif task.protocol == "swap":
                result = swap_task(
                    model, tok, lens, task,
                    band=cfg.evals.band,
                    alpha=cfg.evals.swap_alpha,
                    rcond=cfg.evals.swap_rcond,
                    n_random_seeds=cfg.evals.n_random_seeds,
                )
                swap_results[variant][task.name] = result
                m = result["metrics"]
                print(
                    f"  {task.name}: dp(swap_answer)={m['mean_dp_swap_answer']:+.4f} "
                    f"random_ctrl={m['mean_dp_swap_answer_random_ctrl']:+.4f} "
                    f"dp(answer)={m['mean_dp_answer']:+.4f} "
                    f"top1_flip_rate={m['swap_top1_rate']:.1%}"
                )

    run_dir = make_run_dir(cfg, "lens_evals")
    (run_dir / "probe.json").write_text(json.dumps(probe_results, indent=1))
    (run_dir / "swap.json").write_text(json.dumps(swap_results, indent=1))
    (run_dir / "baselines_used.json").write_text(json.dumps({"path": str(baselines_path)}))
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
