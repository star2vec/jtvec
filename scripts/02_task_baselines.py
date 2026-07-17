"""Measure GPT-2-small's in-context baseline accuracy on each sanity task.

Tasks below the config threshold are marked excluded (they will be skipped by
the lens evals and listed as dropped in the report).

Usage: uv run python scripts/02_task_baselines.py --config configs/gpt2_phase1.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jvec.config import Config
from jvec.evals.baseline import score_task
from jvec.evals.tasks import load_tasks
from jvec.modeling import load_model
from jvec.utils import make_run_dir, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed)
    model, tok, _ = load_model(cfg)
    threshold = cfg.evals.baseline_threshold

    results = {}
    for task in load_tasks():
        scored = score_task(model, tok, task)
        scored["included"] = scored["accuracy"] >= threshold
        results[task.name] = scored
        status = "INCLUDED" if scored["included"] else "DROPPED"
        print(
            f"{task.name:20s} {scored['accuracy']:6.1%}  ({len(task.items)} items)  "
            f"threshold={threshold:.0%}  -> {status}"
        )
        wrong = [i for i in scored["per_item"] if not i["correct"]]
        for item in wrong[:8]:
            print(f"    miss {item['name']:24s} top1={item['top1']!r} expected={item['expected']!r}")
        if len(wrong) > 8:
            print(f"    ... and {len(wrong) - 8} more misses")

    run_dir = make_run_dir(cfg, "task_baselines")
    (run_dir / "baselines.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {run_dir / 'baselines.json'}")


if __name__ == "__main__":
    main()
