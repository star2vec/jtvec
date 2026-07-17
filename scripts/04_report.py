"""Generate the Phase 1 report (readouts, eval tables, gate verdict).

Usage: uv run python scripts/04_report.py --config configs/gpt2_phase1.yaml
       [--evals results/lens_evals/<ts>] [--baselines .../baselines.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jvec.calibration import select_prompts
from jvec.config import Config
from jvec.lens_cache import lens_dir, load_lens
from jvec.modeling import load_model
from jvec.report import build_report
from jvec.utils import REPO_ROOT, make_run_dir, set_seed


def latest(cfg: Config, experiment: str, filename: str) -> Path:
    hits = sorted((REPO_ROOT / cfg.results_dir / experiment).glob(f"*/{filename}"))
    if not hits:
        sys.exit(f"no {experiment}/{filename} found; run the earlier scripts first")
    return hits[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--evals", default=None, help="results/lens_evals/<ts> dir")
    parser.add_argument("--baselines", default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed)
    evals_dir = Path(args.evals) if args.evals else latest(cfg, "lens_evals", "probe.json").parent
    baselines_path = (
        Path(args.baselines) if args.baselines else latest(cfg, "task_baselines", "baselines.json")
    )
    probe = json.loads((evals_dir / "probe.json").read_text())
    swap = json.loads((evals_dir / "swap.json").read_text())
    baselines = json.loads(baselines_path.read_text())

    model, tok, revision = load_model(cfg)
    prompts = select_prompts(cfg, tok)
    lenses, manifests = {}, {}
    for skip_first in cfg.fit.skip_first_variants:
        variant = f"skip{skip_first}"
        lenses[variant] = load_lens(cfg, skip_first, prompts, revision)
        manifests[variant] = json.loads(
            (lens_dir(cfg, skip_first) / "manifest.json").read_text()
        )

    report_md, verdicts = build_report(
        cfg, model, tok, lenses, manifests, prompts.heldout, baselines, probe, swap
    )
    run_dir = make_run_dir(cfg, "phase1_report")
    (run_dir / "report.md").write_text(report_md)
    (run_dir / "sources.json").write_text(
        json.dumps({"evals": str(evals_dir), "baselines": str(baselines_path)}, indent=2)
    )
    print(f"report: {run_dir / 'report.md'}")
    print(f"gate verdicts: {verdicts}")


if __name__ == "__main__":
    main()
