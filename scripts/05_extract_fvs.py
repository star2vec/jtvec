"""Extract function vectors (Todd attention-head method + Hendel hidden-state
method) for the config's ICL tasks on the config's model.

Gating (LABNOTES 2026-07-14): a task is extracted if 10-shot test top-1 >=
fv.min_accuracy AND the model answers >= fv.min_correct_valid items of the
valid split correctly; extraction then uses Todd's canonical filter_set
(correct valid items only). Headline membership for Experiments 1-2 is
decided afterwards from zero-shot FV induction strength, which this script
measures and reports per task.

Order of operations: (1) gate; (2) timing probe with a wall-clock estimate
before the full extraction; (3) per-task extraction with caching + manifests;
(4) induction check (zero-shot accuracy with the FV added at edit_layer vs
without).

Usage: uv run python scripts/05_extract_fvs.py --config configs/pythia410m_phase2.yaml
       [--refit] [--test-acc-file results/fv_gate_pythia410m.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from jvec.config import Config
from jvec.fv import FV_REPO, extract_task_fvs, load_cached_fv, load_fv_model
from jvec.utils import make_run_dir, peak_rss_gb, set_seed

MAX_LOCAL_HOURS = 12.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--test-acc-file", default=None,
                        help="JSON {task: top1} of precomputed 10-shot test accuracies")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed)
    model, tokenizer, model_config, revision = load_fv_model(cfg)

    from utils.eval_utils import n_shot_eval, n_shot_eval_no_intervention  # noqa: PLC0415
    from utils.prompt_utils import load_dataset  # noqa: PLC0415

    from jvec.fv import correct_valid_filter_set  # noqa: PLC0415

    root_data_dir = str(FV_REPO / "dataset_files")
    test_accs = json.loads(Path(args.test_acc_file).read_text()) if args.test_acc_file else {}

    # --- 1. Gate: test accuracy floor + enough correct valid items --------------
    gate, filter_sets = {}, {}
    for task in cfg.fv.tasks:
        dataset = load_dataset(task, root_data_dir=root_data_dir, seed=cfg.seed)
        if task in test_accs:
            top1 = test_accs[task]
        else:
            result = n_shot_eval_no_intervention(
                dataset, cfg.fv.n_shots, model, model_config, tokenizer,
                compute_ppl=False, test_split="test",
            )
            top1 = dict(result["clean_topk"])[1]
        fs = correct_valid_filter_set(cfg, dataset, model, tokenizer, model_config)
        filter_sets[task] = fs
        included = top1 >= cfg.fv.min_accuracy and len(fs) >= cfg.fv.min_correct_valid
        gate[task] = {
            "top1": top1,
            "n_correct_valid": int(len(fs)),
            "n_valid": len(dataset["valid"]),
            "included": included,
            "high_accuracy_stratum": top1 >= cfg.fv.baseline_threshold,
        }
        reason = "" if included else (
            f"  (acc < {cfg.fv.min_accuracy:.0%})" if top1 < cfg.fv.min_accuracy
            else f"  (only {len(fs)} correct valid items < {cfg.fv.min_correct_valid})"
        )
        print(f"{task:20s} test top-1 {top1:6.1%}  correct-valid {len(fs):>4}/{gate[task]['n_valid']:<4} "
              f"-> {'EXTRACT' if included else 'EXCLUDED'}{reason}")

    included = [t for t in cfg.fv.tasks if gate[t]["included"]]
    if not included:
        sys.exit("no task passed the extraction floors; nothing to extract")

    # --- 2. Timing probe --------------------------------------------------------
    def cached(task):
        try:
            return None if args.refit else load_cached_fv(cfg, task, revision)
        except Exception:
            raise

    to_extract = [t for t in included if cached(t) is None]
    if to_extract:
        probe_task = to_extract[0]
        from utils.extract_utils import get_mean_head_activations  # noqa: PLC0415
        from compute_indirect_effect import compute_indirect_effect  # noqa: PLC0415

        dataset = load_dataset(probe_task, root_data_dir=root_data_dir, seed=cfg.seed)
        t0 = time.perf_counter()
        probe_acts = get_mean_head_activations(
            dataset, model, model_config, tokenizer,
            n_icl_examples=cfg.fv.n_shots, N_TRIALS=5,
            filter_set=filter_sets[probe_task],
        )
        mean_s = (time.perf_counter() - t0) / 5
        t0 = time.perf_counter()
        compute_indirect_effect(
            dataset, probe_acts, model, model_config, tokenizer,
            n_shots=cfg.fv.n_shots, n_trials=1, last_token_only=True,
            filter_set=filter_sets[probe_task],
        )
        aie_s = time.perf_counter() - t0
        per_task_s = (
            mean_s * cfg.fv.n_trials_mean * 2  # head + layer mean passes
            + aie_s * cfg.fv.n_trials_aie
        )
        total_s = per_task_s * len(to_extract) * 1.15
        print(
            f"probe: mean-act {mean_s:.1f}s/trial, AIE {aie_s:.1f}s/trial -> "
            f"~{per_task_s / 60:.1f} min/task, ~{total_s / 60:.0f} min for "
            f"{len(to_extract)} task(s); peak RSS {peak_rss_gb():.2f} GB"
        )
        if total_s > MAX_LOCAL_HOURS * 3600:
            sys.exit(f"projected {total_s / 3600:.1f}h > {MAX_LOCAL_HOURS:.0f}h — flag for A100.")

    # --- 3+4. Extract and steering-check each task ------------------------------
    summary = {"gate": gate, "model_revision": revision, "tasks": {}}
    for task in included:
        artifacts = cached(task)
        if artifacts is not None:
            print(f"[cache hit] {task}")
        else:
            print(f"[extracting] {task}")
            set_seed(cfg.seed)  # deterministic trial sampling per task
            artifacts = extract_task_fvs(
                cfg, task, model, tokenizer, model_config, revision,
                filter_set=filter_sets[task],
            )

        dataset = load_dataset(task, root_data_dir=root_data_dir, seed=cfg.seed)
        fv = artifacts["fv_todd"].to(model.device)
        zs = n_shot_eval_no_intervention(
            dataset, 0, model, model_config, tokenizer, compute_ppl=False, test_split="test"
        )
        steered = n_shot_eval(
            dataset, fv.reshape(1, -1), cfg.fv.edit_layer, 0, model, model_config, tokenizer
        )
        zs_top1 = dict(zs["clean_topk"])[1]
        steer_top1 = dict(steered["intervention_topk"])[1]
        summary["tasks"][task] = {
            "test_top1_10shot": gate[task]["top1"],
            "n_correct_valid": gate[task]["n_correct_valid"],
            "zero_shot_top1": zs_top1,
            "zero_shot_plus_fv_top1": steer_top1,
            "induction_gain": round(steer_top1 - zs_top1, 4),
            "top_heads": artifacts["manifest"]["top_heads"][: cfg.fv.n_top_heads],
            "timings": artifacts["manifest"].get("timings"),
        }
        print(
            f"  induction: zero-shot {zs_top1:.1%} -> +FV@L{cfg.fv.edit_layer} "
            f"{steer_top1:.1%} (gain {steer_top1 - zs_top1:+.1%})"
        )

    run_dir = make_run_dir(cfg, "fv_extraction")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
