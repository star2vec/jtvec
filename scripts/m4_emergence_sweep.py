"""M4 emergence sweep: stability-gated FV emergence over Pythia checkpoints.

Per revision (checkpoint) of one scale (model name), runs the M2 stability
gate at the {25,50} rung (extraction at n_trials_aie=50 — reproduces
converged_at=25 WITH the required witness rung at 1/4 the full-ladder cost),
and records 10-shot execution accuracy, 0-shot FV induction gain, and E1-style
label/output-vocab decodability through that checkpoint's own lens. Fixes v1's
confounds: per-checkpoint AIE re-extraction; head re-selection is automatic in
compute_function_vector. Appends one line per checkpoint to sweep.jsonl
(crash-resumable; revision-keyed caches resume). Per-model teardown + a
too-weak-checkpoint guard follow the vendored scripts/13 skeleton.

Multi-scale = one invocation per scale (config sets model.name). The A100
launch is gated on Ecaterina's compute ruling; this script also runs a cheap
single-checkpoint dry-run on the laptop for wiring validation.

Prereg: harness/preregs/EXP-M4-emergence.md (committed before the scientific
run; start_run enforces).

Usage: uv run python scripts/m4_emergence_sweep.py --config configs/m4_emergence_pythia410m.yaml
       --revisions step143000[,step72000,...]
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from jvec.calibration import select_prompts
from jvec.config import Config
from jvec.evals.fvprobe import output_token_ids
from jvec.fv import FV_REPO, extract_task_fvs, load_cached_fv, load_fv_model
from jvec.lens_cache import fit_lens
from jvec.utils import peak_rss_gb, set_seed
from jtvec.core.draws import DrawSet
from jtvec.core.runctx import start_run
from jtvec.e1_decodability import LABEL_SETS, full_vocab_rank, label_rank
from jtvec.emergence import CheckpointRecord, EmergenceRule
from jtvec.fv_stability import (
    ConvergenceRule,
    RungStats,
    convergence_verdict,
    fv_at_rung,
    pairwise_cosines,
    sham_twin,
)

SWEEP_TASKS = ("capitalize", "singular-plural", "english-french")
SWEEP_RUNGS = (25, 50)          # extraction at 50; 25 witnessed by 50 (D-019)
DRAW_KS = (1, 2, 3)
EVAL_SEED = 999
SHAM_SEED_BASE = 9000
RULE = ConvergenceRule(min_pairwise_cosine=0.95, max_gain_iqr=0.05)
POSITIVE_MIN_SEP = 0.10
NEGATIVE_MAX_ABS_SHAM = 0.02


def draw_seed(cfg: Config, k: int) -> int:
    return cfg.seed * 1000 + k


def parse_step(revision: str) -> int:
    """Training step from a 'stepN' revision (0 if not a step label)."""
    return int(revision[4:]) if revision.startswith("step") and revision[4:].isdigit() else 0


def run_checkpoint(cfg, revision, ctx, out) -> dict:
    """Stability-gated gate + execution/induction/decodability at one checkpoint."""
    rcfg = dataclasses.replace(
        cfg,
        model=dataclasses.replace(cfg.model, revision=revision),
        fv=dataclasses.replace(cfg.fv, tasks=tuple(SWEEP_TASKS), n_trials_aie=max(SWEEP_RUNGS)),
    )
    set_seed(rcfg.seed)
    model, tokenizer, model_config, resolved = load_fv_model(rcfg)
    import jlens as jlens_pkg  # noqa: PLC0415
    model_j = jlens_pkg.from_hf(model, tokenizer)
    prompts = select_prompts(rcfg, tokenizer)
    lens = fit_lens(rcfg, rcfg.fit.skip_first_variants[0], prompts, model_j, resolved)
    device = model_j.input_device
    lo, hi = rcfg.evals.band
    band = [l for l in lens.source_layers if lo <= l <= hi]

    from utils.eval_utils import n_shot_eval, n_shot_eval_no_intervention  # noqa: PLC0415
    from utils.prompt_utils import load_dataset  # noqa: PLC0415

    def exec_top1(dataset, n_shots):
        set_seed(EVAL_SEED)
        res = n_shot_eval_no_intervention(dataset, n_shots, model, model_config,
                                          tokenizer, compute_ppl=False, test_split="test")
        return dict(res["clean_topk"])[1], len(res["clean_rank_list"])

    def inject_top1(dataset, vec):
        set_seed(EVAL_SEED)
        res = n_shot_eval(dataset, vec.reshape(1, -1).to(model.device),
                          rcfg.fv.edit_layer, 0, model, model_config, tokenizer)
        return dict(res["intervention_topk"])[1]

    step = parse_step(revision)
    record = {"revision": revision, "resolved": resolved, "step": step, "tasks": {}}

    for task in SWEEP_TASKS:
        dataset = load_dataset(task, root_data_dir=str(FV_REPO / "dataset_files"), seed=rcfg.seed)
        zs_top1, _ = exec_top1(dataset, 0)
        icl_top1, n_test = exec_top1(dataset, rcfg.fv.n_shots)
        entry = {"exec_10shot": icl_top1, "zero_shot": zs_top1, "n_test": n_test}

        # per-checkpoint AIE re-extraction, 3 draws (too-weak guard)
        try:
            arts = {}
            for k in DRAW_KS:
                dcfg = dataclasses.replace(rcfg, cache_dir=f"{rcfg.cache_dir}/{revision}/draw{k}")
                art = load_cached_fv(dcfg, task, resolved)
                if art is None:
                    set_seed(draw_seed(rcfg, k))
                    art = extract_task_fvs(dcfg, task, model, tokenizer, model_config, resolved)
                arts[k] = art
        except ValueError as e:
            entry["fv"] = f"too weak: {e}"
            record["tasks"][task] = entry
            print(f"  {task}: exec {icl_top1:.1%}, FV skipped ({e})", flush=True)
            continue

        per_rung_stats, rung_detail = [], {}
        for rung_idx, T in enumerate(SWEEP_RUNGS):
            fvs, gains, sham_gains = {}, {}, {}
            for k in DRAW_KS:
                fv, _heads = fv_at_rung(arts[k]["mean_head_activations"],
                                        arts[k]["indirect_effect"], T, model,
                                        model_config, rcfg.fv.n_top_heads)
                fvs[f"draw{k}"] = fv
                gains[k] = inject_top1(dataset, fv) - zs_top1
                sham_gains[k] = inject_top1(dataset, sham_twin(fv, SHAM_SEED_BASE + 10 * k + rung_idx)) - zs_top1
            seeds = tuple(draw_seed(rcfg, k) for k in DRAW_KS)
            gain_ds = DrawSet(values=tuple(gains[k] for k in DRAW_KS), seeds=seeds)
            sham_ds = DrawSet(values=tuple(sham_gains[k] for k in DRAW_KS), seeds=seeds)
            cosines = pairwise_cosines(fvs)
            per_rung_stats.append(RungStats(n_trials=T, min_pairwise_cosine=min(cosines.values()),
                                            gain_iqr=gain_ds.iqr, min_top_head_overlap=0))
            rung_detail[T] = {"min_cos": min(cosines.values()), "gain_median": gain_ds.median,
                              "gain_iqr": gain_ds.iqr, "sham_median": sham_ds.median}
        verdict = convergence_verdict(per_rung_stats, RULE)

        # per-checkpoint controls: positive = ICL vs 0-shot sep; negative = sham ~ 0
        positive = (icl_top1 - zs_top1) >= POSITIVE_MIN_SEP
        negative = all(abs(rung_detail[T]["sham_median"]) <= max(NEGATIVE_MAX_ABS_SHAM, 1.0 / n_test)
                       for T in SWEEP_RUNGS)
        gate_passed = bool(verdict["converged"] and positive and negative)

        # decodability through this checkpoint's lens (E1 statistic; draw-1 FV @ top rung)
        fv1, _ = fv_at_rung(arts[1]["mean_head_activations"], arts[1]["indirect_effect"],
                            max(SWEEP_RUNGS), model, model_config, rcfg.fv.n_top_heads)
        v = fv1.float().to(device)
        readouts = {l: model_j.unembed(lens.transport(v, l)).float().cpu() for l in band}
        out_ids = output_token_ids(dataset, tokenizer)
        best_ov = min(float(np.mean([full_vocab_rank(readouts[l], t) for t in out_ids])) for l in band)
        lab = label_rank(readouts, tokenizer, LABEL_SETS["set1"][task]) if task in LABEL_SETS["set1"] else None

        entry.update({
            "converged_at": verdict["converged_at"], "gate_passed": gate_passed,
            "positive_control": positive, "negative_control": negative,
            "induction_gain_toprung": rung_detail[max(SWEEP_RUNGS)]["gain_median"],
            "label_rank_jlens": lab, "outvocab_rank_jlens": best_ov, "rungs": rung_detail,
        })
        record["tasks"][task] = entry
        print(f"  {task}: exec {icl_top1:.1%}, converged_at={verdict['converged_at']}, "
              f"gate={'PASS' if gate_passed else 'no'}, outvocab {best_ov:.0f}", flush=True)

    # scale-level CheckpointRecord for the emergence classifier
    task_entries = [e for e in record["tasks"].values() if "exec_10shot" in e]
    gated = [e for e in task_entries if e.get("gate_passed")]
    cr = CheckpointRecord(
        step=step,
        exec_acc=max((e["exec_10shot"] for e in task_entries), default=0.0),
        gate_passed=len(gated) > 0,
        converged_tasks=len(gated),
        outvocab_rank=min((e["outvocab_rank_jlens"] for e in gated), default=None),
    )
    record["checkpoint_record"] = dataclasses.asdict(cr)
    out.write(json.dumps(record, default=str) + "\n")
    out.flush()

    del model, model_j, lens
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/m4_emergence_pythia410m.yaml"))
    parser.add_argument("--revisions", required=True, help="comma list, e.g. step143000,step72000")
    parser.add_argument("--dry-run", action="store_true",
                        help="wiring validation: capitalize only, no prereg (post_hoc run)")
    args = parser.parse_args()
    t_start = time.perf_counter()

    cfg = Config.load(args.config)
    revisions = args.revisions.split(",")
    prereg = None if args.dry_run else REPO_ROOT / "harness/preregs/EXP-M4-emergence.md"
    ctx = start_run(
        repo_root=REPO_ROOT, config_path=Path(args.config),
        results_root=REPO_ROOT / cfg.results_dir,
        run_name=f"emergence-{cfg.model.name.split('/')[-1]}",
        prereg_path=prereg, post_hoc=args.dry_run,
    )
    print(f"emergence sweep run dir: {ctx.results_dir}; scale {cfg.model.name}; "
          f"revisions {revisions}{' [DRY-RUN]' if args.dry_run else ''}", flush=True)

    global SWEEP_TASKS
    if args.dry_run:
        SWEEP_TASKS = ("capitalize",)

    out = (ctx.results_dir / "sweep.jsonl").open("a", encoding="utf-8")
    records = []
    for rev in revisions:
        print(f"\n=== {cfg.model.name} @ {rev} ===", flush=True)
        records.append(run_checkpoint(cfg, rev, ctx, out))
    out.close()

    crs = [CheckpointRecord(**r["checkpoint_record"]) for r in records if "checkpoint_record" in r]
    emergence = EmergenceRule().classify_scale(crs) if len(crs) >= 2 else {
        "verdict": "INSUFFICIENT-CHECKPOINTS", "n_checkpoints": len(crs)}
    (ctx.results_dir / "emergence.json").write_text(
        json.dumps({"scale": cfg.model.name, "revisions": revisions, "emergence": emergence,
                    "checkpoint_records": [r.get("checkpoint_record") for r in records]},
                   indent=2, default=str), encoding="utf-8")

    ctx.finalize(scale=cfg.model.name, revisions=revisions, emergence=emergence,
                 dry_run=args.dry_run, wall_clock_s=round(time.perf_counter() - t_start, 1),
                 peak_rss_gb=round(peak_rss_gb(), 2))
    print(f"\nemergence verdict ({cfg.model.name}): {emergence.get('verdict')}")
    print(f"records: {ctx.results_dir / 'emergence.json'}")


if __name__ == "__main__":
    main()
