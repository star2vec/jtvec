"""M4-E2 orchestrator: execution/verbalization dissociation on singular-plural.

Applies the two M3-gated ablations (fv-direction, jspace) + their shams to the
two measures (execution accuracy; report_score under P3, gated this session)
on singular-plural, cross-draw over the 3 M2-certified FV draws (fv arm) and
the 3 M1 lens draws (jspace arm). Every ablation effect is a 3-draw DrawSet;
the double-dissociation verdict follows the pre-registered DissociationRule.
Read-only on the instruments it consumes — each is asserted gated
(require_controlled) against its committed ControlRecord before use.

Context sets (exec queries; report contexts) are sampled ONCE and reused
across every condition and draw, so each effect is a paired clean-vs-ablated
contrast (lower variance — the weak P3 report signal, D-016).

Prereg: harness/preregs/EXP-M4-E2-dissociation.md (committed before first run;
start_run enforces).

Usage: uv run python scripts/m4_e2_dissociation.py [--config configs/m4_e2_dissociation_pythia410m.yaml]
"""

from __future__ import annotations

import argparse
import dataclasses
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
from jvec.evals.exp3 import REPORT_LABELS, REPORT_PROBES, final_logits_under, make_hooks
from jvec.fv import FV_REPO, load_fv_model
from jvec.lens_cache import load_lens
from jvec.utils import peak_rss_gb, set_seed
from jtvec.core.draws import DrawSet
from jtvec.core.instruments import ControlRecord, Instrument, require_controlled
from jtvec.core.reporting import scoped
from jtvec.core.runctx import start_run
from jtvec.e2_dissociation import DissociationRule, effect_drawset
from jtvec.m3_instruments import answer_first_tokens, load_certified_fv, verify_lens_manifest

M1_RUN = REPO_ROOT / "results/m1/20260718-010559-lens-gate"
M2_RUN = REPO_ROOT / "results/m2/20260718-114950-fv-stability-gate"
M3_RUN = REPO_ROOT / "results/m3/20260718-174954-instrument-gate"
RPTGATE_RUN = REPO_ROOT / "results/m4/20260719-053911-e2-reportgate"
PREREG = REPO_ROOT / "harness/preregs/EXP-M4-E2-dissociation.md"

TASK = "singular-plural"
FV_DRAWS = (1, 2, 3)     # M2 certified FV cache draws (fv arm)
LENS_DRAWS = (0, 1, 2)   # M1 lens draws (jspace arm; jspace reads the lens)
N_EXEC = 50
N_REPORT = 80
CTX_RNG_SEED = 5252
M_TOP = 10
PROBE_NAME = "P3"        # the report instrument gated under P3 only (report-gate)
RULE = DissociationRule(min_exec_drop=0.15, min_report_drop=0.10)
FV_SHAM_SEED0 = 200      # make_hooks random-direction seeds (distinct per draw)
JS_SHAM_SEED0 = 300


def assert_gated(name: str, controls_path: Path, task_hint: str = "") -> None:
    """Load a committed ControlRecord and require_controlled before use."""
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    if name not in controls:
        sys.exit(f"instrument '{name}' not found in {controls_path}")
    entry = controls[name]
    today = time.strftime("%Y-%m-%d")
    require_controlled(
        Instrument(
            name=name,
            positive_control=ControlRecord(run=str(controls_path.parent),
                                           passed=bool(entry["positive"]), date=today),
            negative_control=ControlRecord(run=str(controls_path.parent),
                                           passed=bool(entry["negative"]), date=today),
        )
    )
    print(f"[gated] {name} {task_hint}(ControlRecord {controls_path.parent.name})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs/m4_e2_dissociation_pythia410m.yaml")
    )
    args = parser.parse_args()
    t_start = time.perf_counter()

    cfg = Config.load(args.config)
    ctx = start_run(
        repo_root=REPO_ROOT,
        config_path=Path(args.config),
        results_root=REPO_ROOT / cfg.results_dir,
        run_name="e2-dissociation",
        prereg_path=PREREG,
    )
    print(f"E2 dissociation run dir: {ctx.results_dir}", flush=True)

    # --- instruments consumed: assert each is gated ----------------------------
    assert_gated("fv-direction-ablation", M3_RUN / "controls.json")
    assert_gated("jspace-ablation", M3_RUN / "controls.json")
    assert_gated("report-score-prior-corrected@singular-plural",
                 RPTGATE_RUN / "controls.json")

    set_seed(cfg.seed)
    hf_model, tokenizer, model_config, revision = load_fv_model(cfg)
    import jlens as jlens_pkg  # noqa: PLC0415

    model_j = jlens_pkg.from_hf(hf_model, tokenizer)
    W_U = hf_model.embed_out.weight.detach().float().cpu()
    model_scope = f"{cfg.model.name}@{revision[:7]}"
    config_scope = f"EXP-M4-E2-dissociation ({Path(args.config).name})"
    bos = tokenizer.bos_token or ""
    lo, hi = cfg.evals.band
    probe = REPORT_PROBES[PROBE_NAME]
    label_id = tokenizer(" " + REPORT_LABELS[TASK], add_special_tokens=False).input_ids[0]

    from utils.prompt_utils import load_dataset  # noqa: PLC0415

    # --- lens draws (cache hits, identity-checked): draw 0 is the M3-verified
    #     fit under cache/m3 (as E1 loaded it read-only); draws 1-2 are E1's
    #     fits under cache/m4e1/lensdraw{j}. The lens cache key excludes seed.
    from jvec.lens_cache import lens_dir  # noqa: PLC0415

    lenses = {}
    for j in LENS_DRAWS:
        cache = "cache/m3" if j == 0 else f"cache/m4e1/lensdraw{j}"
        dcfg = dataclasses.replace(cfg, seed=j, cache_dir=cache)
        set_seed(dcfg.seed)
        prompts = select_prompts(dcfg, tokenizer)
        lens = load_lens(dcfg, 4, prompts, revision)
        manifest = json.loads((lens_dir(dcfg, 4) / "manifest.json").read_text(encoding="utf-8"))
        reference = json.loads((M1_RUN / f"draws/draw{j}/manifest.json").read_text(encoding="utf-8"))
        mism = verify_lens_manifest(manifest, reference)
        if mism:
            sys.exit(f"lens draw{j} identity mismatch vs M1: {mism}")
        lenses[j] = lens
    band_layers = [l for l in lenses[0].source_layers if lo <= l <= hi]
    print(f"[lens] 3 draws loaded + identity-checked; band layers {band_layers}", flush=True)

    # --- certified FVs (fv arm) ------------------------------------------------
    certificates = json.loads((M2_RUN / "certificates.json").read_text(encoding="utf-8"))
    fvs = {k: load_certified_fv(cfg, TASK, revision, certificates, draw_k=k) for k in FV_DRAWS}
    print(f"[fvs] certified {TASK} FVs loaded: draws {FV_DRAWS}", flush=True)

    set_seed(cfg.seed)
    dataset = load_dataset(TASK, root_data_dir=str(FV_REPO / "dataset_files"), seed=cfg.seed)
    neutral_tasks = [t for t in cfg.fv.tasks if t != TASK]
    neutral_ds = {t: load_dataset(t, root_data_dir=str(FV_REPO / "dataset_files"), seed=cfg.seed)
                  for t in neutral_tasks}
    rng = np.random.default_rng(CTX_RNG_SEED)
    n_shots = cfg.fv.n_shots

    def sp_pairs(split, n):
        ds = dataset[split]
        idx = rng.choice(len(ds), n, replace=False)
        chosen = ds[idx]
        return list(zip(chosen["input"], chosen["output"]))

    def context(pairs):
        return bos + "".join(f"Q: {x}\nA: {y}\n\n" for x, y in pairs)

    # --- fixed, paired context sets (sampled once; reused across conditions) ---
    exec_items = []
    for _ in range(N_EXEC):
        pairs = sp_pairs("train", n_shots)
        qx, qy = sp_pairs("test", 1)[0]
        exec_items.append((pairs, str(qx), str(qy)))
    report_items = [sp_pairs("train", n_shots) for _ in range(N_REPORT)]

    # --- P3 neutral baseline (mixed other-task contexts), once -----------------
    def neutral_pairs(n):
        chosen = rng.choice(neutral_tasks, n, replace=True)
        out = []
        for t in chosen:
            ds = neutral_ds[t]["train"]
            i = int(rng.integers(len(ds)))
            out.append((ds[i]["input"], ds[i]["output"]))
        return out

    @torch.no_grad()
    def logp_label(prompt, hooks):
        logits = final_logits_under(model_j, prompt, hooks)
        return float(torch.log_softmax(logits, dim=-1)[label_id]), logits

    baseline_vals = []
    nb_rows = []
    for i in range(N_REPORT):
        lp, _ = logp_label(context(neutral_pairs(n_shots)) + probe, {})
        baseline_vals.append(lp)
        nb_rows.append({"trial": i, "label_logprob": round(lp, 4)})
    baseline = float(np.mean(baseline_vals))
    ctx.save_raw_completions(f"neutral_{PROBE_NAME}", nb_rows)
    print(f"[neutral] P3 baseline log p(' plural') = {baseline:.3f}", flush=True)

    # --- measurement primitives (save raw cells) -------------------------------
    def measure_exec(hooks, cell) -> float:
        hits, rows = 0, []
        for pairs, qx, qy in exec_items:
            logits = final_logits_under(model_j, context(pairs) + f"Q: {qx}\nA:", hooks)
            top1 = int(logits.argmax())
            hit = top1 in answer_first_tokens(tokenizer, qy, case_sensitive=True)
            hits += hit
            rows.append({"query": qx, "target": qy, "top1": tokenizer.decode([top1]),
                         "hit": bool(hit)})
        ctx.save_raw_completions(cell, rows)
        return hits / len(exec_items)

    def measure_report(hooks, cell) -> float:
        scores, rows = [], []
        for pairs in report_items:
            lp, logits = logp_label(context(pairs) + probe, hooks)
            score = lp - baseline
            scores.append(score)
            rows.append({"label_logprob": round(lp, 4), "report_score": round(score, 4),
                         "top1": tokenizer.decode([int(logits.argmax())])})
        ctx.save_raw_completions(cell, rows)
        return float(np.mean(scores))

    # --- clean (shared) --------------------------------------------------------
    exec_clean = measure_exec({}, "exec_none")
    report_clean = measure_report({}, "report_none")
    print(f"[clean] exec {exec_clean:.3f} | report_score {report_clean:+.3f}", flush=True)

    # --- fv arm: per certified FV draw -----------------------------------------
    fv_exec_abl, fv_exec_sham, fv_rep_abl, fv_rep_sham = {}, {}, {}, {}
    for k in FV_DRAWS:
        fv_hooks = make_hooks("fv", band_layers, lenses[0], W_U, fvs[k].vector,
                              m_top=M_TOP, seed=cfg.seed)
        sham_hooks = make_hooks("sham_fv", band_layers, lenses[0], W_U, fvs[k].vector,
                                m_top=M_TOP, seed=FV_SHAM_SEED0 + k)
        fv_exec_abl[k] = measure_exec(fv_hooks, f"exec_fv_draw{k}")
        fv_exec_sham[k] = measure_exec(sham_hooks, f"exec_sham_fv_draw{k}")
        fv_rep_abl[k] = measure_report(fv_hooks, f"report_fv_draw{k}")
        fv_rep_sham[k] = measure_report(sham_hooks, f"report_sham_fv_draw{k}")
        print(f"[fv draw{k}] exec {exec_clean:.3f}->{fv_exec_abl[k]:.3f} "
              f"(sham {fv_exec_sham[k]:.3f}) | report {report_clean:+.3f}->"
              f"{fv_rep_abl[k]:+.3f} (sham {fv_rep_sham[k]:+.3f})", flush=True)

    # --- jspace arm: per lens draw ---------------------------------------------
    js_exec_abl, js_exec_sham, js_rep_abl, js_rep_sham = {}, {}, {}, {}
    for j in LENS_DRAWS:
        js_hooks = make_hooks("jspace", band_layers, lenses[j], W_U, fvs[FV_DRAWS[0]].vector,
                              m_top=M_TOP, seed=cfg.seed)
        sham_hooks = make_hooks("sham_jspace", band_layers, lenses[j], W_U,
                                fvs[FV_DRAWS[0]].vector, m_top=M_TOP, seed=JS_SHAM_SEED0 + j)
        js_exec_abl[j] = measure_exec(js_hooks, f"exec_jspace_lens{j}")
        js_exec_sham[j] = measure_exec(sham_hooks, f"exec_sham_jspace_lens{j}")
        js_rep_abl[j] = measure_report(js_hooks, f"report_jspace_lens{j}")
        js_rep_sham[j] = measure_report(sham_hooks, f"report_sham_jspace_lens{j}")
        print(f"[jspace lens{j}] exec {exec_clean:.3f}->{js_exec_abl[j]:.3f} "
              f"(sham {js_exec_sham[j]:.3f}) | report {report_clean:+.3f}->"
              f"{js_rep_abl[j]:+.3f} (sham {js_rep_sham[j]:+.3f})", flush=True)

    # --- effects (DrawSets) + verdict ------------------------------------------
    verdict = RULE.verdict(
        fv_exec=effect_drawset(exec_clean, fv_exec_abl),
        fv_exec_sham=effect_drawset(exec_clean, fv_exec_sham),
        fv_report=effect_drawset(report_clean, fv_rep_abl),
        fv_report_sham=effect_drawset(report_clean, fv_rep_sham),
        jspace_exec=effect_drawset(exec_clean, js_exec_abl),
        jspace_exec_sham=effect_drawset(exec_clean, js_exec_sham),
        jspace_report=effect_drawset(report_clean, js_rep_abl),
        jspace_report_sham=effect_drawset(report_clean, js_rep_sham),
    )
    (ctx.results_dir / "e2_results.json").write_text(
        json.dumps({"exec_clean": exec_clean, "report_clean": report_clean,
                    "baseline_P3": baseline, "verdict": verdict}, indent=2, default=str),
        encoding="utf-8",
    )

    # --- report ----------------------------------------------------------------
    eff = verdict["effects"]
    lines = [
        "# EXP-M4-E2 report: execution/verbalization dissociation on singular-plural",
        "",
        f"- model: {model_scope} (full sha in run.json/config)",
        "- prereg: harness/preregs/EXP-M4-E2-dissociation.md (constants D-017)",
        f"- clean: execution {exec_clean:.3f} (N={N_EXEC}); report_score {report_clean:+.3f} "
        f"(P3, N={N_REPORT}, baseline {baseline:+.3f})",
        "- ablations: fv-direction (3 certified FV draws), jspace (3 M1 lens draws); "
        "each vs matched sham; paired context sets across conditions",
        f"- decision (D-017): an ablation hurts a measure iff effect_median - sham_median "
        f">= delta (exec {RULE.min_exec_drop}, report {RULE.min_report_drop})",
        "",
        "## Effects (clean - ablated; median [IQR] over 3 draws, vs sham)",
        "",
        "| ablation x measure | effect med | effect IQR | sham med | effect-sham | hurts? |",
        "|---|---|---|---|---|---|",
    ]
    flags = {"fv_exec": verdict["fv_hurts_exec"], "fv_report": verdict["fv_hurts_report"],
             "jspace_exec": verdict["jspace_hurts_exec"],
             "jspace_report": verdict["jspace_hurts_report"]}
    for key in ("fv_exec", "fv_report", "jspace_exec", "jspace_report"):
        e = eff[key]
        lines.append(
            f"| {key.replace('_', ' x ')} | {e['effect_median']:+.3f} | {e['effect_iqr']:.3f} "
            f"| {e['sham_median']:+.3f} | {e['effect_minus_sham']:+.3f} | "
            f"{'YES' if flags[key] else 'no'} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        "- " + scoped(
            f"E2 {TASK}: fv-ablation hurts execution={verdict['fv_hurts_exec']} "
            f"report={verdict['fv_hurts_report']}; jspace-ablation hurts "
            f"report={verdict['jspace_hurts_report']} execution={verdict['jspace_hurts_exec']}; "
            f"verdict {verdict['verdict']} (cross-draw transfer "
            f"fv_exec={verdict['cross_draw_transfer']['fv_exec']}, "
            f"jspace_report={verdict['cross_draw_transfer']['jspace_report']})",
            float(eff["fv_exec"]["effect_minus_sham"]),
            model=model_scope, config=config_scope, n=N_EXEC,
        ),
        "",
        f"**E2 verdict: {verdict['verdict']}** "
        f"(direction1 fv-exec-not-report={verdict['direction1_fv_exec_not_report']}, "
        f"direction2 jspace-report-not-exec={verdict['direction2_jspace_report_not_exec']})",
        "",
        f"wall-clock {round(time.perf_counter() - t_start, 1)} s; peak RSS "
        f"{peak_rss_gb():.2f} GB; device {cfg.device}; grids in e2_results.json; "
        "raw per-item cells under raw_completions/.",
        "",
    ]
    (ctx.results_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    ctx.finalize(
        verdict=verdict["verdict"],
        directions={"fv_exec_not_report": verdict["direction1_fv_exec_not_report"],
                    "jspace_report_not_exec": verdict["direction2_jspace_report_not_exec"]},
        cross_draw_transfer=verdict["cross_draw_transfer"],
        exec_clean=exec_clean, report_clean=report_clean,
        model_revision=revision,
        wall_clock_s=round(time.perf_counter() - t_start, 1),
        peak_rss_gb=round(peak_rss_gb(), 2),
    )
    print(f"\nE2 verdict: {verdict['verdict']}")
    print(f"report: {ctx.results_dir / 'report.md'}")


if __name__ == "__main__":
    main()
