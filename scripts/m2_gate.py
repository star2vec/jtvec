"""M2 orchestrator: FV extraction-stability gate on Pythia-410M.

Decides the CONSTRAINTS known-unknown "at which AIE trial count does Todd-FV
extraction converge on the target model (if at all)". Per task (3), per draw
(3, distinct extraction seeds, everything else fixed): one extraction at the
max rung (n_trials_aie = 200) through vendored jvec.fv; every lower rung
(25/50/100) derives from the stored per-trial AIE tensor. Per (draw, rung):
zero-shot induction gain on fixed eval contexts, with a norm-matched random
sham twin at the same layer/position (sham LAW). Cross-draw agreement per
rung feeds the preregistered convergence rule; converged tasks get gate
certificates (jtvec.core.gate payloads in certificates.json).

The induction readout is treated as an instrument: its positive control
(10-shot ICL separates from 0-shot) and negative control (sham gains near
zero) are evaluated in-run and gate the verdict computation, M1-style.

Prereg: harness/preregs/EXP-M2-fv-stability.md (committed before first run;
start_run enforces). Design reference: design_input/
15_fv_stability_v1_untracked.py (v1, unvalidated, never imported).

Usage: uv run python scripts/m2_gate.py [--config configs/m2_pythia410m.yaml]
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

import torch

from jvec.config import Config
from jvec.fv import FV_REPO, extract_task_fvs, load_cached_fv, load_fv_model
from jvec.utils import peak_rss_gb, set_seed
from jtvec.core.draws import DrawSet
from jtvec.core.instruments import ControlRecord, Instrument, require_controlled
from jtvec.core.intervention import InterventionResult, InterventionSpec, ShamResult
from jtvec.core.reporting import scoped, scoped_intervention
from jtvec.core.runctx import start_run
from jtvec.fv_stability import (
    RUNGS,
    ConvergenceRule,
    RungStats,
    certificate_payload,
    convergence_verdict,
    fv_at_rung,
    pairwise_cosines,
    sham_twin,
    top_head_overlap,
)

DRAW_KS = (1, 2, 3)
EVAL_SEED = 999  # design-ref constant: evaluation contexts held fixed
SHAM_SEED_BASE = 9000  # sham seed = base + 10*draw_k + rung_index (preregistered)

# Tolerances live in the prereg (EXP-M2, Decision rule); mirrored here.
RULE = ConvergenceRule(min_pairwise_cosine=0.95, max_gain_iqr=0.05)
POSITIVE_CONTROL_MIN_SEPARATION = 0.10  # icl10 top1 - zeroshot top1, per task
# D-010: per-task bound max(0.02, 1/N_test) — the sham may move the median by
# at most one readout quantum (prereg Deviations; amended after run 1).
NEGATIVE_CONTROL_MAX_ABS_SHAM = 0.02


def draw_seed(cfg: Config, k: int) -> int:
    return cfg.seed * 1000 + k  # design-ref derivation: only this stream varies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/m2_pythia410m.yaml"))
    args = parser.parse_args()
    t_start = time.perf_counter()

    cfg = Config.load(args.config)
    ctx = start_run(
        repo_root=REPO_ROOT,
        config_path=Path(args.config),
        results_root=REPO_ROOT / cfg.results_dir,
        run_name="fv-stability-gate",
        prereg_path=REPO_ROOT / "harness/preregs/EXP-M2-fv-stability.md",
    )
    print(f"M2 run dir: {ctx.results_dir}", flush=True)

    model, tokenizer, model_config, revision = load_fv_model(cfg)
    model_scope = f"{cfg.model.name}@{revision[:7]}"
    config_scope = f"m2 fv gate ({Path(args.config).name})"

    from utils.eval_utils import n_shot_eval, n_shot_eval_no_intervention  # noqa: PLC0415
    from utils.prompt_utils import load_dataset  # noqa: PLC0415

    def eval_no_fv(dataset, n_shots: int):
        set_seed(EVAL_SEED)
        res = n_shot_eval_no_intervention(
            dataset, n_shots, model, model_config, tokenizer,
            compute_ppl=False, test_split="test",
        )
        return dict(res["clean_topk"])[1], res["clean_rank_list"]

    def eval_with_vector(dataset, vec: torch.Tensor):
        set_seed(EVAL_SEED)  # identical eval contexts for every draw/rung/arm
        res = n_shot_eval(
            dataset, vec.reshape(1, -1).to(model.device), cfg.fv.edit_layer,
            0, model, model_config, tokenizer,
        )
        return dict(res["intervention_topk"])[1], res["intervention_rank_list"]

    # --- extraction: 3 tasks x 3 draws at the max rung, cached per draw -------
    artifacts: dict[tuple[str, int], dict] = {}
    for task in cfg.fv.tasks:
        for k in DRAW_KS:
            dcfg = dataclasses.replace(cfg, cache_dir=f"{cfg.cache_dir}/draw{k}")
            art = load_cached_fv(dcfg, task, revision)  # manifest mismatch raises
            if art is None:
                print(f"[extracting] {task}/draw{k}", flush=True)
                t0 = time.perf_counter()
                set_seed(draw_seed(cfg, k))  # vary ONLY the extraction stream
                art = extract_task_fvs(dcfg, task, model, tokenizer, model_config, revision)
                print(
                    f"  {task}/draw{k}: {(time.perf_counter() - t0) / 60:.1f} min",
                    flush=True,
                )
            else:
                print(f"[cache hit] {task}/draw{k}", flush=True)
            artifacts[(task, k)] = art

    # --- per-task evals, agreement stats, controls ----------------------------
    tasks_out: dict[str, dict] = {}
    for task in cfg.fv.tasks:
        dataset = load_dataset(task, root_data_dir=str(FV_REPO / "dataset_files"), seed=cfg.seed)

        zs_top1, zs_ranks = eval_no_fv(dataset, 0)
        icl10_top1, icl10_ranks = eval_no_fv(dataset, cfg.fv.n_shots)
        n_test = len(zs_ranks)
        ctx.save_raw_completions(
            f"{task}_zeroshot", [{"item": j, "rank": int(r)} for j, r in enumerate(zs_ranks)]
        )
        ctx.save_raw_completions(
            f"{task}_icl{cfg.fv.n_shots}shot",
            [{"item": j, "rank": int(r)} for j, r in enumerate(icl10_ranks)],
        )

        per_rung_stats: list[RungStats] = []
        rungs_out: dict[int, dict] = {}
        for rung_idx, T in enumerate(RUNGS):
            fvs, heads, gains, sham_gains, norms = {}, {}, {}, {}, {}
            for k in DRAW_KS:
                art = artifacts[(task, k)]
                fv, top_heads = fv_at_rung(
                    art["mean_head_activations"], art["indirect_effect"], T,
                    model, model_config, cfg.fv.n_top_heads,
                )
                fvs[f"draw{k}"] = fv
                heads[k] = top_heads
                norms[k] = float(fv.norm())

                top1, ranks = eval_with_vector(dataset, fv)
                # unrounded: the D-010 bound is exact (1/N_test); rounding
                # belongs in display only (run 2 lesson, LABNOTES)
                gains[k] = top1 - zs_top1
                ctx.save_raw_completions(
                    f"{task}_rung{T}_induction",
                    [
                        {"draw": k, "seed": draw_seed(cfg, k), "item": j, "rank": int(r)}
                        for j, r in enumerate(ranks)
                    ],
                )

                sham = sham_twin(fv, SHAM_SEED_BASE + 10 * k + rung_idx)
                sham_top1, sham_ranks = eval_with_vector(dataset, sham)
                sham_gains[k] = sham_top1 - zs_top1
                ctx.save_raw_completions(
                    f"{task}_rung{T}_sham",
                    [
                        {"draw": k, "seed": draw_seed(cfg, k), "item": j, "rank": int(r)}
                        for j, r in enumerate(sham_ranks)
                    ],
                )

            seeds = tuple(draw_seed(cfg, k) for k in DRAW_KS)
            gain_ds = DrawSet(values=tuple(gains[k] for k in DRAW_KS), seeds=seeds)
            sham_ds = DrawSet(values=tuple(sham_gains[k] for k in DRAW_KS), seeds=seeds)
            cosines = pairwise_cosines(fvs)
            overlaps = {
                f"draw{a}|draw{b}": top_head_overlap(heads[a], heads[b])
                for a, b in ((1, 2), (1, 3), (2, 3))
            }
            per_rung_stats.append(
                RungStats(
                    n_trials=T,
                    min_pairwise_cosine=min(cosines.values()),
                    gain_iqr=gain_ds.iqr,
                    min_top_head_overlap=min(overlaps.values()),
                )
            )
            rungs_out[T] = {
                "cosines": cosines,
                "top_head_overlaps": overlaps,
                "top_heads": {f"draw{k}": heads[k] for k in DRAW_KS},
                "fv_norms": norms,
                "gains": gains,
                "sham_gains": sham_gains,
                "gain_median": gain_ds.median,
                "gain_iqr": gain_ds.iqr,
                "sham_median": sham_ds.median,
                "sham_iqr": sham_ds.iqr,
                "zero_shot_top1": zs_top1,
                "n_test": n_test,
            }
            print(
                f"{task:16s} T={T:<3d} min_cos {min(cosines.values()):+.3f}  "
                f"gain {gain_ds.median:+.3f} (IQR {gain_ds.iqr:.3f})  "
                f"sham {sham_ds.median:+.3f}  overlap>={min(overlaps.values())}",
                flush=True,
            )

        # Hendel vectors: descriptive stability only (fixed n_trials_mean;
        # no certificate — scoped out in the prereg).
        hendel = {f"draw{k}": artifacts[(task, k)]["fv_hendel"] for k in DRAW_KS}
        hendel_stats = {
            "flat_cosines": pairwise_cosines({n: v.flatten() for n, v in hendel.items()}),
            "edit_layer_cosines": pairwise_cosines(
                {n: v[cfg.fv.edit_layer] for n, v in hendel.items()}
            ),
        }

        tasks_out[task] = {
            "zero_shot_top1": zs_top1,
            "icl10_top1": icl10_top1,
            "n_test": n_test,
            "rungs": rungs_out,
            "per_rung_stats": per_rung_stats,
            "hendel": hendel_stats,
        }

    # --- instrument controls gate the verdicts (instruments LAW) --------------
    positive_pass = all(
        t["icl10_top1"] - t["zero_shot_top1"] >= POSITIVE_CONTROL_MIN_SEPARATION
        for t in tasks_out.values()
    )
    negative_pass = all(
        abs(rung["sham_median"]) <= max(NEGATIVE_CONTROL_MAX_ABS_SHAM, 1.0 / t["n_test"])
        for t in tasks_out.values()
        for rung in t["rungs"].values()
    )
    today = time.strftime("%Y-%m-%d")
    readout = Instrument(
        name="fv-induction-readout",
        positive_control=ControlRecord(run=str(ctx.results_dir), passed=positive_pass, date=today),
        negative_control=ControlRecord(run=str(ctx.results_dir), passed=negative_pass, date=today),
    )
    require_controlled(readout)  # raises -> no verdicts on an uncontrolled readout

    # --- verdicts and certificates ---------------------------------------------
    verdicts = {
        task: convergence_verdict(t["per_rung_stats"], RULE) for task, t in tasks_out.items()
    }
    certificates = {
        task: certificate_payload(
            estimator="fv_todd",
            task=task,
            model=model_scope,
            converged_at=v["converged_at"],
            n_draws=len(DRAW_KS),
            evidence_run=str(ctx.results_dir),
            issued=today,
        )
        for task, v in verdicts.items()
        if v["converged"]
    }
    (ctx.results_dir / "certificates.json").write_text(
        json.dumps(certificates, indent=2), encoding="utf-8"
    )
    all_converged = all(v["converged"] for v in verdicts.values())

    # --- report ------------------------------------------------------------------
    lines = [
        "# M2 report: FV extraction-stability gate",
        "",
        f"- model: {model_scope} (full sha in run.json/config)",
        "- prereg: harness/preregs/EXP-M2-fv-stability.md",
        f"- draws: {len(DRAW_KS)} independent extraction streams "
        f"(seeds {[draw_seed(cfg, k) for k in DRAW_KS]}); datasets, weights, and "
        "eval contexts held fixed (0-shot evals are context-deterministic)",
        f"- ladder: extraction at n_trials_aie={max(RUNGS)}; rungs {list(RUNGS)} from "
        "stored per-trial AIE prefixes",
        "- sham twins: norm-matched random directions at the same layer/position, "
        f"seeds {SHAM_SEED_BASE}+10k+rung_index",
        "- instrument controls (fv-induction-readout): positive = 10-shot ICL vs "
        f"0-shot separation >= {POSITIVE_CONTROL_MIN_SEPARATION} per task -> "
        f"{'PASS' if positive_pass else 'FAIL'}; negative = |median sham gain| <= "
        f"max({NEGATIVE_CONTROL_MAX_ABS_SHAM}, 1/N_test) at every rung (D-010) -> "
        f"{'PASS' if negative_pass else 'FAIL'}",
        "",
        "## Verdicts (rule: min pairwise cosine >= "
        f"{RULE.min_pairwise_cosine} AND gain IQR <= {RULE.max_gain_iqr} at T and "
        "every larger rung; largest rung alone does not converge)",
        "",
        "| task | converged | converged_at | max-rung-only pass |",
        "|---|---|---|---|",
    ]
    for task, v in verdicts.items():
        lines.append(
            f"| {task} | {'YES' if v['converged'] else 'NO'} | "
            f"{v['converged_at'] or '—'} | {v['passes_at_max_rung_only']} |"
        )
    lines += ["", "## Cross-draw agreement per rung", ""]
    for task, t in tasks_out.items():
        lines += [
            f"### {task} (zero-shot top-1 {t['zero_shot_top1']:.3f}, "
            f"10-shot {t['icl10_top1']:.3f}, N={t['n_test']})",
            "",
            "| rung | min cos | min overlap | gain median | gain IQR | sham median | sham IQR | pass |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for T in RUNGS:
            r = t["rungs"][T]
            ok = verdicts[task]["per_rung"][T]["pass"]
            lines.append(
                f"| {T} | {min(r['cosines'].values()):+.3f} | "
                f"{min(r['top_head_overlaps'].values())}/{cfg.fv.n_top_heads} | "
                f"{r['gain_median']:+.4f} | {r['gain_iqr']:.4f} | "
                f"{r['sham_median']:+.4f} | {r['sham_iqr']:.4f} | "
                f"{'PASS' if ok else 'FAIL'} |"
            )
        lines.append("")
        lines.append(
            "Hendel (descriptive, fixed n_trials_mean="
            f"{cfg.fv.n_trials_mean}): flat cosines {t['hendel']['flat_cosines']}, "
            f"edit-layer cosines {t['hendel']['edit_layer_cosines']}"
        )
        if t["n_test"] < 100:  # D-009: small-task caveat travels with the numbers
            lines.append(
                f"Caveat (D-009): N={t['n_test']} makes the gain criterion coarse "
                f"(top-1 granularity {1 / t['n_test']:.3f}); ruled kept-and-flagged."
            )
        lines.append("")

    lines += ["## Headline numbers (median/IQR over draws; sham in the same line)", ""]
    seeds = tuple(draw_seed(cfg, k) for k in DRAW_KS)
    for task, t in tasks_out.items():
        v = verdicts[task]
        T = v["converged_at"] or max(RUNGS)
        r = t["rungs"][T]
        gain_ds = DrawSet(values=tuple(r["gains"][k] for k in DRAW_KS), seeds=seeds)
        sham_ds = DrawSet(values=tuple(r["sham_gains"][k] for k in DRAW_KS), seeds=seeds)
        lines.append(
            "- "
            + scoped_intervention(
                f"{task} induction gain @T={T}",
                gain_ds,
                sham_ds,
                model=model_scope,
                config=config_scope,
                n=r["n_test"],
            )
        )
        median_norm = sorted(r["fv_norms"].values())[1]
        spec = InterventionSpec(
            kind="inject", layers=(cfg.fv.edit_layer,), positions=(-1,),
            n_directions=1, norm=median_norm,
        )
        row = InterventionResult(
            spec=spec,
            effect=gain_ds,
            sham=ShamResult(spec=spec, effect=sham_ds),
        )
        lines.append(f"  {row.table_row()}")
    lines += [
        "",
        "- "
        + scoped(
            "M2 gate verdict (1=all tasks converged)",
            float(all_converged),
            model=model_scope,
            config=config_scope,
            n=len(tasks_out),
        ),
        "",
        "## Provenance",
        "",
        "| task | draw | seed | mean-acts (s) | AIE (s) | hendel (s) |",
        "|---|---|---|---|---|---|",
    ]
    for task in cfg.fv.tasks:
        for k in DRAW_KS:
            timings = artifacts[(task, k)]["manifest"].get("timings", {})
            lines.append(
                f"| {task} | {k} | {draw_seed(cfg, k)} | "
                f"{timings.get('mean_head_activations_s', '—')} | "
                f"{timings.get('indirect_effect_s', '—')} | "
                f"{timings.get('hendel_residuals_s', '—')} |"
            )
    wall_s = round(time.perf_counter() - t_start, 1)
    lines += [
        "",
        f"wall-clock {wall_s} s; peak RSS {peak_rss_gb():.2f} GB; device {cfg.device}; "
        f"todd_commit {artifacts[(cfg.fv.tasks[0], 1)]['manifest']['todd_commit']}; "
        "per-draw FV caches under cache/m2/draw*/fvs/; raw per-item cells under "
        "raw_completions/.",
        "",
    ]
    (ctx.results_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    def _stats_dict(stats: list[RungStats]) -> list[dict]:
        return [dataclasses.asdict(s) for s in stats]

    (ctx.results_dir / "stability.json").write_text(
        json.dumps(
            {
                task: {
                    "verdict": verdicts[task],
                    "rungs": t["rungs"],
                    "per_rung_stats": _stats_dict(t["per_rung_stats"]),
                    "hendel": t["hendel"],
                    "zero_shot_top1": t["zero_shot_top1"],
                    "icl10_top1": t["icl10_top1"],
                    "n_test": t["n_test"],
                }
                for task, t in tasks_out.items()
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    ctx.finalize(
        draw_seeds=[draw_seed(cfg, k) for k in DRAW_KS],
        rungs=list(RUNGS),
        rule={"min_pairwise_cosine": RULE.min_pairwise_cosine, "max_gain_iqr": RULE.max_gain_iqr},
        instrument_controls={"positive": positive_pass, "negative": negative_pass},
        verdicts={t: v["converged_at"] for t, v in verdicts.items()},
        m2_all_converged=all_converged,
        model_revision=revision,
        wall_clock_s=wall_s,
        peak_rss_gb=round(peak_rss_gb(), 2),
    )
    print(f"\nM2 verdict: {'all tasks converged' if all_converged else 'NOT all converged'}")
    for task, v in verdicts.items():
        print(f"  {task}: converged_at={v['converged_at']}")
    print(f"report: {ctx.results_dir / 'report.md'}")


if __name__ == "__main__":
    main()
