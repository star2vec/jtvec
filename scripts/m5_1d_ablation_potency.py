"""EXP-M5-1d: S1 ablation-potency probe (410M) — the REMOVE arm of the S1
potency dissociation. Projects the EXP-M5-1b concept direction d(c) out of the
residual at the final position of band layers 4-16 (E2 machinery) and measures
whether the model can still produce c on capital-recall.

Per concept, 3 draws (the same seeds as 1b), sham twin, IDENTICAL E2 statistic:
  g = (clean_acc - ablated_acc).median - (clean_acc - sham_acc).median
Split, non-adjacent poles (Ecaterina 2026-07-22): ABLATION-POTENT iff g >= 0.15
WITH cross-draw transfer; ABLATION-INERT iff g <= 0.05 (at floor); 0.05<g<0.15
weak/ambiguous, neither pole. Roster: >=6/8 potent -> H-POTENT; >=6/8 inert ->
H-INERT; else MIXED. Positive control (same statistic + 3-draw marginalization):
ablating the unembed direction of c must drop c-accuracy >= 0.30 above sham, else
INCONCLUSIVE (not inert). Mechanism positive control on record = the null-check
+0.80 "Paris" injection. Issues NO certificate under either branch.

Prereg: harness/preregs/EXP-M5-1d-ablation-potency.md (RATIFIED, committed 05b3490).
Usage: uv run python scripts/m5_1d_ablation_potency.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from jvec.config import Config
from jvec.evals.exp3 import ProjectOutHook, final_logits_under
from jvec.evals.swap import _unembed_direction
from jvec.evals.tasks import surface_token_ids
from jvec.modeling import load_model
from jvec.utils import peak_rss_gb, set_seed
from jtvec.concept_gate import (
    CAPITAL_ROSTER, capital_context_stream, classify_ablation,
    mean_difference_by_layer,
)
from jtvec.core.draws import DrawSet
from jtvec.core.runctx import start_run
from jtvec.e2_dissociation import effect_drawset
from scripts.m5_1_concept_gate import neg_pool_stream
from jvec.evals.concept import answer_states

PREREG = REPO_ROOT / "harness/preregs/EXP-M5-1d-ablation-potency.md"
CFG = REPO_ROOT / "configs/m5_1d_ablation_pythia410m.yaml"
DRAWS = (1, 2, 3)
N_EVAL, EVAL_SEED, SHAM_SEED0, N_EXTRACT = 30, 9000, 4200, 64
POTENT_BAR, INERT_BAR, POS_BAR = 1.0, 0.3, 1.0  # LOGIT units (Option B, 2026-07-22)
ROSTER_RULE = 6  # of 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CFG))
    args = parser.parse_args()
    t0 = time.perf_counter()
    cfg = Config.load(args.config)
    ctx = start_run(repo_root=REPO_ROOT, config_path=Path(args.config),
                    results_root=REPO_ROOT / cfg.results_dir, run_name="m5-1d-ablation-potency",
                    prereg_path=PREREG)
    print(f"M5.1d run dir: {ctx.results_dir}", flush=True)
    set_seed(cfg.seed)
    model, tok, revision = load_model(cfg)
    model_scope = f"{cfg.model.name}@{revision[:7]}"
    lo, hi = cfg.evals.band
    band = [l for l in range(model.n_layers) if lo <= l <= hi]
    print(f"[model] {model_scope}; band {band}", flush=True)

    def mean_logit(prompts, ans_id, hooks) -> float:
        """Mean final-position logit of the answer's primary surface token (Option
        B: the argmax-insensitive measure the pre-run diagnostic forced)."""
        return sum(float(final_logits_under(model, p, hooks)[ans_id]) for p in prompts) / len(prompts)

    def proj_hooks(dirs_by_layer) -> dict:
        return {l: ProjectOutHook(dirs_by_layer[l].reshape(1, -1)) for l in band}

    def sham_hooks(seed) -> dict:
        g = torch.Generator().manual_seed(seed)
        return {l: ProjectOutHook(torch.randn(1, model.d_model, generator=g)) for l in band}

    reports, other_capitals = {}, [c for c in CAPITAL_ROSTER]
    for country, capital in CAPITAL_ROSTER:
        target = (country, capital)
        ans_id = surface_token_ids(tok, capital)[0]  # primary surface token
        prompts = capital_context_stream(target, seed=EVAL_SEED, n=N_EVAL)  # answer = c
        clean_logit = mean_logit(prompts, ans_id, {})

        ablated, sham = {}, {}
        for k in DRAWS:
            pos = answer_states(model, capital_context_stream(target, seed=k, n=N_EXTRACT), band)
            neg = answer_states(model, neg_pool_stream(target, seed=k, n=N_EXTRACT), band)
            raw = mean_difference_by_layer(pos, neg)  # per-layer d_l(c)
            ablated[k] = mean_logit(prompts, ans_id, proj_hooks(raw))
            sham[k] = mean_logit(prompts, ans_id, sham_hooks(SHAM_SEED0 + 10 * k))
        eff = effect_drawset(clean_logit, ablated)    # clean - ablated logit-drop per draw
        sh = effect_drawset(clean_logit, sham)
        g = eff.median - sh.median                    # sham-controlled logit-drop
        transfer = all((v - sh.median) >= POTENT_BAR for v in eff.values)

        # positive control: ablate the unembed direction of c (same statistic, 3-draw marg.)
        u = _unembed_direction(model, tok, capital)
        pos_logit = mean_logit(prompts, ans_id, proj_hooks({l: u for l in band}))
        pos_eff = effect_drawset(clean_logit, {k: pos_logit for k in DRAWS})
        pos_g = pos_eff.median - sh.median
        control_ok = pos_g >= POS_BAR

        # specificity (descriptive): ablate d(c) draw-1 on OTHER-answer prompts (logit-drop)
        other = other_capitals[(other_capitals.index(target) + 1) % len(other_capitals)]
        oprompts = capital_context_stream(other, seed=EVAL_SEED, n=N_EVAL)
        oid = surface_token_ids(tok, other[1])[0]
        pos1 = answer_states(model, capital_context_stream(target, seed=1, n=N_EXTRACT), band)
        neg1 = answer_states(model, neg_pool_stream(target, seed=1, n=N_EXTRACT), band)
        raw1 = mean_difference_by_layer(pos1, neg1)
        spec_drop = mean_logit(oprompts, oid, {}) - mean_logit(oprompts, oid, proj_hooks(raw1))

        cls = classify_ablation(g, transfer, control_ok, POTENT_BAR, INERT_BAR)
        reports[capital] = {
            "concept": capital, "n_eval": N_EVAL, "clean_logit": round(clean_logit, 4),
            "ablated_logit_by_draw": {k: round(ablated[k], 4) for k in DRAWS},
            "sham_logit_by_draw": {k: round(sham[k], 4) for k in DRAWS},
            "effect_median": round(eff.median, 4), "sham_median": round(sh.median, 4),
            "sham_ctrl_logit_drop_g": round(g, 4), "cross_draw_transfer": bool(transfer),
            "pos_control_unembed_logit": round(pos_logit, 4), "pos_control_g": round(pos_g, 4),
            "control_ok": bool(control_ok), "specificity_other_logit_drop": round(spec_drop, 4),
            "classification": cls,
        }
        ctx.save_raw_completions(f"{capital}_ablation", [
            {"draw": k, "clean_logit": round(clean_logit, 4), "ablated_logit": round(ablated[k], 4),
             "sham_logit": round(sham[k], 4)} for k in DRAWS]
            + [{"arm": "pos_unembed", "logit": round(pos_logit, 4)},
               {"arm": "specificity_other", "other": other[1], "logit_drop": round(spec_drop, 4)}])
        print(f"[{capital}] clean {clean_logit:.2f} | g {g:+.3f} (transfer {transfer}) | "
              f"pos-ctrl g {pos_g:+.3f} ({'ok' if control_ok else 'FAIL'}) | spec {spec_drop:+.3f} "
              f"-> {cls}", flush=True)

    n_potent = sum(r["classification"] == "ablation-potent" for r in reports.values())
    n_inert = sum(r["classification"] == "ablation-inert" for r in reports.values())
    n_weak = sum(r["classification"] == "weak-ambiguous" for r in reports.values())
    n_inconc = sum(r["classification"] == "inconclusive" for r in reports.values())
    if n_potent >= ROSTER_RULE:
        roster = "H-POTENT"
    elif n_inert >= ROSTER_RULE:
        roster = "H-INERT"
    else:
        roster = "MIXED"
    summary = {"reports": reports, "roster_verdict": roster, "model": model_scope,
               "counts": {"potent": n_potent, "inert": n_inert, "weak": n_weak, "inconclusive": n_inconc},
               "bars": {"potent": POTENT_BAR, "inert": INERT_BAR, "pos_control": POS_BAR, "roster": ROSTER_RULE},
               "mechanism_positive_control": "null-check unembed('Paris') residual-add injection Δp=+0.80 "
               "(results/m5/20260722-012727-m5-1c-nullcheck); ADD mechanism works, so 1b injection-null is real",
               "certificate": "NONE under either branch; H-POTENT opens the cert path gated on Ecaterina's sign-off",
               "peak_rss_gb": round(peak_rss_gb(), 2), "wall_clock_s": round(time.perf_counter() - t0, 1)}
    (ctx.results_dir / "ablation_potency.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = ["# EXP-M5-1d S1 ablation-potency (410M) — REMOVE arm (Δlogit, Option B)", "",
             f"- model {model_scope}; band {band}; N_eval {N_EVAL}; 3 draws; E2 project-out, sham-ctrl Δlogit",
             f"- bars (logit): potent g>=1.0+transfer | inert g<=0.3 | 0.3-1.0 weak | pos-ctrl>=1.0",
             f"- **roster verdict: {roster}** (potent {n_potent}, inert {n_inert}, weak {n_weak}, inconclusive {n_inconc}) — NO certificate", "",
             "| concept | clean logit | sham-ctrl Δlogit g | transfer | pos-ctrl g | class |",
             "|---|---|---|---|---|---|"]
    for _, cap in CAPITAL_ROSTER:
        r = reports[cap]
        lines.append(f"| {cap} | {r['clean_logit']} | {r['sham_ctrl_logit_drop_g']:+.3f} | "
                     f"{'y' if r['cross_draw_transfer'] else 'n'} | {r['pos_control_g']:+.3f} "
                     f"{'ok' if r['control_ok'] else 'FAIL'} | {r['classification']} |")
    lines += ["", "mechanism positive control (on record): null-check unembed('Paris') injection Δp=+0.80.",
              f"", f"wall {summary['wall_clock_s']} s; peak {summary['peak_rss_gb']} GB. raw under raw_completions/."]
    (ctx.results_dir / "report.md").write_text("\n".join(lines))
    print(f"\n=== EXP-M5-1d roster verdict: {roster} "
          f"(potent {n_potent}, inert {n_inert}, weak {n_weak}, inconc {n_inconc}) ===", flush=True)
    ctx.finalize(roster_verdict=roster, counts=summary["counts"], model_revision=revision,
                 wall_clock_s=summary["wall_clock_s"], peak_rss_gb=summary["peak_rss_gb"])


if __name__ == "__main__":
    main()
