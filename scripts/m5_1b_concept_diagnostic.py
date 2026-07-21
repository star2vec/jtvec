"""EXP-M5-1b: S1 concept-gate diagnostic (410M). HELD until the EXP-M5-1c
null-check passes AND Ecaterina releases the hold — this file is the ready-to-
fire orchestrator; it does not run before that.

Extractor UNCHANGED from EXP-M5-1 (residual mean-difference over certified
capital-recall answer states, band 4-16, 3 draws). Diagnostic changes (ratified
conditions): (a) extended ladder {8..256} with a fixed ceiling + plateau test —
a plateau below 0.95 is a NEGATIVE result on S1, not a call to extend again;
(b) the p-floor is a RESOLUTION failure — resolution via an injection-strength
sweep alpha {1,2,4,8} at N=200, Δp in probability space (NOT log); if Δp stays
within max(0.005,1/N) at every alpha, S1 potency is DECLARED UNMEASURABLE;
(c) no verdict pre-named.

Per concept: convergence (crossed / plateaued-negative / ceiling-limited) x
potency (potent / unmeasurable / sub-bar). Issues NO certificate itself (a
diagnostic, not the gate); a certificate is a later EXP-M5-1 amendment + re-run.

Prereg: harness/preregs/EXP-M5-1b-concept-diagnostic.md (RATIFIED WITH CONDITIONS,
HELD). Usage (only post-null-check + release): uv run python scripts/m5_1b_concept_diagnostic.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from jvec.config import Config
from jvec.evals.concept import answer_states, injected_final_probs
from jvec.evals.tasks import surface_token_ids
from jvec.modeling import load_model
from jvec.utils import peak_rss_gb, set_seed
from jtvec.concept_gate import (
    CAPITAL_ROSTER, ConceptConvergenceRule, ConceptRungStats, capital_context_stream,
    convergence_verdict, identity_direction, injection_deltas, mean_difference_by_layer,
    min_pairwise_cosine, natural_norms, negative_control, plateau_below_bar, rung_prefix,
)
from jtvec.core.draws import DrawSet
from jtvec.core.runctx import start_run
from jtvec.fv_stability import sham_twin
from scripts.m5_1_concept_gate import eval_carriers, neg_pool_stream

PREREG = REPO_ROOT / "harness/preregs/EXP-M5-1b-concept-diagnostic.md"
CFG = REPO_ROOT / "configs/m5_1b_concept_diagnostic_pythia410m.yaml"
DRAWS = (1, 2, 3)
RUNGS = (8, 16, 32, 64, 128, 256)
ALPHAS = (1.0, 2.0, 4.0, 8.0)
N_EVAL, EVAL_SEED, SHAM_SEED0, RANDOM_SEED0 = 200, 9000, 9100, 9500
RULE = ConceptConvergenceRule(min_pairwise_cosine=0.95, max_effect_iqr=0.05)
COS_BAR, POT_BASE, POS_MIN_GAIN = 0.95, 0.005, 0.10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CFG))
    args = parser.parse_args()
    t0 = time.perf_counter()
    cfg = Config.load(args.config)
    ctx = start_run(repo_root=REPO_ROOT, config_path=Path(args.config),
                    results_root=REPO_ROOT / cfg.results_dir, run_name="m5-1b-concept-diagnostic",
                    prereg_path=PREREG)
    print(f"M5.1b run dir: {ctx.results_dir}", flush=True)
    set_seed(cfg.seed)
    model, tok, revision = load_model(cfg)
    model_scope = f"{cfg.model.name}@{revision[:7]}"
    lo, hi = cfg.evals.band
    band = [l for l in range(model.n_layers) if lo <= l <= hi]
    n_max = max(RUNGS)
    band_ok = max(POT_BASE, 1.0 / N_EVAL)
    print(f"[model] {model_scope}; band {band}; ladder ceiling {n_max}", flush=True)

    reports = {}
    for country, capital in CAPITAL_ROSTER:
        target = (country, capital)
        ans_id = surface_token_ids(tok, capital)[0]
        carriers = eval_carriers(target, N_EVAL)
        base_p = [float(injected_final_probs(model, p, {})[ans_id]) for p in carriers]

        def dp(deltas):  # mean Δp(answer) over carriers vs cached base
            return float(np.mean([float(injected_final_probs(model, p, deltas)[ans_id]) - b
                                  for p, b in zip(carriers, base_p)]))

        # extract per draw at the ceiling; cache the raw direction + injection deltas
        raw_by_draw, deltas_by_draw, dirs_by_rung, eff1_by_rung = {}, {}, {t: {} for t in RUNGS}, {t: {} for t in RUNGS}
        for k in DRAWS:
            pos = answer_states(model, capital_context_stream(target, seed=k, n=n_max), band)
            neg = answer_states(model, neg_pool_stream(target, seed=k, n=n_max), band)
            for t in RUNGS:
                pos_t = {l: rung_prefix(v, t) for l, v in pos.items()}
                neg_t = {l: rung_prefix(v, t) for l, v in neg.items()}
                raw = mean_difference_by_layer(pos_t, neg_t)
                dirs_by_rung[t][k] = identity_direction(raw, band)
                d1 = injection_deltas(raw, natural_norms(pos_t))
                eff1_by_rung[t][k] = dp(d1)                 # alpha=1 effect for the ladder IQR
                if t == n_max:
                    raw_by_draw[k] = raw
                    deltas_by_draw[k] = d1
            print(f"[{capital}] draw{k} cos-ready; eff@1 " +
                  " ".join(f"T{t}:{eff1_by_rung[t][k]:+.3f}" for t in RUNGS), flush=True)

        # convergence ladder + plateau
        per_rung = [ConceptRungStats(
            n_contexts=t,
            min_pairwise_cosine=min_pairwise_cosine([dirs_by_rung[t][k] for k in DRAWS]),
            effect_iqr=DrawSet(tuple(eff1_by_rung[t][k] for k in DRAWS), DRAWS).iqr) for t in RUNGS]
        verdict = convergence_verdict(per_rung, RULE)
        cos_by_rung = {s.n_contexts: s.min_pairwise_cosine for s in per_rung}
        plateau = plateau_below_bar(cos_by_rung, COS_BAR)
        conv_outcome = ("crossed" if verdict["converged"]
                        else "plateaued-negative" if plateau["plateaued_below_bar"]
                        else "ceiling-limited")

        # potency: alpha sweep on the ceiling direction, sham-controlled Δp
        gain_by_alpha, potency_rows = {}, []
        for a_i, alpha in enumerate(ALPHAS):
            gains = []
            for k in DRAWS:
                deltas = {l: d * alpha for l, d in deltas_by_draw[k].items()}
                sham = {l: sham_twin(d, SHAM_SEED0 + 1000 * a_i + 10 * k) for l, d in deltas.items()}
                gains.append(dp(deltas) - dp(sham))
            g = DrawSet(tuple(gains), DRAWS)
            gain_by_alpha[alpha] = g
            potency_rows.append({"alpha": alpha, "sham_ctrl_dp_median": round(g.median, 5),
                                 "iqr": round(g.iqr, 5)})
        best_alpha = max(ALPHAS, key=lambda a: gain_by_alpha[a].median)
        best_gain = gain_by_alpha[best_alpha].median
        monotone = all(gain_by_alpha[ALPHAS[i]].median <= gain_by_alpha[ALPHAS[i + 1]].median + 1e-6
                       for i in range(len(ALPHAS) - 1))
        potent = best_gain >= POS_MIN_GAIN and monotone
        unmeasurable = max(abs(gain_by_alpha[a].median) for a in ALPHAS) <= band_ok
        pot_outcome = "potent" if potent else "unmeasurable-potency" if unmeasurable else "sub-bar"

        # negative control (draw-1 ceiling direction, 10 random dirs)
        rand_dps = []
        for s in range(cfg.evals.n_random_seeds):
            rd = {l: sham_twin(d, RANDOM_SEED0 + 100 * s + l) for l, d in deltas_by_draw[1].items()}
            rand_dps.append(dp(rd))
        neg = negative_control(rand_dps, N_EVAL, base=POT_BASE)

        ctx.save_raw_completions(f"{capital}_potency", potency_rows)
        ctx.save_raw_completions(f"{capital}_cosine",
            [{"rung": t, "min_pairwise_cosine": round(cos_by_rung[t], 4),
              "effect_iqr": round(next(s.effect_iqr for s in per_rung if s.n_contexts == t), 5)} for t in RUNGS])
        reports[capital] = {
            "concept": capital, "convergence": conv_outcome, "converged_at": verdict["converged_at"],
            "cosine_by_rung": {t: round(cos_by_rung[t], 4) for t in RUNGS}, "plateau": plateau,
            "potency": pot_outcome, "best_alpha": best_alpha, "best_sham_ctrl_dp": round(best_gain, 5),
            "monotone_dose_response": bool(monotone), "gain_by_alpha": {a: round(gain_by_alpha[a].median, 5) for a in ALPHAS},
            "negative_control": {"passed": neg.passed, "detail": neg.detail},
        }
        print(f"[{capital}] convergence={conv_outcome} (conv_at={verdict['converged_at']}, "
              f"cos@256={cos_by_rung[max(RUNGS)]:.3f}) | potency={pot_outcome} "
              f"(best Δp {best_gain:+.3f} @a{best_alpha:.0f}) | neg {'ok' if neg.passed else 'FAIL'}", flush=True)

    # per-concept outcomes; no certificate issued by the diagnostic
    tally = {}
    for r in reports.values():
        key = f"{r['convergence']}/{r['potency']}"
        tally[key] = tally.get(key, 0) + 1
    summary = {"reports": reports, "outcome_tally": tally, "model": model_scope,
               "rungs": list(RUNGS), "alphas": list(ALPHAS), "n_eval": N_EVAL,
               "cos_bar": COS_BAR, "pos_min_gain": POS_MIN_GAIN, "potency_band": round(band_ok, 5),
               "peak_rss_gb": round(peak_rss_gb(), 2), "wall_clock_s": round(time.perf_counter() - t0, 1),
               "note": "diagnostic — issues no certificate; a certificate is a later ratified EXP-M5-1 amendment + re-run"}
    (ctx.results_dir / "diagnostic.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = ["# EXP-M5-1b S1 concept-gate diagnostic (410M)", "",
             f"- model {model_scope}; ladder {list(RUNGS)}; alphas {list(ALPHAS)}; N_eval {N_EVAL}",
             "- issues NO certificate (diagnostic); per-concept convergence x potency below", "",
             "| concept | convergence | conv_at | cos@256 | potency | best Δp | neg |",
             "|---|---|---|---|---|---|---|"]
    for _, cap in CAPITAL_ROSTER:
        r = reports[cap]
        lines.append(f"| {cap} | {r['convergence']} | {r['converged_at']} | "
                     f"{r['cosine_by_rung'][max(RUNGS)]} | {r['potency']} | {r['best_sham_ctrl_dp']:+.3f} | "
                     f"{'ok' if r['negative_control']['passed'] else 'FAIL'} |")
    lines += ["", f"outcome tally: {tally}", "",
              f"wall {summary['wall_clock_s']} s; peak {summary['peak_rss_gb']} GB. raw under raw_completions/."]
    (ctx.results_dir / "report.md").write_text("\n".join(lines))
    print(f"\n=== EXP-M5-1b diagnostic: tally {tally} ===", flush=True)
    ctx.finalize(outcome_tally=tally, model_revision=revision,
                 wall_clock_s=summary["wall_clock_s"], peak_rss_gb=summary["peak_rss_gb"])


if __name__ == "__main__":
    main()
