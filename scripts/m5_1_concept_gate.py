"""EXP-M5-1 orchestrator: S1 concept-direction stability gate.

Per roster concept (>= 8 capitals), extract the mean-difference concept
direction over certified capital-recall answer states at the final position of
band layers 4-16, on 3 draws (seeds 1/2/3; only the context-resampling stream
varies). Ladder the extraction over n_contexts {8,16,32,64} by prefix slicing,
and per rung compute (a) the min pairwise cosine of the 3 draws' identity
directions and (b) the IQR over draws of the downstream Δp(concept answer)
under injection. A rung passes iff min pairwise cosine >= 0.95 AND effect IQR
<= 0.05; converged_at = smallest passing rung with every larger rung passing
(a pass at 64 alone is not convergence). The S1 species certificate issues iff
every roster concept converges. Instruments: positive control (the direction
moves its own readout, Δp median >= +0.10 over sham), negative control
(norm-matched random directions, |Δp| <= max(0.02, 1/N)), sham twin per cell.

Injection is norm-preserving residual activation-addition (jvec.evals.concept):
the concept direction is residual-space, so the prereg's truncated-pinv clause
is vacuous (design note in jtvec.concept_gate) — flagged, no rule change.

Readout design (flagged for Ecaterina): the downstream Δp(concept answer) is
measured on N=40 FIXED held-out capital-recall carriers querying OTHER roster
countries (seed EVAL_SEED, independent of the 1/2/3 extraction draws), so the
concept's base probability is low and "moves its own readout" has room to show
the +0.10 gain. Injecting d(Paris) on "...The capital of Japan is" and reading
Δp(Paris) tests steering toward the concept, decoupled from the answer the
carrier already prefers.

Prereg: harness/preregs/EXP-M5-1-concept-gate.md (committed; thresholds ratified
as drafted). Usage: uv run python scripts/m5_1_concept_gate.py [--config ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from jvec.config import Config
from jvec.evals.concept import answer_states, injected_final_probs
from jvec.evals.tasks import surface_token_ids
from jvec.modeling import load_model
from jvec.utils import peak_rss_gb, set_seed
from jtvec.concept_gate import (
    CAPITAL_ROSTER,
    RUNGS,
    ConceptConvergenceRule,
    ConceptRungStats,
    capital_context_stream,
    certificate_payload,
    convergence_verdict,
    identity_direction,
    injection_deltas,
    mean_difference_by_layer,
    min_pairwise_cosine,
    natural_norms,
    negative_control,
    positive_control,
    rung_prefix,
)
from jtvec.core.draws import DrawSet
from jtvec.core.reporting import scoped_intervention
from jtvec.core.runctx import start_run
from jtvec.fv_stability import sham_twin

PREREG = REPO_ROOT / "harness/preregs/EXP-M5-1-concept-gate.md"
DRAWS = (1, 2, 3)                 # extraction draws; seeds == draw ids (distinct)
N_EVAL = 40                       # fixed held-out readout carriers per concept
EVAL_SEED = 9000                  # readout carriers, independent of the draws
SHAM_SEED0 = 9100                 # sham seed = 9100 + 10*draw + rung_index
RANDOM_SEED0 = 9500               # negative-control random directions
RULE = ConceptConvergenceRule(min_pairwise_cosine=0.95, max_effect_iqr=0.05)
POS_MIN_GAIN = 0.10


def eval_carriers(target: tuple[str, str], n: int) -> list[str]:
    """N fixed carriers querying OTHER roster countries (base p(concept) low)."""
    others = [c for c in CAPITAL_ROSTER if c != target]
    out: list[str] = []
    per = n // len(others) + 1
    for c in others:
        out += capital_context_stream(c, seed=EVAL_SEED, n=per)
    return out[:n]


def neg_pool_stream(target: tuple[str, str], seed: int, n: int) -> list[str]:
    """Ordered negative extraction stream: round-robin over the other roster
    concepts (prefix-stable — each concept's sub-stream is prefix-stable and the
    round-robin interleave preserves that)."""
    others = [c for c in CAPITAL_ROSTER if c != target]
    subs = {c: capital_context_stream(c, seed=seed, n=n) for c in others}
    out: list[str] = []
    i = 0
    while len(out) < n:
        out.append(subs[others[i % len(others)]][i // len(others)])
        i += 1
    return out[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/m5_1_concept_pythia410m.yaml"))
    args = parser.parse_args()
    t_start = time.perf_counter()

    cfg = Config.load(args.config)
    ctx = start_run(repo_root=REPO_ROOT, config_path=Path(args.config),
                    results_root=REPO_ROOT / cfg.results_dir, run_name="m5-1-concept",
                    prereg_path=PREREG)
    print(f"M5.1 concept gate run dir: {ctx.results_dir}", flush=True)

    set_seed(cfg.seed)
    model, tokenizer, revision = load_model(cfg)
    model_scope = f"{cfg.model.name}@{revision[:7]}"
    config_scope = f"EXP-M5-1 ({Path(args.config).name})"
    lo, hi = cfg.evals.band
    band_layers = [l for l in range(model.n_layers) if lo <= l <= hi]
    n_max = max(RUNGS)
    print(f"[model] {model_scope}; band layers {band_layers}", flush=True)

    reports: dict[str, dict] = {}
    certificates: dict[str, dict] = {}
    today = time.strftime("%Y-%m-%d")

    for country, capital in CAPITAL_ROSTER:
        target = (country, capital)
        concept = capital
        answer_id = surface_token_ids(tokenizer, capital)[0]

        # fixed held-out readout carriers + cached base probs for this concept
        carriers = eval_carriers(target, N_EVAL)
        base_p = [float(injected_final_probs(model, p, {})[answer_id]) for p in carriers]

        def readout(deltas: dict, cell: str) -> tuple[float, list[float]]:
            rows, dps = [], []
            for p, pb in zip(carriers, base_p):
                pi = float(injected_final_probs(model, p, deltas)[answer_id])
                rows.append({"concept": concept, "prompt_tail": p[-48:],
                             "p_base": round(pb, 5), "p_inj": round(pi, 5),
                             "dp": round(pi - pb, 5)})
                dps.append(pi - pb)
            ctx.save_raw_completions(cell, rows)
            return sum(dps) / len(dps), dps

        # per draw: extract states at n_max once, slice per rung
        dirs = {t: {} for t in RUNGS}
        effects = {t: {} for t in RUNGS}
        sham_effects_at_max: dict[int, float] = {}
        real_effects_at_max: dict[int, float] = {}
        for k in DRAWS:
            pos = answer_states(model, capital_context_stream(target, seed=k, n=n_max), band_layers)
            neg = answer_states(model, neg_pool_stream(target, seed=k, n=n_max), band_layers)
            for ri, t in enumerate(RUNGS):
                pos_t = {l: rung_prefix(v, t) for l, v in pos.items()}
                neg_t = {l: rung_prefix(v, t) for l, v in neg.items()}
                raw = mean_difference_by_layer(pos_t, neg_t)
                dirs[t][k] = identity_direction(raw, band_layers)
                deltas = injection_deltas(raw, natural_norms(pos_t))
                eff, _ = readout(deltas, f"{concept}_rung{t}_draw{k}_inject")
                effects[t][k] = eff
                if t == n_max:
                    real_effects_at_max[k] = eff
                    sham = {l: sham_twin(d, SHAM_SEED0 + 10 * k + ri) for l, d in deltas.items()}
                    sham_eff, _ = readout(sham, f"{concept}_rung{t}_draw{k}_sham")
                    sham_effects_at_max[k] = sham_eff
            print(f"[{concept}] draw{k} effects " +
                  " ".join(f"T{t}:{effects[t][k]:+.3f}" for t in RUNGS), flush=True)

        # ladder stats + convergence verdict
        per_rung = [ConceptRungStats(
            n_contexts=t,
            min_pairwise_cosine=min_pairwise_cosine([dirs[t][k] for k in DRAWS]),
            effect_iqr=DrawSet(tuple(effects[t][k] for k in DRAWS), DRAWS).iqr,
        ) for t in RUNGS]
        verdict = convergence_verdict(per_rung, RULE)

        # controls at the max rung
        effect_ds = DrawSet(tuple(real_effects_at_max[k] for k in DRAWS), DRAWS)
        sham_ds = DrawSet(tuple(sham_effects_at_max[k] for k in DRAWS), DRAWS)
        pos_ctrl = positive_control(effect_ds, sham_ds, POS_MIN_GAIN)
        # negative control: draw-1 direction, 10 norm-matched random dirs at max rung
        pos1 = answer_states(model, capital_context_stream(target, seed=1, n=n_max), band_layers)
        neg1 = answer_states(model, neg_pool_stream(target, seed=1, n=n_max), band_layers)
        raw1 = mean_difference_by_layer(pos1, neg1)
        deltas1 = injection_deltas(raw1, natural_norms(pos1))
        rand_dps = []
        for s in range(cfg.evals.n_random_seeds):
            rdeltas = {l: sham_twin(d, RANDOM_SEED0 + 100 * s + l) for l, d in deltas1.items()}
            rd, _ = readout(rdeltas, f"{concept}_neg_random{s}")
            rand_dps.append(rd)
        neg_ctrl = negative_control(rand_dps, N_EVAL)
        controlled = pos_ctrl.passed and neg_ctrl.passed

        reports[concept] = {
            "concept": concept, "answer_token_id": answer_id, "n_eval": N_EVAL,
            "verdict": verdict,
            "effects_by_rung": {t: {k: round(effects[t][k], 5) for k in DRAWS} for t in RUNGS},
            "sham_by_draw": {k: round(sham_effects_at_max[k], 5) for k in DRAWS},
            "controls": {
                "positive": {"passed": pos_ctrl.passed, "detail": pos_ctrl.detail,
                             "effect_median": effect_ds.median, "sham_median": sham_ds.median},
                "negative": {"passed": neg_ctrl.passed, "detail": neg_ctrl.detail},
                "controlled": controlled,
            },
        }
        if controlled and verdict["converged"]:
            certificates[f"s1_concept@{concept}@{model_scope}"] = certificate_payload(
                concept=concept, model=model_scope, converged_at=verdict["converged_at"],
                n_draws=len(DRAWS), evidence_run=str(ctx.results_dir), issued=today)
        print(f"[{concept}] converged_at={verdict['converged_at']} "
              f"pos={pos_ctrl.passed} neg={neg_ctrl.passed} "
              f"-> {'CERT' if (controlled and verdict['converged']) else 'no-cert'}", flush=True)

    all_converged = all(r["verdict"]["converged"] and r["controls"]["controlled"]
                        for r in reports.values())
    species_cert = {
        "species": "S1", "model": model_scope,
        "roster": [c for _, c in CAPITAL_ROSTER],
        "issued": all_converged,
        "concepts_converged": sum(r["verdict"]["converged"] for r in reports.values()),
        "concepts_controlled": sum(r["controls"]["controlled"] for r in reports.values()),
    }
    (ctx.results_dir / "concept_gate.json").write_text(json.dumps(
        {"reports": reports, "species_certificate": species_cert,
         "rule": {"min_pairwise_cosine": RULE.min_pairwise_cosine,
                  "max_effect_iqr": RULE.max_effect_iqr, "positive_min_gain": POS_MIN_GAIN}},
        indent=2, default=str))
    (ctx.results_dir / "certificates.json").write_text(json.dumps(certificates, indent=2))

    # report
    lines = [
        "# EXP-M5-1 report: S1 concept-direction stability gate",
        "", f"- model: {model_scope}; band layers {band_layers}; N_eval={N_EVAL}",
        "- prereg: harness/preregs/EXP-M5-1-concept-gate.md (thresholds ratified)",
        f"- rule: min pairwise cosine >= {RULE.min_pairwise_cosine} AND effect IQR "
        f"<= {RULE.max_effect_iqr}; witness rung required",
        "", "## Per-concept convergence + controls", "",
        "| concept | converged_at | pos ctrl | neg ctrl | certificate |",
        "|---|---|---|---|---|",
    ]
    for _, capital in CAPITAL_ROSTER:
        r = reports[capital]
        lines.append(
            f"| {capital} | {r['verdict']['converged_at']} | "
            f"{'pass' if r['controls']['positive']['passed'] else 'FAIL'} | "
            f"{'pass' if r['controls']['negative']['passed'] else 'FAIL'} | "
            f"{'issued' if (r['controls']['controlled'] and r['verdict']['converged']) else '—'} |")
    lines += ["", "## Instrument lines (effect with sham)", ""]
    for _, capital in CAPITAL_ROSTER:
        r = reports[capital]
        eff = DrawSet(tuple(r["effects_by_rung"][max(RUNGS)][k] for k in DRAWS), DRAWS)
        sham = DrawSet(tuple(r["sham_by_draw"][k] for k in DRAWS), DRAWS)
        lines.append("- " + scoped_intervention(
            f"S1 {capital} Δp(answer) @rung{max(RUNGS)}",
            eff, sham, model=model_scope, config=config_scope, n=N_EVAL))
    lines += ["", f"**S1 species certificate: {'ISSUED' if all_converged else 'NOT issued'}** "
              f"({species_cert['concepts_converged']}/{len(CAPITAL_ROSTER)} concepts converged, "
              f"{species_cert['concepts_controlled']}/{len(CAPITAL_ROSTER)} controlled)",
              "", f"wall-clock {round(time.perf_counter() - t_start, 1)} s; peak RSS "
              f"{peak_rss_gb():.2f} GB; device {cfg.device}; grid in concept_gate.json; "
              "raw cells under raw_completions/.", ""]
    (ctx.results_dir / "report.md").write_text("\n".join(lines))

    ctx.finalize(species_certificate_issued=all_converged,
                 concepts_converged=species_cert["concepts_converged"],
                 model_revision=revision,
                 wall_clock_s=round(time.perf_counter() - t_start, 1),
                 peak_rss_gb=round(peak_rss_gb(), 2))
    print(f"\nS1 species certificate: {'ISSUED' if all_converged else 'NOT issued'}")
    print(f"report: {ctx.results_dir / 'report.md'}")


if __name__ == "__main__":
    main()
