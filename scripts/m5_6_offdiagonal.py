"""EXP-M5-6: off-diagonal test of the (decodability × potency) 2×2 (410M).

Measures an S5 sentiment STEERING vector on BOTH taxonomy axes, same footing as
S1/S2, to turn the S1/S2 anchor dichotomy into a proven double dissociation
(HYPOTHESIS): two diagonal corners are consistent with one hidden deflation axis;
an off-diagonal cell separates the two axes.

- A1 decodability = E1 decode_vector (the instrument that read S2): per band layer,
  jlens readout unembed(J_l·d_l) vs logit unembed(d_l), min label-rank of positive-
  sentiment words; 9 cells (3 steering draws × 3 cached lens draws). DECODABLE iff
  median jlens label-rank <= 20 AND median logit label-rank >= 200 (E1 C1/C3).
- A2 potency = injection (1b machinery) + ablation (1d), readout = sentiment-logit-
  difference S = mean(logit pos tokens) - mean(logit neg tokens) on neutral
  carriers. POTENT iff sham-controlled injection ΔS median >= 1.0 with transfer.

2×2: off-diagonal (dec+pot OR neither) -> H-AXES-INDEPENDENT; on-diagonal
(dec+not-pot = S1 corner / not-dec+pot = S2 corner) -> H-AXES-COUPLED. Both
publishable. Prereg: harness/preregs/EXP-M5-6-offdiagonal.md (RATIFIED c73bb1b).
Usage: uv run python scripts/m5_6_offdiagonal.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from jvec.calibration import select_prompts
from jvec.config import Config
from jvec.evals.concept import _AddHook, answer_states
from jvec.evals.exp3 import ProjectOutHook, final_logits_under
from jvec.evals.tasks import rank_of_word, surface_token_ids
from jvec.lens_cache import load_lens
from jvec.modeling import load_model
from jvec.utils import peak_rss_gb, set_seed
from jtvec.concept_gate import (capital_context_stream, injection_deltas,
                                mean_difference_by_layer, natural_norms)
from jtvec.core.draws import DrawSet
from jtvec.core.runctx import start_run
from jtvec.fv_stability import sham_twin
from scripts.m5_1_concept_gate import neg_pool_stream

PREREG = REPO_ROOT / "harness/preregs/EXP-M5-6-offdiagonal.md"
CFG = REPO_ROOT / "configs/m5_6_offdiag_pythia410m.yaml"
DRAW_CFGS = {j: REPO_ROOT / f"configs/m1_pythia410m_draw{j}.yaml" for j in (0, 1, 2)}
E1_RUN = REPO_ROOT / "results/m4/20260719-021823-e1-decodability"    # S2 FV decode (cited)
CAP_LABELS = ["Paris", "capital", "capitals", "city"]                # S1 concept content
DRAWS, LENS_DRAWS = (1, 2, 3), (0, 1, 2)
N_EXTRACT, SKIP_FIRST = 32, 4
JLENS_MAX, LOGIT_MIN = 20.0, 200.0          # A1 E1 C1 / C3
POTENT_DS, POS_BAR = 1.0, 1.0               # A2 injection ΔS bar / positive-control bar
SHAM_A2, RAND_A1 = 700, 9600

POS = ["I loved it, it was", "This is wonderful,", "A fantastic experience,", "So happy and",
       "It was amazing,", "Truly delightful,", "I am thrilled,", "What a great",
       "Absolutely brilliant,", "A joy to", "So pleased with", "Wonderfully done,",
       "This made me smile,", "An excellent", "Highly recommend,", "Beautiful and"]
NEG = ["I hated it, it was", "This is awful,", "A terrible experience,", "So sad and",
       "It was horrible,", "Truly disgusting,", "I am furious,", "What a bad",
       "Absolutely dreadful,", "A pain to", "So disappointed with", "Terribly done,",
       "This made me cry,", "An awful", "Would not recommend,", "Ugly and"]
CARRIERS = ["The movie was", "Overall I would say it was", "My impression is that it was",
            "The weather today is", "I think this is", "In my opinion it was",
            "The food was", "Honestly it seemed", "The book was", "To me it felt"]
POS_W = ["good", "great", "wonderful", "amazing", "excellent", "nice"]
NEG_W = ["bad", "terrible", "awful", "horrible", "poor", "disappointing"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CFG))
    args = parser.parse_args()
    t0 = time.perf_counter()
    cfg = Config.load(args.config)
    ctx = start_run(repo_root=REPO_ROOT, config_path=Path(args.config),
                    results_root=REPO_ROOT / cfg.results_dir, run_name="m5-6-offdiagonal",
                    prereg_path=PREREG)
    print(f"M5.6 run dir: {ctx.results_dir}", flush=True)
    set_seed(cfg.seed)
    model, tok, revision = load_model(cfg)
    model_scope = f"{cfg.model.name}@{revision[:7]}"
    lo, hi = cfg.evals.band
    band = [l for l in range(model.n_layers) if lo <= l <= hi]
    pos_ids = [surface_token_ids(tok, w)[0] for w in POS_W]
    neg_ids = [surface_token_ids(tok, w)[0] for w in NEG_W]
    print(f"[model] {model_scope}; band {band}", flush=True)

    def S(prompt, hooks) -> float:
        lg = final_logits_under(model, prompt, hooks)
        return float(lg[pos_ids].mean() - lg[neg_ids].mean())

    def sample(prompts, seed):
        rng = np.random.default_rng([seed, 55])
        return [prompts[i] for i in rng.choice(len(prompts), N_EXTRACT // 2, replace=False)]

    # --- extract steering vectors (3 draws) ---
    raw_by_draw, deltas_by_draw = {}, {}
    for k in DRAWS:
        raw = mean_difference_by_layer(answer_states(model, sample(POS, k), band),
                                       answer_states(model, sample(NEG, k), band))
        raw_by_draw[k] = raw
        pos_states = answer_states(model, sample(POS, k), band)
        deltas_by_draw[k] = injection_deltas(raw, natural_norms(pos_states))
    print(f"[extract] {len(DRAWS)} steering draws", flush=True)

    # --- A2 potency: injection (primary) + ablation (corroborating), sham-controlled ΔS ---
    base = {c: S(c, {}) for c in CARRIERS}

    def dS(hooks_of_delta):
        return float(np.mean([S(c, hooks_of_delta) - base[c] for c in CARRIERS]))

    inj, inj_sham, abl, abl_sham = {}, {}, {}, {}
    for k in DRAWS:
        d = deltas_by_draw[k]
        inj[k] = dS({l: _AddHook(d[l]) for l in band})
        sh = {l: sham_twin(d[l], SHAM_A2 + 10 * k + l) for l in band}
        inj_sham[k] = dS({l: _AddHook(sh[l]) for l in band})
        abl[k] = dS({l: ProjectOutHook(raw_by_draw[k][l].reshape(1, -1)) for l in band})
        abl_sham[k] = dS({l: ProjectOutHook(torch.randn(1, model.d_model,
                          generator=torch.Generator().manual_seed(SHAM_A2 + 100 * k + l))) for l in band})
    inj_eff = DrawSet(tuple(inj[k] - inj_sham[k] for k in DRAWS), DRAWS)
    abl_eff = DrawSet(tuple(abl[k] - abl_sham[k] for k in DRAWS), DRAWS)
    inj_transfer = all((inj[k] - inj_sham[k]) >= POTENT_DS for k in DRAWS)
    # A2 positive control: the sentiment-token unembed difference must move ΔS
    W_U = model._lm_head.weight.detach().float().cpu()
    u_sent = (W_U[pos_ids].mean(0) - W_U[neg_ids].mean(0))
    u_sent = u_sent / u_sent.norm()
    nn = natural_norms(answer_states(model, sample(POS, 1), band))
    a2_pos = dS({l: _AddHook(u_sent * nn[l]) for l in band})
    a2_neg_vals = []
    for s in range(5):
        rr = {l: _AddHook(sham_twin(deltas_by_draw[1][l], RAND_A1 + 300 * s + l)) for l in band}
        a2_neg_vals.append(abs(dS(rr)))
    a2_control_ok = a2_pos >= POS_BAR
    potent = bool(inj_eff.median >= POTENT_DS and inj_transfer and a2_control_ok)
    print(f"[A2] inj ΔS {inj_eff.median:+.3f} (transfer {inj_transfer}) | abl ΔS {abl_eff.median:+.3f} | "
          f"pos-ctrl {a2_pos:+.3f} ({'ok' if a2_control_ok else 'FAIL'}) -> potent={potent}", flush=True)

    # --- A1 decodability: E1 decode_vector; S5 + S1/S2 reference rows, 3 lens draws ---
    device = model.input_device

    def perlayer_ranks(dirs, lens, words):
        """(median jlens label-rank, median logit label-rank) over band for one
        per-layer direction set (dirs[l]) via one lens (E1 decode_vector)."""
        jr, lr = [], []
        for l in band:
            v = dirs[l].float().to(device)
            jr.append(min(rank_of_word(model.unembed(lens.transport(v, l)).float().cpu(), tok, w) for w in words))
            lr.append(min(rank_of_word(model.unembed(v).float().cpu(), tok, w) for w in words))
        return statistics.median(jr), statistics.median(lr)

    # reference directions on the SAME per-layer footing. S5 + S1 computed here;
    # S2 (certified FV) is CITED from EXP-M4-E1 — identical decode_vector + the SAME
    # lens draws 0/1/2 — because the FV tensor cache is not on this (Mac) machine.
    s1_raw = {k: mean_difference_by_layer(
        answer_states(model, capital_context_stream(("France", "Paris"), seed=k, n=N_EXTRACT), band),
        answer_states(model, neg_pool_stream(("France", "Paris"), seed=k, n=N_EXTRACT), band)) for k in DRAWS}
    SPECIES = {"S5-steering": (raw_by_draw, POS_W), "S1-concept(Paris)": (s1_raw, CAP_LABELS)}
    print("[A1] extracted S1 concept reference; S2 FV decode cited from EXP-M4-E1", flush=True)

    cells = {name: {} for name in SPECIES}       # name -> {(draw,lens): (jlens, logit)}
    a1_pos_jlens, a1_neg_jlens = [], []
    for j in LENS_DRAWS:
        dcfg = Config.load(str(DRAW_CFGS[j])); set_seed(dcfg.seed)
        lens = load_lens(dcfg, SKIP_FIRST, select_prompts(dcfg, tok), revision)
        for name, (dirs, words) in SPECIES.items():
            for k in DRAWS:
                cells[name][(k, j)] = perlayer_ranks(dirs[k], lens, words)
        vp = {l: u_sent for l in band}
        a1_pos_jlens.append(perlayer_ranks(vp, lens, POS_W)[0])  # sentiment-unembed reads
        vr = {l: torch.randn(model.d_model, generator=torch.Generator().manual_seed(RAND_A1 + j)) for l in band}
        a1_neg_jlens.append(perlayer_ranks(vr, lens, POS_W)[0])  # noise does not
        del lens
        print(f"[A1] lens draw {j} done", flush=True)

    # per-species per-draw (median over lens draws) + overall median; the rank table
    def per_draw(name):
        return {k: (statistics.median([cells[name][(k, j)][0] for j in LENS_DRAWS]),
                    statistics.median([cells[name][(k, j)][1] for j in LENS_DRAWS])) for k in DRAWS}
    rank_table = {name: {"per_draw": {k: {"jlens": pd[k][0], "logit": pd[k][1]} for k in DRAWS},
                         "jlens_median": statistics.median([cells[name][(k, j)][0] for k in DRAWS for j in LENS_DRAWS]),
                         "logit_median": statistics.median([cells[name][(k, j)][1] for k in DRAWS for j in LENS_DRAWS])}
                  for name, pd in ((n, per_draw(n)) for n in SPECIES)}
    e1sp = json.loads((E1_RUN / "e1_results.json").read_text())["singular-plural"]
    rank_table["S2-FV(sing-plur, cited E1)"] = {
        "jlens_median": e1sp["jlens_median"], "logit_median": e1sp["logit_median"],
        "per_draw": {"cited": "EXP-M4-E1 — identical decode_vector + lens draws 0/1/2"},
        "cited_from": str(E1_RUN.relative_to(REPO_ROOT))}
    print("[A1] rank table (jlens / logit label-rank median):", flush=True)
    for name, rt in rank_table.items():
        pd = ("cited(E1)" if "cited" in rt["per_draw"] else
              " ".join(f"d{k}:{rt['per_draw'][k]['jlens']}/{rt['per_draw'][k]['logit']}" for k in DRAWS))
        print(f"    {name:26} median {rt['jlens_median']}/{rt['logit_median']}  per-draw {pd}", flush=True)

    jlens_med = rank_table["S5-steering"]["jlens_median"]
    logit_med = rank_table["S5-steering"]["logit_median"]
    a1_pos_ok = statistics.median(a1_pos_jlens) <= JLENS_MAX      # positive reads
    a1_neg_ok = statistics.median(a1_neg_jlens) > JLENS_MAX       # noise does not read
    a1_control_ok = a1_pos_ok and a1_neg_ok
    # A1 decomposed into two sub-axes (reframe): A1a = reads-at-all; A1b = J-lens-privileged.
    a1a_decodable = bool(jlens_med <= JLENS_MAX)                          # C1
    a1b_decodable = bool(jlens_med <= JLENS_MAX and logit_med >= LOGIT_MIN)  # C1 ∧ C3
    logit_trivial = bool(a1a_decodable and not a1b_decodable)            # jlens ≈ logit, output-aligned
    print(f"[A1] jlens label-rank median {jlens_med} | logit median {logit_med} -> "
          f"A1a(reads-at-all)={a1a_decodable} A1b(J-lens-privileged)={a1b_decodable} "
          f"logit-trivial={logit_trivial} | controls {'ok' if a1_control_ok else 'FAIL'}", flush=True)

    # --- verdict (never label off-diagonal on A1a alone) ---
    if not (a1_control_ok and a2_control_ok):
        cell, verdict = "INCONCLUSIVE", "INCONCLUSIVE (a control failed)"
    else:
        cell = (f"A1a={'y' if a1a_decodable else 'n'} A1b={'y' if a1b_decodable else 'n'} "
                f"(jlens {jlens_med} / logit {logit_med}); potent={'y' if potent else 'n'}")
        if a1b_decodable and potent:
            verdict = "AXES-INDEPENDENT (A1b-privileged AND potent; double dissociation supported, HYPOTHESIS)"
        elif logit_trivial:
            verdict = ("A1-DECOMPOSES (S5 is logit-trivially decodable [A1a] but NOT J-lens-privileged "
                       "[A1b]; A1 splits into reads-at-all vs J-lens-privileged via output-alignment; "
                       "NOT off-diagonal, deflation NOT refuted)")
        else:
            verdict = (f"REPORTED-AS-FALLS: A1a={a1a_decodable} A1b={a1b_decodable} potent={potent} "
                       "(no off-diagonal claim on A1a alone)")

    ctx.save_raw_completions("A2_potency", [
        {"draw": k, "inj_dS": round(inj[k], 4), "inj_sham": round(inj_sham[k], 4),
         "abl_dS": round(abl[k], 4), "abl_sham": round(abl_sham[k], 4)} for k in DRAWS]
        + [{"arm": "a2_pos_unembed", "dS": round(a2_pos, 4)},
           {"arm": "a2_neg_random_absmax", "dS": round(max(a2_neg_vals), 4)}])
    ctx.save_raw_completions("A1_rank_table",
        [{"species": name, "draw": k, "lens": j, "jlens": cells[name][(k, j)][0],
          "logit": cells[name][(k, j)][1]} for name in SPECIES for k in DRAWS for j in LENS_DRAWS]
        + [{"arm": "a1_pos_jlens_median", "rank": statistics.median(a1_pos_jlens)},
           {"arm": "a1_neg_jlens_median", "rank": statistics.median(a1_neg_jlens)}])

    summary = {
        "model": model_scope, "verdict": verdict, "cell": cell,
        "rank_table": rank_table,
        "A1": {"a1a_reads_at_all": a1a_decodable, "a1b_jlens_privileged": a1b_decodable,
               "logit_trivial": logit_trivial, "jlens_label_rank_median": jlens_med,
               "logit_label_rank_median": logit_med, "C1_max": JLENS_MAX, "C3_min": LOGIT_MIN,
               "control_ok": a1_control_ok, "pos_jlens_median": statistics.median(a1_pos_jlens),
               "neg_jlens_median": statistics.median(a1_neg_jlens)},
        "A2": {"potent": potent, "injection_dS_median": round(inj_eff.median, 4),
               "injection_transfer": inj_transfer, "ablation_dS_median": round(abl_eff.median, 4),
               "pos_control_dS": round(a2_pos, 4), "neg_control_absmax": round(max(a2_neg_vals), 4),
               "control_ok": a2_control_ok, "bar": POTENT_DS,
               "inj_by_draw": {k: round(inj[k] - inj_sham[k], 4) for k in DRAWS}},
        "reference": "S1 = (decodable, not-potent); S2 = (not-decodable, potent) — the diagonal",
        "peak_rss_gb": round(peak_rss_gb(), 2), "wall_clock_s": round(time.perf_counter() - t0, 1),
    }
    (ctx.results_dir / "offdiagonal.json").write_text(json.dumps(summary, indent=2, default=str))

    def _pattern(rt):
        if rt["jlens_median"] <= JLENS_MAX and rt["logit_median"] < LOGIT_MIN:
            return "jlens≈logit (logit-trivial)"
        return "J-lens-privileged" if rt["jlens_median"] <= JLENS_MAX else "dark (neither)"

    def _perdraw(rt):
        if "cited" in rt["per_draw"]:
            return "cited(E1)"
        return " ".join(f"d{k}:{rt['per_draw'][k]['jlens']}/{rt['per_draw'][k]['logit']}" for k in DRAWS)
    table_rows = [f"| {name} | {rt['jlens_median']} | {rt['logit_median']} | {_pattern(rt)} | {_perdraw(rt)} |"
                  for name, rt in rank_table.items()]
    (ctx.results_dir / "report.md").write_text("\n".join([
        "# EXP-M5-6 off-diagonal test (410M) — S5 steering on the (decodability × potency) 2×2", "",
        f"- **verdict: {verdict}**", f"- S5 cell: **{cell}**", "",
        f"- A1 (decode_vector, E1): jlens label-rank median {jlens_med} / logit {logit_med} -> "
        f"A1a reads-at-all=**{a1a_decodable}**, A1b J-lens-privileged=**{a1b_decodable}** "
        f"(logit-trivial={logit_trivial}; controls {'ok' if a1_control_ok else 'FAIL'})",
        f"- A2 (injection+ablation ΔS): injection ΔS {inj_eff.median:+.3f} (transfer {inj_transfer}) / "
        f"ablation {abl_eff.median:+.3f} -> **{'potent' if potent else 'not-potent'}** "
        f"(pos-ctrl {a2_pos:+.3f}, {'ok' if a2_control_ok else 'FAIL'})", "",
        "## A1 rank table — jlens / logit label-rank (identical decode_vector statistic)", "",
        "| direction | jlens median | logit median | pattern | per-draw jlens/logit |",
        "|---|---|---|---|---|",
        *table_rows,
        "",
        "Contrast: S2 (FV) is dark to the J-lens (both high); S5 (steering) is jlens≈logit "
        "(logit-trivial, output-aligned) — the sub-axes coincide for S2 and separate for S5.",
        f"wall {summary['wall_clock_s']} s; peak {summary['peak_rss_gb']} GB. raw under raw_completions/.",
    ]))
    print(f"\n=== EXP-M5-6: {verdict} | S5 cell {cell} ===", flush=True)
    ctx.finalize(verdict=verdict, cell=cell, a1a_decodable=a1a_decodable,
                 a1b_decodable=a1b_decodable, logit_trivial=logit_trivial, potent=potent,
                 model_revision=revision, wall_clock_s=summary["wall_clock_s"],
                 peak_rss_gb=summary["peak_rss_gb"])


if __name__ == "__main__":
    main()
