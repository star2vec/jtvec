"""EXP-M5-8 within-species breadth (410M, Mac arm: S1 concepts + S5 steering).

Runs the IDENTICAL 4-axis apparatus over multiple NAMED instances per species and
asks whether each species' n=1 profile reproduces within type or the species is
internally heterogeneous. All named instances are reported (anti-harvesting); an
off-profile instance is reported as heterogeneity, never curated out.

Axes (identical instruments + bars to the n=1 runs; 3 extraction draws, 3 cached
lens draws, sham twins):
1. DRAW-STABILITY   — jtvec.concept_gate.min_pairwise_cosine over the 3 per-draw
   identity_direction units (the ONE method used identically for S1/S2/S5).
2. LENS-READOUT     — decode_vector jlens label-rank over the 3 cached lens draws
   (the EXP-M5-6 / E1 A1 statistic); the direction's own content as the label.
3. POTENCY          — injection + ablation, sham-controlled: S1 = Δp(answer token)
   alpha-sweep (m5_1b) + project-out Δp; S5 = sentiment-style logit-difference ΔS
   (m5_6) + project-out ΔS.
4. OUTPUT-ALIGNMENT — the logit-lens label-rank (low = output-aligned; the logit
   arm of axis 2).

Reference profiles + reproduction bars (prereg, ratified):
- S1 (5 concepts) = draw-stable ∧ lens-dark (jlens > 20) ∧ inert  → reproduces iff ≥ 4/5.
- S5 (4 attrs)    = output-aligned (logit ≤ 20) ∧ logit-trivial (jlens ≤ 20 ∧ logit < 200)
  ∧ potent (LOAD-BEARING)                                          → reproduces iff ≥ 3/4.

S2 (3 certified FVs) runs on the RTX (FV tensor cache); it is NOT in this file.
Prereg: harness/preregs/EXP-M5-8-within-species-breadth.md (RATIFIED, committed).
Usage: uv run python scripts/m5_8_breadth.py
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
from jtvec.concept_gate import (capital_context_stream, identity_direction,
                                injection_deltas, mean_difference_by_layer,
                                min_pairwise_cosine, natural_norms)
from jtvec.core.draws import DrawSet
from jtvec.core.runctx import start_run
from jtvec.fv_stability import sham_twin
from scripts.m5_1_concept_gate import eval_carriers, neg_pool_stream

PREREG = REPO_ROOT / "harness/preregs/EXP-M5-8-within-species-breadth.md"
CFG = REPO_ROOT / "configs/m5_8_breadth_pythia410m.yaml"
DRAW_CFGS = {j: REPO_ROOT / f"configs/m1_pythia410m_draw{j}.yaml" for j in (0, 1, 2)}

DRAWS, LENS_DRAWS = (1, 2, 3), (0, 1, 2)
N_POOL_S5, N_SUB_S5 = 16, 12      # S5 pools; subsample per draw so the 3 draws genuinely differ
N_EXTRACT_S1, N_EVAL_S1 = 256, 200   # n=256 = m5_1b's converged regime (all 5 concepts cross 0.95 there)
N_EXTRACT_S1_SMOKE, N_EVAL_S1_SMOKE = 64, 20   # smoke plumbing only (not the real bars)
SKIP_FIRST = 4
COS_BAR = 0.95                    # draw-stability
JLENS_MAX, LOGIT_MIN = 20.0, 200.0
OUT_ALIGN_MAX = 20.0              # output-aligned = logit label-rank <= this
ALPHAS = (1.0, 2.0, 4.0, 8.0)     # S1 injection sweep (m5_1b)
S1_GAIN, S1_ABL = 0.10, 0.10      # S1 injection / ablation |Δp| potency bars
S5_DS, S5_POS = 1.0, 1.0          # S5 injection ΔS bar / positive-control bar
SHAM0, RAND0 = 4200, 4800

# S1 concepts — the 5 NAMED capitals (subset of the certified roster).
S1_CONCEPTS = [("France", "Paris"), ("England", "London"), ("Italy", "Rome"),
               ("Germany", "Berlin"), ("Spain", "Madrid")]

# ---- S5 steering attribute stimulus sets (NAMED in the prereg; ALL reported) ----
# sentiment reuses the EXP-M5-6 sets verbatim; formality / politeness / excitement
# are authored here on the same shape (pos/neg extraction pairs, neutral carriers,
# single-token pos_w/neg_w readout words for the logit-difference metric).
SENTIMENT = {
    "pos": ["I loved it, it was", "This is wonderful,", "A fantastic experience,", "So happy and",
            "It was amazing,", "Truly delightful,", "I am thrilled,", "What a great",
            "Absolutely brilliant,", "A joy to", "So pleased with", "Wonderfully done,",
            "This made me smile,", "An excellent", "Highly recommend,", "Beautiful and"],
    "neg": ["I hated it, it was", "This is awful,", "A terrible experience,", "So sad and",
            "It was horrible,", "Truly disgusting,", "I am furious,", "What a bad",
            "Absolutely dreadful,", "A pain to", "So disappointed with", "Terribly done,",
            "This made me cry,", "An awful", "Would not recommend,", "Ugly and"],
    "carriers": ["The movie was", "Overall I would say it was", "My impression is that it was",
                 "The weather today is", "I think this is", "In my opinion it was",
                 "The food was", "Honestly it seemed", "The book was", "To me it felt"],
    "pos_w": ["good", "great", "wonderful", "amazing", "excellent", "nice"],
    "neg_w": ["bad", "terrible", "awful", "horrible", "poor", "disappointing"],
}
FORMALITY = {
    "pos": ["I would like to formally", "Pursuant to our agreement,", "It is with great pleasure that",
            "We hereby acknowledge", "Please be advised that", "I am writing to inform you",
            "In accordance with the", "Kindly find enclosed", "We respectfully request",
            "It would be prudent to", "Allow me to elaborate", "I trust this message finds you",
            "With reference to your", "We are pleased to announce", "Furthermore, it is essential",
            "I should be grateful if"],
    "neg": ["hey so basically", "yeah I dunno,", "gonna grab some", "lol that was",
            "wanna hang out", "kinda tired,", "nah it's cool", "gotta run,",
            "sup, what's", "that's so dumb", "omg I can't", "whatever,",
            "idk maybe later", "cool cool,", "yeah nah,", "ugh so annoying"],
    "carriers": ["The report stated that", "The email began by saying", "The message was phrased as",
                 "The letter opened,", "The reply came across as", "The note read,",
                 "The document said", "The response was", "The memo continued,", "The wording was"],
    "pos_w": ["therefore", "furthermore", "regarding", "accordingly", "moreover", "hereby"],
    "neg_w": ["yeah", "gonna", "wanna", "kinda", "gotta", "stuff"],
}
POLITENESS = {
    "pos": ["Could you please", "I would be grateful if", "Would you mind", "Thank you so much for",
            "I really appreciate your", "If it is not too much trouble,", "Please kindly",
            "May I ask you to", "I am sorry to bother you, but", "It would be wonderful if you could",
            "Excuse me, would you", "I hope you do not mind, but", "Pardon me, could you",
            "I would be delighted if", "With your permission,", "Thank you kindly for"],
    "neg": ["Just do it", "Give me that", "Move out of my", "I do not care what",
            "Shut up and", "Hurry up already,", "That is your problem,", "Do it now,",
            "Stop wasting my", "Whatever, just", "You had better", "I said give me",
            "Get out of my", "Quit complaining and", "Obviously you should", "Deal with it,"],
    "carriers": ["He turned and said", "The customer told the clerk,", "She responded by saying",
                 "The request was phrased,", "At the desk the person said", "The note read,",
                 "In the message he wrote,", "The reply to the waiter was", "The tone was",
                 "When asking for help she said"],
    "pos_w": ["please", "kindly", "thank", "appreciate", "grateful", "sorry"],
    "neg_w": ["now", "just", "obviously", "whatever", "shut", "quit"],
}
EXCITEMENT = {
    "pos": ["Wow, this is", "I cannot believe how", "This is absolutely", "So thrilling and",
            "What an incredible", "I am so pumped about", "This is mind-blowing,", "Amazing, I just",
            "So exciting, I cannot", "This is the best", "Unbelievable, it was", "I am bursting with",
            "What a rush,", "Electrifying and", "We finally did it,", "This is spectacular,"],
    "neg": ["This is so boring,", "Meh, it was", "Nothing much happened,", "It was quite dull,",
            "So tedious and", "Yawn, another", "It was pretty mundane,", "Nothing special,",
            "Just the usual,", "It dragged on and", "Rather uneventful,", "So monotonous,",
            "It was forgettable,", "Nothing to see,", "Bland and", "Utterly tiresome,"],
    "carriers": ["The event was", "Overall the day felt", "The presentation was", "My reaction was",
                 "The trip turned out", "The game seemed", "The lecture was", "The party felt",
                 "The announcement was", "To be honest it was"],
    "pos_w": ["thrilling", "exciting", "amazing", "incredible", "awesome", "wow"],
    "neg_w": ["boring", "dull", "tedious", "mundane", "bland", "dreary"],
}
S5_ATTRS = {"sentiment": SENTIMENT, "formality": FORMALITY,
            "politeness": POLITENESS, "excitement": EXCITEMENT}


def _sub(pool, k):
    """Subsample N_SUB_S5 of N_POOL_S5 for draw k (genuinely distinct per draw)."""
    rng = np.random.default_rng([k, 55])
    return [pool[i] for i in rng.choice(len(pool), N_SUB_S5, replace=False)]


def _median_pair(cells):
    """cells: list of (jlens, logit) -> (median jlens, median logit)."""
    return statistics.median([c[0] for c in cells]), statistics.median([c[1] for c in cells])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CFG))
    parser.add_argument("--smoke", action="store_true",
                        help="extract 1 S5 attr + 1 S1 concept, print cosine + 1 lens-draw ranks; no start_run")
    args = parser.parse_args()
    t0 = time.perf_counter()
    cfg = Config.load(args.config)
    smoke = args.smoke
    if not smoke:
        ctx = start_run(repo_root=REPO_ROOT, config_path=Path(args.config),
                        results_root=REPO_ROOT / cfg.results_dir, run_name="m5-8-breadth",
                        prereg_path=PREREG)
        print(f"M5.8 run dir: {ctx.results_dir}", flush=True)
    set_seed(cfg.seed)
    model, tok, revision = load_model(cfg)
    model_scope = f"{cfg.model.name}@{revision[:7]}"
    lo, hi = cfg.evals.band
    band = [l for l in range(model.n_layers) if lo <= l <= hi]
    device = model.input_device
    n_extract_s1 = N_EXTRACT_S1_SMOKE if smoke else N_EXTRACT_S1
    n_eval_s1 = N_EVAL_S1_SMOKE if smoke else N_EVAL_S1
    print(f"[model] {model_scope}; band {band}; smoke={smoke}; "
          f"n_extract_s1={n_extract_s1} n_eval_s1={n_eval_s1}", flush=True)

    # ---------- extract every instance direction (S1 + S5), 3 draws ----------
    instances = {}   # name -> dict(species, raw_by_draw, deltas_by_draw, units, label_words, extras)
    s1_list = S1_CONCEPTS[:1] if smoke else S1_CONCEPTS
    s5_list = list(S5_ATTRS.items())[:1] if smoke else list(S5_ATTRS.items())

    for country, capital in s1_list:
        target = (country, capital)
        raw_by_draw, deltas_by_draw = {}, {}
        for k in DRAWS:
            pos = answer_states(model, capital_context_stream(target, seed=k, n=n_extract_s1), band)
            neg = answer_states(model, neg_pool_stream(target, seed=k, n=n_extract_s1), band)
            raw = mean_difference_by_layer(pos, neg)
            raw_by_draw[k] = raw
            deltas_by_draw[k] = injection_deltas(raw, natural_norms(pos))
        units = [identity_direction(raw_by_draw[k], band) for k in DRAWS]
        instances[f"S1:{capital}"] = {
            "species": "S1", "raw_by_draw": raw_by_draw, "deltas_by_draw": deltas_by_draw,
            "units": units, "label_words": [capital, "capital", "capitals", "city"],
            "target": target}
        print(f"[extract S1:{capital}] 3 draws; draw-stability cos {min_pairwise_cosine(units):.3f}", flush=True)

    for name, A in s5_list:
        raw_by_draw, deltas_by_draw, pos_states_by_draw = {}, {}, {}
        for k in DRAWS:
            pos_states = answer_states(model, _sub(A["pos"], k), band)
            neg_states = answer_states(model, _sub(A["neg"], k), band)
            raw = mean_difference_by_layer(pos_states, neg_states)
            raw_by_draw[k] = raw
            deltas_by_draw[k] = injection_deltas(raw, natural_norms(pos_states))
            pos_states_by_draw[k] = pos_states
        units = [identity_direction(raw_by_draw[k], band) for k in DRAWS]
        instances[f"S5:{name}"] = {
            "species": "S5", "raw_by_draw": raw_by_draw, "deltas_by_draw": deltas_by_draw,
            "units": units, "label_words": A["pos_w"], "attr": A,
            "pos_states_by_draw": pos_states_by_draw}
        print(f"[extract S5:{name}] 3 draws; draw-stability cos {min_pairwise_cosine(units):.3f}", flush=True)

    # ---------- potency (species-specific readout) ----------
    def s1_potency(inst):
        target = inst["target"]
        ans_id = surface_token_ids(tok, target[1])[0]
        carriers = eval_carriers(target, n_eval_s1)

        def prob(prompt, hooks):     # p(answer token) at the final position under hooks
            lg = final_logits_under(model, prompt, hooks)
            return float(torch.softmax(lg.float(), dim=-1)[ans_id])
        base_p = [prob(p, {}) for p in carriers]

        def dp(hooks):
            return float(np.mean([prob(p, hooks) - b for p, b in zip(carriers, base_p)]))
        # injection alpha sweep, sham-controlled (m5_1b)
        gain_by_alpha = {}
        for ai, alpha in enumerate(ALPHAS):
            gains = []
            for k in DRAWS:
                d = {l: v * alpha for l, v in inst["deltas_by_draw"][k].items()}
                sh = {l: sham_twin(v, SHAM0 + 1000 * ai + 10 * k + l) for l, v in d.items()}
                gains.append(dp({l: _AddHook(v) for l, v in d.items()})
                             - dp({l: _AddHook(v) for l, v in sh.items()}))
            gain_by_alpha[alpha] = DrawSet(tuple(gains), DRAWS)
        best_alpha = max(ALPHAS, key=lambda a: gain_by_alpha[a].median)
        best_gain = gain_by_alpha[best_alpha].median
        monotone = all(gain_by_alpha[ALPHAS[i]].median <= gain_by_alpha[ALPHAS[i + 1]].median + 1e-6
                       for i in range(len(ALPHAS) - 1))
        inj_potent = best_gain >= S1_GAIN and monotone
        # ablation: project out the raw direction at band, Δp vs sham project-out
        abl = []
        for k in DRAWS:
            real = dp({l: ProjectOutHook(inst["raw_by_draw"][k][l].reshape(1, -1)) for l in band})
            sh = dp({l: ProjectOutHook(torch.randn(1, model.d_model,
                     generator=torch.Generator().manual_seed(SHAM0 + 100 * k + l))) for l in band})
            abl.append(real - sh)
        abl_eff = DrawSet(tuple(abl), DRAWS)
        abl_potent = abs(abl_eff.median) >= S1_ABL
        potent = bool(inj_potent or abl_potent)
        return {"potent": potent, "inj_best_dp": round(best_gain, 5), "inj_best_alpha": best_alpha,
                "inj_monotone": bool(monotone), "inj_potent": bool(inj_potent),
                "abl_dp_median": round(abl_eff.median, 5), "abl_potent": bool(abl_potent),
                "gain_by_alpha": {a: round(gain_by_alpha[a].median, 5) for a in ALPHAS}}

    def s5_potency(inst):
        A = inst["attr"]
        pos_ids = [surface_token_ids(tok, w)[0] for w in A["pos_w"]]
        neg_ids = [surface_token_ids(tok, w)[0] for w in A["neg_w"]]

        def Sc(prompt, hooks):
            lg = final_logits_under(model, prompt, hooks)
            return float(lg[pos_ids].mean() - lg[neg_ids].mean())
        base = {c: Sc(c, {}) for c in A["carriers"]}

        def dS(hooks):
            return float(np.mean([Sc(c, hooks) - base[c] for c in A["carriers"]]))
        inj, inj_sham, abl, abl_sham = {}, {}, {}, {}
        for k in DRAWS:
            d = inst["deltas_by_draw"][k]
            inj[k] = dS({l: _AddHook(d[l]) for l in band})
            sh = {l: sham_twin(d[l], SHAM0 + 10 * k + l) for l in band}
            inj_sham[k] = dS({l: _AddHook(sh[l]) for l in band})
            abl[k] = dS({l: ProjectOutHook(inst["raw_by_draw"][k][l].reshape(1, -1)) for l in band})
            abl_sham[k] = dS({l: ProjectOutHook(torch.randn(1, model.d_model,
                            generator=torch.Generator().manual_seed(SHAM0 + 100 * k + l))) for l in band})
        inj_eff = DrawSet(tuple(inj[k] - inj_sham[k] for k in DRAWS), DRAWS)
        abl_eff = DrawSet(tuple(abl[k] - abl_sham[k] for k in DRAWS), DRAWS)
        transfer = all((inj[k] - inj_sham[k]) >= S5_DS for k in DRAWS)
        # positive control: the pos/neg unembed-difference direction must move ΔS
        W_U = model._lm_head.weight.detach().float().cpu()
        u = (W_U[pos_ids].mean(0) - W_U[neg_ids].mean(0)); u = u / u.norm()
        nn = natural_norms(inst["pos_states_by_draw"][1])
        pos_ctrl = dS({l: _AddHook(u * nn[l]) for l in band})
        control_ok = pos_ctrl >= S5_POS
        potent = bool(inj_eff.median >= S5_DS and transfer and control_ok)
        return {"potent": potent, "inj_dS_median": round(inj_eff.median, 4), "transfer": bool(transfer),
                "abl_dS_median": round(abl_eff.median, 4), "pos_ctrl_dS": round(pos_ctrl, 4),
                "control_ok": bool(control_ok),
                "inj_by_draw": {k: round(inj[k] - inj_sham[k], 4) for k in DRAWS}}

    for name, inst in instances.items():
        inst["potency"] = s1_potency(inst) if inst["species"] == "S1" else s5_potency(inst)
        p = inst["potency"]
        print(f"[potency {name}] potent={p['potent']} " +
              (f"inj Δp {p['inj_best_dp']:+.3f}@a{p['inj_best_alpha']:.0f} abl Δp {p['abl_dp_median']:+.3f}"
               if inst["species"] == "S1" else
               f"inj ΔS {p['inj_dS_median']:+.3f} (transfer {p['transfer']}) abl ΔS {p['abl_dS_median']:+.3f} "
               f"pos-ctrl {p['pos_ctrl_dS']:+.3f}({'ok' if p['control_ok'] else 'FAIL'})"), flush=True)

    # ---------- lens-readout + output-alignment (loop 3 lens draws, all instances) ----------
    def perlayer_ranks(dirs_by_layer, lens, words):
        jr, lr = [], []
        for l in band:
            v = dirs_by_layer[l].float().to(device)
            jr.append(min(rank_of_word(model.unembed(lens.transport(v, l)).float().cpu(), tok, w) for w in words))
            lr.append(min(rank_of_word(model.unembed(v).float().cpu(), tok, w) for w in words))
        return statistics.median(jr), statistics.median(lr)

    lens_draws = LENS_DRAWS[:1] if smoke else LENS_DRAWS
    cells = {name: [] for name in instances}   # name -> list of (jlens, logit) over (draw, lens)
    a1_pos, a1_neg = [], []
    for j in lens_draws:
        dcfg = Config.load(str(DRAW_CFGS[j])); set_seed(dcfg.seed)
        lens = load_lens(dcfg, SKIP_FIRST, select_prompts(dcfg, tok), revision)
        for name, inst in instances.items():
            for k in DRAWS:
                cells[name].append(perlayer_ranks(inst["raw_by_draw"][k], lens, inst["label_words"]))
        # controls: a readable positive (a sentiment-word unembed direction) + noise negative
        W_U = model._lm_head.weight.detach().float().cpu()
        u = W_U[[surface_token_ids(tok, w)[0] for w in SENTIMENT["pos_w"]]].mean(0)
        a1_pos.append(perlayer_ranks({l: u for l in band}, lens, SENTIMENT["pos_w"])[0])
        vr = {l: torch.randn(model.d_model, generator=torch.Generator().manual_seed(RAND0 + j)) for l in band}
        a1_neg.append(perlayer_ranks(vr, lens, SENTIMENT["pos_w"])[0])
        del lens
        print(f"[lens] draw {j} done", flush=True)
    a1_control_ok = statistics.median(a1_pos) <= JLENS_MAX and statistics.median(a1_neg) > JLENS_MAX

    # ---------- profiles + species verdicts ----------
    per_instance = {}
    for name, inst in instances.items():
        jl, lg = _median_pair(cells[name])
        draw_cos = min_pairwise_cosine(inst["units"])
        p = inst["potency"]
        rec = {"species": inst["species"], "draw_stability_cos": round(draw_cos, 4),
               "draw_stable": bool(draw_cos >= COS_BAR),
               "jlens_label_rank": jl, "logit_label_rank": lg,
               "lens_dark": bool(jl > JLENS_MAX), "output_aligned": bool(lg <= OUT_ALIGN_MAX),
               "logit_trivial": bool(jl <= JLENS_MAX and lg < LOGIT_MIN),
               "potent": p["potent"], "potency": p}
        if inst["species"] == "S1":
            rec["profile_match"] = bool(rec["draw_stable"] and rec["lens_dark"] and not p["potent"])
        else:
            rec["profile_match"] = bool(rec["output_aligned"] and rec["logit_trivial"] and p["potent"])
        per_instance[name] = rec

    s1 = [r for r in per_instance.values() if r["species"] == "S1"]
    s5 = [r for r in per_instance.values() if r["species"] == "S5"]
    s1_match = sum(r["profile_match"] for r in s1)
    s5_match = sum(r["profile_match"] for r in s5)
    s1_verdict = ("PROFILE-REPRODUCES" if s1_match >= 4 else "HETEROGENEOUS") + f" ({s1_match}/{len(s1)})" if s1 else "n/a"
    s5_verdict = ("PROFILE-REPRODUCES" if s5_match >= 3 else "HETEROGENEOUS") + f" ({s5_match}/{len(s5)})" if s5 else "n/a"

    if smoke:
        print("\n=== SMOKE ===")
        for name, r in per_instance.items():
            print(f"  {name}: cos {r['draw_stability_cos']} stable={r['draw_stable']} | "
                  f"jlens {r['jlens_label_rank']} logit {r['logit_label_rank']} | potent={r['potent']} | "
                  f"match={r['profile_match']}")
        print(f"  a1 controls ok={a1_control_ok} (pos {statistics.median(a1_pos)} / neg {statistics.median(a1_neg)})")
        print(f"  S1 {s1_verdict} | S5 {s5_verdict}")
        return

    summary = {"model": model_scope, "a1_control_ok": a1_control_ok,
               "a1_pos_median": statistics.median(a1_pos), "a1_neg_median": statistics.median(a1_neg),
               "bars": {"cos": COS_BAR, "jlens_max": JLENS_MAX, "logit_min": LOGIT_MIN,
                        "out_align_max": OUT_ALIGN_MAX, "s1_gain": S1_GAIN, "s5_dS": S5_DS},
               "per_instance": per_instance,
               "S1": {"verdict": s1_verdict, "match": s1_match, "n": len(s1)},
               "S5": {"verdict": s5_verdict, "match": s5_match, "n": len(s5)},
               "peak_rss_gb": round(peak_rss_gb(), 2), "wall_clock_s": round(time.perf_counter() - t0, 1)}
    (ctx.results_dir / "breadth.json").write_text(json.dumps(summary, indent=2, default=str))
    for name, inst in instances.items():
        ctx.save_raw_completions(name.replace(":", "_"),
            [{"instance": name, "axis": "lens", "cell": i, "jlens": c[0], "logit": c[1]}
             for i, c in enumerate(cells[name])] + [{"instance": name, "potency": inst["potency"]}])

    def row(name, r):
        prof = "MATCH" if r["profile_match"] else "off"
        return (f"| {name} | {r['draw_stability_cos']} | {r['jlens_label_rank']} | {r['logit_label_rank']} | "
                f"{r['potent']} | {prof} |")
    lines = ["# EXP-M5-8 within-species breadth (410M, Mac arm: S1 + S5)", "",
             f"- model {model_scope}; 3 extraction draws; 3 cached lens draws; A1 controls "
             f"{'ok' if a1_control_ok else 'FAIL'}", "",
             f"- **S1: {s1_verdict}** (profile = draw-stable ∧ lens-dark ∧ inert; bar ≥4/5)",
             f"- **S5: {s5_verdict}** (profile = output-aligned ∧ logit-trivial ∧ potent; bar ≥3/4, LOAD-BEARING)",
             "", "| instance | draw-stab cos | jlens rank | logit rank | potent | profile |",
             "|---|---|---|---|---|---|"]
    lines += [row(n, per_instance[n]) for n in per_instance]
    lines += ["", "S2 (3 certified FVs) runs on the RTX (FV tensor cache) — folded in on that result.",
              f"wall {summary['wall_clock_s']} s; peak {summary['peak_rss_gb']} GB. raw under raw_completions/."]
    (ctx.results_dir / "report.md").write_text("\n".join(lines))
    print(f"\n=== EXP-M5-8 (Mac): S1 {s1_verdict} | S5 {s5_verdict} ===", flush=True)
    ctx.finalize(s1_verdict=s1_verdict, s5_verdict=s5_verdict, a1_control_ok=a1_control_ok,
                 model_revision=revision, wall_clock_s=summary["wall_clock_s"],
                 peak_rss_gb=summary["peak_rss_gb"])


if __name__ == "__main__":
    main()
