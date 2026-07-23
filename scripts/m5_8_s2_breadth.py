"""EXP-M5-8 S2 arm: within-species breadth over the 3 M2-certified FVs (410M, RTX).

S2 = capitalize, singular-plural, english-french (the ONLY 3 M2-certified FVs;
ALL reported; heterogeneity reported as heterogeneity, none curated out). Per FV,
four axes under the IDENTICAL instruments used in the n=1 runs:

  1. DRAW-STABILITY  jtvec.concept_gate.min_pairwise_cosine over the 3 cached
     certified fv_todd draw tensors (cache/m2/draw{1,2,3}) — the SAME statistic
     S1/S5 use, NOT M2's internal metric. draw-UNSTABLE match iff cos < 0.95.
  2. LENS-READOUT    decode_vector jlens label-rank of the FV's task-label words,
     over the 3 cached lens draws (cache/draw{0,1,2}); lens-dark iff jlens > 20.
  3. OUTPUT-ALIGN    the logit arm of decode_vector (logit label-rank); reported.
  4. POTENCY         injection (n_shot_eval + FV @ edit_layer, gain vs sham_twin)
     AND ablation (exp3 project-out-FV vs sham_fv, execution drop); potent iff
     either clears sham by 0.15 (>=3 draws, sham twins).

S2 reference profile = draw-UNSTABLE AND lens-dark AND potent; reproduces iff
>= 2/3 FVs match. Fresh experiment (budget n/a). NO certificate / NO sign-off
(Ecaterina's). Subcommands: probe (one-FV timing) | gate (full run via start_run).
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import glob
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from jlens import JacobianLens

from jvec.config import Config, _from_dict
from jtvec.core.runctx import start_run
from jtvec.core.draws import DrawSet
from jtvec.concept_gate import min_pairwise_cosine

PREREG = REPO_ROOT / "harness/preregs/EXP-M5-8-within-species-breadth.md"


# --- pure functions (model-free; unit-tested) -------------------------------
def draw_unstable(min_cos: float, bar: float) -> bool:
    """S2 profile leg: draw-UNSTABLE = fails the S1 stability bar."""
    return min_cos < bar


def lens_dark(jlens_label_rank: float, bar: float) -> bool:
    return jlens_label_rank > bar


def cleared(effect_median: float, sham_median: float, delta: float) -> bool:
    return bool((effect_median - sham_median) >= delta)


def _json_default(o):
    """Coerce numpy scalars (n_shot_eval returns them) to native for json.dumps."""
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (bool,)):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def potent(inj_gain_med: float, inj_sham_med: float,
           abl_drop_med: float, abl_sham_med: float, delta: float) -> dict:
    """Potent iff EITHER injection induces OR ablation cuts, each beyond its sham
    by `delta`. Both sub-measures reported."""
    inj = cleared(inj_gain_med, inj_sham_med, delta)
    abl = cleared(abl_drop_med, abl_sham_med, delta)
    return {"injection_potent": inj, "ablation_potent": abl, "potent": bool(inj or abl)}


def s2_match(min_cos: float, jlens_rank: float, pot: dict, bars: dict) -> dict:
    du = draw_unstable(min_cos, bars["stability"])
    ld = lens_dark(jlens_rank, bars["lens_dark"])
    p = pot["potent"]
    return {"draw_unstable": du, "lens_dark": ld, "potent": p, "match": bool(du and ld and p)}


def roster_verdict(matches: list[bool], n_certify: int) -> str:
    return "PROFILE-REPRODUCES" if sum(matches) >= n_certify else "HETEROGENEOUS"


# --- config -----------------------------------------------------------------
def load_cfg(path: str) -> tuple[Config, dict]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name for f in dataclasses.fields(Config)}
    extra = {k: raw.pop(k) for k in list(raw) if k not in known}
    return _from_dict(Config, raw), extra


def _fv_path(k: int, task: str) -> str:
    hits = glob.glob(str(REPO_ROOT / f"cache/m2/draw{k}/fvs/EleutherAI/*/{task}/fv_todd.pt"))
    if not hits:
        raise FileNotFoundError(f"no fv_todd for {task} draw{k} under cache/m2")
    return hits[0]


def load_fv_todd(k: int, task: str) -> torch.Tensor:
    return torch.load(_fv_path(k, task), map_location="cpu", weights_only=True).float().flatten()


def _lens_path(k: int) -> str:
    hits = glob.glob(str(REPO_ROOT / f"cache/draw{k}/lenses/EleutherAI/*/skip4_n10/lens.pt"))
    if not hits:
        raise FileNotFoundError(f"no 410M lens under cache/draw{k}")
    return hits[0]


# --- axis 1: draw-stability (no model) --------------------------------------
def axis_draw_stability(task: str, m2_draws: list[int]) -> dict:
    units = []
    for k in m2_draws:
        v = load_fv_todd(k, task)
        units.append(v / v.norm())
    return {"min_pairwise_cosine": round(min_pairwise_cosine(units), 4),
            "fv_norms": [round(float(load_fv_todd(k, task).norm()), 3) for k in m2_draws],
            "n_draws": len(m2_draws)}


# --- axes 2/3: lens-readout + output-alignment (model_j + lens draws) --------
def axis_lens(model_j, tok, task: str, fv: torch.Tensor, lenses: list, band_layers, out_ids) -> dict:
    from jvec.evals.fvprobe import decode_vector, random_like
    jlens_ranks, logit_ranks, rand_jlens = [], [], []
    for lens in lenses:
        per_layer = decode_vector(model_j, tok, lens, fv, task, out_ids, layers=band_layers)
        jlens_ranks.append(min(per_layer[l]["jlens"]["label_rank"] for l in band_layers))
        logit_ranks.append(min(per_layer[l]["logit"]["label_rank"] for l in band_layers))
        # negative control: a norm-matched random vector should NOT read as the label
        rl = random_like(fv, seed=1234)
        pr = decode_vector(model_j, tok, lens, rl, task, out_ids, layers=band_layers)
        rand_jlens.append(min(pr[l]["jlens"]["label_rank"] for l in band_layers))
    med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
    return {"jlens_label_rank_median": med(jlens_ranks), "jlens_per_lensdraw": jlens_ranks,
            "logit_label_rank_median": med(logit_ranks), "logit_per_lensdraw": logit_ranks,
            "random_jlens_label_rank_median": med(rand_jlens)}


# --- axis 4a: injection potency (FV HF model + n_shot_eval) ------------------
def injection_drawset(cfg, task, model, model_config, tok, m2_draws) -> tuple[DrawSet, DrawSet, dict]:
    from utils.eval_utils import n_shot_eval, n_shot_eval_no_intervention  # vendored FV repo
    from utils.prompt_utils import load_dataset
    from jtvec.fv_stability import sham_twin
    from jvec.utils import set_seed
    EVAL_SEED = 42
    ds = load_dataset(task, root_data_dir=str(REPO_ROOT / "third_party/function_vectors/dataset_files"), seed=cfg.seed)
    set_seed(EVAL_SEED)
    zs = dict(n_shot_eval_no_intervention(ds, 0, model, model_config, tok, compute_ppl=False, test_split="test")["clean_topk"])[1]
    gains, sham_gains = {}, {}
    for k in m2_draws:
        fv = load_fv_todd(k, task).to(model.device)
        set_seed(EVAL_SEED)
        g = dict(n_shot_eval(ds, fv.reshape(1, -1), cfg.fv.edit_layer, 0, model, model_config, tok)["intervention_topk"])[1]
        gains[k] = g - zs
        sham = sham_twin(fv, 700 + k).to(model.device)
        set_seed(EVAL_SEED)
        sg = dict(n_shot_eval(ds, sham.reshape(1, -1), cfg.fv.edit_layer, 0, model, model_config, tok)["intervention_topk"])[1]
        sham_gains[k] = sg - zs
    seeds = tuple(m2_draws)
    gd = DrawSet(values=tuple(gains[k] for k in m2_draws), seeds=seeds)
    sd = DrawSet(values=tuple(sham_gains[k] for k in m2_draws), seeds=seeds)
    return gd, sd, {"zero_shot_top1": round(zs, 4), "gains": {k: round(v, 4) for k, v in gains.items()},
                    "sham_gains": {k: round(v, 4) for k, v in sham_gains.items()}}


# --- axis 4b: ablation potency (model_j + exp3 project-out-FV) ---------------
def ablation_drawset(cfg, task, model_j, tok, lens0, W_U, band_layers, m2_draws, n_exec, m_top):
    from jvec.evals.exp3 import make_hooks, final_logits_under
    from jtvec.m3_instruments import answer_first_tokens
    from utils.prompt_utils import load_dataset
    import numpy as np
    ds = load_dataset(task, root_data_dir=str(REPO_ROOT / "third_party/function_vectors/dataset_files"), seed=cfg.seed)
    bos = tok.bos_token or ""
    rng = np.random.default_rng(5858)
    n_shots = cfg.fv.n_shots

    def pairs(split, n):
        d = ds[split]
        idx = rng.choice(len(d), n, replace=False)
        c = d[idx]
        return list(zip(c["input"], c["output"]))

    def context(ps):
        return bos + "".join(f"Q: {x}\nA: {y}\n\n" for x, y in ps)

    exec_items = []
    for _ in range(n_exec):
        ps = pairs("train", n_shots)
        qx, qy = pairs("test", 1)[0]
        exec_items.append((ps, str(qx), str(qy)))

    def measure(hooks) -> float:
        hits = 0
        for ps, qx, qy in exec_items:
            logits = final_logits_under(model_j, context(ps) + f"Q: {qx}\nA:", hooks)
            top1 = int(logits.argmax())
            hits += top1 in answer_first_tokens(tok, qy, case_sensitive=True)
        return hits / len(exec_items)

    clean = measure({})
    effects, sham_effects = {}, {}
    for k in m2_draws:
        fv = load_fv_todd(k, task)
        abl = measure(make_hooks("fv", band_layers, lens0, W_U, fv, m_top=m_top, seed=200 + k))
        sh = measure(make_hooks("sham_fv", band_layers, lens0, W_U, fv, m_top=m_top, seed=300 + k))
        effects[k] = clean - abl
        sham_effects[k] = clean - sh
    seeds = tuple(m2_draws)
    ed = DrawSet(values=tuple(effects[k] for k in m2_draws), seeds=seeds)
    sd = DrawSet(values=tuple(sham_effects[k] for k in m2_draws), seeds=seeds)
    return ed, sd, {"clean_exec": round(clean, 4), "effects": {k: round(v, 4) for k, v in effects.items()},
                    "sham_effects": {k: round(v, 4) for k, v in sham_effects.items()}}


# --- orchestration ----------------------------------------------------------
def run_all(cfg, s2, ctx=None, tasks=None, log=print) -> dict:
    from jvec.modeling import load_model
    from jvec.fv import load_fv_model
    from jvec.evals.fvprobe import output_token_ids
    from utils.prompt_utils import load_dataset
    bars = s2["bars"]
    tasks = tasks or s2["fvs"]
    lo, hi = cfg.evals.band

    results = {}
    # axis 1 (no model)
    for task in tasks:
        results[task] = {"draw_stability": axis_draw_stability(task, s2["m2_draws"])}
        log(f"[draw-stability] {task}: min_cos={results[task]['draw_stability']['min_pairwise_cosine']}")

    # axis 4a injection (FV HF model)
    t0 = time.time()
    fmodel, ftok, fmconfig, frev = load_fv_model(cfg)
    for task in tasks:
        gd, sd, raw = injection_drawset(cfg, task, fmodel, fmconfig, ftok, s2["m2_draws"])
        results[task]["injection"] = {"gain_median": round(gd.median, 4), "gain_values": list(gd.values),
                                      "sham_median": round(sd.median, 4), "sham_values": list(sd.values),
                                      "cleared": cleared(gd.median, sd.median, bars["potency_delta"]), **raw}
        log(f"[injection] {task}: gain_med={gd.median:+.3f} sham_med={sd.median:+.3f}")
        if ctx:
            ctx.save_raw_completions(f"{task}_injection", [{"draw": k, "gain": raw["gains"][k], "sham_gain": raw["sham_gains"][k]} for k in s2["m2_draws"]])
    del fmodel
    gc.collect(); torch.cuda.empty_cache()
    log(f"[injection] all done {time.time()-t0:.1f}s")

    # axes 2/3 lens + 4b ablation (model_j + lenses)
    model_j, tok, rev = load_model(cfg)
    lenses = [JacobianLens.load(_lens_path(k)) for k in s2["lens_draws"]]
    band_layers = [l for l in lenses[0].source_layers if lo <= l <= hi]
    W_U = model_j._lm_head.weight.detach()
    log(f"[model_j] loaded; band layers {band_layers[0]}..{band_layers[-1]} ({len(lenses)} lens draws)")
    for task in tasks:
        ds = load_dataset(task, root_data_dir=str(REPO_ROOT / "third_party/function_vectors/dataset_files"), seed=cfg.seed)
        out_ids = output_token_ids(ds, tok)
        fv1 = load_fv_todd(s2["m2_draws"][0], task)  # E1 lesson: marginalize over LENS draws for a fixed FV
        lens_res = axis_lens(model_j, tok, task, fv1, lenses, band_layers, out_ids)
        results[task]["lens_readout"] = {"jlens_label_rank_median": lens_res["jlens_label_rank_median"],
                                         "jlens_per_lensdraw": lens_res["jlens_per_lensdraw"],
                                         "random_jlens_label_rank_median": lens_res["random_jlens_label_rank_median"]}
        results[task]["output_alignment"] = {"logit_label_rank_median": lens_res["logit_label_rank_median"],
                                             "logit_per_lensdraw": lens_res["logit_per_lensdraw"]}
        ed, sd, raw = ablation_drawset(cfg, task, model_j, tok, lenses[0], W_U, band_layers, s2["m2_draws"], s2["n_exec"], s2["m_top"])
        results[task]["ablation"] = {"effect_median": round(ed.median, 4), "effect_values": list(ed.values),
                                    "sham_median": round(sd.median, 4), "sham_values": list(sd.values),
                                    "cleared": cleared(ed.median, sd.median, bars["potency_delta"]), **raw}
        log(f"[lens] {task}: jlens_rank_med={lens_res['jlens_label_rank_median']} logit_rank_med={lens_res['logit_label_rank_median']} | "
            f"[ablation] effect_med={ed.median:+.3f} sham_med={sd.median:+.3f}")
        if ctx:
            ctx.save_raw_completions(f"{task}_ablation", [{"draw": k, "effect": raw["effects"][k], "sham_effect": raw["sham_effects"][k]} for k in s2["m2_draws"]])

    # verdicts
    for task in tasks:
        r = results[task]
        pot = potent(r["injection"]["gain_median"], r["injection"]["sham_median"],
                     r["ablation"]["effect_median"], r["ablation"]["sham_median"], bars["potency_delta"])
        r["potency"] = pot
        r["profile"] = s2_match(r["draw_stability"]["min_pairwise_cosine"],
                                r["lens_readout"]["jlens_label_rank_median"], pot, bars)
    return results


def gate(cfg_path: str):
    cfg, extra = load_cfg(cfg_path)
    s2 = extra["s2"]
    ctx = start_run(repo_root=REPO_ROOT, config_path=Path(cfg_path),
                    results_root=REPO_ROOT / "results/m5", run_name="m5-8-s2-breadth", prereg_path=PREREG)
    print(f"EXP-M5-8 S2 run dir: {ctx.results_dir}", flush=True)
    results = run_all(cfg, s2, ctx=ctx)
    matches = [results[t]["profile"]["match"] for t in s2["fvs"]]
    verdict = roster_verdict(matches, s2["bars"]["n_certify"])
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2) if cfg.device == "cuda" else None
    summary = {
        "experiment": "EXP-M5-8", "arm": "S2", "substrate": cfg.model.name, "revision": cfg.model.revision,
        "profile_reference": "draw-UNSTABLE AND lens-dark AND potent", "bars": s2["bars"],
        "n_fvs": len(s2["fvs"]), "n_match": sum(matches), "roster_verdict": verdict,
        "peak_vram_gb": peak, "per_fv": results,
        "note": "draw-stability is min_pairwise_cosine over the cached certified fv_todd tensors "
                "(prereg-mandated identical method); the prereg's expected 0.43-0.61 range was the v1 "
                "cross-code-path number, not the M2 same-pipeline certified tensors — flagged.",
    }
    (ctx.results_dir / "s2_breadth.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    _write_table(ctx.results_dir, summary)
    ctx.finalize(n_match=int(sum(bool(x) for x in matches)), roster_verdict=verdict, peak_vram_gb=peak)
    print(f"\n=== EXP-M5-8 S2: {sum(matches)}/{len(s2['fvs'])} match profile -> {verdict} "
          f"(bar {s2['bars']['n_certify']}/3) ===\nrun dir: {ctx.results_dir}", flush=True)


def _write_table(run_dir: Path, s: dict):
    L = [f"# EXP-M5-8 S2 within-species breadth — {s['n_match']}/{s['n_fvs']} match -> **{s['roster_verdict']}**\n",
         f"Reference profile: {s['profile_reference']}. Bars: {s['bars']}. All 3 certified FVs reported.",
         "Certificate + sign-off are Ecaterina's.\n",
         "| FV | draw-stab min-cos | draw-UNSTABLE? | jlens label-rank | lens-dark? | logit label-rank (out-align) | inj gain v sham | abl drop v sham | potent? | MATCH |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for t, r in s["per_fv"].items():
        ds_, lr, oa, inj, abl, pf = (r["draw_stability"], r["lens_readout"], r["output_alignment"],
                                     r["injection"], r["ablation"], r["profile"])
        L.append(f"| {t} | {ds_['min_pairwise_cosine']} | {pf['draw_unstable']} | "
                 f"{lr['jlens_label_rank_median']} | {pf['lens_dark']} | {oa['logit_label_rank_median']} | "
                 f"{inj['gain_median']:+.3f}v{inj['sham_median']:+.3f} | {abl['effect_median']:+.3f}v{abl['sham_median']:+.3f} | "
                 f"{pf['potent']} | {'YES' if pf['match'] else 'no'} |")
    L.append(f"\n{s['note']}")
    (run_dir / "s2_breadth_table.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def probe(cfg_path: str, task: str):
    cfg, extra = load_cfg(cfg_path)
    s2 = extra["s2"]
    t0 = time.time()
    results = run_all(cfg, s2, ctx=None, tasks=[task])
    dt = time.time() - t0
    r = results[task]
    print(f"\nprobe {task}: done {dt:.1f}s; full-run projection ~{dt*len(s2['fvs'])/3600:.2f} h "
          f"({'UNDER' if dt*len(s2['fvs']) < 12*3600 else 'OVER'} 12 h LAW)", flush=True)
    print(f"  draw_stab={r['draw_stability']['min_pairwise_cosine']} jlens={r['lens_readout']['jlens_label_rank_median']} "
          f"logit={r['output_alignment']['logit_label_rank_median']} "
          f"inj={r['injection']['gain_median']:+.3f}/{r['injection']['sham_median']:+.3f} "
          f"abl={r['ablation']['effect_median']:+.3f}/{r['ablation']['sham_median']:+.3f} profile={r['profile']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe"); p.add_argument("--config", required=True); p.add_argument("--task", required=True)
    g = sub.add_parser("gate"); g.add_argument("--config", required=True)
    a = ap.parse_args()
    if a.cmd == "probe":
        probe(a.config, a.task)
    else:
        gate(a.config)


if __name__ == "__main__":
    main()
