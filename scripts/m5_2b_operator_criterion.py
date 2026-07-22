"""EXP-M5-2b: S3 draw-marginalized / cosine operator criterion (Pythia-1.4B).

FINAL budgeted amendment cycle. The estimator is UNCHANGED from EXP-M5-2
(JacobianIclMeanEstimator, h_layer=12, 3 draws) — only the convergence CRITERION
changes, from top-1 output agreement to:
  (i)  cross-draw output-STATE cosine >= 0.95, AND
  (ii) draw-marginalized faithfulness >= 0.60 (per-probe majority-vote top-1
       across the 3 draws' operators).
The positive criterion is a pre-registered RE-ANALYSIS of EXP-M5-2's retained raw
(top1_by_draw majority vote) + its deterministic stored output-state cosine
(labeled post-hoc). The D-034 negative control (an UNRELATED relation's operator
applied to the target's probe) is estimated FRESH and MUST FAIL (<= 0.10);
label-shuffling is banned. Certify per relation iff cosine >= 0.95 AND
marginalized-faithful >= 0.60 AND the D-034 negative fails. Roster: >= 6/8 ->
H-S3-DIRECTION-STABLE, else H-S3-HARD. NO SIXTH CRITERION: close-but-under = FAIL.

The layer is NOT re-selected. Whatever this returns is the S3 disposition. The
certificate + sign-off are Ecaterina's; this writes neither.

Subcommands: probe (donor timing) | gate (full run via start_run).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from jvec.config import Config, _from_dict
from jtvec.core.runctx import start_run

# reuse the EXP-M5-2 estimator/wrapper (unchanged estimator, per prereg)
_spec = importlib.util.spec_from_file_location("m5_2", REPO_ROOT / "scripts" / "m5_2_operator_gate.py")
m52 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m52)

PREREG = REPO_ROOT / "harness/preregs/EXP-M5-2b-operator-criterion.md"


# --- pure functions (model-free; unit-tested) -------------------------------
def majority_vote(tokens: list[str]) -> str:
    """Draw-ensemble top-1: most common normalized token across draws; ties break
    to the earliest draw (stable)."""
    norm = [m52.norm_word(t) for t in tokens]
    counts = Counter(norm)
    top = max(counts.values())
    for t in norm:  # earliest-draw tiebreak
        if counts[t] == top:
            return t
    return norm[0]


def marginalized_faithfulness(top1_by_subject: list[list[str]], objs: list[str]) -> float:
    """Majority-vote top-1 across draws per subject, prefix-matched to the object."""
    if not top1_by_subject:
        return 0.0
    hits = sum(m52.prefix_match(majority_vote(draws), o) for draws, o in zip(top1_by_subject, objs))
    return hits / len(top1_by_subject)


def per_draw_faithfulness(top1_by_subject: list[list[str]], objs: list[str]) -> list[float]:
    """Descriptive: faithfulness of each individual draw (shows the argmax churn)."""
    n_draws = len(top1_by_subject[0]) if top1_by_subject else 0
    out = []
    for d in range(n_draws):
        hits = sum(m52.prefix_match(draws[d], o) for draws, o in zip(top1_by_subject, objs))
        out.append(round(hits / len(top1_by_subject), 4))
    return out


def raw_top1_agreement(top1_by_subject: list[list[str]]) -> float:
    return m52.top1_agreement(top1_by_subject)


def certify(cosine: float, marg_faith: float, neg_faith: float, bars: dict) -> bool:
    return bool(cosine >= bars["output_cosine"] and marg_faith >= bars["marginalized_faith"]
                and neg_faith <= bars["negative"])


def require_controlled(marg_faith: float, neg_faith: float, bars: dict) -> tuple[bool, dict]:
    """Positive: draw-marginalized faithfulness >= bar. Negative (D-034): the
    unrelated-relation operator's marginalized faithfulness must be <= bar."""
    pos_ok = marg_faith >= bars["marginalized_faith"]
    neg_ok = neg_faith <= bars["negative"]
    return (pos_ok and neg_ok), {
        "positive_marginalized_faith": round(marg_faith, 4), "positive_ok": bool(pos_ok),
        "negative_unrelated_faith": round(neg_faith, 4), "negative_ok": bool(neg_ok),
    }


# --- config + reuse loaders -------------------------------------------------
def load_cfg(path: str) -> tuple[Config, dict]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name for f in dataclasses.fields(Config)}
    extra = {k: raw.pop(k) for k in list(raw) if k not in known}
    return _from_dict(Config, raw), extra


def load_m52_reuse(run_dir: Path) -> dict:
    """Per-relation retained raw (subjects/objects/top1_by_draw) + stored cosine."""
    gate = json.loads((run_dir / "operator_gate.json").read_text())
    out = {}
    for rel, v in gate["per_relation"].items():
        rows = [json.loads(l) for l in
                (run_dir / "raw_completions" / f"{rel.replace('/', '_')}.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()]
        out[rel] = {
            "cosine": v["output_state_cosine"],
            "subjects": [r["subject"] for r in rows],
            "objects": [r["object"] for r in rows],
            "top1_by_draw": [r["top1_by_draw"] for r in rows],
        }
    return out


# --- D-034 negative: unrelated-relation operator applied to the target probe --
def donor_marginal_faith(mt, donor_ops, subjects: list[str], objs: list[str]) -> tuple[float, list[str]]:
    """Apply each donor draw-operator to the target subjects; majority-vote across
    donor draws per subject; prefix-match the TARGET objects (must fail)."""
    per_draw_top1 = []
    for op in donor_ops:
        top1, _ = m52.apply_operator(op, subjects)
        per_draw_top1.append(top1)
    marg_preds = []
    hits = 0
    for i, o in enumerate(objs):
        draws = [per_draw_top1[d][i] for d in range(len(donor_ops))]
        mv = majority_vote(draws)
        marg_preds.append(mv)
        hits += m52.prefix_match(mv, o)
    return hits / len(objs) if objs else 0.0, marg_preds


def estimate_donor_ops(mt, rel_path: str, seeds, k_estimate, h_layer, z_layer, n_probe, log=print):
    """Estimate the donor relation's per-draw operators (same estimator/config as
    EXP-M5-2, so deterministically identical to that relation's operators)."""
    rj = m52.load_relation_json(rel_path)
    samples = [(s["subject"], s["object"]) for s in rj["samples"]]
    _, pool = m52.fixed_split(samples, n_probe, k_estimate)
    ops = []
    for seed in seeds:
        rng = random.Random(seed)
        est = rng.sample(pool, min(k_estimate, len(pool)))
        t0 = time.time()
        ops.append(m52.estimate_operator(mt, m52.to_relation(rj, est), h_layer, z_layer))
        log(f"    donor {rel_path} seed={seed} estimated {time.time()-t0:.1f}s")
    return ops


def _is_factual(rel: str) -> bool:
    return rel.startswith("factual/")


# --- probe (donor timing) ---------------------------------------------------
def probe(cfg_path: str, target_rel: str):
    cfg, extra = load_cfg(cfg_path)
    op = extra["operator"]
    mt, rev = m52.build_mt(cfg)
    donor_rel = op["donor_for_factual"] if _is_factual(target_rel) else op["donor_for_other"]
    print(f"probe: donor={donor_rel} -> target={target_rel} h_layer={op['h_layer']}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ops = estimate_donor_ops(mt, donor_rel, op["seeds"], op["k_estimate"], op["h_layer"], op["z_layer"], op["n_probe"] if "n_probe" in op else 30)
    est_s = time.time() - t0
    reuse = load_m52_reuse(REPO_ROOT / op["reuse_m5_2_run"])
    tgt = reuse[target_rel]
    t1 = time.time()
    neg, _ = donor_marginal_faith(mt, ops, tgt["subjects"], tgt["objects"])
    apply_s = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 1e9
    # full run: 2 donor relations estimated once (3 draws each) + 8 targets cross-applied
    proj_h = (2 * est_s + 8 * apply_s) / 3600
    print(f"  donor 3-draw estimate {est_s:.1f}s; cross-apply to {target_rel} ({len(tgt['subjects'])} probe) "
          f"{apply_s:.1f}s; neg-faith={neg:.3f}; peakVRAM={peak:.2f}GB", flush=True)
    print(f"  full-run projection ~{proj_h:.2f} h ({'UNDER' if proj_h < 12 else 'OVER'} 12 h LAW)", flush=True)


# --- gate (full run via start_run) ------------------------------------------
def gate(cfg_path: str):
    cfg, extra = load_cfg(cfg_path)
    op = extra["operator"]
    bars = op["bars"]
    ctx = start_run(repo_root=REPO_ROOT, config_path=Path(cfg_path),
                    results_root=REPO_ROOT / "results/m5", run_name="m5-2b-operator-criterion",
                    prereg_path=PREREG)
    print(f"EXP-M5-2b run dir: {ctx.results_dir} (h_layer={op['h_layer']})", flush=True)
    reuse = load_m52_reuse(REPO_ROOT / op["reuse_m5_2_run"])
    mt, rev = m52.build_mt(cfg)

    # estimate the two cross-category donor operator sets once (reused across targets)
    donors = {}
    for dr in {op["donor_for_factual"], op["donor_for_other"]}:
        print(f"[donor] estimating {dr}", flush=True)
        donors[dr] = estimate_donor_ops(mt, dr, op["seeds"], op["k_estimate"], op["h_layer"], op["z_layer"], op.get("n_probe", 30))

    per_relation, n_cert = {}, 0
    for rel in op["relations"]:
        r = reuse[rel]
        cosine = r["cosine"]
        marg = marginalized_faithfulness(r["top1_by_draw"], r["objects"])
        per_draw = per_draw_faithfulness(r["top1_by_draw"], r["objects"])
        agree = raw_top1_agreement(r["top1_by_draw"])
        donor_rel = op["donor_for_factual"] if _is_factual(rel) else op["donor_for_other"]
        neg, neg_preds = donor_marginal_faith(mt, donors[donor_rel], r["subjects"], r["objects"])
        controlled, ctrl = require_controlled(marg, neg, bars)
        cert = certify(cosine, marg, neg, bars)
        n_cert += cert
        per_relation[rel] = {
            "n_probe": len(r["subjects"]), "output_state_cosine_reused": cosine,
            "marginalized_faithfulness": round(marg, 4), "per_draw_faithfulness": per_draw,
            "raw_top1_agreement": round(agree, 4),
            "donor_relation": donor_rel, "negative_unrelated_faith": round(neg, 4),
            "controls": ctrl, "controlled": controlled, "certified": cert,
        }
        ctx.save_raw_completions(rel.replace("/", "_"), [
            {"relation": rel, "subject": s, "object": o,
             "top1_by_draw": r["top1_by_draw"][i], "marginal_top1": majority_vote(r["top1_by_draw"][i]),
             "donor": donor_rel, "donor_marginal_top1": neg_preds[i]}
            for i, (s, o) in enumerate(zip(r["subjects"], r["objects"]))])
        print(f"  {rel}: cos={cosine} marg_faith={marg:.3f} (per-draw {per_draw}) "
              f"neg={neg:.3f} CERTIFIED={cert}", flush=True)

    roster = "H-S3-DIRECTION-STABLE" if n_cert >= bars["n_relations_certify"] else "H-S3-HARD"
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2) if cfg.device == "cuda" else None
    summary = {
        "experiment": "EXP-M5-2b", "substrate": cfg.model.name, "revision": rev,
        "h_layer": op["h_layer"], "criterion": "output-state cosine>=0.95 AND draw-marginalized(majority-vote) faith>=0.60 AND D-034 negative<=0.10",
        "reused_m5_2_run": op["reuse_m5_2_run"], "positive_is_reanalysis_posthoc": True,
        "bars": bars, "n_relations": len(op["relations"]), "n_certified": n_cert,
        "roster_verdict": roster, "peak_vram_gb": peak,
        "no_sixth_criterion": "close-but-under = FAIL (H-S3-HARD); final budgeted cycle",
        "per_relation": per_relation,
    }
    (ctx.results_dir / "operator_criterion.json").write_text(json.dumps(summary, indent=2))
    _write_table(ctx.results_dir, summary)
    ctx.finalize(n_certified=n_cert, roster_verdict=roster, peak_vram_gb=peak)
    print(f"\n=== EXP-M5-2b: {n_cert}/{len(op['relations'])} certified -> {roster} "
          f"(bar {bars['n_relations_certify']}/8) ===\nrun dir: {ctx.results_dir}", flush=True)


def _write_table(run_dir: Path, s: dict):
    L = [f"# EXP-M5-2b S3 marginalized/cosine criterion — {s['n_certified']}/{s['n_relations']} certified -> **{s['roster_verdict']}**\n",
         "Final budgeted amendment cycle. Positive criterion = pre-registered re-analysis of "
         f"{s['reused_m5_2_run']} (post-hoc); D-034 negative estimated fresh. Certificate + sign-off are Ecaterina's.",
         f"Substrate {s['substrate']}@{s['revision']}, h_layer={s['h_layer']}. Bars: {s['bars']}\n",
         "| relation | n_probe | out-cos (>=0.95) | marg-faith (>=0.60) | per-draw faith | raw-agree | donor neg (<=0.10) | ctrl | CERTIFIED |",
         "|---|---|---|---|---|---|---|---|---|"]
    for rel, r in s["per_relation"].items():
        L.append(f"| {rel} | {r['n_probe']} | {r['output_state_cosine_reused']} | {r['marginalized_faithfulness']} | "
                 f"{r['per_draw_faithfulness']} | {r['raw_top1_agreement']} | {r['negative_unrelated_faith']} "
                 f"({r['donor_relation'].split('/')[-1]}) | {'ok' if r['controlled'] else 'FAIL'} | "
                 f"{'YES' if r['certified'] else 'no'} |")
    L.append(f"\n{s['no_sixth_criterion']}")
    (run_dir / "operator_criterion_table.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--config", required=True)
    p.add_argument("--relation", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--config", required=True)
    a = ap.parse_args()
    if a.cmd == "probe":
        probe(a.config, a.relation)
    else:
        gate(a.config)


if __name__ == "__main__":
    main()
