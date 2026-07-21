"""EXP-M5-2: S3 relational-operator stability gate (Pythia-1.4B, cuda).

Certifies the cross-draw FUNCTIONAL convergence of the vendored linear
relational-operator extractor (Hernandez/Hendel; third_party/relations
@1b9ec3c JacobianIclMeanEstimator, D-022), called as a library and UNMODIFIED.
Purely functional/behavioural — no max-contrast / concept instrument — so
independent of the EXP-M5-1c null-check and not counted against the amendment
budget (prereg §Independence).

Per relation, over 3 draws (seeds vary only the ICL-context + subject sampling;
weights/data/probe-set fixed), on a FIXED held-out probe set:
  - top-1 output-token agreement across draws  >= 0.90
  - output-state cosine across draws           >= 0.95
  - positive control: held-out faithfulness    >= 0.60
  - negative control: shuffled-relation operator faithfulness <= 0.10
converged/certified iff both functional bars hold AND both controls pass.
S3 certified as a species iff >= 6 of the 8 relations converge. Raw W cosine is
descriptive only.

LANDING (prereg, required before any gate number): the wrapped operator applied
to held-out subjects reproduces sensible faithfulness (>= the positive-control
bar) on a relation the model knows — the wrapper is not trusted otherwise. Run
in `probe`; the gate re-checks the positive control per relation.

Subcommands:
  probe --config <cfg> --relation <cat/name> --layers L1 L2 ...   (layer pick + timing + landing)
  gate  --config <cfg>                                            (full run via start_run)

The gate certificate + sign-off are Ecaterina's; this produces the evidence.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
_REL = str(REPO_ROOT / "third_party" / "relations")


def _ensure_relations_path() -> None:
    """Put the vendored relations repo on sys.path ONLY when its `src` is about to
    be imported (inside the model-side functions). It ships a top-level `scripts/`
    regular package that hijacks `import scripts...` for the whole process if it
    sits on sys.path — so importing THIS module (e.g. for the model-free unit
    tests) must NOT touch sys.path, or it breaks other suites' `from scripts...`."""
    if _REL not in sys.path:
        sys.path.append(_REL)

import torch
import yaml

from jvec.config import Config, _from_dict
from jtvec.core.runctx import start_run

PREREG = REPO_ROOT / "harness/preregs/EXP-M5-2-operator-gate.md"
RELATIONS_ROOT = REPO_ROOT / "third_party" / "relations" / "data"


# --- pure functions (model-free; unit-tested) -------------------------------
def norm_word(w: str) -> str:
    return w.strip().lower()


def prefix_match(pred_word: str, obj: str) -> bool:
    """Faithfulness match: the (nonempty) predicted top-1 token is a prefix of the
    object, or vice-versa — the repo's is_nontrivial_prefix convention, adapted
    to a single predicted token (operator returns one object token)."""
    p, o = norm_word(pred_word), norm_word(obj)
    return len(p) > 0 and (o.startswith(p) or p.startswith(o))


def faithfulness(pred_words: list[str], objs: list[str]) -> float:
    if not pred_words:
        return 0.0
    return sum(prefix_match(p, o) for p, o in zip(pred_words, objs)) / len(pred_words)


def top1_agreement(top1_by_subject: list[list[str]]) -> float:
    """Fraction of probe subjects where ALL draws predict the same top-1 token."""
    if not top1_by_subject:
        return 0.0
    agree = sum(len(set(norm_word(t) for t in draws)) == 1 for draws in top1_by_subject)
    return agree / len(top1_by_subject)


def _cos(a, b) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))


def output_state_cosine(z_by_subject: list[list[torch.Tensor]]) -> float:
    """Mean over probe subjects of the mean pairwise cross-draw cosine of the
    operator-output states."""
    if not z_by_subject:
        return 0.0
    per = []
    for draws in z_by_subject:
        pairs = [(i, j) for i in range(len(draws)) for j in range(i + 1, len(draws))]
        if not pairs:
            continue
        per.append(sum(_cos(draws[i], draws[j]) for i, j in pairs) / len(pairs))
    return sum(per) / len(per) if per else 0.0


def raw_w_cosine(w_list: list[torch.Tensor]) -> float:
    """Descriptive only (prereg): mean pairwise cosine of the flattened W matrices."""
    pairs = [(i, j) for i in range(len(w_list)) for j in range(i + 1, len(w_list))]
    if not pairs:
        return 0.0
    return sum(_cos(w_list[i], w_list[j]) for i, j in pairs) / len(pairs)


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def require_controlled(pos_faith_draws: list[float], neg_faith_draws: list[float], bars: dict) -> tuple[bool, dict]:
    """Positive: median held-out faithfulness >= bar. Negative: shuffled-relation
    median faithfulness <= bar. Both gate the per-relation verdict."""
    pos_med = _median(pos_faith_draws) if pos_faith_draws else 0.0
    neg_med = _median(neg_faith_draws) if neg_faith_draws else 1.0
    pos_ok = pos_med >= bars["faithfulness_pos"]
    neg_ok = neg_med <= bars["shuffled_neg"]
    return (pos_ok and neg_ok), {
        "positive_faithfulness_median": round(pos_med, 4), "positive_ok": bool(pos_ok),
        "negative_shuffled_faithfulness_median": round(neg_med, 4), "negative_ok": bool(neg_ok),
        "positive_per_draw": [round(x, 4) for x in pos_faith_draws],
        "negative_per_draw": [round(x, 4) for x in neg_faith_draws],
    }


def converged(agreement: float, out_cos: float, controlled: bool, bars: dict) -> bool:
    return bool(agreement >= bars["agreement"] and out_cos >= bars["output_cosine"] and controlled)


# --- config -----------------------------------------------------------------
def load_cfg(path: str) -> tuple[Config, dict]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name for f in dataclasses.fields(Config)}
    extra = {k: raw.pop(k) for k in list(raw) if k not in known}
    return _from_dict(Config, raw), extra


def load_relation_json(rel_path: str) -> dict:
    return json.loads((RELATIONS_ROOT / f"{rel_path}.json").read_text(encoding="utf-8"))


# --- model / vendored-estimator wrapper -------------------------------------
def build_mt(cfg: Config):
    """Revision-pinned ModelAndTokenizer for the vendored estimator (unmodified
    library). GPTNeoX lm_head path handles Pythia; fp32 per config."""
    _ensure_relations_path()
    import transformers
    from src import models  # vendored

    device = cfg.torch_device()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        cfg.model.name, revision=cfg.model.revision, dtype=cfg.torch_dtype()
    ).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(cfg.model.name, revision=cfg.model.revision)
    tok.pad_token = tok.eos_token
    resolved = getattr(hf.config, "_commit_hash", None) or cfg.model.revision
    return models.ModelAndTokenizer(model=hf, tokenizer=tok), resolved


def to_relation(rel_json: dict, samples: list[tuple[str, str]]):
    """Minimal vendored data.Relation over a given (subject,object) subset. The
    JSONs carry 4 property fields; RelationProperties needs 6 — fill the two
    absent (fn_type/disambiguating) with inert defaults (unused by estimation)."""
    _ensure_relations_path()
    from src import data  # vendored
    p = rel_json.get("properties", {})
    return data.Relation(
        name=rel_json["name"],
        prompt_templates=rel_json["prompt_templates"],
        prompt_templates_zs=rel_json.get("prompt_templates_zs", rel_json["prompt_templates"]),
        samples=[data.RelationSample(subject=s, object=o) for s, o in samples],
        properties=data.RelationProperties(
            relation_type=p.get("relation_type", ""), domain_name=p.get("domain_name", ""),
            range_name=p.get("range_name", ""), symmetric=p.get("symmetric", False),
            fn_type=p.get("fn_type", ""), disambiguating=p.get("disambiguating", False),
        ),
    )


def estimate_operator(mt, relation, h_layer: int, z_layer):
    _ensure_relations_path()
    from src.operators import JacobianIclMeanEstimator  # vendored
    est = JacobianIclMeanEstimator(mt=mt, h_layer=h_layer, z_layer=z_layer)
    op = est(relation)
    gc.collect()
    torch.cuda.empty_cache()
    return op


@torch.no_grad()
def apply_operator(op, subjects: list[str]):
    """Apply operator to each held-out subject; return (top1_words, z_vectors)."""
    top1, zs = [], []
    for subj in subjects:
        out = op(subject=subj, k=1)
        top1.append(out.predictions[0].token)
        zs.append(out.z.detach().flatten().float().cpu())
    return top1, zs


# --- draw / split helpers ---------------------------------------------------
def fixed_split(samples: list[tuple[str, str]], n_probe: int, k_estimate: int):
    """Draw-independent held-out probe (seed 0) + estimation pool, disjoint.

    The pool is floored at max(k_estimate+4, half) so the 3 draws sample DIFFERENT
    estimation subsets (a stability gate needs the operators to be able to differ;
    a pool == k_estimate would make every draw identical). Small relations
    therefore get a smaller probe — the achieved N is reported per relation."""
    rng = random.Random(0)
    s = samples[:]
    rng.shuffle(s)
    pool_min = min(len(s) - 1, max(k_estimate + 4, (len(s) + 1) // 2))
    probe_n = min(n_probe, len(s) - pool_min)
    probe = s[:probe_n]
    pool = s[probe_n:]
    return probe, pool


def derange_objects(pairs: list[tuple[str, str]], seed: int) -> list[tuple[str, str]]:
    """Shuffled-relation control: permute objects across subjects (value-different
    where possible)."""
    rng = random.Random(seed)
    subs = [s for s, _ in pairs]
    objs = [o for _, o in pairs]
    for _ in range(200):
        perm = objs[:]
        rng.shuffle(perm)
        if all(perm[i] != objs[i] for i in range(len(objs))):
            return list(zip(subs, perm))
    return list(zip(subs, objs[::-1]))


# --- one relation, all draws ------------------------------------------------
def run_relation(mt, rel_path: str, opcfg: dict, log=print) -> dict:
    rj = load_relation_json(rel_path)
    samples = [(s["subject"], s["object"]) for s in rj["samples"]]
    probe, pool = fixed_split(samples, opcfg["n_probe"], opcfg["k_estimate"])
    probe_subjects = [s for s, _ in probe]
    probe_objs = [o for _, o in probe]
    h_layer, z_layer, k = opcfg["h_layer"], opcfg["z_layer"], opcfg["k_estimate"]

    top1_draws, z_draws, w_list = [], [], []
    pos_faith, neg_faith = [], []
    t0 = time.time()
    for seed in opcfg["seeds"]:
        rng = random.Random(seed)
        est_samples = rng.sample(pool, min(k, len(pool)))
        op = estimate_operator(mt, to_relation(rj, est_samples), h_layer, z_layer)
        w_list.append(op.weight.detach().flatten().float().cpu())
        top1, zs = apply_operator(op, probe_subjects)
        top1_draws.append(top1)
        z_draws.append(zs)
        pos_faith.append(faithfulness(top1, probe_objs))
        # shuffled-relation negative control (same seed's estimation subjects)
        sh = derange_objects(est_samples, seed)
        op_sh = estimate_operator(mt, to_relation(rj, sh), h_layer, z_layer)
        top1_sh, _ = apply_operator(op_sh, probe_subjects)
        neg_faith.append(faithfulness(top1_sh, probe_objs))
        log(f"  [{rel_path}] draw seed={seed}: faith={pos_faith[-1]:.3f} shuffled={neg_faith[-1]:.3f} "
            f"(est n={len(est_samples)}, probe n={len(probe_subjects)})")
    dt = time.time() - t0

    per_subject_top1 = [[top1_draws[d][i] for d in range(len(top1_draws))] for i in range(len(probe_subjects))]
    per_subject_z = [[z_draws[d][i] for d in range(len(z_draws))] for i in range(len(probe_subjects))]
    agreement = top1_agreement(per_subject_top1)
    out_cos = output_state_cosine(per_subject_z)
    w_cos = raw_w_cosine(w_list)
    controlled, ctrl = require_controlled(pos_faith, neg_faith, opcfg["bars"])
    conv = converged(agreement, out_cos, controlled, opcfg["bars"])
    return {
        "relation": rel_path, "n_probe": len(probe_subjects), "k_estimate": k,
        "n_pool": len(pool), "seeds": opcfg["seeds"], "wall_s": round(dt, 1),
        "top1_agreement": round(agreement, 4), "output_state_cosine": round(out_cos, 4),
        "raw_w_cosine_descriptive": round(w_cos, 4),
        "controls": ctrl, "controlled": controlled, "converged": conv,
        "probe_subjects": probe_subjects, "probe_objects": probe_objs,
        "top1_by_draw": top1_draws,
    }


# --- probe (layer pick + timing + landing) ----------------------------------
def probe(cfg_path: str, rel_path: str, layers: list[int]):
    cfg, extra = load_cfg(cfg_path)
    opcfg = extra["operator"]
    mt, revision = build_mt(cfg)
    print(f"probe: {cfg.model.name}@{revision} relation={rel_path} layers={layers}", flush=True)
    rj = load_relation_json(rel_path)
    samples = [(s["subject"], s["object"]) for s in rj["samples"]]
    probe_set, pool = fixed_split(samples, opcfg["n_probe"], opcfg["k_estimate"])
    subs = [s for s, _ in probe_set]
    objs = [o for _, o in probe_set]
    rng = random.Random(1)
    est_samples = rng.sample(pool, min(opcfg["k_estimate"], len(pool)))
    torch.cuda.reset_peak_memory_stats()
    results = []
    for L in layers:
        t0 = time.time()
        op = estimate_operator(mt, to_relation(rj, est_samples), L, opcfg["z_layer"])
        est_s = time.time() - t0
        top1, _ = apply_operator(op, subs)
        f = faithfulness(top1, objs)
        peak = torch.cuda.max_memory_allocated() / 1e9
        results.append({"h_layer": L, "faithfulness": round(f, 4), "estimate_s": round(est_s, 1), "peak_vram_gb": round(peak, 2)})
        print(f"  h_layer={L}: faithfulness={f:.3f} estimate={est_s:.1f}s peakVRAM={peak:.2f}GB (probe n={len(subs)}, est n={len(est_samples)})", flush=True)
    best = max(results, key=lambda r: r["faithfulness"])
    # full-run projection: 8 relations x 3 draws x 2 arms x est_s(best) + eval
    proj_h = 8 * 3 * 2 * best["estimate_s"] / 3600
    print(f"\nBEST h_layer={best['h_layer']} faith={best['faithfulness']:.3f}; "
          f"per-operator ~{best['estimate_s']:.1f}s -> full-run estimate ~{proj_h:.2f} h "
          f"({'UNDER' if proj_h < 12 else 'OVER'} 12 h LAW)", flush=True)
    print(f"LANDING: wrapper faithfulness at best layer >= {opcfg['bars']['faithfulness_pos']}? "
          f"{best['faithfulness'] >= opcfg['bars']['faithfulness_pos']}", flush=True)
    return best, proj_h


# --- gate (full run via start_run) ------------------------------------------
def gate(cfg_path: str):
    cfg, extra = load_cfg(cfg_path)
    opcfg = extra["operator"]
    if opcfg.get("h_layer") is None:
        sys.exit("operator.h_layer is null — run `probe` first and record the chosen layer in the config.")
    ctx = start_run(repo_root=REPO_ROOT, config_path=Path(cfg_path),
                    results_root=REPO_ROOT / "results/m5", run_name="m5-2-operator-gate",
                    prereg_path=PREREG)
    print(f"EXP-M5-2 run dir: {ctx.results_dir} (h_layer={opcfg['h_layer']})", flush=True)
    mt, revision = build_mt(cfg)

    per_relation = {}
    for rel in opcfg["relations"]:
        print(f"\n===== {rel} =====", flush=True)
        r = run_relation(mt, rel, opcfg)
        per_relation[rel] = r
        ctx.save_raw_completions(
            rel.replace("/", "_"),
            [{"relation": rel, "subject": s, "object": o,
              "top1_by_draw": [r["top1_by_draw"][d][i] for d in range(len(r["seeds"]))]}
             for i, (s, o) in enumerate(zip(r["probe_subjects"], r["probe_objects"]))],
        )
        print(f"  -> agreement={r['top1_agreement']} cos={r['output_state_cosine']} "
              f"faith={r['controls']['positive_faithfulness_median']} "
              f"shuffled={r['controls']['negative_shuffled_faithfulness_median']} "
              f"CONVERGED={r['converged']}", flush=True)

    n_conv = sum(r["converged"] for r in per_relation.values())
    s3_certified = n_conv >= opcfg["bars"]["n_relations_certify"]
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2) if cfg.device == "cuda" else None
    summary = {
        "experiment": "EXP-M5-2", "substrate": cfg.model.name, "revision": revision,
        "h_layer": opcfg["h_layer"], "z_layer": opcfg["z_layer"], "k_estimate": opcfg["k_estimate"],
        "n_draws": len(opcfg["seeds"]), "bars": opcfg["bars"],
        "n_relations": len(opcfg["relations"]), "n_converged": n_conv,
        "s3_certified_set": s3_certified, "peak_vram_gb": peak,
        "per_relation": {k: {kk: vv for kk, vv in v.items()
                             if kk not in ("probe_subjects", "probe_objects", "top1_by_draw")}
                         for k, v in per_relation.items()},
    }
    (ctx.results_dir / "operator_gate.json").write_text(json.dumps(summary, indent=2))
    _write_table(ctx.results_dir, summary)
    ctx.finalize(n_converged=n_conv, s3_certified_set=s3_certified, peak_vram_gb=peak)
    print(f"\n=== EXP-M5-2: {n_conv}/{len(opcfg['relations'])} relations converged; "
          f"S3-set-certified={s3_certified} (bar {opcfg['bars']['n_relations_certify']}) ===", flush=True)
    print(f"run dir: {ctx.results_dir}", flush=True)


def _write_table(run_dir: Path, s: dict):
    L = [f"# EXP-M5-2 S3 operator-gate — {s['n_converged']}/{s['n_relations']} converged; "
         f"S3-set-certified={s['s3_certified_set']}\n",
         "Functional gate (prereg EXP-M5-2). Certificate + sign-off are Ecaterina's.",
         f"Substrate {s['substrate']}@{s['revision']}, h_layer={s['h_layer']}, "
         f"k_estimate={s['k_estimate']}, n_draws={s['n_draws']}. Bars: {s['bars']}\n",
         "| relation | n_probe | top1-agree | out-cos | faith(pos) | shuffled(neg) | ctrl | CONVERGED | W-cos(desc) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for rel, r in s["per_relation"].items():
        c = r["controls"]
        L.append(f"| {rel} | {r['n_probe']} | {r['top1_agreement']} | {r['output_state_cosine']} | "
                 f"{c['positive_faithfulness_median']} | {c['negative_shuffled_faithfulness_median']} | "
                 f"{'ok' if r['controlled'] else 'FAIL'} | {'yes' if r['converged'] else 'no'} | "
                 f"{r['raw_w_cosine_descriptive']} |")
    (run_dir / "operator_gate_table.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--config", required=True)
    p.add_argument("--relation", required=True)
    p.add_argument("--layers", nargs="+", type=int, required=True)
    g = sub.add_parser("gate")
    g.add_argument("--config", required=True)
    a = ap.parse_args()
    if a.cmd == "probe":
        probe(a.config, a.relation, a.layers)
    else:
        gate(a.config)


if __name__ == "__main__":
    main()
