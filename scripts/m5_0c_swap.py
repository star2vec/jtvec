"""EXP-M5-0c: swap-intervention decomposition (410M vs 1.4B).

Instrument diagnostic (D-029): decide whether 1.4B's EXP-M5-0 Q2/Q6 failure is
a genuine potency-scaling effect (H-potency) or a flip-rate/base-margin
confound (H-confound). Puts a verdict on record; changes NO Q2/Q6 bar.

Estimator: the M1 lens-based causal swap (jvec.evals.swap — truncated pinv
rcond 0.05, source-token-position edit, norm preservation), reusing its hook +
pinv machinery, but capturing final-position LOGITS under three arms per item:
  - base  : unhooked forward
  - swap  : A-component moved onto e(swap_to)              [real intervention]
  - sham  : A-component moved onto a random unit direction [norm-matched twin]
    (n_random_seeds seeds; the item's sham = mean over seeds)

Per item (gap := logit(swap_answer) - logit(answer) at the final position):
  gap_shift_real          = gap_swap - gap_base
  gap_shift_sham          = gap_sham - gap_base
  sham_controlled_shift   = gap_shift_real - gap_shift_sham   (= gap_swap - gap_sham)
  base_margin             = logit_base(answer) - logit_base(runner_up)
  margin_normalized_flip  = sham_controlled_shift > base_margin
  top1_flip               = argmax(logit_swap) == swap_answer_id   (raw Q2 metric)

The three prereg quantities, IDENTICAL statistic across substrates:
  (1) sham-controlled logit-gap-shift distribution (median/IQR over draws),
  (2) top-1 flip rate,
  (3) margin-normalized flip fraction.

Statistics restricted to the MATCHED item set (items both substrates get right
pre-swap); full-set numbers reported too. base logits are draw-independent
(no hook), so base-correctness is per substrate.

margin_normalized_flip uses the prereg's literal definition; "gap(original) -
gap(runner-up)" is read as logit_base(answer) - logit_base(runner_up) (the
base dominance margin). Flagged in the writeup as a wording interpretation.

Subcommands:
  run     --swap-config <cfg> --fit-configs c0 c1 c2 --out <run_dir>
  verdict --run <run_dir> --ref <slug> --test <slug>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch
import yaml
from jlens import ActivationRecorder

from jvec.calibration import select_prompts
from jvec.config import Config, _from_dict
from jvec.evals.controls import random_unit_vector
from jvec.evals.swap import _SwapHook, _unembed_direction, pinv_jacobians
from jvec.evals.tasks import surface_token_ids
from jvec.lens_cache import load_lens
from jvec.modeling import load_model
from jvec.utils import REPO_ROOT, set_seed

# --- prereg thresholds (ratified as drafted) --------------------------------
POS_CONTROL_MIN = 0.30           # 410m sham-controlled gap-shift median (logit units)
NEG_CONTROL_BAND = (-0.03, 0.03)  # sham gap-shift median must lie here (both substrates)
POTENCY_DELTA = 0.15             # 410m median - 1.4b median >= this for H-potency
# margin-normalized flip low/high is qualitative in the prereg; these name the
# judgment used in decide() and are reported so Ecaterina can adjudicate.
MARGIN_FLIP_HIGH = 0.60          # "high" margin-normalized flip
MARGIN_FLIP_EXCESS = 0.15        # margin-flip exceeds raw top1-flip by this => confound-shaped


# --- pure statistic functions (model-free; covered by the landing test) -----
def item_stats(logit_base, logit_swap, logit_sham_list, answer_id: int, swap_answer_id: int) -> dict:
    """Per-item swap-decomposition statistics from three arms' final logits."""
    lb = torch.as_tensor(logit_base, dtype=torch.float64)
    ls = torch.as_tensor(logit_swap, dtype=torch.float64)
    gap_base = float(lb[swap_answer_id] - lb[answer_id])
    gap_swap = float(ls[swap_answer_id] - ls[answer_id])
    gap_shams = []
    for lsh in logit_sham_list:
        t = torch.as_tensor(lsh, dtype=torch.float64)
        gap_shams.append(float(t[swap_answer_id] - t[answer_id]))
    gap_sham = sum(gap_shams) / len(gap_shams)
    sham_controlled_shift = gap_swap - gap_sham
    # base runner-up = highest base logit other than the answer token
    masked = lb.clone()
    masked[answer_id] = float("-inf")
    runner_up_id = int(masked.argmax())
    base_margin = float(lb[answer_id] - lb[runner_up_id])
    # per-arm answer / swap-answer logits (retained so any control quantity is
    # post-hoc computable). The swap-answer-specific SHAM push isolates whether a
    # random-direction edit spuriously pushes the target, decoupled from the
    # source-ablation that (identically in both arms) suppresses the answer.
    lsa_base = float(lb[swap_answer_id])
    lsa_swap = float(ls[swap_answer_id])
    sham_sa = [float(torch.as_tensor(x, dtype=torch.float64)[swap_answer_id]) for x in logit_sham_list]
    lsa_sham = sum(sham_sa) / len(sham_sa)
    # probability-space swap-answer push: the M-series sham quantity (Q3 used
    # Δp(swap_answer) under the random direction ≈ 0). A norm-matched sham is an
    # ACTIVE edit, so its logit-space effect is O(1) even when the target-prob
    # effect is ~0; this is the reading that matches the [-0.03,0.03] band.
    def p_sa(t):
        return float(torch.softmax(torch.as_tensor(t, dtype=torch.float64), dim=-1)[swap_answer_id])
    p_base = p_sa(logit_base)
    p_swap = p_sa(logit_swap)
    p_sham = sum(p_sa(x) for x in logit_sham_list) / len(logit_sham_list)
    return {
        "gap_base": gap_base,
        "gap_swap": gap_swap,
        "gap_sham_mean": gap_sham,
        "gap_sham_seeds": gap_shams,
        "gap_shift_real": gap_swap - gap_base,
        "gap_shift_sham": gap_sham - gap_base,
        "sham_controlled_shift": sham_controlled_shift,
        "base_margin": base_margin,
        "runner_up_id": runner_up_id,
        "base_top1_id": int(lb.argmax()),
        "base_top1_correct": int(lb.argmax()) == answer_id,
        "margin_normalized_flip": bool(sham_controlled_shift > base_margin),
        "top1_flip": int(ls.argmax()) == swap_answer_id,
        "logit_swap_answer_base": lsa_base,
        "logit_swap_answer_swap": lsa_swap,
        "logit_swap_answer_sham_mean": lsa_sham,
        "real_swap_answer_push": lsa_swap - lsa_base,        # logit-space (informational)
        "sham_swap_answer_push": lsa_sham - lsa_base,        # logit-space (informational)
        "p_swap_answer_base": p_base,
        "p_swap_answer_swap": p_swap,
        "p_swap_answer_sham_mean": p_sham,
        "dp_swap_answer_real": p_swap - p_base,              # prob-space
        "dp_swap_answer_sham": p_sham - p_base,              # prob-space NEG CONTROL (M-series Q3 quantity)
    }


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _iqr(xs: list[float]) -> tuple[float, float, float]:
    """Return (q1, q3, iqr) via linear interpolation (numpy 'linear')."""
    s = sorted(xs)
    n = len(s)
    def q(p):
        if n == 1:
            return s[0]
        pos = p * (n - 1)
        lo = int(pos)
        frac = pos - lo
        hi = min(lo + 1, n - 1)
        return s[lo] + frac * (s[hi] - s[lo])
    q1, q3 = q(0.25), q(0.75)
    return q1, q3, q3 - q1


def draw_summary(items: list[dict]) -> dict:
    """One draw's item-level summary (over a given item set)."""
    n = len(items)
    scs = [i["sham_controlled_shift"] for i in items]
    shamshift = [i["gap_shift_sham"] for i in items]
    sham_push = [i["sham_swap_answer_push"] for i in items]
    sham_dp = [i["dp_swap_answer_sham"] for i in items]
    return {
        "n": n,
        "mean_sham_controlled_shift": sum(scs) / n if n else None,
        "median_sham_controlled_shift": _median(scs) if n else None,
        "mean_sham_gap_shift": sum(shamshift) / n if n else None,          # neg ctrl: logit gap-shift (confounded)
        "mean_sham_swap_answer_push": sum(sham_push) / n if n else None,   # neg ctrl: logit target-push (informational)
        "mean_dp_swap_answer_sham": sum(sham_dp) / n if n else None,       # neg ctrl: prob target-push (gating; M-series)
        "top1_flip_rate": sum(i["top1_flip"] for i in items) / n if n else None,
        "margin_norm_flip_rate": sum(i["margin_normalized_flip"] for i in items) / n if n else None,
    }


def aggregate_over_draws(draw_summaries: list[dict]) -> dict:
    """Median/IQR over draws of the per-draw item-means (the DrawSet statistic)."""
    scs = [d["mean_sham_controlled_shift"] for d in draw_summaries]
    sham = [d["mean_sham_gap_shift"] for d in draw_summaries]
    sham_push = [d["mean_sham_swap_answer_push"] for d in draw_summaries]
    sham_dp = [d["mean_dp_swap_answer_sham"] for d in draw_summaries]
    flip = [d["top1_flip_rate"] for d in draw_summaries]
    mflip = [d["margin_norm_flip_rate"] for d in draw_summaries]
    q1, q3, iqr = _iqr(scs)
    return {
        "n_draws": len(draw_summaries),
        "sham_controlled_shift": {
            "median": _median(scs), "q1": q1, "q3": q3, "iqr": iqr,
            "min": min(scs), "max": max(scs), "per_draw": scs,
        },
        "sham_gap_shift_median": _median(sham),                    # neg ctrl: logit gap-shift (confounded)
        "sham_swap_answer_push_median": _median(sham_push),        # neg ctrl: logit target-push (informational)
        "dp_swap_answer_sham_median": _median(sham_dp),            # neg ctrl: prob target-push (GATING)
        "top1_flip_rate_median": _median(flip),
        "margin_norm_flip_rate_median": _median(mflip),
        "per_draw_flip": flip,
        "per_draw_margin_flip": mflip,
    }


def require_controlled(ref_agg: dict, test_agg: dict) -> tuple[bool, dict]:
    """Instrument controls (instruments LAW).

    Positive: reference (410m) sham-controlled gap-shift median >= 0.30.
    Negative: a random-direction sham must give no spurious push TOWARD THE
    SWAP TARGET. Two readings are reported:
      - literal (prereg wording): sham gap-shift median in [-0.03, 0.03].
        NOTE this quantity is confounded by the source-component ablation the
        sham shares with the real swap (it suppresses the original answer,
        raising the gap) -- so it is NOT a clean target-specific null. Reported
        for the record; a FLAGGED clarification for Ecaterina (see writeup).
      - clarified (target-specific): sham swap-answer push (Δlogit(swap_answer)
        under the sham) median in [-0.03, 0.03]. This isolates the random
        direction's push toward the target and is the M-series-consistent null.
    The verdict gates on positive + clarified-negative; literal-negative is
    informational pending the ruling.
    """
    lo, hi = NEG_CONTROL_BAND
    pos_ok = ref_agg["sham_controlled_shift"]["median"] >= POS_CONTROL_MIN
    # prob-space target push (M-series Q3 quantity) -- GATING negative control
    neg_prob = {n: a["dp_swap_answer_sham_median"] for n, a in (("ref", ref_agg), ("test", test_agg))}
    # logit-space readings -- informational (a norm-matched sham is an active
    # edit, so both are O(1) even under a true null; they cannot meet a
    # prob-scale band). Reported for the record.
    neg_logit_push = {n: a["sham_swap_answer_push_median"] for n, a in (("ref", ref_agg), ("test", test_agg))}
    neg_logit_gap = {n: a["sham_gap_shift_median"] for n, a in (("ref", ref_agg), ("test", test_agg))}
    neg_prob_ok = all(lo <= v <= hi for v in neg_prob.values())
    report = {
        "positive_control_ok": bool(pos_ok),
        "positive_control_value": ref_agg["sham_controlled_shift"]["median"],
        "negative_control_prob_ok": bool(neg_prob_ok),
        "negative_control_prob_push": neg_prob,        # gating (band is in prob units)
        "negative_control_logit_push": neg_logit_push,  # informational
        "negative_control_logit_gap_shift": neg_logit_gap,  # informational (source-ablation confounded)
        "band": list(NEG_CONTROL_BAND),
        "flag": "prereg negative control reads 'sham gap-shift median within [-0.03,0.03] logit units'. "
                "A norm-matched sham is an ACTIVE edit whose logit effects are O(1) (it ablates the source "
                "component, suppressing the answer), so no logit-space sham quantity meets a 0.03 band even "
                "on the certified 410m. The band matches the M-series Q3 sham, which is Δp(swap_answer) under "
                "the random direction (~0). Gating on the PROB-space target push; logit readings reported. "
                "FLAGGED clarification for Ecaterina (does not affect the sham-controlled decision statistic).",
    }
    return (pos_ok and neg_prob_ok), report


def decide(ref_agg: dict, test_agg: dict) -> dict:
    """Apply the prereg decision rule. Returns label + the numbers it used."""
    ref_med = ref_agg["sham_controlled_shift"]["median"]
    test_med = test_agg["sham_controlled_shift"]["median"]
    delta = ref_med - test_med
    ref_iv = (ref_agg["sham_controlled_shift"]["q1"], ref_agg["sham_controlled_shift"]["q3"])
    test_iv = (test_agg["sham_controlled_shift"]["q1"], test_agg["sham_controlled_shift"]["q3"])
    non_overlap = ref_iv[0] > test_iv[1] or test_iv[0] > ref_iv[1]
    test_mflip = test_agg["margin_norm_flip_rate_median"]
    test_flip = test_agg["top1_flip_rate_median"]
    mflip_low = test_mflip < MARGIN_FLIP_HIGH
    mflip_high = test_mflip >= MARGIN_FLIP_HIGH and (test_mflip - test_flip) >= MARGIN_FLIP_EXCESS

    if delta >= POTENCY_DELTA and non_overlap and mflip_low:
        label = "H-POTENCY"
    elif (not non_overlap) and mflip_high:
        label = "H-CONFOUND"
    else:
        label = "MIXED"
    return {
        "label": label,
        "ref_median": ref_med, "test_median": test_med, "delta": delta,
        "potency_delta_bar": POTENCY_DELTA, "iqr_non_overlap": non_overlap,
        "ref_iqr": ref_iv, "test_iqr": test_iv,
        "test_margin_norm_flip": test_mflip, "test_top1_flip": test_flip,
        "margin_flip_low": mflip_low, "margin_flip_high": mflip_high,
        "note": "H-POTENCY: reduced gap-shift at 1.4B (HYPOTHESIS tier, NOT a gate failure). "
                "H-CONFOUND: comparable gap-shift, low raw flip but high margin-normalized flip. "
                "margin-flip low/high cutoff is a judgment (prereg leaves it qualitative); raw numbers reported.",
    }


# --- model-side swap pass ---------------------------------------------------
@torch.no_grad()
def _final_logits(model, input_ids) -> torch.Tensor:
    final = model.n_layers - 1
    with ActivationRecorder(model.layers, at=[final]) as rec:
        model.forward(input_ids)
        residual = rec.activations[final][0, -1].detach()
    return model.unembed(residual.float()).float().cpu()


@torch.no_grad()
def swap_logit_pass(model, tokenizer, lens, task, *, band, alpha, rcond, n_random_seeds) -> list[dict]:
    """Run base/swap/sham arms per item; return per-item logit-derived stats."""
    device = model.input_device
    band_layers = [l for l in lens.source_layers if band[0] <= l <= band[1]]
    J = {l: lens.jacobians[l].to(device) for l in band_layers}
    J_pinv = {l: p.to(device) for l, p in pinv_jacobians(lens, band_layers, rcond=rcond).items()}

    out = []
    for item in task.items:
        input_ids = model.encode(item["prompt"])
        src_token = surface_token_ids(tokenizer, item["intermediate"])[0]
        positions = (input_ids[0] == src_token).nonzero().flatten().tolist()
        if not positions:
            out.append({"name": item["name"], "skipped": "source token not in prompt"})
            continue
        e_src = _unembed_direction(model, tokenizer, item["intermediate"]).to(device)
        e_swap = _unembed_direction(model, tokenizer, item["swap_to"]).to(device)
        answer_id = surface_token_ids(tokenizer, item["answer"])[0]
        swap_answer_id = surface_token_ids(tokenizer, item["swap_answer"])[0]

        def run(e_dst):
            handles = [
                model.layers[l].register_forward_hook(
                    _SwapHook(J[l], J_pinv[l], e_src, e_dst, alpha, positions)
                )
                for l in band_layers
            ]
            try:
                return _final_logits(model, input_ids)
            finally:
                for h in handles:
                    h.remove()

        logit_base = _final_logits(model, input_ids)
        logit_swap = run(e_swap)
        logit_shams = [
            run(random_unit_vector(lens.d_model, seed=s).to(device))
            for s in range(n_random_seeds)
        ]
        stats = item_stats(logit_base, logit_swap, logit_shams, answer_id, swap_answer_id)
        stats["name"] = item["name"]
        stats["n_edit_positions"] = len(positions)
        out.append(stats)
    return out


def load_swap_config(path: str) -> tuple[Config, dict]:
    """Load the swap config, tolerating 0c-only keys the strict Config schema
    rejects (decomposition / draws_cache_dirs / tasks). Those are either the
    defaults this orchestrator already implements, or unused (draws are driven
    by the fit configs). Returns (Config, extra_keys)."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name for f in dataclasses.fields(Config)}
    extra = {k: raw.pop(k) for k in list(raw) if k not in known}
    return _from_dict(Config, raw), extra


def run_substrate(swap_cfg_path: str, fit_cfg_paths: list[str], out_dir: Path):
    swap_cfg, extra = load_swap_config(swap_cfg_path)
    set_seed(swap_cfg.seed)
    ev = swap_cfg.evals
    band = tuple(ev.band)
    # one swap-capitals task
    task_raw = json.loads((REPO_ROOT / "tasks" / "swap-capitals.json").read_text())
    from jvec.evals.tasks import Task
    task = Task(name=task_raw["task"], protocol=task_raw["protocol"], items=task_raw["items"])

    model, tok, revision = load_model(swap_cfg)
    slug = swap_cfg.experiment
    print(f"[{slug}] model {swap_cfg.model.name}@{revision} band={band} "
          f"alpha={ev.swap_alpha} rcond={ev.swap_rcond} sham_seeds={ev.n_random_seeds}", flush=True)

    draws = []
    for i, fcp in enumerate(fit_cfg_paths):
        fit_cfg = Config.load(fcp)
        prompts = select_prompts(fit_cfg, tok)
        skip = fit_cfg.fit.skip_first_variants[0]
        lens = load_lens(fit_cfg, skip, prompts, revision)
        t0 = time.perf_counter()
        recs = swap_logit_pass(
            model, tok, lens, task,
            band=band, alpha=ev.swap_alpha, rcond=ev.swap_rcond,
            n_random_seeds=ev.n_random_seeds,
        )
        dt = time.perf_counter() - t0
        scored = [r for r in recs if "skipped" not in r]
        print(f"  draw{i} ({fit_cfg.seed}) lens={lens.source_layers[0]}..{lens.source_layers[-1]} "
              f"n={len(scored)} {dt:.1f}s", flush=True)
        draws.append({"draw": i, "fit_config": fcp, "seed": fit_cfg.seed,
                      "cache_dir": fit_cfg.cache_dir, "wall_s": round(dt, 1), "items": recs})

    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir = out_dir / slug
    sub_dir.mkdir(parents=True, exist_ok=True)
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2) if swap_cfg.device == "cuda" else None
    manifest = {
        "experiment": "EXP-M5-0c", "substrate": slug,
        "model": swap_cfg.model.name, "revision": revision, "dtype": swap_cfg.model.dtype,
        "band": list(band), "alpha": ev.swap_alpha, "rcond": ev.swap_rcond,
        "n_random_seeds": ev.n_random_seeds, "task": task.name, "n_items": len(task.items),
        "peak_vram_gb": peak, "device": swap_cfg.device,
    }
    (sub_dir / "records.json").write_text(json.dumps({"manifest": manifest, "draws": draws}, indent=1))
    # copy the swap config into the results dir (LAW)
    (sub_dir / Path(swap_cfg_path).name).write_text(Path(swap_cfg_path).read_text())
    # raw completions per (substrate, draw) — one jsonl per cell (LAW convention)
    raw_dir = out_dir / "raw_completions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for d in draws:
        with (raw_dir / f"{slug}_draw{d['draw']}.jsonl").open("w", encoding="utf-8") as f:
            for r in d["items"]:
                f.write(json.dumps(r) + "\n")
    print(f"[{slug}] wrote {sub_dir/'records.json'} (peak VRAM {peak} GB)", flush=True)


# --- verdict ----------------------------------------------------------------
def _load_records(run_dir: Path, slug: str) -> dict:
    return json.loads((run_dir / slug / "records.json").read_text())


def _base_correct_names(records: dict) -> set[str]:
    # base logits are draw-independent; use draw 0
    d0 = records["draws"][0]["items"]
    return {r["name"] for r in d0 if "skipped" not in r and r["base_top1_correct"]}


def verdict(run_dir: Path, ref_slug: str, test_slug: str):
    ref, test = _load_records(run_dir, ref_slug), _load_records(run_dir, test_slug)
    matched = sorted(_base_correct_names(ref) & _base_correct_names(test))

    def agg_for(records: dict, names: set[str]):
        summaries = []
        for d in records["draws"]:
            items = [r for r in d["items"] if "skipped" not in r and r["name"] in names]
            summaries.append(draw_summary(items))
        return aggregate_over_draws(summaries), summaries

    matched_set = set(matched)
    ref_agg, ref_draws = agg_for(ref, matched_set)
    test_agg, test_draws = agg_for(test, matched_set)
    controlled, control_report = require_controlled(ref_agg, test_agg)
    decision = decide(ref_agg, test_agg)
    if not controlled:
        decision = {**decision, "gated_note": "controls did not fully pass; verdict reported PROVISIONAL", "provisional": True}

    # also compute full-set (all base-correct per substrate, unmatched) for transparency
    ref_full, _ = agg_for(ref, _base_correct_names(ref))
    test_full, _ = agg_for(test, _base_correct_names(test))

    out = {
        "experiment": "EXP-M5-0c", "tier": "instrument-diagnostic",
        "changes_q2_q6_bar": False,
        "ref_substrate": ref_slug, "test_substrate": test_slug,
        "matched_item_set": matched, "n_matched": len(matched),
        "controls": {"passed": controlled, **control_report},
        "verdict": decision,
        "matched": {"ref": ref_agg, "test": test_agg},
        "matched_per_draw": {"ref": ref_draws, "test": test_draws},
        "full_unmatched": {"ref": ref_full, "test": test_full},
    }
    (run_dir / "verdict.json").write_text(json.dumps(out, indent=2))
    _write_table(run_dir, out, ref, test, matched)
    print(json.dumps({"verdict": decision["label"], "controls_passed": controlled,
                      "n_matched": len(matched),
                      "ref_median": round(ref_agg["sham_controlled_shift"]["median"], 4),
                      "test_median": round(test_agg["sham_controlled_shift"]["median"], 4)}, indent=2))


def _write_table(run_dir: Path, out: dict, ref: dict, test: dict, matched: list[str]):
    v = out["verdict"]
    c = out["controls"]
    L = [f"# EXP-M5-0c swap-decomposition verdict — **{v['label']}**"
         + ("  (PROVISIONAL)" if v.get("provisional") else "") + "\n",
         "Instrument diagnostic (D-029). Puts a verdict on record; changes NO Q2/Q6 bar.",
         f"Controls passed (positive + prob-space negative): {c['passed']}",
         f"- positive (410m sham-ctrl gap-shift median >= {POS_CONTROL_MIN}): {c['positive_control_ok']} "
         f"(value {c['positive_control_value']:.3f})",
         f"- negative GATING, prob-space dp(swap_answer) under sham in {c['band']}: "
         f"{c['negative_control_prob_ok']} ({ {k: round(x,4) for k,x in c['negative_control_prob_push'].items()} })",
         f"- negative informational, logit target-push: "
         f"{ {k: round(x,3) for k,x in c['negative_control_logit_push'].items()} }; "
         f"logit gap-shift: { {k: round(x,3) for k,x in c['negative_control_logit_gap_shift'].items()} }",
         f"- FLAG: {c['flag']}",
         f"\nMatched item set (base-correct on both): N={out['n_matched']} — {', '.join(matched)}\n"]
    L.append("| substrate | sham-ctrl gap-shift median [q1,q3] (per-draw) | neg dp(sham) | top1-flip med | margin-norm-flip med |")
    L.append("|---|---|---|---|---|")
    for slug, agg in (("ref " + out["ref_substrate"], out["matched"]["ref"]),
                      ("test " + out["test_substrate"], out["matched"]["test"])):
        s = agg["sham_controlled_shift"]
        pd = ",".join(f"{x:.3f}" for x in s["per_draw"])
        L.append(f"| {slug} | {s['median']:.3f} [{s['q1']:.3f},{s['q3']:.3f}] ({pd}) | "
                 f"{agg['dp_swap_answer_sham_median']:.4f} | "
                 f"{agg['top1_flip_rate_median']:.3f} | {agg['margin_norm_flip_rate_median']:.3f} |")
    L.append(f"\nDecision numbers: 410m_median={v.get('ref_median'):.3f}, "
             f"1.4b_median={v.get('test_median'):.3f}, delta={v.get('delta'):.3f} "
             f"(bar {v.get('potency_delta_bar')}), IQR non-overlap={v.get('iqr_non_overlap')}, "
             f"1.4b margin-flip={v.get('test_margin_norm_flip')}, 1.4b raw-flip={v.get('test_top1_flip')}.")
    L.append(f"\n{v.get('note','')}")
    (run_dir / "verdict_table.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def experiment(ref_swap_cfg, ref_fits, test_swap_cfg, test_fits, prereg, results_root):
    """Conformant single-experiment entry: start_run (clean tree + committed
    prereg + run.json + copied config) once, run both substrates into the run
    dir, then the verdict. One experiment -> one results dir -> one commit."""
    from jtvec.core.runctx import start_run
    ctx = start_run(
        repo_root=REPO_ROOT, config_path=Path(ref_swap_cfg),
        results_root=Path(results_root), run_name="0c-swap-decomposition",
        prereg_path=Path(prereg),
    )
    print(f"0c run dir: {ctx.results_dir}", flush=True)
    for extra in (test_swap_cfg, *ref_fits, *test_fits):
        import shutil
        shutil.copy2(extra, ctx.results_dir / Path(extra).name)
    ref_slug, _ = load_swap_config(ref_swap_cfg)
    test_slug, _ = load_swap_config(test_swap_cfg)
    run_substrate(ref_swap_cfg, ref_fits, ctx.results_dir)
    run_substrate(test_swap_cfg, test_fits, ctx.results_dir)
    verdict(ctx.results_dir, ref_slug.experiment, test_slug.experiment)
    out = json.loads((ctx.results_dir / "verdict.json").read_text())
    ctx.finalize(verdict=out["verdict"]["label"], controls_passed=out["controls"]["passed"],
                 n_matched=out["n_matched"])
    return ctx.results_dir


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--swap-config", required=True)
    r.add_argument("--fit-configs", nargs=3, required=True)
    r.add_argument("--out", required=True)
    v = sub.add_parser("verdict")
    v.add_argument("--run", required=True)
    v.add_argument("--ref", required=True)
    v.add_argument("--test", required=True)
    e = sub.add_parser("experiment")
    e.add_argument("--ref-swap-config", required=True)
    e.add_argument("--ref-fits", nargs=3, required=True)
    e.add_argument("--test-swap-config", required=True)
    e.add_argument("--test-fits", nargs=3, required=True)
    e.add_argument("--prereg", required=True)
    e.add_argument("--results-root", default="results/m5")
    a = ap.parse_args()
    if a.cmd == "run":
        run_substrate(a.swap_config, a.fit_configs, Path(a.out))
    elif a.cmd == "verdict":
        verdict(Path(a.run), a.ref, a.test)
    else:
        experiment(a.ref_swap_config, a.ref_fits, a.test_swap_config, a.test_fits,
                   a.prereg, a.results_root)


if __name__ == "__main__":
    main()
