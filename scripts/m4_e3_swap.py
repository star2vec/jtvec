"""M4-E3 orchestrator: cross-basis FV swap on capitalize->singular-plural.

On capitalize (task A) 10-shot prompts, swap the FV_A component onto FV_B
(singular-plural) at the final position of band layers 4-16 and measure the
task-B answer rate, under none / lens_swap / direct_swap / random_target,
cross-draw over the 3 M2-certified FV draws (draw k uses fv_A_k and fv_B_k).
The M3-gated fv-swap instrument is asserted controlled before use; scoring is
exact-match case-sensitive (D-012). Context sets sampled ONCE and reused
across conditions (paired). Scope: the gated capitalize->singular-plural pair
only (Ecaterina 2026-07-19); the translation pair is deferred.

Prereg: harness/preregs/EXP-M4-E3-swap.md (committed before first run;
start_run enforces).

Usage: uv run python scripts/m4_e3_swap.py [--config configs/m4_e3_swap_pythia410m.yaml]
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

import numpy as np

from jvec.calibration import select_prompts
from jvec.config import Config
from jvec.evals.fvswap import final_logits, make_swap_hooks
from jvec.evals.swap import pinv_jacobians
from jvec.fv import FV_REPO, load_fv_model
from jvec.lens_cache import lens_dir, load_lens
from jvec.utils import peak_rss_gb, set_seed
from jtvec.core.instruments import ControlRecord, Instrument, require_controlled
from jtvec.core.reporting import scoped
from jtvec.core.runctx import start_run
from jtvec.e3_swap import SwapRedirectionRule
from jtvec.m3_instruments import answer_first_tokens, load_certified_fv, shared_query_map, verify_lens_manifest

M1_RUN = REPO_ROOT / "results/m1/20260718-010559-lens-gate"
M2_RUN = REPO_ROOT / "results/m2/20260718-114950-fv-stability-gate"
M3_RUN = REPO_ROOT / "results/m3/20260718-174954-instrument-gate"
PREREG = REPO_ROOT / "harness/preregs/EXP-M4-E3-swap.md"

TASK_A, TASK_B = "capitalize", "singular-plural"  # M3-gated swap direction
FV_DRAWS = (1, 2, 3)
N_QUERIES = 30
CTX_RNG_SEED = 6363
RANDOM_SEED0 = 400
RULE = SwapRedirectionRule(min_b_gain=0.20, max_random_elevation=0.05, min_j_specificity=0.15)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/m4_e3_swap_pythia410m.yaml"))
    args = parser.parse_args()
    t_start = time.perf_counter()

    cfg = Config.load(args.config)
    ctx = start_run(
        repo_root=REPO_ROOT, config_path=Path(args.config),
        results_root=REPO_ROOT / cfg.results_dir, run_name="e3-swap", prereg_path=PREREG,
    )
    print(f"E3 swap run dir: {ctx.results_dir}", flush=True)

    # --- assert the fv-swap instrument is gated (M3) ---------------------------
    m3c = json.loads((M3_RUN / "controls.json").read_text(encoding="utf-8"))["fv-swap"]
    today = time.strftime("%Y-%m-%d")
    require_controlled(Instrument(
        name="fv-swap",
        positive_control=ControlRecord(run=str(M3_RUN), passed=bool(m3c["positive"]), date=today),
        negative_control=ControlRecord(run=str(M3_RUN), passed=bool(m3c["negative"]), date=today),
    ))
    print(f"[gated] fv-swap (ControlRecord {M3_RUN.name})", flush=True)

    set_seed(cfg.seed)
    hf_model, tokenizer, model_config, revision = load_fv_model(cfg)
    import jlens as jlens_pkg  # noqa: PLC0415
    model_j = jlens_pkg.from_hf(hf_model, tokenizer)
    model_scope = f"{cfg.model.name}@{revision[:7]}"
    config_scope = f"EXP-M4-E3-swap ({Path(args.config).name})"
    bos = tokenizer.bos_token or ""
    lo, hi = cfg.evals.band

    # --- lens draw 0 (cache/m3), identity-checked; pinvs -----------------------
    dcfg = dataclasses.replace(cfg, seed=0, cache_dir="cache/m3")
    set_seed(0)
    prompts = select_prompts(dcfg, tokenizer)
    lens = load_lens(dcfg, 4, prompts, revision)
    man = json.loads((lens_dir(dcfg, 4) / "manifest.json").read_text(encoding="utf-8"))
    ref = json.loads((M1_RUN / "draws/draw0/manifest.json").read_text(encoding="utf-8"))
    mism = verify_lens_manifest(man, ref)
    if mism:
        sys.exit(f"lens draw0 identity mismatch vs M1: {mism}")
    band_layers = [l for l in lens.source_layers if lo <= l <= hi]
    pinvs = pinv_jacobians(lens, band_layers, rcond=cfg.evals.swap_rcond)
    print(f"[lens] draw0 identity ok; band {band_layers}", flush=True)

    # --- certified FV pairs (draw k of both) -----------------------------------
    certs = json.loads((M2_RUN / "certificates.json").read_text(encoding="utf-8"))
    fv_a = {k: load_certified_fv(cfg, TASK_A, revision, certs, draw_k=k) for k in FV_DRAWS}
    fv_b = {k: load_certified_fv(cfg, TASK_B, revision, certs, draw_k=k) for k in FV_DRAWS}
    print(f"[fvs] certified {TASK_A} + {TASK_B} FVs loaded: draws {FV_DRAWS}", flush=True)

    from utils.prompt_utils import load_dataset  # noqa: PLC0415
    set_seed(cfg.seed)
    ds_a = load_dataset(TASK_A, root_data_dir=str(FV_REPO / "dataset_files"), seed=cfg.seed)
    ds_b = load_dataset(TASK_B, root_data_dir=str(FV_REPO / "dataset_files"), seed=cfg.seed)
    shared = shared_query_map(ds_a, ds_b)  # q -> (y_a, y_b), distinct
    if len(shared) < 10:
        sys.exit(f"{TASK_A}->{TASK_B}: only {len(shared)} shared queries; pair unusable")
    rng = np.random.default_rng(CTX_RNG_SEED)
    queries = list(rng.choice(sorted(shared), min(N_QUERIES, len(shared)), replace=False))

    # fixed, paired task-A contexts (sampled once, reused across conditions/draws)
    train_a = ds_a["train"]
    def a_context():
        idx = rng.choice(len(train_a), cfg.fv.n_shots, replace=False)
        ch = train_a[idx]
        return bos + "".join(f"Q: {x}\nA: {y}\n\n" for x, y in zip(ch["input"], ch["output"]))
    swap_items = [(a_context(), q, shared[q][0], shared[q][1]) for q in queries]

    def b_rate(hooks, cell) -> float:
        b_hits, rows = 0, []
        for c, q, y_a, y_b in swap_items:
            logits = final_logits(model_j, c + f"Q: {q}\nA:", hooks)
            top1 = int(logits.argmax())
            b_hit = top1 in answer_first_tokens(tokenizer, y_b, case_sensitive=True)
            a_hit = top1 in answer_first_tokens(tokenizer, y_a, case_sensitive=True)
            b_hits += b_hit
            rows.append({"query": q, "target_a": y_a, "target_b": y_b,
                         "top1": tokenizer.decode([top1]), "b_hit": bool(b_hit), "a_hit": bool(a_hit)})
        ctx.save_raw_completions(cell, rows)
        return b_hits / len(swap_items)

    # --- none (shared clean) ---------------------------------------------------
    none_b = b_rate({}, "swap_none")
    print(f"[none] B-rate {none_b:.3f}", flush=True)

    # --- per FV draw: lens_swap / direct_swap / random_target ------------------
    lens_b, direct_b, random_b = {}, {}, {}
    for k in FV_DRAWS:
        lens_hooks = make_swap_hooks("lens_swap", band_layers, lens, fv_a[k].vector, fv_b[k].vector, pinvs, seed=cfg.seed)
        direct_hooks = make_swap_hooks("direct_swap", band_layers, lens, fv_a[k].vector, fv_b[k].vector, pinvs, seed=cfg.seed)
        rand_hooks = make_swap_hooks("random_target", band_layers, lens, fv_a[k].vector, fv_b[k].vector, pinvs, seed=RANDOM_SEED0 + k)
        lens_b[k] = b_rate(lens_hooks, f"swap_lens_draw{k}")
        direct_b[k] = b_rate(direct_hooks, f"swap_direct_draw{k}")
        random_b[k] = b_rate(rand_hooks, f"swap_random_draw{k}")
        print(f"[draw{k}] none {none_b:.3f} | lens_swap {lens_b[k]:.3f} | "
              f"direct_swap {direct_b[k]:.3f} | random {random_b[k]:.3f}", flush=True)

    verdict = RULE.verdict(none_b=none_b, lens_b=lens_b, direct_b=direct_b, random_b=random_b)
    (ctx.results_dir / "e3_results.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")

    # --- report ----------------------------------------------------------------
    lines = [
        "# EXP-M4-E3 report: cross-basis FV swap (capitalize -> singular-plural)",
        "",
        f"- model: {model_scope} (full sha in run.json/config)",
        "- prereg: harness/preregs/EXP-M4-E3-swap.md (constants D-018)",
        f"- clean task-B rate {none_b:.3f}; swap moves fv_A component onto fv_B at "
        f"band layers 4-16, final position; cross-draw over 3 certified FV draws",
        f"- decision (D-018): redirects iff best-swap gain median >= {RULE.min_b_gain} and "
        f"random gain <= {RULE.max_random_elevation}; J-specific iff lens-direct median "
        f">= {RULE.min_j_specificity}",
        "",
        "## Task-B answer rate by condition (per FV draw + median)",
        "",
        "| condition | draw1 | draw2 | draw3 | median |",
        "|---|---|---|---|---|",
        f"| none | {none_b:.3f} | {none_b:.3f} | {none_b:.3f} | {none_b:.3f} |",
        f"| lens_swap | {lens_b[1]:.3f} | {lens_b[2]:.3f} | {lens_b[3]:.3f} | {verdict['lens_b_median']:.3f} |",
        f"| direct_swap | {direct_b[1]:.3f} | {direct_b[2]:.3f} | {direct_b[3]:.3f} | {verdict['direct_b_median']:.3f} |",
        f"| random_target | {random_b[1]:.3f} | {random_b[2]:.3f} | {random_b[3]:.3f} | {verdict['random_b_median']:.3f} |",
        "",
        "## Verdict",
        "",
        "- " + scoped(
            f"E3 {TASK_A}->{TASK_B}: best-swap B-gain median "
            f"{verdict['best_swap_gain_median']:+.3f} (random {verdict['random_gain_median']:+.3f}); "
            f"lens-direct gap {verdict['j_specificity_gap']:+.3f}; verdict {verdict['verdict']} "
            f"(transfer {verdict['cross_draw_transfer']})",
            float(verdict["best_swap_gain_median"]), model=model_scope, config=config_scope, n=N_QUERIES,
        ),
        "",
        f"**E3 verdict: {verdict['verdict']}** "
        f"(redirects={verdict['redirects']}, J-specific={verdict['j_specific']}, "
        f"cross-draw transfer={verdict['cross_draw_transfer']})",
        "",
        f"wall-clock {round(time.perf_counter() - t_start, 1)} s; peak RSS "
        f"{peak_rss_gb():.2f} GB; device {cfg.device}; grid in e3_results.json; raw cells "
        "under raw_completions/.",
        "",
    ]
    (ctx.results_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    ctx.finalize(
        verdict=verdict["verdict"], redirects=verdict["redirects"],
        j_specific=verdict["j_specific"], cross_draw_transfer=verdict["cross_draw_transfer"],
        none_b=none_b, model_revision=revision,
        wall_clock_s=round(time.perf_counter() - t_start, 1), peak_rss_gb=round(peak_rss_gb(), 2),
    )
    print(f"\nE3 verdict: {verdict['verdict']}")
    print(f"report: {ctx.results_dir / 'report.md'}")


if __name__ == "__main__":
    main()
