"""EXP-M5-1c null-check orchestrator (410M): do the recalibrated instruments
report NULL on a known-null? Governance gate (CONSTRAINTS instrument-amendment
LAW). Two instruments, reported separately:

I1 — the amended-Q5 max-contrast statistic on a SCRAMBLED-LABEL latent probe
(capital-operand with permuted intermediates) over the cached 410M lenses. Pass:
median-over-draws ratio_jlens < 5.0 (no manufactured advantage). Sanity: the
GENUINE (unscrambled) anchor clears 5.0 (pipeline can fire).

I2 — the D-033 extended-ladder concept readout on SCRAMBLED-LABEL directions
(random grouping, ladder to 256, alpha-swept resolvable Δp). Pass: scrambled
min-pairwise cosine < 0.95 at every rung AND sham-controlled Δp within
max(0.005,1/N) at every alpha (no false convergence / no manufactured steering).
Sanity: genuine-label cosine materially exceeds scrambled (discrimination); an
unembed-direction injection moves Δp resolvably (injection can steer).

Prereg: harness/preregs/EXP-M5-1c-null-check.md (RATIFIED, committed bb1dcb2).
Usage: uv run python scripts/m5_1c_null_check.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from jvec.calibration import select_prompts
from jvec.config import Config
from jvec.evals.baseline import score_task
from jvec.evals.probe import probe_task
from jvec.evals.swap import _unembed_direction
from jvec.evals.tasks import Task, load_tasks, surface_token_ids
from jvec.lens_cache import load_lens
from jvec.modeling import load_model
from jvec.utils import peak_rss_gb, set_seed
from jtvec.concept_gate import (
    CAPITAL_ROSTER, capital_context_stream, group_by_label, identity_direction,
    injection_deltas, mean_difference_by_layer, min_pairwise_cosine,
    natural_norms, rung_prefix, scrambled_labels,
)
from jvec.evals.concept import answer_states, injected_final_probs
from jtvec.core.draws import DrawSet
from jtvec.core.runctx import start_run
from jtvec.fv_stability import sham_twin
from jtvec.lens_diagnostic import task_arm_ratios

PREREG = REPO_ROOT / "harness/preregs/EXP-M5-1c-null-check.md"
NULL_CFG = REPO_ROOT / "configs/m5_1c_nullcheck_pythia410m.yaml"
DRAW_CFGS = {k: REPO_ROOT / f"configs/m1_pythia410m_draw{k}.yaml" for k in (0, 1, 2)}
# I1
SKIP_FIRST, CAP, ADV, PASS_K, N_RANDOM = 4, 5.0, 5.0, 10, 10
I1_ANCHOR = "capital-operand"
# I2
RUNGS_EXT = (8, 16, 32, 64, 128, 256)
CTX_DRAWS = (1, 2, 3)
ALPHAS = (1.0, 2.0, 4.0, 8.0)
N_EVAL, EVAL_SEED, SHAM_SEED0 = 200, 9000, 9100
COS_BAR, POT_BASE, DISCRIM_MARGIN = 0.95, 0.005, 0.15


# ---------- I1: max-contrast on a scrambled-label latent probe ----------

def scramble_intermediates(task: Task, seed: int) -> Task:
    """Permute each item's probed intermediate onto another item's (a derangement
    where possible), so the probe reads a target with no alignment to the state."""
    n = len(task.items)
    perm = np.random.default_rng([seed, 11]).permutation(n)
    for i in range(n):  # avoid fixed points where possible
        if perm[i] == i:
            perm[i], perm[(i + 1) % n] = perm[(i + 1) % n], perm[i]
    items = [dict(it, intermediates=task.items[perm[i]]["intermediates"])
             for i, it in enumerate(task.items)]
    return Task(name=f"{task.name}-scrambled", protocol=task.protocol, items=items)


def i1_null_check(ctx, model, tok, revision) -> dict:
    tasks = {t.name: t for t in load_tasks(REPO_ROOT / "tasks")}
    anchor = tasks[I1_ANCHOR]
    scored = score_task(model, tok, anchor)
    correct = {it["name"] for it in scored["per_item"] if it["correct"]}
    genuine = Task(anchor.name, anchor.protocol,
                   [it for it in anchor.items if it["name"] in correct])
    print(f"[I1] {I1_ANCHOR} behavioural {len(genuine.items)}/{len(anchor.items)}", flush=True)

    gen_ratios, scr_ratios = [], []
    for k in (0, 1, 2):
        cfg_k = Config.load(str(DRAW_CFGS[k]))
        set_seed(cfg_k.seed)
        prompts = select_prompts(cfg_k, tok)
        lens = load_lens(cfg_k, SKIP_FIRST, prompts, revision)
        layers = list(lens.source_layers)
        scrambled = scramble_intermediates(genuine, seed=k)
        g = task_arm_ratios(json.loads(json.dumps(probe_task(
            model, tok, lens, genuine, pass_k=PASS_K, n_random_seeds=N_RANDOM)["metrics"])), layers, CAP)
        s = task_arm_ratios(json.loads(json.dumps(probe_task(
            model, tok, lens, scrambled, pass_k=PASS_K, n_random_seeds=N_RANDOM)["metrics"])), layers, CAP)
        gen_ratios.append(g["jlens_ratio"]); scr_ratios.append(s["jlens_ratio"])
        ctx.save_raw_completions(f"I1_draw{k}", [
            {"arm": "genuine", **g}, {"arm": "scrambled", **s}])
        print(f"[I1] draw{k} genuine ratio_jlens={g['jlens_ratio']:.2f} | "
              f"scrambled={s['jlens_ratio']:.2f} (rand_max={s['random_max_ratio']:.2f})", flush=True)

    gen_med = float(np.median(gen_ratios)); scr_med = float(np.median(scr_ratios))
    sanity = gen_med >= ADV
    passed = scr_med < ADV
    return {"instrument": "I1-max-contrast", "genuine_ratio_median": round(gen_med, 3),
            "scrambled_ratio_median": round(scr_med, 3), "bar": ADV,
            "sanity_pipeline_fires": sanity, "null_reported": passed,
            "passed": bool(passed and sanity), "genuine_ratios": gen_ratios,
            "scrambled_ratios": scr_ratios}


# ---------- I2: scrambled-label concept direction ----------

def eval_carriers(n: int) -> list[str]:
    """N fixed mixed capital-recall carriers (varied countries/prefixes)."""
    return [capital_context_stream(CAPITAL_ROSTER[i % len(CAPITAL_ROSTER)],
                                   seed=EVAL_SEED + i, n=1)[0] for i in range(n)]


def cosine_ladder(states_by_draw: dict[int, dict], labels_by_draw: dict[int, list],
                  band: list[int], n_labels: int) -> dict[int, float]:
    """Max over roster labels of the min-pairwise cosine across draws, per rung."""
    out = {}
    for t in RUNGS_EXT:
        per_label = []
        for c in range(n_labels):
            dirs = []
            for k, states in states_by_draw.items():
                sliced = {l: rung_prefix(v, t) for l, v in states.items()}
                grp = group_by_label(sliced, labels_by_draw[k][:t], c)
                if grp is None:
                    break
                dirs.append(identity_direction(mean_difference_by_layer(*grp), band))
            if len(dirs) == len(states_by_draw):
                per_label.append(min_pairwise_cosine(dirs))
        out[t] = max(per_label) if per_label else 0.0
    return out


def i2_null_check(ctx, model, tok) -> dict:
    cfg = Config.load(str(NULL_CFG))
    lo, hi = cfg.evals.band
    band = [l for l in range(model.n_layers) if lo <= l <= hi]
    n_lab = len(CAPITAL_ROSTER)
    pool_n = max(RUNGS_EXT)

    # extract the mixed context pool once per draw; record true + scrambled labels
    states_by_draw, true_lab, scr_lab = {}, {}, {}
    for k in CTX_DRAWS:
        pool, tlab = [], []
        per = pool_n // n_lab
        for ci, cap in enumerate(CAPITAL_ROSTER):
            strm = capital_context_stream(cap, seed=k, n=per)
            pool += strm; tlab += [ci] * len(strm)
        states_by_draw[k] = answer_states(model, pool[:pool_n], band)
        true_lab[k] = tlab[:pool_n]
        scr_lab[k] = scrambled_labels(pool_n, n_lab, seed=k)
    print(f"[I2] pool {pool_n} contexts x {len(CTX_DRAWS)} draws extracted; band {band}", flush=True)

    scr_cos = cosine_ladder(states_by_draw, scr_lab, band, n_lab)
    gen_cos = cosine_ladder(states_by_draw, true_lab, band, n_lab)
    max_scr_cos = max(scr_cos.values())
    ceil = max(RUNGS_EXT)
    discrim_ok = gen_cos[ceil] - scr_cos[ceil] >= DISCRIM_MARGIN
    print(f"[I2] cosine@256 genuine {gen_cos[ceil]:.3f} vs scrambled {scr_cos[ceil]:.3f} "
          f"(max scrambled over rungs {max_scr_cos:.3f})", flush=True)

    # potency null: scrambled direction (full pool) injected on carriers, alpha sweep.
    # Precompute per-label base p(answer) and the unscaled scrambled deltas once.
    carriers = eval_carriers(N_EVAL)
    ans_ids = [surface_token_ids(tok, cap)[0] for _, cap in CAPITAL_ROSTER]
    base_p = {c: [float(injected_final_probs(model, p, {})[ans_ids[c]]) for p in carriers]
              for c in range(n_lab)}
    unscaled = {}  # (c,k) -> per-layer unit*natural-norm deltas of the scrambled direction
    for c in range(n_lab):
        for k in CTX_DRAWS:
            grp = group_by_label(states_by_draw[k], scr_lab[k], c)
            if grp is not None:
                unscaled[(c, k)] = injection_deltas(mean_difference_by_layer(*grp),
                                                    natural_norms(grp[0]))

    def sham_ctrl_dp(deltas, sham, c):
        ans = ans_ids[c]
        dp_i = np.mean([float(injected_final_probs(model, p, deltas)[ans]) - b
                        for p, b in zip(carriers, base_p[c])])
        dp_s = np.mean([float(injected_final_probs(model, p, sham)[ans]) - b
                        for p, b in zip(carriers, base_p[c])])
        return float(dp_i - dp_s)

    worst_abs_dp, pot_rows, alpha_report = 0.0, [], {}
    for a_i, alpha in enumerate(ALPHAS):
        per_label_med = []
        for c in range(n_lab):
            eff, seeds = [], []
            for k in CTX_DRAWS:
                if (c, k) not in unscaled:
                    continue
                deltas = {l: d * alpha for l, d in unscaled[(c, k)].items()}
                sham = {l: sham_twin(d, SHAM_SEED0 + 1000 * a_i + 10 * k + c)
                        for l, d in deltas.items()}
                eff.append(sham_ctrl_dp(deltas, sham, c)); seeds.append(k)
            if len(eff) >= 3:
                med = float(DrawSet(tuple(eff), tuple(seeds)).median)
                per_label_med.append(med)
                pot_rows.append({"alpha": alpha, "label": CAPITAL_ROSTER[c][1],
                                 "sham_ctrl_dp_median": round(med, 5)})
        worst = max((abs(m) for m in per_label_med), default=0.0)
        worst_abs_dp = max(worst_abs_dp, worst)
        alpha_report[alpha] = round(worst, 5)
        print(f"[I2] alpha={alpha:.0f} worst |sham-ctrl Δp| over labels = {worst:.5f}", flush=True)

    # injection sanity: a capital's own unembed direction must move Δp resolvably
    c0, ans0 = 0, ans_ids[0]
    nn = natural_norms(states_by_draw[CTX_DRAWS[0]])
    u = _unembed_direction(model, tok, CAPITAL_ROSTER[c0][1]).float()
    u_deltas = {l: u * nn[l] for l in band}
    inj_sanity_dp = float(np.mean([float(injected_final_probs(model, p, u_deltas)[ans0]) - b
                                   for p, b in zip(carriers, base_p[c0])]))
    ctx.save_raw_completions("I2_potency", pot_rows)
    ctx.save_raw_completions("I2_cosine", [
        {"rung": t, "scrambled_cos": round(scr_cos[t], 4), "genuine_cos": round(gen_cos[t], 4)}
        for t in RUNGS_EXT])

    band_ok = max(POT_BASE, 1.0 / N_EVAL)
    cos_null = max_scr_cos < COS_BAR
    pot_null = worst_abs_dp <= band_ok
    inj_sanity = inj_sanity_dp > band_ok
    print(f"[I2] injection sanity: unembed Δp={inj_sanity_dp:.4f} vs band {band_ok:.4f} "
          f"-> {'ok' if inj_sanity else 'FAIL'}", flush=True)
    return {"instrument": "I2-concept-readout",
            "scrambled_cosine_by_rung": {t: round(scr_cos[t], 4) for t in RUNGS_EXT},
            "genuine_cosine_by_rung": {t: round(gen_cos[t], 4) for t in RUNGS_EXT},
            "max_scrambled_cosine": round(max_scr_cos, 4), "cos_bar": COS_BAR,
            "worst_abs_sham_ctrl_dp": round(worst_abs_dp, 5), "potency_band": round(band_ok, 5),
            "worst_dp_by_alpha": alpha_report,
            "convergence_null_reported": bool(cos_null), "potency_null_reported": bool(pot_null),
            "sanity_discrimination": bool(discrim_ok),
            "sanity_injection_dp": round(inj_sanity_dp, 5), "sanity_injection_fires": bool(inj_sanity),
            "passed": bool(cos_null and pot_null and discrim_ok and inj_sanity)}


def main() -> None:
    t0 = time.perf_counter()
    cfg = Config.load(str(NULL_CFG))
    ctx = start_run(repo_root=REPO_ROOT, config_path=NULL_CFG,
                    results_root=REPO_ROOT / cfg.results_dir, run_name="m5-1c-nullcheck",
                    prereg_path=PREREG)
    print(f"null-check run dir: {ctx.results_dir}", flush=True)
    set_seed(cfg.seed)
    model, tok, revision = load_model(cfg)
    print(f"[model] {cfg.model.name}@{revision[:7]}, {model.n_layers} layers", flush=True)

    i1 = i1_null_check(ctx, model, tok, revision)
    i2 = i2_null_check(ctx, model, tok)
    overall = bool(i1["passed"] and i2["passed"])
    summary = {"overall_null_check_passed": overall, "I1": i1, "I2": i2,
               "peak_rss_gb": round(peak_rss_gb(), 2),
               "wall_clock_s": round(time.perf_counter() - t0, 1)}
    (ctx.results_dir / "null_check.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# EXP-M5-1c null-check report (410M)", "",
        f"**overall: {'PASS' if overall else 'FAIL'}** (both instruments must report null on null)", "",
        "## I1 — amended-Q5 max-contrast (scrambled-label latent probe)",
        f"- genuine ratio_jlens median {i1['genuine_ratio_median']} (sanity ≥ {ADV}: "
        f"{'ok' if i1['sanity_pipeline_fires'] else 'FAIL'})",
        f"- scrambled ratio_jlens median {i1['scrambled_ratio_median']} (null < {ADV}: "
        f"{'PASS' if i1['null_reported'] else 'FAIL — manufactures advantage'})",
        f"- **I1: {'PASS' if i1['passed'] else 'FAIL'}**", "",
        "## I2 — D-033 extended-ladder concept readout (scrambled labels)",
        f"- max scrambled cosine over rungs {i2['max_scrambled_cosine']} (null < {COS_BAR}: "
        f"{'PASS' if i2['convergence_null_reported'] else 'FAIL — false convergence'})",
        f"- worst |sham-ctrl Δp| {i2['worst_abs_sham_ctrl_dp']} over alphas {i2['worst_dp_by_alpha']} "
        f"(null ≤ {i2['potency_band']}: {'PASS' if i2['potency_null_reported'] else 'FAIL — manufactures steering'})",
        f"- sanity: genuine cosine@256 {i2['genuine_cosine_by_rung'][256]} vs scrambled "
        f"{i2['scrambled_cosine_by_rung'][256]} (discrimination: {'ok' if i2['sanity_discrimination'] else 'FAIL'}); "
        f"unembed-injection Δp {i2['sanity_injection_dp']} ({'ok' if i2['sanity_injection_fires'] else 'FAIL'})",
        f"- **I2: {'PASS' if i2['passed'] else 'FAIL'}**", "",
        f"Per-instrument withdrawal on failure (CONSTRAINTS LAW). wall "
        f"{summary['wall_clock_s']} s, peak {summary['peak_rss_gb']} GB. raw under raw_completions/.",
    ]
    (ctx.results_dir / "report.md").write_text("\n".join(lines))
    print(f"\n=== null-check: {'PASS' if overall else 'FAIL'} | I1 {'PASS' if i1['passed'] else 'FAIL'} | "
          f"I2 {'PASS' if i2['passed'] else 'FAIL'} ===", flush=True)
    ctx.finalize(overall_null_check_passed=overall, i1_passed=i1["passed"], i2_passed=i2["passed"],
                 model_revision=revision, wall_clock_s=summary["wall_clock_s"],
                 peak_rss_gb=summary["peak_rss_gb"])


if __name__ == "__main__":
    main()
