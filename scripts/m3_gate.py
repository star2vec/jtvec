"""M3 orchestrator: intervention-instrument gate on Pythia-410M.

Scope (D-011): gate the intervention instruments the M4 experiments consume
— FV-direction ablation, J-space ablation, forced-choice report probe, FV
swap — each with an in-run positive AND negative control on certified FVs
(M2), after re-materializing the lens on this machine and verifying it
against M1's committed manifest (identity: calibration hashes et al.;
function: capital-recall band-min HMR spot-check). FV injection was gated at
M2. E1-E4 stay in M4.

Controls are deterministic given the preregistered seeds (context sampling
rng 9090, sham seeds cfg.seed): no stochastic estimator is drawn here — the
estimators involved (lens, FVs) carry their own M1/M2 gates. Per-item raw
records are retained for every reported number. Bounds are
quantization-aware (D-010): sham/deviation bounds are max(base, 1/N).

Prereg: harness/preregs/EXP-M3-intervention-instruments.md (committed before
first run; start_run enforces).

Usage: uv run python scripts/m3_gate.py [--config configs/m3_pythia410m.yaml]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from jvec.calibration import select_prompts
from jvec.config import Config
from jvec.evals.exp3 import (
    REPORT_LABELS,
    REPORT_PROBES,
    final_logits_under,
    label_token_ids,
    make_hooks,
)
from jvec.evals.fvswap import final_logits, make_swap_hooks
from jvec.evals.probe import probe_task
from jvec.evals.swap import pinv_jacobians
from jvec.evals.tasks import load_tasks
from jvec.fv import FV_REPO, load_fv_model
from jvec.lens_cache import load_lens
from jvec.utils import peak_rss_gb, set_seed
from jtvec.core.instruments import ControlRecord, Instrument, require_controlled
from jtvec.core.reporting import scoped
from jtvec.core.runctx import start_run
from jtvec.m3_instruments import (
    AblationControlRule,
    ReportProbeControlRule,
    SwapControlRule,
    answer_first_tokens,
    execution_answer,
    explicit_rule_context,
    load_certified_fv,
    random_word_null_context,
    shared_query_map,
    verify_lens_manifest,
)

M1_RUN = REPO_ROOT / "results/m1/20260718-010559-lens-gate"
M2_RUN = REPO_ROOT / "results/m2/20260718-114950-fv-stability-gate"
PREREG = REPO_ROOT / "harness/preregs/EXP-M3-intervention-instruments.md"

CTX_RNG_SEED = 9090  # all context sampling (preregistered)
N_EXEC = 30
N_REPORT_CTRL = 12
N_SWAP = 30
M_TOP = 10
SWAP_PAIR = ("capitalize", "singular-plural")  # D-011: both certified
ABLATION_TASKS = ("capitalize", "singular-plural")
# D-012: jspace anchored on capital-recall (36 items, the M1-VERIFIED probing
# task) with exact-match scoring, replacing swap-capitals (16 items, coarse).
JSPACE_TASK = "capital-recall"

# Tolerances live in the prereg (EXP-M3, Decision rule); mirrored here.
LENS_HMR_FACTOR = 1.5  # refit band-min jlens HMR <= factor x committed M1 value
LENS_LOGIT_SEP = 5.0  # logit HMR at that layer >= sep x refit jlens HMR
ABLATION_RULE = AblationControlRule(min_exec_drop=0.15, max_sham_dev=0.05)
REPORT_RULE = ReportProbeControlRule(min_detection=0.8, max_null_above_prior=0.15)
SWAP_RULE = SwapControlRule(min_b_gain=0.2, max_random_elevation=0.05)


def bandmin_hmr(per_item: list[dict], arm: str, band: range) -> tuple[int, float]:
    """(layer, HMR) minimizing the harmonic-mean rank over the band."""

    def hmr(layer: int) -> float:
        ranks = [item["ranks"][arm][layer] if layer in item["ranks"][arm]
                 else item["ranks"][arm][str(layer)] for item in per_item]
        return len(ranks) / sum(1.0 / r for r in ranks)

    layer = min(band, key=hmr)
    return layer, hmr(layer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/m3_pythia410m.yaml"))
    args = parser.parse_args()
    t_start = time.perf_counter()

    cfg = Config.load(args.config)
    ctx = start_run(
        repo_root=REPO_ROOT,
        config_path=Path(args.config),
        results_root=REPO_ROOT / cfg.results_dir,
        run_name="instrument-gate",
        prereg_path=PREREG,
    )
    print(f"M3 run dir: {ctx.results_dir}", flush=True)

    set_seed(cfg.seed)
    hf_model, tokenizer, model_config, revision = load_fv_model(cfg)
    import jlens as jlens_pkg  # noqa: PLC0415

    model_j = jlens_pkg.from_hf(hf_model, tokenizer)
    W_U = hf_model.embed_out.weight.detach().float().cpu()
    model_scope = f"{cfg.model.name}@{revision[:7]}"
    config_scope = f"m3 instrument gate ({Path(args.config).name})"
    bos = tokenizer.bos_token or ""
    rng = np.random.default_rng(CTX_RNG_SEED)

    from utils.prompt_utils import load_dataset  # noqa: PLC0415

    # --- Stage 1: lens re-materialization, verified against M1 ----------------
    prompts = select_prompts(cfg, tokenizer)
    skip = cfg.fit.skip_first_variants[0]
    try:
        lens = load_lens(cfg, skip, prompts, revision)
        print("[lens] cache hit", flush=True)
    except FileNotFoundError:
        print("[lens] fitting on this machine via vendored 01_fit_lens.py", flush=True)
        subprocess.run(
            [sys.executable, "scripts/01_fit_lens.py", "--config", args.config],
            cwd=REPO_ROOT, check=True,
        )
        lens = load_lens(cfg, skip, prompts, revision)

    refit_manifest = json.loads(
        (REPO_ROOT / cfg.cache_dir / "lenses" / f"{cfg.model.name}@{revision}"
         / f"skip{skip}_n{cfg.calibration.n_prompts}" / "manifest.json"
         ).read_text(encoding="utf-8")
    )
    m1_manifest = json.loads(
        (M1_RUN / "draws/draw0/manifest.json").read_text(encoding="utf-8")
    )
    mismatches = verify_lens_manifest(refit_manifest, m1_manifest)
    if mismatches:
        sys.exit(f"lens identity mismatch vs committed M1 draw0 manifest: {mismatches}")
    print("[lens] identity matches M1 draw 0 (incl. calibration sha256)", flush=True)

    lo, hi = cfg.evals.band
    band = range(lo, hi + 1)
    band_layers = [l for l in lens.source_layers if lo <= l <= hi]

    cr_task = next(t for t in load_tasks() if t.name == "capital-recall")
    spot = probe_task(model_j, tokenizer, lens, cr_task, pass_k=cfg.evals.pass_k,
                      n_random_seeds=0)
    layer, refit_hmr = bandmin_hmr(spot["per_item"], "jlens", band)
    _, refit_logit_hmr = bandmin_hmr(spot["per_item"], "logit", range(layer, layer + 1))
    m1_probe = json.loads((M1_RUN / "draws/draw0/probe.json").read_text(encoding="utf-8"))
    m1_layer, m1_hmr = bandmin_hmr(m1_probe["skip4"]["capital-recall"]["per_item"], "jlens", band)
    lens_ok = (refit_hmr <= LENS_HMR_FACTOR * m1_hmr
               and refit_logit_hmr >= LENS_LOGIT_SEP * refit_hmr)
    ctx.save_raw_completions(
        "lens-spotcheck_capital-recall",
        [{"name": it["name"],
          "bandmin_rank_jlens": min(it["ranks"]["jlens"][l] if l in it["ranks"]["jlens"]
                                    else it["ranks"]["jlens"][str(l)] for l in band)}
         for it in spot["per_item"]],
    )
    print(f"[lens] spot-check: refit band-min HMR {refit_hmr:.2f} (L{layer}) vs M1 "
          f"{m1_hmr:.2f} (L{m1_layer}); logit at L{layer} {refit_logit_hmr:.1f} -> "
          f"{'PASS' if lens_ok else 'FAIL'}", flush=True)
    if not lens_ok:
        sys.exit("lens functional spot-check failed; instruments not gated")

    # --- Stage 2: certified FVs -------------------------------------------------
    certificates = json.loads((M2_RUN / "certificates.json").read_text(encoding="utf-8"))
    fvs = {t: load_certified_fv(cfg, t, revision, certificates) for t in cfg.fv.tasks}
    print(f"[fvs] certified artifacts loaded: {sorted(fvs)} (M2 evidence run ok)", flush=True)

    datasets = {t: load_dataset(t, root_data_dir=str(FV_REPO / "dataset_files"), seed=cfg.seed)
                for t in cfg.fv.tasks}
    labels = label_token_ids(tokenizer)
    candidate_ids = list(labels.values())
    if len(set(candidate_ids)) != len(candidate_ids):
        sys.exit("report-probe label first-tokens are not distinct; probe unusable")
    prior = 1.0 / len(candidate_ids)

    def sample_pairs(task: str, split: str, n: int):
        ds = datasets[task]
        idx = rng.choice(len(ds[split]), n, replace=False)
        chosen = ds[split][idx]
        return list(zip(chosen["input"], chosen["output"]))

    def build_context(pairs) -> str:
        return bos + "".join(f"Q: {x}\nA: {y}\n\n" for x, y in pairs)

    def hit_exact(top1: int, answer: str) -> bool:
        # D-012: exact-match (case-sensitive) scoring uniformly across the
        # execution controls; surface_token_ids retained only where the M1
        # probing protocol requires it (lens spot-check).
        return top1 in answer_first_tokens(tokenizer, str(answer), case_sensitive=True)

    def exec_accuracy(task: str, hooks, cell: str) -> float:
        hits, rows = 0, []
        for _ in range(N_EXEC):
            pairs = sample_pairs(task, "train", cfg.fv.n_shots)
            qx, qy = sample_pairs(task, "test", 1)[0]
            logits = final_logits_under(model_j, build_context(pairs) + f"Q: {qx}\nA:", hooks)
            top1 = int(logits.argmax())
            hit = hit_exact(top1, qy)
            hits += hit
            rows.append({"query": str(qx), "target": str(qy),
                         "top1": tokenizer.decode([top1]), "hit": bool(hit)})
        ctx.save_raw_completions(cell, rows)
        return hits / N_EXEC

    def report_correct(logits: torch.Tensor, task: str) -> bool:
        cand = torch.tensor([logits[i] for i in candidate_ids])
        return candidate_ids[int(cand.argmax())] == labels[task]

    # --- Stage 3a: FV-direction ablation controls --------------------------------
    fv_ablation = {}
    for task in ABLATION_TASKS:
        accs = {}
        for cond in ("none", "fv", "sham_fv"):
            hooks = make_hooks(cond, band_layers, lens, W_U, fvs[task].vector,
                               m_top=M_TOP, seed=cfg.seed)
            accs[cond] = exec_accuracy(task, hooks, f"fvablate_{task}_{cond}")
        fv_ablation[task] = ABLATION_RULE.verdict(
            none_acc=accs["none"], ablated_acc=accs["fv"], sham_acc=accs["sham_fv"],
            n=N_EXEC,
        )
        print(f"[fv-ablation] {task}: {fv_ablation[task]}", flush=True)

    # --- Stage 3b: J-space ablation controls (M1-anchored lens task) -------------
    sc_task = next(t for t in load_tasks() if t.name == JSPACE_TASK)
    jspace_accs = {}
    for cond in ("none", "jspace", "sham_jspace"):
        hooks = make_hooks(cond, band_layers, lens, W_U, fvs["capitalize"].vector,
                           m_top=M_TOP, seed=cfg.seed)
        hits, rows = 0, []
        for item in sc_task.items:
            logits = final_logits_under(model_j, item["prompt"], hooks)
            top1 = int(logits.argmax())
            answer = execution_answer(item)
            hit = hit_exact(top1, answer)
            hits += hit
            rows.append({"name": item["name"], "target": answer,
                         "top1": tokenizer.decode([top1]), "hit": bool(hit)})
        ctx.save_raw_completions(f"jspace_{JSPACE_TASK}_{cond}", rows)
        jspace_accs[cond] = hits / len(sc_task.items)
    jspace = ABLATION_RULE.verdict(
        none_acc=jspace_accs["none"], ablated_acc=jspace_accs["jspace"],
        sham_acc=jspace_accs["sham_jspace"], n=len(sc_task.items),
    )
    print(f"[jspace-ablation] {JSPACE_TASK}: {jspace}", flush=True)

    # --- Stage 3c: report-probe controls ------------------------------------------
    # D-012 null pool per task: the OTHER tasks' train outputs (real words, no
    # coherent class), so the negative arm removes the task signal even for
    # morphological-output tasks (which the v1 shuffled baseline did not).
    task_outputs = {t: sorted({str(y) for y in datasets[t]["train"][:]["output"]})
                    for t in cfg.fv.tasks}
    null_pool = {t: sorted(set().union(*(task_outputs[o] for o in cfg.fv.tasks if o != t)))
                 for t in cfg.fv.tasks}
    report_ctrl = {}
    for task in cfg.fv.tasks:
        detection = {}
        for pname, probe in REPORT_PROBES.items():
            det, rows = 0, []
            for _ in range(N_REPORT_CTRL):
                pairs = sample_pairs(task, "train", cfg.fv.n_shots)
                ys = [y for _, y in pairs]
                rng.shuffle(ys)
                shuffled = [(x, y) for (x, _), y in zip(pairs, ys)]
                context = explicit_rule_context(bos, REPORT_LABELS[task], shuffled)
                logits = final_logits_under(model_j, context + probe, {})
                ok = report_correct(logits, task)
                det += ok
                rows.append({"phrasing": pname, "correct": bool(ok)})
            ctx.save_raw_completions(f"report_{task}_{pname}_explicit-rule", rows)
            detection[pname] = det / N_REPORT_CTRL
        # negative arm (D-012): random-word-output context, no coherent task
        nul, rows = 0, []
        for pname, probe in REPORT_PROBES.items():
            for _ in range(N_REPORT_CTRL):
                pairs = sample_pairs(task, "train", cfg.fv.n_shots)
                inputs = [x for x, _ in pairs]
                context = random_word_null_context(bos, inputs, null_pool[task], rng)
                logits = final_logits_under(model_j, context + probe, {})
                ok = report_correct(logits, task)
                nul += ok
                rows.append({"phrasing": pname, "correct": bool(ok)})
        ctx.save_raw_completions(f"report_{task}_null", rows)
        n_null = N_REPORT_CTRL * len(REPORT_PROBES)
        report_ctrl[task] = REPORT_RULE.verdict(
            detection_by_phrasing=detection, null_acc=nul / n_null,
            prior=prior, n=n_null,
        )
        print(f"[report-probe] {task}: {report_ctrl[task]}", flush=True)

    # --- Stage 3d: FV-swap controls -------------------------------------------------
    task_a, task_b = SWAP_PAIR
    shared = shared_query_map(datasets[task_a], datasets[task_b])
    if len(shared) < 10:
        sys.exit(f"swap pair {SWAP_PAIR}: only {len(shared)} shared queries; pair unusable")
    queries = list(rng.choice(sorted(shared), min(N_SWAP, len(shared)), replace=False))
    pinvs = pinv_jacobians(lens, band_layers, rcond=cfg.evals.swap_rcond)
    b_rates = {}
    for cond in ("none", "lens_swap", "direct_swap", "random_target"):
        hooks = make_swap_hooks(cond, band_layers, lens, fvs[task_a].vector,
                                fvs[task_b].vector, pinvs, seed=cfg.seed)
        hits, rows = 0, []
        for q in queries:
            pairs = sample_pairs(task_a, "train", cfg.fv.n_shots)
            logits = final_logits(model_j, build_context(pairs) + f"Q: {q}\nA:", hooks)
            top1 = int(logits.argmax())
            y_a, y_b = shared[q]
            b_hit = hit_exact(top1, y_b)  # D-012: case-sensitive, no A/B collision
            hits += b_hit
            rows.append({"query": q, "target_a": y_a, "target_b": y_b,
                         "top1": tokenizer.decode([top1]), "b_hit": bool(b_hit)})
        ctx.save_raw_completions(f"swap_{task_a}-to-{task_b}_{cond}", rows)
        b_rates[cond] = hits / len(queries)
    swap = SWAP_RULE.verdict(b_rates=b_rates, n=len(queries))
    print(f"[fv-swap] {task_a}->{task_b}: {swap}", flush=True)

    # --- Stage 4: instrument verdicts (instruments LAW) ---------------------------
    today = time.strftime("%Y-%m-%d")
    run = str(ctx.results_dir)
    verdicts = {
        "fv-direction-ablation": {
            "positive": all(v["positive_pass"] for v in fv_ablation.values()),
            "negative": all(v["negative_pass"] for v in fv_ablation.values()),
            "detail": fv_ablation,
        },
        "jspace-ablation": {
            "positive": jspace["positive_pass"],
            "negative": jspace["negative_pass"],
            "detail": jspace,
        },
        "report-probe-forced-choice": {
            "positive": all(v["positive_pass"] for v in report_ctrl.values()),
            "negative": all(v["negative_pass"] for v in report_ctrl.values()),
            "detail": report_ctrl,
        },
        "fv-swap": {
            "positive": swap["positive_pass"],
            "negative": swap["negative_pass"],
            "detail": swap,
        },
    }
    instruments = {
        name: Instrument(
            name=name,
            positive_control=ControlRecord(run=run, passed=v["positive"], date=today),
            negative_control=ControlRecord(run=run, passed=v["negative"], date=today),
        )
        for name, v in verdicts.items()
    }
    gated = {name: inst.is_controlled() for name, inst in instruments.items()}
    for name, inst in instruments.items():
        if gated[name]:
            require_controlled(inst)  # sanity: constructible AND passing
    all_gated = all(gated.values())

    (ctx.results_dir / "controls.json").write_text(
        json.dumps(
            {name: {"positive": v["positive"], "negative": v["negative"],
                    "run": run, "date": today, "detail": v["detail"]}
             for name, v in verdicts.items()},
            indent=2, default=str,
        ),
        encoding="utf-8",
    )

    # --- report -------------------------------------------------------------------
    lines = [
        "# M3 report: intervention-instrument gate",
        "",
        f"- model: {model_scope} (full sha in run.json/config)",
        "- prereg: harness/preregs/EXP-M3-intervention-instruments.md",
        f"- certified FVs: M2 run {M2_RUN.name}, draw 1 (n_trials_aie=200 >= "
        "converged_at=25)",
        f"- lens: re-materialized on this machine; identity equals M1 draw 0's "
        f"committed manifest (incl. calibration sha256); functional spot-check "
        f"band-min jlens HMR {refit_hmr:.2f} (L{layer}) vs M1 {m1_hmr:.2f} "
        f"(bound {LENS_HMR_FACTOR}x), logit {refit_logit_hmr:.1f} "
        f"(>= {LENS_LOGIT_SEP}x separation)",
        f"- context rng {CTX_RNG_SEED}; sham seeds {cfg.seed}; bounds are "
        "quantization-aware, max(base, 1/N) (D-010)",
        "",
        "## Instrument verdicts",
        "",
        "| instrument | positive | negative | gated |",
        "|---|---|---|---|",
    ]
    for name, v in verdicts.items():
        lines.append(
            f"| {name} | {'PASS' if v['positive'] else 'FAIL'} | "
            f"{'PASS' if v['negative'] else 'FAIL'} | "
            f"{'YES' if gated[name] else 'NO'} |"
        )
    lines += ["", "## Control readings (every ablated/swapped number next to its sham)", ""]
    for task, v in fv_ablation.items():
        lines.append("- " + scoped(
            f"fv-ablation {task}: exec none {v['none']:.3f} -> fv {v['ablated']:.3f} "
            f"(sham_fv {v['sham']:.3f}, bound {v['sham_bound']:.3f})",
            float(v["none"] - v["ablated"]), model=model_scope, config=config_scope,
            n=v["n"],
        ))
    lines.append("- " + scoped(
        f"jspace-ablation {JSPACE_TASK}: exec none {jspace['none']:.3f} -> jspace "
        f"{jspace['ablated']:.3f} (sham_jspace {jspace['sham']:.3f}, bound "
        f"{jspace['sham_bound']:.3f})",
        float(jspace["none"] - jspace["ablated"]), model=model_scope,
        config=config_scope, n=jspace["n"],
    ))
    for task, v in report_ctrl.items():
        det = ", ".join(f"{p} {a:.2f}" for p, a in v["detection_by_phrasing"].items())
        lines.append("- " + scoped(
            f"report-probe {task}: explicit-rule detection [{det}], random-word "
            f"null {v['null_acc']:.3f} vs prior {v['prior']:.3f} "
            f"(margin {v['null_margin']:.3f})",
            float(v["best_detection"]), model=model_scope, config=config_scope,
            n=v["n"],
        ))
    lines.append("- " + scoped(
        f"fv-swap {task_a}->{task_b}: b-rate none {swap['b_rates']['none']:.3f}, "
        f"direct {swap['b_rates']['direct_swap']:.3f}, lens "
        f"{swap['b_rates']['lens_swap']:.3f} (random_target "
        f"{swap['b_rates']['random_target']:.3f}, bound {swap['random_bound']:.3f})",
        float(swap["best_gain"]), model=model_scope, config=config_scope, n=swap["n"],
    ))
    wall_s = round(time.perf_counter() - t_start, 1)
    lines += [
        "",
        f"**M3 verdict: {'all instruments gated' if all_gated else 'NOT all instruments gated'}**",
        "",
        f"wall-clock {wall_s} s; peak RSS {peak_rss_gb():.2f} GB; device {cfg.device}; "
        "ControlRecord pairs in controls.json; raw per-item cells under "
        "raw_completions/.",
        "",
    ]
    (ctx.results_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    ctx.finalize(
        lens_spotcheck={"refit_hmr": refit_hmr, "layer": layer,
                        "m1_hmr": m1_hmr, "logit_hmr": refit_logit_hmr},
        instruments={n: {"positive": v["positive"], "negative": v["negative"]}
                     for n, v in verdicts.items()},
        m3_all_gated=all_gated,
        model_revision=revision,
        wall_clock_s=wall_s,
        peak_rss_gb=round(peak_rss_gb(), 2),
    )
    print(f"\nM3 verdict: {'all instruments gated' if all_gated else 'NOT all gated'}")
    for name, ok in gated.items():
        print(f"  {name}: {'GATED' if ok else 'not gated'}")
    print(f"report: {ctx.results_dir / 'report.md'}")


if __name__ == "__main__":
    main()
