"""M4-E1 orchestrator: FV label decodability (J-lens vs logit lens) on
Pythia-410M, per EXP-M4-E1 (constants ratified D-014).

Reads the three M2-certified Todd FVs (3 tasks x 3 FV draws, full-trial
tensors) through five lens instances — the three M1 lens draws (primary
variant skip4_n10, identity-checked against M1's committed manifests) and
the two v1 robustness variants (skip16_n10, skip4_n5; in-run controls only)
— and evaluates the preregistered per-task criteria C1-C4. Read-only: no
model computation is modified; no interventions, hence no shams (the
norm-matched random-vector arm is the reading-level null).

Prereg: harness/preregs/EXP-M4-E1-decodability.md (committed before first
run; start_run enforces).

Usage: uv run python scripts/m4_e1_gate.py [--config configs/m4_e1_pythia410m.yaml]
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

import torch

from jvec.calibration import select_prompts
from jvec.config import Config
from jvec.evals.fvprobe import output_token_ids, random_like
from jvec.evals.swap import pinv_jacobians
from jvec.fv import FV_REPO  # noqa: F401  (sys.path side-effect for the Todd repo)
from jvec.lens_cache import ManifestMismatch, fit_lens, lens_dir, load_lens
from jvec.modeling import load_model
from jvec.utils import peak_rss_gb, set_seed
from jtvec.core.draws import DrawSet
from jtvec.core.instruments import ControlRecord, Instrument, require_controlled
from jtvec.core.reporting import scoped
from jtvec.core.runctx import start_run
from jtvec.e1_decodability import (
    E1_TASKS,
    LABEL_SETS,
    RANDOM_SEEDS,
    E1DecisionRule,
    ReadoutNegativeRule,
    ReadoutPositiveRule,
    canonical_label_token,
    full_vocab_rank,
    label_rank,
    layer_set_stability,
)
from jtvec.m3_instruments import load_certified_fv, verify_lens_manifest

M1_RUN = REPO_ROOT / "results/m1/20260718-010559-lens-gate"
M2_RUN = REPO_ROOT / "results/m2/20260718-114950-fv-stability-gate"
PREREG = REPO_ROOT / "harness/preregs/EXP-M4-E1-decodability.md"

FV_DRAWS = (1, 2, 3)  # M2 cache draws
LENS_DRAWS = (0, 1, 2)  # M1 seeds
VARIANTS = ("skip16_n10", "skip4_n5")  # v1's registered robustness variants
POS_RULE = ReadoutPositiveRule()  # rank <= 10 at >= ceil(0.75*n) layers (D-014)
NEG_RULE = ReadoutNegativeRule()  # random median rank >= 100
RULE = E1DecisionRule()  # C1 <= 20, C3 >= 200, C4 >= 95/100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/m4_e1_pythia410m.yaml"))
    args = parser.parse_args()
    t_start = time.perf_counter()

    cfg = Config.load(args.config)
    ctx = start_run(
        repo_root=REPO_ROOT,
        config_path=Path(args.config),
        results_root=REPO_ROOT / cfg.results_dir,
        run_name="e1-decodability",
        prereg_path=PREREG,
    )
    print(f"E1 run dir: {ctx.results_dir}", flush=True)

    set_seed(cfg.seed)
    model_j, tokenizer, revision = load_model(cfg)
    device = model_j.input_device
    W_U = model_j._lm_head.weight.detach().float().cpu()
    model_scope = f"{cfg.model.name}@{revision[:7]}"
    config_scope = f"EXP-M4-E1 ({Path(args.config).name})"
    lo, hi = cfg.evals.band

    # --- Stage 1: lens instances -------------------------------------------------
    # Primary variant, three M1 draws. The lens cache key excludes the seed,
    # so each draw gets its own cache_dir (M1 precedent). Draw 0 reuses the
    # M3-verified fit read-only when present.
    instances: dict[str, dict] = {}

    def register(name: str, lens, m1_manifest_path: Path | None, manifest_path: Path):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if m1_manifest_path is not None:
            reference = json.loads(m1_manifest_path.read_text(encoding="utf-8"))
            mismatches = verify_lens_manifest(manifest, reference)
            if mismatches:
                sys.exit(f"lens identity mismatch ({name}) vs committed M1 manifest: "
                         f"{mismatches}")
            print(f"[lens {name}] identity matches M1 (incl. calibration sha256)",
                  flush=True)
        band_layers = [l for l in lens.source_layers if lo <= l <= hi]
        if not band_layers:
            sys.exit(f"lens instance {name} has no band overlap; unusable")
        instances[name] = {
            "lens": lens,
            "band_layers": band_layers,
            "pinvs": pinv_jacobians(lens, band_layers, rcond=cfg.evals.swap_rcond),
        }

    for j in LENS_DRAWS:
        dcfg = dataclasses.replace(
            cfg, seed=j, cache_dir=f"cache/m4e1/lensdraw{j}"
        )
        set_seed(dcfg.seed)
        prompts = select_prompts(dcfg, tokenizer)
        lens = None
        if j == 0:
            m3cfg = dataclasses.replace(cfg, cache_dir="cache/m3")
            try:
                lens = load_lens(m3cfg, 4, prompts, revision)
                mpath = lens_dir(m3cfg, 4) / "manifest.json"
                print("[lens draw0] cache hit (M3 fit, read-only)", flush=True)
            except (FileNotFoundError, ManifestMismatch):
                lens = None
        if lens is None:
            t0 = time.perf_counter()
            try:
                lens = load_lens(dcfg, 4, prompts, revision)
                print(f"[lens draw{j}] cache hit", flush=True)
            except FileNotFoundError:
                print(f"[lens draw{j}] fitting on this machine ...", flush=True)
                lens = fit_lens(dcfg, 4, prompts, model_j, revision)
                print(f"[lens draw{j}] fitted in "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
            mpath = lens_dir(dcfg, 4) / "manifest.json"
        register(f"draw{j}", lens, M1_RUN / f"draws/draw{j}/manifest.json", mpath)

    set_seed(cfg.seed)
    prompts0 = select_prompts(cfg, tokenizer)
    v16cfg = dataclasses.replace(
        cfg, cache_dir="cache/m4e1/lensdraw0",
        fit=dataclasses.replace(cfg.fit, skip_first_variants=(16,)),
    )
    try:
        lens16 = load_lens(v16cfg, 16, prompts0, revision)
        print("[lens skip16_n10] cache hit", flush=True)
    except FileNotFoundError:
        print("[lens skip16_n10] fitting ...", flush=True)
        lens16 = fit_lens(v16cfg, 16, prompts0, model_j, revision)
    register("skip16_n10", lens16, None, lens_dir(v16cfg, 16) / "manifest.json")

    v45cfg = dataclasses.replace(
        cfg, cache_dir="cache/m4e1/lensdraw0",
        calibration=dataclasses.replace(cfg.calibration, n_prompts=5),
    )
    prompts45 = select_prompts(v45cfg, tokenizer)
    try:
        lens45 = load_lens(v45cfg, 4, prompts45, revision)
        print("[lens skip4_n5] cache hit", flush=True)
    except FileNotFoundError:
        print("[lens skip4_n5] fitting ...", flush=True)
        lens45 = fit_lens(v45cfg, 4, prompts45, model_j, revision)
    register("skip4_n5", lens45, None, lens_dir(v45cfg, 4) / "manifest.json")

    # --- Stage 2: certified FVs ---------------------------------------------------
    certificates = json.loads((M2_RUN / "certificates.json").read_text(encoding="utf-8"))
    fvs = {t: {k: load_certified_fv(cfg, t, revision, certificates, draw_k=k)
               for k in FV_DRAWS} for t in E1_TASKS}
    print(f"[fvs] certified artifacts loaded: {sorted(fvs)} x draws {FV_DRAWS}",
          flush=True)

    @torch.no_grad()
    def readout(vector: torch.Tensor, name: str, layers) -> dict[int, torch.Tensor]:
        lens = instances[name]["lens"]
        v = vector.float().to(device)
        return {l: model_j.unembed(lens.transport(v, l)).float().cpu() for l in layers}

    @torch.no_grad()
    def readout_logit(vector: torch.Tensor) -> dict[int, torch.Tensor]:
        v = vector.float().to(device)
        return {-1: model_j.unembed(v).float().cpu()}

    # --- Stage 3: in-run instrument controls (before any FV is read) --------------
    pos: dict[str, dict[str, dict]] = {n: {} for n in instances}
    for name, inst in instances.items():
        for task in E1_TASKS:
            t_star = canonical_label_token(tokenizer, LABEL_SETS["set2"][task][0])
            per_layer_rank = {}
            rows = []
            for l in inst["band_layers"]:
                v_plus = inst["pinvs"][l] @ W_U[t_star]
                logits = readout(v_plus, name, [l])[l]
                r = full_vocab_rank(logits, t_star)
                per_layer_rank[l] = r
                rows.append({"instance": name, "layer": int(l), "rank": int(r),
                             "label_token": tokenizer.decode([t_star])})
            pos[name][task] = POS_RULE.evaluate(per_layer_rank)
            ctx.save_raw_completions(f"poscontrol_{task}", rows)
        print(f"[poscontrol {name}] " + ", ".join(
            f"{t}: {len(pos[name][t]['included_layers'])}/{pos[name][t]['n_layers']}"
            f"{'' if pos[name][t]['passed'] else ' FAIL'}"
            for t in E1_TASKS), flush=True)

    stability = {
        task: layer_set_stability(
            {j: pos[f"draw{j}"][task]["included_layers"] for j in LENS_DRAWS}
        )
        for task in E1_TASKS
    }

    def included(name: str, task: str) -> list[int]:
        layers = pos[name][task]["included_layers"]
        return layers if layers else instances[name]["band_layers"]

    neg: dict[str, dict[str, dict]] = {n: {} for n in instances}
    random_rank_draw0: dict[str, dict[int, list[int]]] = {}  # task -> norm draw -> ranks
    for task in E1_TASKS:
        random_rank_draw0[task] = {}
        for name in instances:
            ranks, rows = [], []
            for seed in RANDOM_SEEDS:
                rv = random_like(fvs[task][1].vector, seed)
                r = label_rank(readout(rv, name, included(name, task)),
                               tokenizer, LABEL_SETS["set1"][task])
                ranks.append(r)
                rows.append({"instance": name, "seed": seed, "fv_draw_norm": 1,
                             "rank": int(r)})
            neg[name][task] = NEG_RULE.evaluate(ranks)
            ctx.save_raw_completions(f"random_{task}", rows)
            if name == "draw0":
                random_rank_draw0[task][1] = ranks
        # C4 companion readings at the other two draw norms (rank-invariant to
        # the rescaling by construction; recorded for the preregistered cell).
        for k in FV_DRAWS[1:]:
            ranks, rows = [], []
            for seed in RANDOM_SEEDS:
                rv = random_like(fvs[task][k].vector, seed)
                r = label_rank(readout(rv, "draw0", included("draw0", task)),
                               tokenizer, LABEL_SETS["set1"][task])
                ranks.append(r)
                rows.append({"instance": "draw0", "seed": seed, "fv_draw_norm": k,
                             "rank": int(r)})
            random_rank_draw0[task][k] = ranks
            ctx.save_raw_completions(f"random_{task}", rows)
        print(f"[negcontrol {task}] " + ", ".join(
            f"{n}: med {neg[n][task]['median_rank']:.0f}"
            f"{'' if neg[n][task]['passed'] else ' FAIL'}" for n in instances),
            flush=True)

    def task_controls_ok(task: str) -> bool:
        primary_ok = all(pos[f"draw{j}"][task]["passed"] for j in LENS_DRAWS)
        neg_ok = all(neg[f"draw{j}"][task]["passed"] for j in LENS_DRAWS)
        return primary_ok and neg_ok and stability[task]["passed"]

    # --- Stage 4: FV readings ------------------------------------------------------
    from utils.prompt_utils import load_dataset  # noqa: PLC0415

    results: dict[str, dict] = {}
    for task in E1_TASKS:
        set1 = LABEL_SETS["set1"][task]
        logit_readouts = {i: readout_logit(fvs[task][i].vector) for i in FV_DRAWS}
        r_logit = {(i, s): label_rank(logit_readouts[i], tokenizer, LABEL_SETS[s][task])
                   for i in FV_DRAWS for s in LABEL_SETS}
        rows = []
        for i in FV_DRAWS:
            per_layer = {"-1": label_rank(logit_readouts[i], tokenizer, set1)}
            rows.append({"fv_draw": i, "instance": "logit", "set": "set1",
                         "arm": "logit", "rank": int(r_logit[(i, "set1")]),
                         "per_layer_rank": per_layer,
                         "top10": [tokenizer.decode([t]) for t in
                                   logit_readouts[i][-1].topk(10).indices]})

        primary_jlens: dict[tuple[int, int], int] = {}
        ordering: dict[str, tuple[int, int]] = {}
        not_evaluable: list[str] = []
        for i in FV_DRAWS:
            v = fvs[task][i].vector
            for j in LENS_DRAWS:
                name = f"draw{j}"
                outs = readout(v, name, included(name, task))
                r = label_rank(outs, tokenizer, set1)
                primary_jlens[(i, j)] = r
                ordering[f"fv{i}_{name}_skip4_n10_set1"] = (r, r_logit[(i, "set1")])
                best_l = min(outs, key=lambda l: label_rank({l: outs[l]}, tokenizer, set1))
                rows.append({"fv_draw": i, "instance": name, "set": "set1",
                             "arm": "jlens", "rank": int(r),
                             "per_layer_rank": {str(l): label_rank({l: outs[l]}, tokenizer, set1)
                                                for l in outs},
                             "top10": [tokenizer.decode([t]) for t in
                                       outs[best_l].topk(10).indices]})
            # robustness cells at lens draw 0: all variants x all sets
            for vname in ("draw0",) + VARIANTS:
                variant_label = "skip4_n10" if vname == "draw0" else vname
                variant_ok = pos[vname][task]["passed"] and neg[vname][task]["passed"]
                for s in LABEL_SETS:
                    cell = f"fv{i}_draw0_{variant_label}_{s}"
                    if vname == "draw0" and s == "set1":
                        continue  # identical to the primary cell above
                    if not variant_ok:
                        not_evaluable.append(cell)
                        continue
                    outs = readout(fvs[task][i].vector, vname, included(vname, task))
                    r = label_rank(outs, tokenizer, LABEL_SETS[s][task])
                    ordering[cell] = (r, r_logit[(i, s)])
                    rows.append({"fv_draw": i, "instance": vname, "set": s,
                                 "arm": "jlens", "rank": int(r),
                                 "per_layer_rank": {str(l): label_rank({l: outs[l]}, tokenizer,
                                                                       LABEL_SETS[s][task])
                                                    for l in outs},
                                 "top10": []})
        ctx.save_raw_completions(f"decode_{task}", rows)

        random_beaten = {
            i: sum(primary_jlens[(i, 0)] < r for r in random_rank_draw0[task][i])
            for i in FV_DRAWS
        }

        if not task_controls_ok(task):
            results[task] = {
                "verdict": "INSTRUMENT-VOID",
                "primary_jlens": {f"{i},{j}": int(r) for (i, j), r in primary_jlens.items()},
                "random_beaten": random_beaten,
            }
            print(f"[verdict {task}] INSTRUMENT-VOID (controls)", flush=True)
            continue

        verdict = RULE.verdict(
            primary_jlens=primary_jlens,
            logit_by_fv_draw={i: r_logit[(i, "set1")] for i in FV_DRAWS},
            ordering_cells=ordering,
            random_beaten_by_fv_draw=random_beaten,
            not_evaluable_cells=tuple(not_evaluable),
        )
        results[task] = {
            **verdict,
            "primary_jlens": {f"{i},{j}": int(r) for (i, j), r in primary_jlens.items()},
            "logit_by_fv_draw": {i: int(r_logit[(i, "set1")]) for i in FV_DRAWS},
            "random_beaten": random_beaten,
        }
        print(f"[verdict {task}] {verdict['verdict']} "
              f"(jlens med {verdict['jlens_median']:.0f}, logit med "
              f"{verdict['logit_median']:.0f}, beaten {random_beaten})", flush=True)

        # descriptive arm: output-vocabulary mean rank (outside the decision rule)
        dataset = load_dataset(task, root_data_dir=str(FV_REPO / "dataset_files"),
                               seed=cfg.seed)
        out_ids = output_token_ids(dataset, tokenizer)
        ov_rows = []
        for i in FV_DRAWS:
            outs = readout(fvs[task][i].vector, "draw0", included("draw0", task))
            per_layer = {str(l): float(sum(full_vocab_rank(outs[l], t) for t in out_ids)
                                       / len(out_ids)) for l in outs}
            ov_rows.append({"fv_draw": i, "arm": "jlens_draw0",
                            "per_layer_mean_rank": per_layer,
                            "best": min(per_layer.values())})
            lg = logit_readouts[i][-1]
            ov_rows.append({"fv_draw": i, "arm": "logit",
                            "mean_rank": float(sum(full_vocab_rank(lg, t) for t in out_ids)
                                               / len(out_ids))})
        ctx.save_raw_completions(f"outputvocab_{task}", ov_rows)

    # --- Stage 5: instrument records + report ---------------------------------------
    today = time.strftime("%Y-%m-%d")
    run = str(ctx.results_dir)
    controls = {}
    for task in E1_TASKS:
        ok = task_controls_ok(task)
        inst = Instrument(
            name=f"jlens-label-rank-readout@{task}",
            positive_control=ControlRecord(
                run=run, date=today,
                passed=all(pos[n][task]["passed"] for n in instances)
                and stability[task]["passed"],
            ),
            negative_control=ControlRecord(
                run=run, date=today,
                passed=all(neg[n][task]["passed"] for n in instances),
            ),
        )
        if ok:
            require_controlled(
                Instrument(
                    name=inst.name,
                    positive_control=ControlRecord(run=run, date=today, passed=True),
                    negative_control=ControlRecord(run=run, date=today, passed=True),
                )
            )
        controls[inst.name] = {
            "positive": inst.positive_control.passed,
            "negative": inst.negative_control.passed,
            "primary_controls_ok": ok,
            "run": run,
            "date": today,
            "detail": {
                "positive": {n: pos[n][task] for n in instances},
                "negative": {n: neg[n][task] for n in instances},
                "layer_set_stability": stability[task],
            },
        }
    (ctx.results_dir / "controls.json").write_text(
        json.dumps(controls, indent=2, default=str), encoding="utf-8"
    )
    (ctx.results_dir / "e1_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "# EXP-M4-E1 report: FV label decodability (J-lens vs logit lens)",
        "",
        f"- model: {model_scope} (full sha in run.json/config)",
        "- prereg: harness/preregs/EXP-M4-E1-decodability.md (constants D-014)",
        f"- certified FVs: M2 run {M2_RUN.name}, draws 1-3 (n_trials_aie=200 >= "
        "converged_at=25); Hendel out of scope (no certificate)",
        "- lens instances: M1 draws 0-2 (identity == committed M1 manifests) + "
        "skip16_n10 + skip4_n5 (in-run controls only)",
        f"- random seeds {RANDOM_SEEDS[0]}-{RANDOM_SEEDS[-1]}; rank statistics are "
        "invariant to the norm-matching rescaling (companion norm cells recorded)",
        "",
        "## Per-task verdicts (C1-C4)",
        "",
        "| task | C1 jlens med | C3 logit med | C2 cells | C4 beaten/100 | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for task, res in results.items():
        if res["verdict"] == "INSTRUMENT-VOID":
            lines.append(f"| {task} | - | - | - | - | INSTRUMENT-VOID |")
            continue
        beaten = "/".join(str(res["random_beaten"][i]) for i in FV_DRAWS)
        lines.append(
            f"| {task} | {res['jlens_median']:.0f} | {res['logit_median']:.0f} | "
            f"{res['n_ordering_cells']} ({len(res['not_evaluable_cells'])} n/e) | "
            f"{beaten} | {res['verdict']} |"
        )
    lines += ["", "## Scoped readings", ""]
    for task, res in results.items():
        if res["verdict"] == "INSTRUMENT-VOID":
            lines.append(f"- {task}: INSTRUMENT-VOID; readings neither for nor "
                         "against (instruments LAW); see controls.json")
            continue
        grid = DrawSet(
            values=tuple(float(v) for v in res["primary_jlens"].values()),
            seeds=tuple(int(k.replace(",", "")) for k in res["primary_jlens"]),
        )
        logit_ds = DrawSet(
            values=tuple(float(v) for v in res["logit_by_fv_draw"].values()),
            seeds=tuple(res["logit_by_fv_draw"]),
        )
        lines.append("- " + scoped(
            f"E1 {task}: jlens label-rank {grid.summary()} vs logit "
            f"{logit_ds.summary()}; random beaten "
            f"{'/'.join(str(res['random_beaten'][i]) for i in FV_DRAWS)}/100 "
            f"per draw; verdict {res['verdict']}",
            float(res["jlens_median"]), model=model_scope, config=config_scope,
            n=len(res["primary_jlens"]),
        ))
    n_supports = sum(r.get("verdict") == "DECODABLE-AND-SEPARATED" for r in results.values())
    n_void = sum(r.get("verdict") == "INSTRUMENT-VOID" for r in results.values())
    clm_moves = n_supports >= 1 and n_void == 0
    wall_s = round(time.perf_counter() - t_start, 1)
    lines += [
        "",
        f"**E1 outcome: {n_supports}/3 tasks DECODABLE-AND-SEPARATED"
        f"{', ' + str(n_void) + ' INSTRUMENT-VOID' if n_void else ''}; "
        f"CLM-001 {'moves to preliminary' if clm_moves else 'stays hypothesis'} "
        "per the preregistered rule (pending the post-run evidence commit"
        f"{' and human verification for any further promotion' if clm_moves else ''})**",
        "",
        f"wall-clock {wall_s} s; peak RSS {peak_rss_gb():.2f} GB; device {cfg.device}; "
        "grids in e1_results.json; ControlRecords in controls.json; raw cells "
        "under raw_completions/.",
        "",
    ]
    (ctx.results_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    ctx.finalize(
        verdicts={t: results[t]["verdict"] for t in results},
        clm001_moves_to_preliminary=clm_moves,
        model_revision=revision,
        wall_clock_s=wall_s,
        peak_rss_gb=round(peak_rss_gb(), 2),
    )
    print(f"\nE1 verdicts: {ctx.record['verdicts']}")
    print(f"report: {ctx.results_dir / 'report.md'}")


if __name__ == "__main__":
    main()
