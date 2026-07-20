"""EXP-M5-0 substrate qualification (D-025 as-authored S1; D-022 LRE; D-026
binding). Per substrate, two model passes to stay within 16 GB: Pass A loads
the jlens model and scores S1 (as-authored, greedy top-1, preserving the M1
anchor) + the binding battery; Pass B loads the HF model and runs FV 10/0-shot
execution (Todd n_shot_eval) + the LRE relation battery. Produces the admitted
species x substrate report.

Under the D-029 scoped admission, 1.4B is A1/A4-admitted; this run qualifies the
capability batteries that gate which species can be measured.

Prereg: harness/preregs/EXP-M5-0-qualification.md.
Usage: uv run python scripts/m5_0_qualification.py
"""

from __future__ import annotations

import dataclasses
import gc
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from jvec.config import Config
from jvec.evals.baseline import score_task
from jvec.evals.tasks import load_tasks
from jvec.fv import FV_REPO, load_fv_model
from jvec.modeling import load_model
from jvec.utils import peak_rss_gb, set_seed
from jtvec.core.runctx import start_run
from jtvec.qualification import lre_relation_accuracy, load_relation

SUBSTRATES = {
    "pythia-410m": REPO_ROOT / "configs/m1_pythia410m_draw0.yaml",
    "pythia-1.4b": REPO_ROOT / "configs/m5_lens_pythia1p4b_draw0.yaml",
}
PREREG = REPO_ROOT / "harness/preregs/EXP-M5-0-qualification.md"
FV_TASKS = ["capitalize", "singular-plural", "english-french"]
LRE_RELATIONS = [
    "factual/country_capital_city", "factual/landmark_in_country",
    "factual/food_from_country", "factual/product_by_company",
    "linguistic/adj_antonym", "linguistic/verb_past_tense",
    "linguistic/word_first_letter", "linguistic/word_last_letter",
    "commonsense/object_superclass", "commonsense/fruit_inside_color",
    "commonsense/task_done_by_tool", "commonsense/work_location",
]
BARS = {"s1": 0.80, "fv": 0.80, "lre": 0.60, "bind2": 0.70}
LRE_N_TEST, EVAL_SEED = 50, 999


def exec_top1(dataset, n_shots, model, mconfig, tok):
    from utils.eval_utils import n_shot_eval_no_intervention  # noqa: PLC0415
    set_seed(EVAL_SEED)
    res = n_shot_eval_no_intervention(dataset, n_shots, model, mconfig, tok,
                                      compute_ppl=False, test_split="test")
    return dict(res["clean_topk"])[1], len(res["clean_rank_list"])


def free(model):
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def qualify(sub: str, cfg_path: Path, ctx) -> dict:
    cfg = Config.load(str(cfg_path))
    set_seed(cfg.seed)
    device = cfg.torch_device()
    report = {"substrate": sub, "revision": cfg.model.revision, "s1": {}, "binding": {},
              "fv": {}, "lre": {}}

    # --- Pass A: jlens model — S1 (as-authored) + binding ---
    model_j, tok_j, _ = load_model(cfg)
    for task in load_tasks(REPO_ROOT / "tasks"):
        scored = score_task(model_j, tok_j, task)
        shots = task.items[0]["prompt"].count(". ") if task.items else 0
        report["s1"][task.name] = {"accuracy": round(scored["accuracy"], 4),
                                   "shots_as_authored": shots, "n": len(task.items),
                                   "admitted": scored["accuracy"] >= BARS["s1"]}
        ctx.save_raw_completions(f"{sub}_s1_{task.name}",
            [{"substrate": sub, **it} for it in scored["per_item"]])
    for task in load_tasks(REPO_ROOT / "tasks" / "binding"):
        scored = score_task(model_j, tok_j, task)
        report["binding"][task.name] = {"accuracy": round(scored["accuracy"], 4),
                                        "n": len(task.items)}
        ctx.save_raw_completions(f"{sub}_binding_{task.name}",
            [{"substrate": sub, **it} for it in scored["per_item"]])
    report["binding"]["s4_admitted"] = report["binding"].get("bind2", {}).get("accuracy", 0) >= BARS["bind2"]
    print(f"[{sub}] pass A done (S1 + binding)", flush=True)
    free(model_j)

    # --- Pass B: HF model — FV 10/0-shot + LRE ---
    model_h, tok_h, mconfig, _ = load_fv_model(cfg)
    from utils.prompt_utils import load_dataset  # noqa: PLC0415
    for task in FV_TASKS:
        ds = load_dataset(task, root_data_dir=str(FV_REPO / "dataset_files"), seed=cfg.seed)
        icl, n = exec_top1(ds, 10, model_h, mconfig, tok_h)
        zs, _ = exec_top1(ds, 0, model_h, mconfig, tok_h)
        report["fv"][task] = {"exec_10shot": round(icl, 4), "zero_shot": round(zs, 4),
                              "n": n, "admitted": icl >= BARS["fv"]}
        print(f"[{sub}] fv {task}: 10-shot {icl:.1%} / 0-shot {zs:.1%}", flush=True)
    n_lre_pass = 0
    for rel in LRE_RELATIONS:
        r = lre_relation_accuracy(model_h, tok_h, load_relation(rel), 10, LRE_N_TEST, device, seed=cfg.seed)
        admitted = r["accuracy"] >= BARS["lre"]
        n_lre_pass += admitted
        report["lre"][rel] = {"accuracy": round(r["accuracy"], 4), "n": r["n"], "admitted": admitted}
        ctx.save_raw_completions(f"{sub}_lre_{rel.replace('/', '_')}",
            [{"substrate": sub, **it} for it in r["per_item"]])
        print(f"[{sub}] lre {rel}: {r['accuracy']:.1%} (N={r['n']})", flush=True)
    report["lre"]["s3_admitted"] = n_lre_pass >= 8  # M5_SPEC: >= 8 relations
    report["lre"]["n_relations_passing"] = n_lre_pass
    free(model_h)
    return report


def main() -> None:
    ctx = start_run(repo_root=REPO_ROOT, config_path=SUBSTRATES["pythia-410m"],
                    results_root=REPO_ROOT / "results/m5", run_name="qualification",
                    prereg_path=PREREG)
    print(f"qualification run dir: {ctx.results_dir}", flush=True)
    import shutil
    shutil.copy2(SUBSTRATES["pythia-1.4b"], ctx.results_dir / SUBSTRATES["pythia-1.4b"].name)

    reports = {}
    for sub, cfg_path in SUBSTRATES.items():
        print(f"\n===== qualifying {sub} =====", flush=True)
        reports[sub] = qualify(sub, cfg_path, ctx)

    summary = {"reports": reports, "bars": BARS, "peak_rss_gb": round(peak_rss_gb(), 2)}
    (ctx.results_dir / "qualification.json").write_text(json.dumps(summary, indent=2))

    print("\n=== EXP-M5-0 qualification summary ===", flush=True)
    for sub, r in reports.items():
        s1_ok = sum(t["admitted"] for t in r["s1"].values())
        fv_ok = sum(t["admitted"] for t in r["fv"].values())
        print(f"  {sub}: S1 {s1_ok}/{len(r['s1'])} admitted | FV {fv_ok}/{len(r['fv'])} | "
              f"LRE {r['lre']['n_relations_passing']}/12 (S3 {'ADMIT' if r['lre']['s3_admitted'] else 'no'}) | "
              f"S4 {'ADMIT' if r['binding']['s4_admitted'] else 'no'} (bind2 {r['binding'].get('bind2',{}).get('accuracy')})")
    print(f"  run dir: {ctx.results_dir}")
    ctx.finalize(peak_rss_gb=summary["peak_rss_gb"])


if __name__ == "__main__":
    main()
