"""EXP-M5-8b: S5 top-k unembed-token diagnostic (410M).

Post-hoc, non-confounded adjudication of EXP-M5-8's S5 HETEROGENEOUS (1/4): dumps
what each S5 steering vector actually POINTS AT (its top unembed tokens under the
logit lens + J-lens), rather than whether it hits the readout words I picked.
Re-specifying S5 with new words is refused (anti-harvesting) — this only reads the
SAME vectors.

Branches (Ecaterina, fixed before the dump): all 4 attribute-aligned in top
tokens → 1/4 was a readout-word artifact, S5 output-alignment TYPE-GENERAL; only
sentiment aligned → HETEROGENEOUS real. The alignment call is the human read of
the raw tokens; the pos_w-in-top-20 flag is a convenience only.

Prereg: harness/preregs/EXP-M5-8b-s5-toptokens.md (ORDERED by Ecaterina).
Usage: uv run python scripts/m5_8b_s5_toptokens.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from jvec.calibration import select_prompts
from jvec.config import Config
from jvec.evals.concept import answer_states
from jvec.evals.tasks import surface_token_ids
from jvec.lens_cache import load_lens
from jvec.modeling import load_model
from jvec.utils import peak_rss_gb, set_seed
from jtvec.concept_gate import mean_difference_by_layer
from jtvec.core.runctx import start_run
from scripts.m5_8_breadth import S5_ATTRS, _sub, DRAWS, SKIP_FIRST

PREREG = REPO_ROOT / "harness/preregs/EXP-M5-8b-s5-toptokens.md"
CFG = REPO_ROOT / "configs/m5_8_breadth_pythia410m.yaml"
DRAW0_CFG = REPO_ROOT / "configs/m1_pythia410m_draw0.yaml"
TOPK = 15


def main() -> None:
    argparse.ArgumentParser().parse_args()
    t0 = time.perf_counter()
    cfg = Config.load(str(CFG))
    ctx = start_run(repo_root=REPO_ROOT, config_path=CFG, results_root=REPO_ROOT / cfg.results_dir,
                    run_name="m5-8b-s5-toptokens", prereg_path=PREREG)
    print(f"M5.8b run dir: {ctx.results_dir}", flush=True)
    set_seed(cfg.seed)
    model, tok, revision = load_model(cfg)
    lo, hi = cfg.evals.band
    band = [l for l in range(model.n_layers) if lo <= l <= hi]
    rep_layers = [band[1], band[len(band) // 2], band[-2]]   # low / mid / high
    device = model.input_device
    print(f"[model] EleutherAI/pythia-410m@{revision[:7]}; band {band}; rep-layers {rep_layers}", flush=True)

    # representative J-lens draw (lens draw 0); top-1 stability noted qualitatively
    dcfg = Config.load(str(DRAW0_CFG)); set_seed(dcfg.seed)
    lens = load_lens(dcfg, SKIP_FIRST, select_prompts(dcfg, tok), revision)

    def top_tokens(vocab_logits, k=TOPK):
        idx = torch.topk(vocab_logits.float().flatten(), k).indices.tolist()
        return [tok.decode([i]).strip() or repr(tok.decode([i])) for i in idx]

    results = {}
    for name, A in S5_ATTRS.items():
        # mean-difference direction, averaged over the 3 draws (representative)
        raws = []
        for kk in DRAWS:
            raws.append(mean_difference_by_layer(answer_states(model, _sub(A["pos"], kk), band),
                                                 answer_states(model, _sub(A["neg"], kk), band)))
        mean_dir = {l: torch.stack([raws[i][l] for i in range(len(DRAWS))]).mean(0) for l in band}

        per_layer = {}
        for l in rep_layers:
            v = mean_dir[l].float().to(device)
            logit_top = top_tokens(model.unembed(v).float().cpu())
            jlens_top = top_tokens(model.unembed(lens.transport(v, l)).float().cpu())
            per_layer[l] = {"logit_top": logit_top, "jlens_top": jlens_top}
        # convenience flag only (NOT decisive): do the authored pos_w appear in top-20 logit tokens anywhere?
        pooled_logit = set()
        for l in band:
            v = mean_dir[l].float().to(device)
            pooled_logit |= {tok.decode([i]).strip().lower()
                             for i in torch.topk(model.unembed(v).float().flatten(), 20).indices.tolist()}
        authored_hit = sorted(w for w in A["pos_w"] if w.lower() in pooled_logit)
        results[name] = {"per_layer": {str(l): per_layer[l] for l in rep_layers},
                         "authored_pos_w": A["pos_w"], "authored_pos_w_in_top20_logit": authored_hit}
        print(f"\n=== S5:{name} ===", flush=True)
        for l in rep_layers:
            print(f"  L{l} logit-top: {per_layer[l]['logit_top']}", flush=True)
            print(f"  L{l} jlens-top: {per_layer[l]['jlens_top']}", flush=True)
        print(f"  [convenience] authored pos_w in any-layer top-20 logit: {authored_hit or 'NONE'}", flush=True)

    del lens
    summary = {"model": f"EleutherAI/pythia-410m@{revision[:7]}", "topk": TOPK,
               "rep_layers": rep_layers, "results": results,
               "note": "alignment is Ecaterina's read of the raw top tokens; the authored-word flag is a convenience only, not decisive",
               "peak_rss_gb": round(peak_rss_gb(), 2), "wall_clock_s": round(time.perf_counter() - t0, 1)}
    (ctx.results_dir / "s5_toptokens.json").write_text(json.dumps(summary, indent=2, default=str))
    ctx.save_raw_completions("s5_toptokens",
        [{"attr": n, "layer": l, **results[n]["per_layer"][l]} for n in results for l in results[n]["per_layer"]])

    lines = ["# EXP-M5-8b — S5 top-k unembed tokens (what each steering vector points at)", "",
             f"- model {summary['model']}; top-{TOPK}; representative layers {rep_layers}; logit lens = output space", "",
             "Alignment is a HUMAN read of the raw tokens below (the authored-word flag is a convenience, not decisive).", ""]
    for name in results:
        lines.append(f"## S5:{name}")
        for l in results[name]["per_layer"]:
            pl = results[name]["per_layer"][l]
            lines.append(f"- **L{l} logit**: {', '.join(pl['logit_top'])}")
            lines.append(f"- L{l} jlens: {', '.join(pl['jlens_top'])}")
        lines.append(f"- convenience — authored pos_w in any top-20 logit: "
                     f"**{results[name]['authored_pos_w_in_top20_logit'] or 'NONE'}** "
                     f"(authored: {results[name]['authored_pos_w']})")
        lines.append("")
    lines.append(f"wall {summary['wall_clock_s']} s; peak {summary['peak_rss_gb']} GB.")
    (ctx.results_dir / "report.md").write_text("\n".join(lines))
    print(f"\n=== EXP-M5-8b done: {ctx.results_dir} ===", flush=True)
    ctx.finalize(model_revision=revision, wall_clock_s=summary["wall_clock_s"], peak_rss_gb=summary["peak_rss_gb"])


if __name__ == "__main__":
    main()
