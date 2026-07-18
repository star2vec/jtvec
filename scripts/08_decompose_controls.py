"""Positive control for Experiment 2 (LABNOTES 2026-07-15, step 1).

The FV decomposition found J-space fractions at/below random. Before trusting
the "FV mass is vocab-dark" reading, decompose residuals we KNOW the J-lens
reads well (Phase-1 evals):

(a) capital-recall answer-position residuals — last prompt token of the
    Phase-1 few-shot capital prompts (J-lens HMR ~2.5 there in Phase 1);
(b) mid-ICL residuals at answer-formation positions — final token of Todd
    10-shot ICL prompts for three strong-induction tasks.

Same k, same band layers, same dictionaries as scripts/07. If (a)/(b) are not
clearly elevated over the FV/random/pile baselines, the instrument is weak and
the vocab-dark claim must be flagged.

Usage: uv run python scripts/08_decompose_controls.py --config configs/pythia410m_phase2.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from jvec.calibration import select_prompts
from jvec.config import Config
from jvec.decompose import gradient_pursuit, jlens_dictionary
from jvec.fv import FV_REPO, load_fv_model
from jvec.lens_cache import load_lens
from jvec.utils import REPO_ROOT, make_run_dir, set_seed

K_ATOMS = 25
N_CAPITAL_PROMPTS = 12
N_ICL_PROMPTS_PER_TASK = 8
ICL_TASKS = ["english-french", "capitalize", "singular-plural"]


@torch.no_grad()
def residuals_at_last(model_j, prompt: str, layers: list[int]) -> dict[int, torch.Tensor]:
    from jlens import ActivationRecorder  # noqa: PLC0415

    ids = model_j.encode(prompt, max_length=512)
    with ActivationRecorder(model_j.layers, at=layers) as rec:
        model_j.forward(ids)
        return {l: rec.activations[l][0, -1].detach().float().cpu() for l in layers}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.seed)
    hf_model, tokenizer, model_config, revision = load_fv_model(cfg)

    import jlens as jlens_pkg
    model_j = jlens_pkg.from_hf(hf_model, tokenizer)
    prompts = select_prompts(cfg, tokenizer)
    lens = load_lens(cfg, cfg.fit.skip_first_variants[0], prompts, revision)
    W_U = hf_model.embed_out.weight.detach().float().cpu()
    lo, hi = cfg.evals.band
    band_layers = [l for l in lens.source_layers if lo <= l <= hi]
    dictionaries = {l: jlens_dictionary(lens, W_U, l) for l in band_layers}

    # (a) Phase-1 capital-recall prompts, answer-formation position.
    capital_items = json.loads((REPO_ROOT / "tasks" / "capital-recall.json").read_text())["items"]
    control_a: dict[int, list[float]] = {l: [] for l in band_layers}
    for item in capital_items[:N_CAPITAL_PROMPTS]:
        res = residuals_at_last(model_j, item["prompt"], band_layers)
        for l in band_layers:
            control_a[l].append(gradient_pursuit(res[l], dictionaries[l], k=K_ATOMS).fraction)

    # (b) Todd 10-shot ICL prompts, final (query-separator/answer-formation) token.
    from utils.prompt_utils import (  # noqa: PLC0415
        get_token_meta_labels,
        load_dataset,
        word_pairs_to_prompt_data,
    )

    control_b: dict[str, dict[int, list[float]]] = {}
    for task in ICL_TASKS:
        dataset = load_dataset(task, root_data_dir=str(FV_REPO / "dataset_files"), seed=cfg.seed)
        control_b[task] = {l: [] for l in band_layers}
        for _ in range(N_ICL_PROMPTS_PER_TASK):
            word_pairs = dataset["train"][
                np.random.choice(len(dataset["train"]), cfg.fv.n_shots, replace=False)
            ]
            word_pairs_test = dataset["valid"][np.random.choice(len(dataset["valid"]), 1)]
            prompt_data = word_pairs_to_prompt_data(
                word_pairs, query_target_pair=word_pairs_test,
                prepend_bos_token=not model_config["prepend_bos"],
            )
            query = prompt_data["query_target"]["input"]
            _, prompt_string = get_token_meta_labels(
                prompt_data, tokenizer, query, prepend_bos=model_config["prepend_bos"]
            )
            res = residuals_at_last(model_j, prompt_string, band_layers)
            for l in band_layers:
                control_b[task][l].append(
                    gradient_pursuit(res[l], dictionaries[l], k=K_ATOMS).fraction
                )

    # Reference numbers from the FV decomposition run.
    fv_run = sorted((REPO_ROOT / cfg.results_dir / "fv_decomposition").glob("*/decomposition.json"))[-1]
    fv_results = json.loads(fv_run.read_text())

    def mean(xs):
        return sum(xs) / len(xs)

    print(f"{'layer':>5} {'(a) capital-ans':>16} {'(b) ICL en-fr':>14} {'(b) ICL capit.':>14} "
          f"{'(b) ICL sg-pl':>14} {'pile ctrl':>10} {'FV todd (mean over tasks)':>26}")
    summary = {"control_a": {}, "control_b": {}, "pile": {}, "fv_mean": {}}
    for l in band_layers:
        fv_fracs = [t["todd"][str(l)]["fraction"] for t in fv_results["tasks"].values()]
        pile = mean(fv_results["pile_controls"][str(l)])
        row = {
            "a": mean(control_a[l]),
            "en_fr": mean(control_b["english-french"][l]),
            "cap": mean(control_b["capitalize"][l]),
            "sg_pl": mean(control_b["singular-plural"][l]),
        }
        summary["control_a"][l] = round(row["a"], 4)
        summary["control_b"][l] = {t: round(mean(control_b[t][l]), 4) for t in ICL_TASKS}
        summary["pile"][l] = round(pile, 4)
        summary["fv_mean"][l] = round(mean(fv_fracs), 4)
        print(f"{l:>5} {row['a']:>16.3f} {row['en_fr']:>14.3f} {row['cap']:>14.3f} "
              f"{row['sg_pl']:>14.3f} {pile:>10.3f} {mean(fv_fracs):>26.3f}")

    run_dir = make_run_dir(cfg, "fv_decomposition_control")
    (run_dir / "control.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {run_dir / 'control.json'}")


if __name__ == "__main__":
    main()
