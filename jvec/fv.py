"""Wrapper around Todd et al. function_vectors (third_party/function_vectors).

Their extraction code (baukit hooks, causal mediation, FV construction) is
used as-is, imported as a library. The only thing we replace is model
loading: their loader forces fp16 for Pythia, which is the MPS-broken path —
we load fp32 on the config device and replicate their MODEL_CONFIG dict
verbatim (see model_utils.py, gpt-neox branch).

Cached artifacts per task under ``cache/fvs/<model>/<task>/``:
mean_head_activations.pt, indirect_effect.pt, fv_todd.pt (top-k-head FV),
fv_hendel.pt (mean last-token residual per layer), manifest.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from jlens import ActivationRecorder

from jvec.config import Config
from jvec.utils import REPO_ROOT, jlens_commit  # noqa: F401  (jlens_commit reused by callers)

FV_REPO = REPO_ROOT / "third_party" / "function_vectors"
FV_SRC = FV_REPO / "src"
if str(FV_SRC) not in sys.path:
    sys.path.insert(0, str(FV_SRC))  # their modules import as `utils.*`


def _patch_todd_for_transformers5() -> None:
    """Fix a silent no-op in Todd's FV intervention under transformers >= 5.

    Their ``add_function_vector`` hook only edits tuple outputs; NeoX blocks
    return plain tensors in transformers 5.x, so the FV was never added
    (observed: induction gain exactly +0.0%). The AIE path is unaffected — it
    edits the attention dense-projection *input*, which is always a tensor.
    """
    import utils.intervention_utils as iu  # noqa: PLC0415

    def add_function_vector(edit_layer, fv_vector, device, idx=-1):
        def add_act(output, layer_name):
            current_layer = int(layer_name.split(".")[2])
            if current_layer != edit_layer:
                return output
            if isinstance(output, tuple):
                output[0][:, idx] += fv_vector.to(device)
                return output
            output[:, idx] += fv_vector.to(device)
            return output

        return add_act

    iu.add_function_vector = add_function_vector


_patch_todd_for_transformers5()


def todd_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(FV_REPO), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def load_fv_model(cfg: Config):
    """fp32 model on the config device + Todd-style MODEL_CONFIG for GPT-NeoX."""
    device = cfg.torch_device()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.model.name, revision=cfg.model.revision
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        cfg.model.name, revision=cfg.model.revision, dtype=cfg.torch_dtype()
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if "neox" not in model.config.model_type:
        raise ValueError(
            f"load_fv_model replicates the Todd MODEL_CONFIG for GPT-NeoX/Pythia "
            f"only; got model_type={model.config.model_type!r}"
        )
    n_layers = model.config.num_hidden_layers
    model_config = {
        "n_heads": model.config.num_attention_heads,
        "n_layers": n_layers,
        "resid_dim": model.config.hidden_size,
        "name_or_path": model.config.name_or_path,
        "attn_hook_names": [
            f"gpt_neox.layers.{l}.attention.dense" for l in range(n_layers)
        ],
        "layer_hook_names": [f"gpt_neox.layers.{l}" for l in range(n_layers)],
        "prepend_bos": False,
    }
    revision = getattr(model.config, "_commit_hash", None) or (cfg.model.revision or "unknown")
    return model, tokenizer, model_config, revision


def fv_dir(cfg: Config, task: str) -> Path:
    model_slug = cfg.model.name
    if cfg.model.revision:  # checkpoint sweeps: one cache per revision
        model_slug = f"{cfg.model.name}@{cfg.model.revision}"
    return REPO_ROOT / cfg.cache_dir / "fvs" / model_slug / task


def fv_identity(cfg: Config, task: str, model_revision: str) -> dict:
    fv = cfg.fv
    return {
        "task": task,
        "model_name": cfg.model.name,
        "model_revision": model_revision,
        "n_shots": fv.n_shots,
        "n_trials_mean": fv.n_trials_mean,
        "n_trials_aie": fv.n_trials_aie,
        "n_top_heads": fv.n_top_heads,
        "seed": cfg.seed,
        "todd_commit": todd_commit(),
    }


class FVManifestMismatch(RuntimeError):
    pass


def load_cached_fv(cfg: Config, task: str, model_revision: str) -> dict | None:
    """Load cached FV artifacts if the manifest matches; None if absent."""
    directory = fv_dir(cfg, task)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    expected = fv_identity(cfg, task, model_revision)
    mismatches = {
        k: (manifest.get(k), v) for k, v in expected.items() if manifest.get(k) != v
    }
    if mismatches:
        raise FVManifestMismatch(
            f"cached FV at {directory} does not match config: {mismatches}; "
            f"rerun scripts/05_extract_fvs.py --refit to re-extract"
        )
    return {
        name: torch.load(directory / f"{name}.pt", map_location="cpu", weights_only=True)
        for name in ("mean_head_activations", "indirect_effect", "fv_todd", "fv_hendel")
    } | {"manifest": manifest}


def correct_valid_filter_set(cfg: Config, dataset, model, tokenizer, model_config) -> "np.ndarray":
    """Indices of valid-split items the model answers correctly at n-shot.

    This is the canonical Todd pipeline's ``filter_set_validation``
    (evaluate_function_vector.py): FV statistics are computed only over ICL
    prompts whose query the model actually solves.
    """
    from utils.eval_utils import n_shot_eval_no_intervention  # noqa: PLC0415

    results = n_shot_eval_no_intervention(
        dataset, cfg.fv.n_shots, model, model_config, tokenizer,
        compute_ppl=False, test_split="valid",
    )
    return np.where(np.array(results["clean_rank_list"]) == 0)[0]


@torch.no_grad()
def hendel_mean_residuals(cfg: Config, dataset, model, tokenizer, model_config, filter_set=None) -> torch.Tensor:
    """Hendel-style task vectors: mean residual at the final (query) token of
    n_shots ICL prompts, one vector per layer -> [n_layers, d_model].

    Reuses Todd's prompt construction (same prompt distribution as the Todd
    FV) but records residuals with jlens's ActivationRecorder — their
    ``get_mean_layer_activations`` assumes blocks return tuples, which
    transformers 5.x NeoX blocks no longer do.
    """
    from utils.prompt_utils import (  # noqa: PLC0415
        get_token_meta_labels,
        word_pairs_to_prompt_data,
    )

    n_layers, d = model_config["n_layers"], model_config["resid_dim"]
    blocks = model.gpt_neox.layers
    storage = torch.zeros(cfg.fv.n_trials_mean, n_layers, d)
    prepend_bos = not model_config["prepend_bos"]
    if filter_set is None:
        filter_set = np.arange(len(dataset["valid"]))
    for n in range(cfg.fv.n_trials_mean):
        word_pairs = dataset["train"][
            np.random.choice(len(dataset["train"]), cfg.fv.n_shots, replace=False)
        ]
        word_pairs_test = dataset["valid"][np.random.choice(filter_set, 1)]
        prompt_data = word_pairs_to_prompt_data(
            word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos
        )
        query = prompt_data["query_target"]["input"]
        _, prompt_string = get_token_meta_labels(
            prompt_data, tokenizer, query, prepend_bos=model_config["prepend_bos"]
        )
        inputs = tokenizer([prompt_string], return_tensors="pt").to(model.device)
        with ActivationRecorder(blocks, at=list(range(n_layers))) as recorder:
            model(**inputs)
            for l in range(n_layers):
                storage[n, l] = recorder.activations[l][0, -1].detach().float().cpu()
    return storage.mean(dim=0)


def extract_task_fvs(
    cfg: Config, task: str, model, tokenizer, model_config, model_revision: str,
    filter_set=None,
) -> dict:
    """Run the Todd pipeline + Hendel mean-residual for one task and cache."""
    from utils.extract_utils import (  # noqa: PLC0415 (their src, path-injected)
        compute_function_vector,
        get_mean_head_activations,
    )
    from utils.prompt_utils import load_dataset  # noqa: PLC0415
    from compute_indirect_effect import compute_indirect_effect  # noqa: PLC0415

    fv_cfg = cfg.fv
    dataset = load_dataset(task, root_data_dir=str(FV_REPO / "dataset_files"), seed=cfg.seed)

    timings = {}
    if filter_set is None:
        t0 = time.perf_counter()
        filter_set = correct_valid_filter_set(cfg, dataset, model, tokenizer, model_config)
        timings["filter_set_s"] = round(time.perf_counter() - t0, 1)
    if len(filter_set) < fv_cfg.n_shots:
        raise ValueError(
            f"{task}: only {len(filter_set)} correctly-answered valid items; "
            f"too few to sample FV trials from"
        )

    t0 = time.perf_counter()
    mean_head_acts = get_mean_head_activations(
        dataset, model, model_config, tokenizer,
        n_icl_examples=fv_cfg.n_shots, N_TRIALS=fv_cfg.n_trials_mean,
        filter_set=filter_set,
    )
    timings["mean_head_activations_s"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    indirect_effect = compute_indirect_effect(
        dataset, mean_head_acts, model, model_config, tokenizer,
        n_shots=fv_cfg.n_shots, n_trials=fv_cfg.n_trials_aie, last_token_only=True,
        filter_set=filter_set,
    )
    timings["indirect_effect_s"] = round(time.perf_counter() - t0, 1)

    fv_todd, top_heads = compute_function_vector(
        mean_head_acts, indirect_effect, model, model_config,
        n_top_heads=fv_cfg.n_top_heads,
    )

    t0 = time.perf_counter()
    fv_hendel = hendel_mean_residuals(
        cfg, dataset, model, tokenizer, model_config, filter_set=filter_set
    )
    timings["hendel_residuals_s"] = round(time.perf_counter() - t0, 1)

    directory = fv_dir(cfg, task)
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "mean_head_activations": mean_head_acts.float().cpu(),
        "indirect_effect": indirect_effect.float().cpu(),
        "fv_todd": fv_todd.squeeze().float().cpu(),
        "fv_hendel": fv_hendel,
    }
    for name, tensor in artifacts.items():
        torch.save(tensor, directory / f"{name}.pt")
    manifest = fv_identity(cfg, task, model_revision) | {
        "top_heads": [(int(l), int(h), float(s)) for l, h, s in top_heads],
        "timings": timings,
        "device": cfg.device,
        "torch_version": torch.__version__,
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return artifacts | {"manifest": manifest}
