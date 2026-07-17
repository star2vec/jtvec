"""Model loading: HF checkpoint -> jlens LensModel wrapper."""

from __future__ import annotations

import jlens
import transformers

from jvec.config import Config


def load_model(cfg: Config) -> tuple[jlens.HFLensModel, transformers.PreTrainedTokenizerBase, str]:
    """Load the config's model on the config's device/dtype and wrap for jlens.

    Returns ``(lens_model, tokenizer, resolved_revision)``; the revision is the
    HF commit sha actually loaded, for the manifest.
    """
    device = cfg.torch_device()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        cfg.model.name, revision=cfg.model.revision, dtype=cfg.torch_dtype()
    ).to(device)
    tok = transformers.AutoTokenizer.from_pretrained(
        cfg.model.name, revision=cfg.model.revision
    )
    resolved_revision = getattr(hf.config, "_commit_hash", None) or (
        cfg.model.revision or "unknown"
    )
    model = jlens.from_hf(hf, tok)
    return model, tok, resolved_revision
