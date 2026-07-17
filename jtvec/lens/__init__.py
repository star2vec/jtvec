"""M1: J-lens fitting + mandatory sanity gate.

Facade over the vendored v1 implementation (see VENDORING.md): the validated
code lives byte-identical in the top-level `jvec` package and
`third_party/jacobian-lens` (submodule @ 581d398). This module is the v2 API
surface; the M1 gate orchestration lives in `scripts/m1_gate.py`.
"""

from jvec.calibration import PromptSet, select_prompts
from jvec.config import Config
from jvec.lens_cache import ManifestMismatch, fit_lens, lens_dir, load_lens
from jvec.modeling import load_model

__all__ = [
    "Config",
    "PromptSet",
    "select_prompts",
    "fit_lens",
    "load_lens",
    "lens_dir",
    "ManifestMismatch",
    "load_model",
]
