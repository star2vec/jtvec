"""Unit tests for config, manifest enforcement, and cache round-trip."""

import json

import pytest
import torch
from jlens import JacobianLens

from jvec.calibration import PromptSet
from jvec.config import Config
from jvec.lens_cache import (
    ManifestMismatch,
    expected_identity,
    lens_dir,
    load_lens,
)


def _tiny_prompts() -> PromptSet:
    return PromptSet(
        calibration=["a" * 200, "b" * 200],
        heldout=["c" * 200],
        calibration_sha256=["ha", "hb"],
        heldout_sha256=["hc"],
        corpus="test",
        seed=0,
    )


def test_config_roundtrip(tmp_path):
    cfg = Config.load("configs/gpt2_phase1.yaml")
    assert cfg.model.name == "gpt2"
    assert cfg.fit.skip_first_variants == (4, 16)
    assert cfg.evals.band == (2, 8)
    cfg.save(tmp_path / "copy.yaml")
    assert Config.load(tmp_path / "copy.yaml") == cfg


def test_config_rejects_unknown_keys(tmp_path):
    (tmp_path / "bad.yaml").write_text("experiment: x\nnot_a_key: 1\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        Config.load(tmp_path / "bad.yaml")


def test_lens_fp32_roundtrip(tmp_path):
    torch.manual_seed(0)
    lens = JacobianLens(
        jacobians={0: torch.randn(8, 8), 1: torch.randn(8, 8)},
        n_prompts=2,
        d_model=8,
    )
    path = tmp_path / "lens.pt"
    lens.save(str(path), dtype=torch.float32)
    loaded = JacobianLens.load(str(path))
    for layer in (0, 1):
        assert torch.equal(loaded.jacobians[layer], lens.jacobians[layer])


def test_load_lens_refuses_mismatched_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr("jvec.lens_cache.REPO_ROOT", tmp_path)
    monkeypatch.setattr("jvec.lens_cache.jlens_commit", lambda: "deadbeef")
    cfg = Config()
    prompts = _tiny_prompts()

    directory = lens_dir(cfg, skip_first=4)
    directory.mkdir(parents=True)
    lens = JacobianLens(jacobians={0: torch.eye(8)}, n_prompts=2, d_model=8)
    lens.save(str(directory / "lens.pt"), dtype=torch.float32)
    manifest = expected_identity(cfg, 4, prompts, "rev0")
    # n_prompts in identity comes from config (5) vs our 2-prompt set — fix it
    # to match so only the deliberate corruption below mismatches.
    manifest["n_prompts"] = cfg.calibration.n_prompts

    # matching manifest loads fine
    (directory / "manifest.json").write_text(json.dumps(manifest))
    assert load_lens(cfg, 4, prompts, "rev0").d_model == 8

    # corrupt one identity field -> hard error naming the field
    manifest["skip_first"] = 16
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ManifestMismatch, match="skip_first"):
        load_lens(cfg, 4, prompts, "rev0")

    # different jlens commit -> hard error too
    manifest["skip_first"] = 4
    manifest["jlens_commit"] = "0ldc0mmit"
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ManifestMismatch, match="jlens_commit"):
        load_lens(cfg, 4, prompts, "rev0")


def test_missing_cache_is_filenotfound(tmp_path, monkeypatch):
    monkeypatch.setattr("jvec.lens_cache.REPO_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="01_fit_lens"):
        load_lens(Config(), 4, _tiny_prompts(), "rev0")
