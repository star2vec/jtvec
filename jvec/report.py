"""Phase-1 report: held-out readouts, eval tables, and the milestone-gate verdict."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from jlens import JacobianLens

from jvec.config import Config
from jvec.evals.tasks import rank_of_word


def heldout_readout_section(
    model, tokenizer, lenses: dict[str, JacobianLens], heldout: list[str], cfg: Config
) -> str:
    """Layer-by-layer top-k lens tokens at the last position of each held-out prompt."""
    k = cfg.evals.topk_report
    lines = ["## Held-out prompt readouts", ""]
    for idx, prompt in enumerate(heldout):
        first_variant = next(iter(lenses.values()))
        input_ids = first_variant.apply(
            model, prompt, layers=[0], positions=[-1], max_seq_len=cfg.fit.max_seq_len
        )[2]
        tail = tokenizer.decode(input_ids[0, -12:])
        lines += [f"### Held-out prompt {idx + 1}", "",
                  f"(128-token window; last 12 tokens: `...{tail}`)", ""]
        for variant, lens in lenses.items():
            lens_logits, model_logits, _ = lens.apply(
                model, prompt, positions=[-1], max_seq_len=cfg.fit.max_seq_len
            )
            logit_logits, _, _ = lens.apply(
                model, prompt, positions=[-1], max_seq_len=cfg.fit.max_seq_len,
                use_jacobian=False,
            )
            model_top = tokenizer.decode([int(model_logits[0].argmax())])
            lines += [f"**{variant}** (model's actual next token: `{model_top!r}`)", "",
                      f"| layer | J-lens top-{k} | logit-lens top-{k} |",
                      "|---|---|---|"]
            for layer in lens.source_layers:
                def top(logits):
                    toks = [tokenizer.decode([t]) for t in logits.topk(k).indices]
                    return " ".join(f"`{t}`" for t in toks).replace("|", "\\|")
                lines.append(
                    f"| {layer} | {top(lens_logits[layer][0])} | {top(logit_logits[layer][0])} |"
                )
            lines.append("")
    return "\n".join(lines)


def baseline_section(baselines: dict, threshold: float) -> str:
    lines = ["## Task baseline gate", "",
             f"Included = in-context top-1 accuracy >= {threshold:.0%}.", "",
             "| task | protocol | accuracy | items | verdict |", "|---|---|---|---|---|"]
    for name, b in sorted(baselines.items()):
        verdict = "**INCLUDED**" if b["included"] else "dropped"
        lines.append(
            f"| {name} | {b['protocol']} | {b['accuracy']:.1%} | {len(b['per_item'])} | {verdict} |"
        )
    return "\n".join(lines + [""])


def probe_section(probe: dict, cfg: Config) -> str:
    k = cfg.evals.pass_k
    lines = ["## Probing eval (rank of the intermediate token in the lens readout)", "",
             "HMR = harmonic mean rank over items (lower is better); "
             f"pass@{k} = fraction of items with rank <= {k}. "
             "`random` = mean over Frobenius-matched Gaussian matrices "
             f"({cfg.evals.n_random_seeds} seeds).", ""]
    for variant, tasks in probe.items():
        for task_name, result in tasks.items():
            m = result["metrics"]
            rand_arms = [a for a in result["arms"] if a.startswith("random")]
            lines += [f"### {variant} / {task_name}", "",
                      f"| layer | J-lens HMR | logit HMR | random HMR | J-lens pass@{k} | logit pass@{k} | random pass@{k} |",
                      "|---|---|---|---|---|---|---|"]
            for layer in result["layers"]:
                l = str(layer)
                rand_hmr = sum(m[a]["per_layer"][l]["hmr"] for a in rand_arms) / len(rand_arms)
                rand_pass = sum(m[a]["per_layer"][l][f"pass@{k}"] for a in rand_arms) / len(rand_arms)
                lines.append(
                    f"| {layer} | {m['jlens']['per_layer'][l]['hmr']:.1f} "
                    f"| {m['logit']['per_layer'][l]['hmr']:.1f} "
                    f"| {rand_hmr:.1f} "
                    f"| {m['jlens']['per_layer'][l][f'pass@{k}']:.2f} "
                    f"| {m['logit']['per_layer'][l][f'pass@{k}']:.2f} "
                    f"| {rand_pass:.2f} |"
                )
            mol = {arm: m[arm]["min_over_layers"] for arm in ("jlens", "logit")}
            lines += ["",
                      f"min-over-layers: J-lens HMR {mol['jlens']['hmr']:.1f} / pass@{k} {mol['jlens'][f'pass@{k}']:.2f}; "
                      f"logit HMR {mol['logit']['hmr']:.1f} / pass@{k} {mol['logit'][f'pass@{k}']:.2f}", ""]
    return "\n".join(lines)


def swap_section(swap: dict) -> str:
    lines = ["## Causal swap eval (pseudoinverse write-back, norm-preserving, truncated pinv)", "",
             "| variant | task | dp(swap_answer) | random ctrl | dp(answer) | top-1 flip rate | n |",
             "|---|---|---|---|---|---|---|"]
    for variant, tasks in swap.items():
        for task_name, result in tasks.items():
            m = result["metrics"]
            lines.append(
                f"| {variant} | {task_name} | {m['mean_dp_swap_answer']:+.4f} "
                f"| {m['mean_dp_swap_answer_random_ctrl']:+.4f} | {m['mean_dp_answer']:+.4f} "
                f"| {m['swap_top1_rate']:.1%} | {m['n_scored']} |"
            )
    return "\n".join(lines + [""])


def gate_verdict(probe: dict, swap: dict, cfg: Config) -> tuple[str, dict]:
    """Milestone gate per lens variant. Returns (markdown, verdicts)."""
    lo, hi = cfg.evals.band
    lines = ["## Milestone gate", "",
             "Criteria (per included probing task): (A) J-lens HMR beats logit-lens HMR at "
             f"some layer in the L{lo}-L{hi} band; (B) J-lens HMR beats the random-matrix "
             f"control (mean over seeds) at every band layer (L{lo}-L{hi}; ruling "
             "2026-07-14 — the earliest layers are excluded, matching the paper's own "
             "caveat). Swap criterion: (C) mean dp(swap_answer) exceeds the "
             "random-direction control.", ""]
    verdicts = {}
    for variant in probe:
        checks = []
        for task_name, result in probe[variant].items():
            m = result["metrics"]
            rand_arms = [a for a in result["arms"] if a.startswith("random")]
            band_layers = [l for l in result["layers"] if lo <= l <= hi]
            a_wins = [
                l for l in band_layers
                if m["jlens"]["per_layer"][str(l)]["hmr"] < m["logit"]["per_layer"][str(l)]["hmr"]
            ]
            cond_a = bool(a_wins)
            b_fails = []
            for l in band_layers:
                rand_hmr = sum(m[a]["per_layer"][str(l)]["hmr"] for a in rand_arms) / len(rand_arms)
                if m["jlens"]["per_layer"][str(l)]["hmr"] >= rand_hmr:
                    b_fails.append(l)
            cond_b = not b_fails
            checks.append((f"{task_name} (A)", cond_a,
                           f"J-lens beats logit at band layers {a_wins}" if cond_a
                           else "J-lens never beats logit lens in the band"))
            checks.append((f"{task_name} (B)", cond_b,
                           "J-lens beats mean random control at every band layer" if cond_b
                           else f"J-lens loses to mean random control at band layers {b_fails}"))
        for task_name, result in swap[variant].items():
            m = result["metrics"]
            cond_c = m["mean_dp_swap_answer"] > m["mean_dp_swap_answer_random_ctrl"]
            checks.append((f"{task_name} (C)", cond_c,
                           f"dp {m['mean_dp_swap_answer']:+.4f} vs random {m['mean_dp_swap_answer_random_ctrl']:+.4f}"))
        passed = all(ok for _, ok, _ in checks)
        verdicts[variant] = passed
        lines += [f"### {variant}: {'**PASS**' if passed else '**FAIL**'}", ""]
        for name, ok, detail in checks:
            lines.append(f"- [{'x' if ok else ' '}] {name}: {detail}")
        lines.append("")
    return "\n".join(lines), verdicts


def build_report(
    cfg: Config,
    model,
    tokenizer,
    lenses: dict[str, JacobianLens],
    manifests: dict[str, dict],
    heldout: list[str],
    baselines: dict,
    probe: dict,
    swap: dict,
) -> tuple[str, dict]:
    any_manifest = next(iter(manifests.values()))
    variant_descs = ", ".join(
        f"{v} (skip_first={m['skip_first']}, fitted {m['fitted_at']}, {m['wall_clock_s']}s)"
        for v, m in manifests.items()
    )
    header = ["# Phase 1 report: J-lens on GPT-2-small", "",
              f"- model: {cfg.model.name} (revision {any_manifest['model_revision']})",
              f"- device: {cfg.device}, dtype: {cfg.model.dtype}, seed: {cfg.seed}",
              f"- calibration: n={cfg.calibration.n_prompts} x {cfg.fit.max_seq_len} tokens from {cfg.calibration.corpus}",
              f"- jlens commit: {any_manifest['jlens_commit']}",
              f"- lens variants: {variant_descs}",
              ""]
    gate_md, verdicts = gate_verdict(probe, swap, cfg)
    parts = [
        "\n".join(header),
        gate_md,
        baseline_section(baselines, cfg.evals.baseline_threshold),
        probe_section(probe, cfg),
        swap_section(swap),
        heldout_readout_section(model, tokenizer, lenses, heldout, cfg),
    ]
    return "\n".join(parts), verdicts
