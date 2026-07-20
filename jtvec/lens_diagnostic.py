"""EXP-M5-0b verdict logic: the max-contrast statistic and the latent-vs-output
decision rule. Pure functions over probe-metric dicts so the verdict has a
model-free landing test (tests/test_lens_diagnostic.py).

The max-contrast statistic (RATIFIED amendment, 2026-07-20) is applied
IDENTICALLY to every readout arm: for arm A over a fixed candidate layer set
S, ratio_A = max over L in S with HMR_A(L) <= cap of logit_HMR(L)/HMR_A(L).
The J-lens verdict only counts if ratio_jlens beats what the SAME procedure
gives the random-matrix arm (ratio_random < advantage_ratio); otherwise the
selection is inflating noise and the statistic is withdrawn.
"""

from __future__ import annotations

import statistics


def arm_max_contrast(arm_per_layer: dict, logit_per_layer: dict,
                     layers: list[int], cap: float) -> tuple[float, int | None]:
    """(ratio, layer) of the max-contrast layer for one arm. Among layers L
    (in S) where the arm's HMR <= cap, the largest logit_HMR/arm_HMR. If no
    layer qualifies, (0.0, None) — no advantage."""
    best_ratio, best_layer = 0.0, None
    for L in layers:
        a = arm_per_layer[str(L)]["hmr"]
        if a <= cap:
            r = logit_per_layer[str(L)]["hmr"] / a
            if r > best_ratio:
                best_ratio, best_layer = r, L
    return best_ratio, best_layer


def task_arm_ratios(task_metrics: dict, layers: list[int], cap: float) -> dict:
    """Per-task, per-draw: jlens max-contrast ratio + the worst-case (max)
    random-arm ratio under the identical statistic."""
    logit = task_metrics["logit"]["per_layer"]
    jl_ratio, jl_layer = arm_max_contrast(task_metrics["jlens"]["per_layer"], logit, layers, cap)
    rand_max = 0.0
    for arm, payload in task_metrics.items():
        if arm.startswith("random-"):
            r, _ = arm_max_contrast(payload["per_layer"], logit, layers, cap)
            rand_max = max(rand_max, r)
    return {"jlens_ratio": jl_ratio, "jlens_layer": jl_layer, "random_max_ratio": rand_max}


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs))


def task_shows_advantage(per_draw: list[dict], advantage_ratio: float) -> tuple[bool, dict]:
    """A task shows the J-lens advantage iff median-over-draws jlens ratio
    >= advantage_ratio AND the random control (median max-random ratio) stays
    below it."""
    jl_med = _median([d["jlens_ratio"] for d in per_draw])
    rand_med = _median([d["random_max_ratio"] for d in per_draw])
    shows = jl_med >= advantage_ratio and rand_med < advantage_ratio
    return shows, {"jlens_ratio_median": round(jl_med, 3),
                   "random_ratio_median": round(rand_med, 3), "n_draws": len(per_draw)}


# matched latent -> output partner (same prompts, different probed token)
MATCHED = {"fresh1hop-operand": "fresh1hop-answer",
           "fresh2hop-bridge": "fresh2hop-answer"}


def amended_q5_verdict(anchor_shows: dict[str, tuple[bool, int]],
                       adequate_n: int = 20) -> dict:
    """EXP-M5-0 amended Q5 (D-027 outcome c). anchor_shows[task] =
    (shows_advantage, N_correct) for LATENT-INTERMEDIATE anchors. Q5 passes iff
    >= 2 latent anchors at adequate N (>= adequate_n) clear the advantage;
    anchors below adequate N are descriptive only (2hop-bridge rider)."""
    clearing = [t for t, (s, n) in anchor_shows.items() if s and n >= adequate_n]
    descriptive = {t: n for t, (s, n) in anchor_shows.items() if n < adequate_n}
    return {
        "passed": len(clearing) >= 2,
        "clearing_adequate_n": sorted(clearing),
        "n_clearing": len(clearing),
        "descriptive_low_n": descriptive,
        "adequate_n": adequate_n,
    }


def diagnostic_verdict(per_task_draws: dict[str, list[dict]],
                       advantage_ratio: float,
                       matched: dict[str, str] | None = None) -> dict:
    """Decide the D-027 fork. per_task_draws[task] = list of per-draw
    task_arm_ratios dicts. GAP RETURNS iff >= 2 FRESH latent tasks show the
    advantage AND their matched output tasks do NOT (the dissociation)."""
    matched = matched or MATCHED
    shows = {t: task_shows_advantage(v, advantage_ratio) for t, v in per_task_draws.items()}

    dissociating = []  # latent tasks that show advantage while their output partner does not
    for latent, output in matched.items():
        if latent not in shows:
            continue
        latent_shows = shows[latent][0]
        output_shows = shows[output][0] if output in shows else False
        if latent_shows and not output_shows:
            dissociating.append(latent)

    gap_returns = len(dissociating) >= 2
    verdict = "GAP-RETURNS" if gap_returns else "NO-GAP"
    return {
        "verdict": verdict,
        "d027_outcome": ("miscalibration (amend Q5, admit 1.4B)" if gap_returns
                         else "convergence (register deflation, justify 2.8B)"),
        "n_dissociating_pairs": len(dissociating),
        "dissociating_latent_tasks": dissociating,
        "per_task": {t: {"shows_advantage": s[0], **s[1]} for t, s in shows.items()},
    }
