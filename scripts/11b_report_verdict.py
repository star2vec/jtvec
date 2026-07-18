"""A1 verdict: evaluate the PREREGISTRATION §A1 criteria on collected trials.

Criteria (registered 2026-07-16, before the run):
  1. coherent > shuffled > other report_score ordering, separable by 95%
     bootstrap CIs (non-overlapping for coherent vs other; coherent vs
     shuffled must be positive at the CI level), on >=3 of 8 tasks;
  2. >=1 task with the label in free-running top-10 under coherent context
     (median over trials);
  3. criterion 1 holds under both probe phrasings.

Prints per-cell means with CIs and the mechanical PASS/FAIL verdict.

Usage: uv run python scripts/11b_report_verdict.py results/report_protocol_a1/<ts>/trials.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np


def bootstrap_ci(values, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    path = sys.argv[1]
    rows = [json.loads(l) for l in open(path) if '"trial"' in l]
    trials = [r for r in rows if r["kind"] == "trial"]

    scores = defaultdict(list)   # (probe, task, ctx) -> [report_score]
    ranks = defaultdict(list)    # (probe, task) -> [vocab rank] (coherent only)
    for r in trials:
        scores[(r["probe"], r["task"], r["ctx_type"])].append(r["report_score"])
        if r["ctx_type"] == "coherent":
            ranks[(r["probe"], r["task"])].append(r["label_vocab_rank"])

    probes = sorted({r["probe"] for r in trials})
    tasks = sorted({r["task"] for r in trials})

    c1_pass_per_probe = {}
    for probe in probes:
        print(f"\n=== probe {probe} ===")
        print(f"{'task':18s} {'coherent':>22} {'shuffled':>22} {'other':>22}  {'ordered+separable':>18}")
        passing = []
        for task in tasks:
            cells = {}
            for ctx in ("coherent", "shuffled", "other"):
                m, lo, hi = bootstrap_ci(scores[(probe, task, ctx)])
                cells[ctx] = (m, lo, hi)
            ordered = cells["coherent"][0] > cells["shuffled"][0] > cells["other"][0]
            separable = cells["coherent"][1] > cells["other"][2]  # CI-disjoint coh vs other
            coh_gt_shuf = cells["coherent"][1] > cells["shuffled"][2]
            ok = ordered and separable and coh_gt_shuf
            if ok:
                passing.append(task)
            fmt = lambda c: f"{c[0]:+6.2f} [{c[1]:+6.2f},{c[2]:+6.2f}]"
            print(f"{task:18s} {fmt(cells['coherent']):>22} {fmt(cells['shuffled']):>22} "
                  f"{fmt(cells['other']):>22}  {'PASS' if ok else 'fail':>18}")
        c1_pass_per_probe[probe] = passing
        print(f"criterion-1 tasks passing under {probe}: {passing}")

    print("\n=== criterion 2: median coherent-context vocab rank of label ===")
    c2_tasks = set()
    for (probe, task), rs in sorted(ranks.items()):
        med = int(np.median(rs))
        if med <= 10:
            c2_tasks.add(task)
        print(f"{probe} {task:18s} median rank {med:>6}  (min {min(rs)}, max {max(rs)})")

    both = set(c1_pass_per_probe[probes[0]]) & set(c1_pass_per_probe[probes[1]]) if len(probes) > 1 else set()
    c1 = len(both) >= 3
    c2 = len(c2_tasks) >= 1
    print(f"\ncriterion 1+3 (>=3 tasks pass under BOTH probes): {sorted(both)} -> {'PASS' if c1 else 'FAIL'}")
    print(f"criterion 2 (>=1 task, median free-running rank <=10): {sorted(c2_tasks)} -> {'PASS' if c2 else 'FAIL'}")
    print(f"\nA1 VERDICT: {'PASS' if (c1 and c2) else 'FAIL'}")


if __name__ == "__main__":
    main()
