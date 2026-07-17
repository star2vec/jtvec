"""LAW: raw model outputs are retained on disk for every reported number.
Configs are copied into every results directory.

CI-side check that a results directory actually contains what the LAW
requires: a copied config, a valid run record, and raw completions.
"""

from __future__ import annotations

import json
from pathlib import Path

from jtvec.core.runctx import RUN_RECORD_REQUIRED_KEYS


def check_results_dir(path: Path) -> list[str]:
    """Return violations for one results directory (empty list = pass)."""
    path = Path(path)
    violations: list[str] = []
    if not path.is_dir():
        return [f"{path}: results directory does not exist"]

    configs = list(path.glob("*.yaml")) + list(path.glob("*.yml"))
    if not configs:
        violations.append(f"{path}: no copied config (*.yaml) found")

    run_json = path / "run.json"
    if not run_json.is_file():
        violations.append(f"{path}: run.json missing")
    else:
        try:
            record = json.loads(run_json.read_text())
            missing = [k for k in RUN_RECORD_REQUIRED_KEYS if k not in record]
            if missing:
                violations.append(f"{path}: run.json missing keys {missing}")
        except json.JSONDecodeError as e:
            violations.append(f"{path}: run.json is not valid JSON ({e})")

    raw = path / "raw_completions"
    if not raw.is_dir() or not any(raw.iterdir()):
        violations.append(f"{path}: raw_completions/ missing or empty")

    return violations


def count_raw_per_cell(path: Path) -> dict[str, int]:
    """Completion counts per result cell (one .jsonl per cell)."""
    raw = Path(path) / "raw_completions"
    if not raw.is_dir():
        return {}
    return {
        f.stem: sum(1 for line in f.read_text().splitlines() if line.strip())
        for f in sorted(raw.glob("*.jsonl"))
    }
