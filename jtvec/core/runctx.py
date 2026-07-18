"""LAWs enforced here:

- Preregistration file per experiment, committed BEFORE the first run,
  containing: hypothesis, decision rule, and what result would count as
  failure. Post-hoc analyses are labeled post-hoc forever.
- One commit per experiment. Raw model outputs are retained on disk for
  every reported number. Configs are copied into every results directory.

`start_run` is the only way scientific code obtains a results directory. It
refuses to start on a dirty git tree (which mechanically forces the
commit-then-run discipline and makes the recorded commit hash meaningful),
refuses to start without a committed prereg containing the required
sections (unless the run is explicitly post_hoc, which is stamped into the
run record forever), copies the config into the results directory
unconditionally, and provides the writer that retains raw completions.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

REQUIRED_PREREG_SECTIONS = (
    "## Hypothesis",
    "## Decision rule",
    "## What counts as failure",
    "## Estimator plan",
    "## Sample plan",
    "## Resource estimate",
)

RUN_RECORD_REQUIRED_KEYS = (
    "run_name",
    "started",
    "git_commit",
    "prereg_path",
    "prereg_commit",
    "config_sha256",
    "post_hoc",
)


class DirtyTreeError(RuntimeError):
    """Raised when a scientific run is started from an uncommitted tree."""


class PreregError(RuntimeError):
    """Raised when the preregistration LAW is not satisfied."""


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def git_head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


def git_is_dirty(repo: Path) -> bool:
    return bool(_git(repo, "status", "--porcelain"))


def prereg_commit(repo: Path, prereg: Path) -> str:
    """Commit that introduced/last touched the prereg; raises if uncommitted."""
    rel = prereg.resolve().relative_to(repo.resolve())
    commit = _git(repo, "log", "-1", "--format=%H", "--", str(rel))
    if not commit:
        raise PreregError(
            f"LAW violation: prereg '{rel}' is not committed; commit it before the first run"
        )
    return commit


def check_prereg_sections(prereg: Path) -> None:
    text = prereg.read_text(encoding="utf-8")
    missing = [s for s in REQUIRED_PREREG_SECTIONS if s not in text]
    if missing:
        raise PreregError(f"prereg '{prereg}' is missing required sections: {missing}")


@dataclass
class RunContext:
    results_dir: Path
    record: dict

    def save_raw_completions(self, cell: str, completions: list[dict]) -> Path:
        """Retain raw model outputs for a result cell (LAW). One jsonl per cell."""
        raw_dir = self.results_dir / "raw_completions"
        raw_dir.mkdir(exist_ok=True)
        path = raw_dir / f"{cell}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            for item in completions:
                f.write(json.dumps(item) + "\n")
        return path

    def finalize(self, **extra) -> None:
        self.record.update(extra)
        self.record["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
        (self.results_dir / "run.json").write_text(json.dumps(self.record, indent=2))


def start_run(
    repo_root: Path,
    config_path: Path,
    results_root: Path,
    run_name: str,
    prereg_path: Path | None = None,
    post_hoc: bool = False,
) -> RunContext:
    """The only entry point to a results directory. Enforces the run LAWs."""
    repo_root = Path(repo_root)

    if git_is_dirty(repo_root):
        raise DirtyTreeError(
            "LAW violation: working tree is dirty; commit the experiment "
            "(one commit per experiment) before running"
        )

    if post_hoc:
        prereg_commit_hash = None
    else:
        if prereg_path is None:
            raise PreregError(
                "LAW violation: no prereg supplied; pass prereg_path, or "
                "post_hoc=True to label this analysis post-hoc forever"
            )
        check_prereg_sections(prereg_path)
        prereg_commit_hash = prereg_commit(repo_root, prereg_path)

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = Path(results_root) / f"{stamp}-{run_name}"
    results_dir.mkdir(parents=True)

    # LAW: configs are copied into every results directory, unconditionally.
    config_path = Path(config_path)
    shutil.copy2(config_path, results_dir / config_path.name)

    record = {
        "run_name": run_name,
        "started": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_head(repo_root),
        "prereg_path": str(prereg_path) if prereg_path else None,
        "prereg_commit": prereg_commit_hash,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "post_hoc": post_hoc,
    }
    (results_dir / "run.json").write_text(json.dumps(record, indent=2))
    return RunContext(results_dir=results_dir, record=record)
