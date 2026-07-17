import json
import subprocess

import pytest

from jtvec.core.runctx import DirtyTreeError, PreregError, start_run

PREREG = """# Preregistration - EXP-test

## Hypothesis
(HYPOTHESIS tier) test hypothesis

## Decision rule
threshold decided before the run

## What counts as failure
the opposite result

## Estimator plan
3 draws, seeds 0,1,2

## Sample plan
N=100 per cell

## Resource estimate
2 min, 1 GB
"""


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "config.yaml").write_text("model: test\nseed: 0\n")
    (repo / "prereg.md").write_text(PREREG)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "experiment setup"], cwd=repo, check=True)
    return repo


def test_dirty_tree_refused(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "uncommitted.txt").write_text("x")
    with pytest.raises(DirtyTreeError):
        start_run(repo, repo / "config.yaml", repo / "results", "t", repo / "prereg.md")


def test_uncommitted_prereg_refused(tmp_path):
    repo = make_repo(tmp_path)
    late = repo / "late_prereg.md"
    late.write_text(PREREG)
    # file exists but was never committed -> the run must not start;
    # note the dirty-tree check alone would also catch this
    with pytest.raises((PreregError, DirtyTreeError)):
        start_run(repo, repo / "config.yaml", repo / "results", "t", late)


def test_prereg_missing_sections_refused(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "prereg.md").write_text("# Preregistration\n\n## Hypothesis\nonly this\n")
    subprocess.run(["git", "commit", "-aqm", "truncate prereg"], cwd=repo, check=True)
    with pytest.raises(PreregError):
        start_run(repo, repo / "config.yaml", repo / "results", "t", repo / "prereg.md")


def test_missing_prereg_requires_explicit_post_hoc(tmp_path):
    repo = make_repo(tmp_path)
    with pytest.raises(PreregError):
        start_run(repo, repo / "config.yaml", repo / "results", "t")


def test_post_hoc_is_stamped_forever(tmp_path):
    repo = make_repo(tmp_path)
    ctx = start_run(repo, repo / "config.yaml", repo / "results", "t", post_hoc=True)
    record = json.loads((ctx.results_dir / "run.json").read_text())
    assert record["post_hoc"] is True
    assert record["prereg_commit"] is None


def test_valid_run_copies_config_and_records_provenance(tmp_path):
    repo = make_repo(tmp_path)
    ctx = start_run(repo, repo / "config.yaml", repo / "results", "t", repo / "prereg.md")

    assert (ctx.results_dir / "config.yaml").read_text() == "model: test\nseed: 0\n"
    record = json.loads((ctx.results_dir / "run.json").read_text())
    assert len(record["git_commit"]) == 40
    assert len(record["prereg_commit"]) == 40
    assert record["post_hoc"] is False

    ctx.save_raw_completions("cell_a", [{"prompt": "p", "completion": "c"}] * 2)
    raw = (ctx.results_dir / "raw_completions" / "cell_a.jsonl").read_text()
    assert len(raw.splitlines()) == 2

    ctx.finalize(seeds=[0, 1, 2])
    record = json.loads((ctx.results_dir / "run.json").read_text())
    assert record["seeds"] == [0, 1, 2]
    assert "finished" in record
