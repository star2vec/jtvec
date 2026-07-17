import json

from jtvec.validators.claims import check_claims
from jtvec.validators.hypotheses import check_hypotheses
from jtvec.validators.language import check_language
from jtvec.validators.results_dirs import check_results_dir

CLAIMS_HEADER = "# CLAIMS\n\n## Claims\n\n"


def make_results_dir(root, n_raw=25):
    rd = root / "results" / "run1"
    (rd / "raw_completions").mkdir(parents=True)
    (rd / "config.yaml").write_text("model: test\n")
    record = {
        "run_name": "t",
        "started": "2026-07-17T12:00:00",
        "git_commit": "a" * 40,
        "prereg_path": "prereg.md",
        "prereg_commit": "b" * 40,
        "config_sha256": "c" * 64,
        "post_hoc": False,
    }
    (rd / "run.json").write_text(json.dumps(record))
    lines = "\n".join(json.dumps({"completion": str(i)}) for i in range(n_raw))
    (rd / "raw_completions" / "headline.jsonl").write_text(lines + "\n")
    return rd


def claim_entry(claim_id="CLM-001", status="verified", **overrides):
    fields = {
        "status": status,
        "statement": "on test-model, config c, N=25, effect observed",
        "scope": "test-model, config c, N=25",
        "evidence-commit": "d" * 40,
        "prereg": "prereg.md",
        "results-dir": "results/run1",
        "raw-completions": "results/run1/raw_completions",
        "verified-by": "LABNOTES 2026-07-17",
    }
    fields.update(overrides)
    lines = "\n".join(f"- {k}: {v}" for k, v in fields.items())
    return f"### {claim_id}\n{lines}\n"


def test_results_dir_check_catches_missing_pieces(tmp_path):
    violations = check_results_dir(tmp_path / "nope")
    assert violations
    rd = make_results_dir(tmp_path)
    assert check_results_dir(rd) == []
    (rd / "run.json").unlink()
    assert any("run.json" in v for v in check_results_dir(rd))


def test_verified_claim_without_labnotes_line_blocked(tmp_path):
    make_results_dir(tmp_path)
    (tmp_path / "CLAIMS.md").write_text(CLAIMS_HEADER + claim_entry())
    (tmp_path / "LABNOTES.md").write_text("# LABNOTES\n\nno verification here\n")
    violations = check_claims(tmp_path)
    assert any("verified-by: Ecaterina" in v for v in violations)


def test_verified_claim_with_labnotes_line_passes(tmp_path):
    make_results_dir(tmp_path)
    (tmp_path / "CLAIMS.md").write_text(CLAIMS_HEADER + claim_entry())
    (tmp_path / "LABNOTES.md").write_text(
        "# LABNOTES\n\nverify: CLM-001 raw-read: 25 re-derived: yes "
        "verified-by: Ecaterina date: 2026-07-17\n"
    )
    assert check_claims(tmp_path) == []


def test_verified_claim_with_too_few_raw_completions_blocked(tmp_path):
    make_results_dir(tmp_path, n_raw=10)
    (tmp_path / "CLAIMS.md").write_text(CLAIMS_HEADER + claim_entry())
    (tmp_path / "LABNOTES.md").write_text(
        "verify: CLM-001 verified-by: Ecaterina date: 2026-07-17\n"
    )
    violations = check_claims(tmp_path)
    assert any("raw completions" in v for v in violations)


def test_hypothesis_status_claim_needs_no_evidence(tmp_path):
    (tmp_path / "CLAIMS.md").write_text(
        CLAIMS_HEADER
        + claim_entry(status="hypothesis", **{"evidence-commit": "none",
                                              "results-dir": "none",
                                              "verified-by": "none"})
    )
    (tmp_path / "LABNOTES.md").write_text("# LABNOTES\n")
    assert check_claims(tmp_path) == []


def test_bad_status_rejected(tmp_path):
    (tmp_path / "CLAIMS.md").write_text(CLAIMS_HEADER + claim_entry(status="proven"))
    (tmp_path / "LABNOTES.md").write_text("# LABNOTES\n")
    assert any("status" in v for v in check_claims(tmp_path))


def test_language_lint_flags_unscoped_claims(tmp_path):
    (tmp_path / "LABNOTES.md").write_text("This proves the effect is real.\n")
    violations = check_language(tmp_path)
    assert len(violations) == 1
    (tmp_path / "LABNOTES.md").write_text(
        "This proves the effect [VERIFIED: swap-ceiling entry].\n"
    )
    assert check_language(tmp_path) == []


def test_language_lint_accepts_waiver(tmp_path):
    (tmp_path / "LABNOTES.md").write_text(
        "quoting the LAW about 'shows' here <!-- lint-ok: quotation -->\n"
    )
    assert check_language(tmp_path) == []


def test_hypothesis_phrases_require_tag(tmp_path):
    (tmp_path / "LABNOTES.md").write_text(
        "execution and verbalization causally dissociate on this task.\n"
    )
    violations = check_hypotheses(tmp_path)
    assert len(violations) == 1
    (tmp_path / "LABNOTES.md").write_text(
        "testing whether execution and verbalization causally dissociate "
        "(HYPOTHESIS tier).\n"
    )
    assert check_hypotheses(tmp_path) == []
