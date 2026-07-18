"""LAWs enforced here:

- A claims ledger (CLAIMS.md) tracks every scientific claim with status in
  {hypothesis, preliminary, verified, withdrawn} and the evidence commit.
  The paper may only contain claims at "verified".
- Human verification gate between "results exist" and "results are claimed":
  promotion to `verified` is blocked unless LABNOTES.md contains Ecaterina's
  verification line for the claim, and the cited results directory holds
  >= MIN_RAW_PER_CELL raw completions per headline cell.

The machine can only block promotion; the reading and hand re-derivation
are Ecaterina's by definition.
"""

from __future__ import annotations

import re
from pathlib import Path

from jtvec.validators.results_dirs import check_results_dir, count_raw_per_cell

STATUSES = {"hypothesis", "preliminary", "verified", "withdrawn"}
REQUIRED_FIELDS = (
    "status",
    "statement",
    "scope",
    "evidence-commit",
    "prereg",
    "results-dir",
    "raw-completions",
    "verified-by",
)
MIN_RAW_PER_CELL = 20
ID_RE = re.compile(r"^CLM-\d{3,}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
TEMPLATE_ID = "CLM-000"


def parse_claims(claims_md: Path) -> dict[str, dict[str, str]]:
    """Parse `### CLM-xxx` entries with `- field: value` lines."""
    entries: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for line in claims_md.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^### (\S+)", line)
        if heading:
            current = {}
            entries[heading.group(1)] = current
            continue
        field = re.match(r"^- ([a-z-]+):\s*(.*)$", line)
        if field and current is not None:
            current[field.group(1)] = field.group(2).strip()
    entries.pop(TEMPLATE_ID, None)
    return entries


def _verification_line(labnotes_text: str, claim_id: str) -> bool:
    pattern = re.compile(
        rf"verify:\s*{claim_id}\b.*verified-by:\s*Ecaterina.*date:\s*\d{{4}}-\d{{2}}-\d{{2}}"
    )
    return any(pattern.search(line) for line in labnotes_text.splitlines())


def check_claims(repo_root: Path) -> list[str]:
    """Return violations for the claims ledger (empty list = pass)."""
    repo_root = Path(repo_root)
    claims_md = repo_root / "CLAIMS.md"
    labnotes = repo_root / "LABNOTES.md"
    if not claims_md.is_file():
        return ["CLAIMS.md missing"]
    if not labnotes.is_file():
        return ["LABNOTES.md missing"]

    violations: list[str] = []
    labnotes_text = labnotes.read_text(encoding="utf-8")

    for claim_id, fields in parse_claims(claims_md).items():
        where = f"CLAIMS.md {claim_id}"
        if not ID_RE.match(claim_id):
            violations.append(f"{where}: id does not match CLM-NNN")
            continue
        missing = [f for f in REQUIRED_FIELDS if f not in fields]
        if missing:
            violations.append(f"{where}: missing fields {missing}")
            continue
        status = fields["status"]
        if status not in STATUSES:
            violations.append(f"{where}: status '{status}' not in {sorted(STATUSES)}")
            continue

        if status in ("preliminary", "verified"):
            results_dir = repo_root / fields["results-dir"]
            violations.extend(f"{where}: {v}" for v in check_results_dir(results_dir))

        if status == "verified":
            if not COMMIT_RE.match(fields["evidence-commit"]):
                violations.append(
                    f"{where}: verified without a valid evidence-commit "
                    f"('{fields['evidence-commit']}')"
                )
            if not _verification_line(labnotes_text, claim_id):
                violations.append(
                    f"{where}: verified without Ecaterina's LABNOTES line "
                    f"('verify: {claim_id} ... verified-by: Ecaterina ... date: YYYY-MM-DD')"
                )
            counts = count_raw_per_cell(repo_root / fields["results-dir"])
            if not counts:
                violations.append(f"{where}: verified but no raw completion cells found")
            for cell, n in counts.items():
                if n < MIN_RAW_PER_CELL:
                    violations.append(
                        f"{where}: cell '{cell}' has {n} raw completions "
                        f"(< {MIN_RAW_PER_CELL})"
                    )
    return violations
