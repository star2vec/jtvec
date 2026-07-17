"""Run every repo validator; nonzero exit on any violation. CI entry point."""

from __future__ import annotations

import sys
from pathlib import Path

from jtvec.validators.claims import check_claims
from jtvec.validators.hypotheses import check_hypotheses
from jtvec.validators.language import check_language


def main(repo_root: str | None = None) -> int:
    root = Path(repo_root) if repo_root else Path.cwd()
    checks = {
        "claims ledger": check_claims,
        "language discipline": check_language,
        "hypothesis-tier discipline": check_hypotheses,
    }
    failed = False
    for name, check in checks.items():
        violations = check(root)
        status = "PASS" if not violations else f"FAIL ({len(violations)})"
        print(f"[{status}] {name}")
        for v in violations:
            print(f"    {v}")
        failed = failed or bool(violations)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
