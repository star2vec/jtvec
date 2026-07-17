"""CONSTRAINTS.md tier discipline: HYPOTHESIS-tier statements must NEVER be
stated as findings in code comments, logs, drafts, or reports until v2
confirms them.

Mechanical rule: a line in prose or source that uses one of the key phrases
below (derived from the seven HYPOTHESIS entries in CONSTRAINTS.md) must
carry the literal tag "HYPOTHESIS" on the same line. CONSTRAINTS.md itself
is exempt (it is the ledger that defines the entries).
"""

from __future__ import annotations

import re
from pathlib import Path

from jtvec.validators.language import prose_files

HYPOTHESIS_PHRASES = re.compile(
    r"(causally dissociat|causally separable|double dissociation"
    r"|invisible to the logit lens|matures early|emerges? late"
    r"|confabulat|steerable-but-not|made to report a task"
    r"|raises? report accuracy)",
    re.IGNORECASE,
)
TAG = "HYPOTHESIS"


def _source_files(repo_root: Path) -> list[Path]:
    # this module is exempt like CONSTRAINTS.md: it defines the patterns
    return sorted(
        p for p in Path(repo_root).glob("jtvec/**/*.py") if p.name != "hypotheses.py"
    )


def check_hypotheses(repo_root: Path) -> list[str]:
    """Return violations (empty list = pass)."""
    violations: list[str] = []
    for path in prose_files(repo_root) + _source_files(repo_root):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            match = HYPOTHESIS_PHRASES.search(line)
            if match and TAG not in line:
                violations.append(
                    f"{path.relative_to(repo_root)}:{lineno}: hypothesis-tier "
                    f"phrase '{match.group(0)}' without the {TAG} tag"
                )
    return violations
