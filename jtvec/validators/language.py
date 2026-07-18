"""LAW (lint half): the words "shows", "proves", "demonstrates" require a
VERIFIED-tier basis; observations are stated with scope, never as general
facts.

A lint cannot judge semantics, so the rule is mechanical: in prose files,
any line using a banned verb stem must cite its basis inline with
`[VERIFIED: <entry>]`, or carry an explicit waiver `<!-- lint-ok: <reason> -->`.
CONSTRAINTS.md (which defines the rule) and templates are exempt. The
hard-block half lives in jtvec/core/reporting.py.
"""

from __future__ import annotations

import re
from pathlib import Path

BANNED = re.compile(
    r"\b(shows?|showed|shown|proves?|proved|proven|demonstrates?|demonstrated"
    r"|the model can)\b",
    re.IGNORECASE,
)
ALLOWED = re.compile(r"\[VERIFIED:|<!-- lint-ok:")

PROSE_GLOBS = ("LABNOTES.md", "CLAIMS.md", "DRAFT*.md", "results/**/*.md")


def prose_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in PROSE_GLOBS:
        files.extend(Path(repo_root).glob(pattern))
    return sorted(set(files))


def check_language(repo_root: Path) -> list[str]:
    """Return violations (empty list = pass)."""
    violations: list[str] = []
    for path in prose_files(repo_root):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = BANNED.search(line)
            if match and not ALLOWED.search(line):
                violations.append(
                    f"{path.relative_to(repo_root)}:{lineno}: '{match.group(0)}' "
                    "without [VERIFIED: ...] basis or lint-ok waiver"
                )
    return violations
