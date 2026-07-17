"""Repo-level validators, run by CI on every push (`python -m jtvec.validators`).

Each validator is the CI-side enforcement point of one LAW:
- claims.py:       claims ledger schema + human-verification gate (blocks
                   promotion to `verified` without Ecaterina's LABNOTES entry)
- results_dirs.py: every results dir cited by a claim has config, run record,
                   and raw completions on disk
- language.py:     lint half of the language-discipline LAW
- hypotheses.py:   HYPOTHESIS-tier statements never appear untagged
"""
