"""jtvec: v2 measurement stack.

Package layout (build order per project brief):
- core/    enforcement primitives for the CONSTRAINTS.md LAWs (M0)
- lens/    J-lens fitting + 9-check sanity gate (M1, vendored from v1)
- fv/      FV extraction behind the mandatory stability gate (M2)
- interv/  ablate / inject / swap with auto-generated sham twins (M3)
- harness/ task battery, preregistration, claims ledger tooling (M0-M4)
"""
