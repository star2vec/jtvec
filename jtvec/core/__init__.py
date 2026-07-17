"""Enforcement primitives for the CONSTRAINTS.md LAWs.

Every module here is the mechanical enforcement point for one LAW; the
docstring of each module quotes the LAW it enforces. Scientific code in
lens/, fv/, and interv/ must build on these types - they are designed so
that a LAW-violating result is unrepresentable, not merely discouraged.
"""
