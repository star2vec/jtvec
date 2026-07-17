"""LAW: every estimator passes a stability gate before its outputs are used:
demonstrate convergence of the estimate under re-sampling at the chosen
sample size, on the target model, before the first scientific number.

`CertifiedArtifact` is the base class for every estimator output (the FV
object in fv/ subclasses it in M2). It cannot be constructed without a valid
`GateCertificate`, and a certificate cannot be constructed without pointing
at an existing convergence-study results directory. An ungated FV is
unrepresentable in the API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jtvec.core.draws import MIN_DRAWS


class UngatedEstimatorError(RuntimeError):
    """Raised when an estimator output is built without a stability gate."""


@dataclass(frozen=True)
class GateCertificate:
    """Proof that an estimator's stability gate passed.

    estimator:    e.g. "fv_todd", "fv_hendel"
    model:        e.g. "EleutherAI/pythia-410m@9879c9b"
    converged_at: the sample size (e.g. AIE trial count) at which convergence
                  was demonstrated
    n_draws:      independent draws used in the convergence study
    evidence_run: results directory of the convergence study; must exist
    issued:       ISO date the gate passed
    """

    estimator: str
    model: str
    converged_at: int
    n_draws: int
    evidence_run: str
    issued: str

    def __post_init__(self) -> None:
        for field_name in ("estimator", "model", "evidence_run", "issued"):
            if not getattr(self, field_name):
                raise UngatedEstimatorError(f"certificate field '{field_name}' is empty")
        if self.converged_at <= 0:
            raise UngatedEstimatorError(
                f"converged_at={self.converged_at}; a gate certificate needs the "
                "sample size at which convergence was demonstrated"
            )
        if self.n_draws < MIN_DRAWS:
            raise UngatedEstimatorError(
                f"n_draws={self.n_draws} < {MIN_DRAWS}; the convergence study "
                "itself is subject to the minimum-draws LAW"
            )
        if not Path(self.evidence_run).exists():
            raise UngatedEstimatorError(
                f"evidence_run '{self.evidence_run}' does not exist; a certificate "
                "must point at the convergence-study results directory"
            )


class CertifiedArtifact:
    """Base class for estimator outputs. Construction requires a certificate."""

    def __init__(self, certificate: GateCertificate) -> None:
        if not isinstance(certificate, GateCertificate):
            raise UngatedEstimatorError(
                "LAW violation: estimator output constructed without a "
                "GateCertificate; run the stability gate first"
            )
        self.certificate = certificate
