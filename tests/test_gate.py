import pytest

from jtvec.core.gate import CertifiedArtifact, GateCertificate, UngatedEstimatorError


def make_cert(tmp_path, **overrides):
    evidence = tmp_path / "results" / "fv_convergence"
    evidence.mkdir(parents=True, exist_ok=True)
    fields = dict(
        estimator="fv_todd",
        model="EleutherAI/pythia-410m@9879c9b",
        converged_at=200,
        n_draws=3,
        evidence_run=str(evidence),
        issued="2026-07-17",
    )
    fields.update(overrides)
    return GateCertificate(**fields)


def test_artifact_without_certificate_is_unrepresentable():
    with pytest.raises(UngatedEstimatorError):
        CertifiedArtifact(certificate=None)


def test_certificate_requires_existing_evidence_run(tmp_path):
    with pytest.raises(UngatedEstimatorError):
        make_cert(tmp_path, evidence_run=str(tmp_path / "does-not-exist"))


def test_certificate_requires_min_draws(tmp_path):
    with pytest.raises(UngatedEstimatorError):
        make_cert(tmp_path, n_draws=2)


def test_certificate_requires_positive_convergence_point(tmp_path):
    with pytest.raises(UngatedEstimatorError):
        make_cert(tmp_path, converged_at=0)


def test_valid_certificate_admits_artifact(tmp_path):
    artifact = CertifiedArtifact(certificate=make_cert(tmp_path))
    assert artifact.certificate.converged_at == 200
