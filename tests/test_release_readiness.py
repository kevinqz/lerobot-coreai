# test_release_readiness.py — release readiness evaluation (v0.9.3).

import json
from importlib.resources import files

import jsonschema

from lerobot_coreai.operator_approval import ApprovalConfig, approve_bundle
from lerobot_coreai.release_readiness import evaluate_release_readiness


def _approve(tmp_path, bundle, **over):
    base = dict(bundle_dir=bundle, operator="Kevin Saltarelli",
                attest_not_physical_safety=True, attest_not_unrestricted_actuation=True)
    base.update(over)
    manifest = approve_bundle(ApprovalConfig(**base))
    path = tmp_path / "approval_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _schema():
    return json.loads(files("lerobot_coreai.schemas").joinpath(
        "release-readiness-report.schema.json").read_text())


def test_valid_bundle_and_approval_ready(sim_evidence_bundle, tmp_path):
    bundle = sim_evidence_bundle()
    approval = _approve(tmp_path, bundle)
    result = evaluate_release_readiness(bundle, approval)
    assert result.ready
    assert result.report["claims"]["proves_release_readiness_for_scope"] is True
    jsonschema.validate(result.report, _schema())


def test_missing_approval_not_ready(sim_evidence_bundle, tmp_path):
    bundle = sim_evidence_bundle()
    result = evaluate_release_readiness(bundle, tmp_path / "nope.json")
    assert not result.ready
    assert result.report["claims"]["proves_release_readiness_for_scope"] is False


def test_expired_approval_not_ready(sim_evidence_bundle, tmp_path):
    bundle = sim_evidence_bundle()
    manifest = approve_bundle(ApprovalConfig(
        bundle_dir=bundle, operator="K", attest_not_physical_safety=True,
        attest_not_unrestricted_actuation=True))
    manifest["expires_at"] = "2000-01-01T00:00:00Z"
    approval = tmp_path / "approval_manifest.json"
    approval.write_text(json.dumps(manifest))
    result = evaluate_release_readiness(bundle, approval)
    assert not result.ready


def test_tampered_bundle_not_ready(sim_evidence_bundle, tmp_path):
    bundle = sim_evidence_bundle()
    approval = _approve(tmp_path, bundle)
    (bundle / "source_run" / "sim_report.json").write_text('{"x": 1}')
    result = evaluate_release_readiness(bundle, approval)
    assert not result.ready


def test_report_claims_do_not_overclaim(sim_evidence_bundle, tmp_path):
    bundle = sim_evidence_bundle()
    approval = _approve(tmp_path, bundle)
    result = evaluate_release_readiness(bundle, approval)
    claims = result.report["claims"]
    assert claims["proves_physical_safety"] is False
    assert claims["proves_real_world_safety"] is False
    assert claims["authorizes_unrestricted_real_world_actuation"] is False


def test_missing_regression_approval_not_release_ready(sim_evidence_bundle, tmp_path):
    # An operator may WAIVE regression at approval time, but release-readiness
    # (the go/no-go) must still block by default.
    bundle = sim_evidence_bundle(with_regression=False)
    manifest = approve_bundle(ApprovalConfig(
        bundle_dir=bundle, operator="K", attest_not_physical_safety=True,
        attest_not_unrestricted_actuation=True, allow_missing_regression=True,
        allow_warnings=True))
    approval = tmp_path / "approval_manifest.json"
    approval.write_text(json.dumps(manifest))
    result = evaluate_release_readiness(bundle, approval)
    assert result.ready is False
    assert any("safety_regression" in b for b in result.blocking_failures)


def test_missing_regression_waived_readiness_with_explicit_flag(sim_evidence_bundle, tmp_path):
    # Only with BOTH the approval waiver AND the explicit readiness flag does the
    # missing regression downgrade to a warning.
    bundle = sim_evidence_bundle(with_regression=False)
    manifest = approve_bundle(ApprovalConfig(
        bundle_dir=bundle, operator="K", attest_not_physical_safety=True,
        attest_not_unrestricted_actuation=True, allow_missing_regression=True,
        allow_warnings=True))
    approval = tmp_path / "approval_manifest.json"
    approval.write_text(json.dumps(manifest))
    result = evaluate_release_readiness(bundle, approval, allow_missing_regression=True)
    assert result.ready is True
    assert any("waived" in w for w in result.warnings)


def test_not_ready_when_quality_failed(sim_evidence_bundle, tmp_path):
    # A bundle whose safety quality failed can't be approved; readiness must be false
    # even if we hand-craft an approval pointing at it.
    bundle = sim_evidence_bundle(passed_quality=False)
    # Build an (invalid) approval by overriding checks off — but readiness re-checks
    # evidence independently, so it stays not-ready.
    manifest = approve_bundle(ApprovalConfig(
        bundle_dir=bundle, operator="K", attest_not_physical_safety=True,
        attest_not_unrestricted_actuation=True, require_safety_quality_passed=False,
        allow_warnings=True))
    approval = tmp_path / "approval_manifest.json"
    approval.write_text(json.dumps(manifest))
    result = evaluate_release_readiness(bundle, approval)
    assert not result.ready
