# test_operator_approval.py — operator approval protocol (v0.9.3).

import json
from importlib.resources import files

import jsonschema
import pytest

from lerobot_coreai.errors import CoreAIPolicyError
from lerobot_coreai.operator_approval import (
    ApprovalConfig,
    approve_bundle,
    build_approval_request,
    verify_approval,
)


def _cfg(bundle, **over):
    base = dict(bundle_dir=bundle, operator="Kevin Saltarelli",
                attest_not_physical_safety=True, attest_not_unrestricted_actuation=True)
    base.update(over)
    return ApprovalConfig(**base)


def _schema():
    return json.loads(files("lerobot_coreai.schemas").joinpath(
        "operator-approval.schema.json").read_text())


class TestApprovalRequest:
    def test_complete_bundle_request_ok(self, sim_evidence_bundle):
        bundle = sim_evidence_bundle()
        req = build_approval_request(ApprovalConfig(bundle_dir=bundle))
        assert req.ok
        assert not [c for c in req.checks if c.severity == "required" and not c.passed]

    def test_missing_bundle_manifest_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(CoreAIPolicyError, match="Bundle manifest not found"):
            build_approval_request(ApprovalConfig(bundle_dir=empty))

    def test_missing_safety_quality_blocks(self, sim_evidence_bundle, tmp_path):
        bundle = sim_evidence_bundle()
        # Remove the safety quality report from the bundle.
        (bundle / "source_run" / "safety_quality_report.json").unlink()
        req = build_approval_request(ApprovalConfig(bundle_dir=bundle))
        assert not req.ok
        assert any(c.name == "safety_quality_report_exists" and not c.passed for c in req.checks)

    def test_failed_safety_quality_blocks(self, sim_evidence_bundle):
        bundle = sim_evidence_bundle(passed_quality=False)
        req = build_approval_request(ApprovalConfig(bundle_dir=bundle))
        assert any(c.name == "safety_quality_passed" and not c.passed for c in req.checks)

    def test_profile_calibration_overclaim_blocks_approval(self, sim_evidence_bundle):
        # A physical-safety overclaim in a profile artifact must block approval.
        bundle = sim_evidence_bundle()
        p = bundle / "source_run" / "calibrated_profile.json"
        obj = json.loads(p.read_text())
        obj["claims"] = {"proves_physical_safety": True}
        p.write_text(json.dumps(obj))
        req = build_approval_request(ApprovalConfig(bundle_dir=bundle))
        assert not req.ok
        assert any(c.name == "no_physical_safety_overclaim" and not c.passed
                   for c in req.checks)

    def test_missing_regression_blocks_by_default(self, sim_evidence_bundle):
        bundle = sim_evidence_bundle(with_regression=False)
        req = build_approval_request(ApprovalConfig(bundle_dir=bundle))
        assert any(c.name == "safety_regression_report_exists" and not c.passed
                   for c in req.checks)

    def test_missing_regression_allowed_with_override(self, sim_evidence_bundle):
        bundle = sim_evidence_bundle(with_regression=False)
        req = build_approval_request(ApprovalConfig(
            bundle_dir=bundle, allow_missing_regression=True, allow_warnings=True))
        assert any("regression report missing but allowed" in w for w in req.warnings)

    def test_request_separates_required_and_warnings(self, sim_evidence_bundle):
        # v0.9.4: required_checks_passed and warnings_present are distinct.
        bundle = sim_evidence_bundle(with_regression=False)
        req = build_approval_request(ApprovalConfig(
            bundle_dir=bundle, allow_missing_regression=True, allow_warnings=False))
        # Required checks pass (regression waived), but a warning is present, so
        # ok is False while required_checks_passed is True.
        assert req.required_checks_passed is True
        assert req.warnings_present is True
        assert req.ok is False


class TestApproveBundle:
    def test_approve_writes_valid_manifest(self, sim_evidence_bundle):
        bundle = sim_evidence_bundle()
        manifest = approve_bundle(_cfg(bundle))
        assert manifest["approved"] is True
        assert manifest["approved_by"] == "Kevin Saltarelli"
        assert manifest["operator_attestation"]["accepted"] is True
        assert manifest["artifacts"]  # hashes bound
        assert "sha256:" in manifest["bundle"]["manifest_sha256"]
        jsonschema.validate(manifest, _schema())

    def test_approve_requires_operator(self, sim_evidence_bundle):
        bundle = sim_evidence_bundle()
        with pytest.raises(CoreAIPolicyError, match="requires --operator"):
            approve_bundle(_cfg(bundle, operator=None))

    def test_approve_requires_attestation(self, sim_evidence_bundle):
        bundle = sim_evidence_bundle()
        with pytest.raises(CoreAIPolicyError, match="explicit attestation"):
            approve_bundle(_cfg(bundle, attest_not_physical_safety=False))

    def test_approve_blocks_on_failed_quality(self, sim_evidence_bundle):
        bundle = sim_evidence_bundle(passed_quality=False)
        with pytest.raises(CoreAIPolicyError, match="required checks failed"):
            approve_bundle(_cfg(bundle))

    def test_approve_blocks_missing_regression(self, sim_evidence_bundle):
        bundle = sim_evidence_bundle(with_regression=False)
        with pytest.raises(CoreAIPolicyError):
            approve_bundle(_cfg(bundle))

    def test_approve_allows_missing_regression_override(self, sim_evidence_bundle):
        bundle = sim_evidence_bundle(with_regression=False)
        manifest = approve_bundle(_cfg(bundle, allow_missing_regression=True,
                                       allow_warnings=True))
        assert manifest["approved"] is True
        assert manifest["operator_overrides"]["allow_missing_regression"] is True

    def test_approve_bad_scope(self, sim_evidence_bundle):
        bundle = sim_evidence_bundle()
        with pytest.raises(CoreAIPolicyError, match="Unknown approval_scope"):
            approve_bundle(_cfg(bundle, approval_scope="real_full_send"))

    def test_manifest_claims_honest(self, sim_evidence_bundle):
        bundle = sim_evidence_bundle()
        manifest = approve_bundle(_cfg(bundle))
        assert manifest["claims"]["proves_physical_safety"] is False
        assert manifest["claims"]["authorizes_unrestricted_real_world_actuation"] is False

    def test_approval_does_not_mutate_bundle(self, sim_evidence_bundle):
        from lerobot_coreai.sim_bundle import verify_sim_bundle
        bundle = sim_evidence_bundle()
        approve_bundle(_cfg(bundle))
        # Bundle still verifies (approval written elsewhere, not into the bundle).
        assert verify_sim_bundle(bundle).ok
        assert not (bundle / "approval_manifest.json").exists()


def _write_approval(tmp_path, bundle, **over):
    manifest = approve_bundle(_cfg(bundle, **over))
    path = tmp_path / "approval_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


class TestVerifyApproval:
    def test_valid_approval_verifies(self, sim_evidence_bundle, tmp_path):
        bundle = sim_evidence_bundle()
        approval = _write_approval(tmp_path, bundle)
        result = verify_approval(bundle, approval)
        assert result.approval_valid
        assert not result.expired

    def test_missing_approval_fails(self, sim_evidence_bundle, tmp_path):
        bundle = sim_evidence_bundle()
        result = verify_approval(bundle, tmp_path / "nope.json")
        assert not result.approval_valid

    def test_expired_approval_fails(self, sim_evidence_bundle, tmp_path):
        bundle = sim_evidence_bundle()
        manifest = approve_bundle(_cfg(bundle))
        manifest["expires_at"] = "2000-01-01T00:00:00Z"  # in the past
        approval = tmp_path / "approval_manifest.json"
        approval.write_text(json.dumps(manifest))
        result = verify_approval(bundle, approval)
        assert result.expired
        assert not result.approval_valid

    def test_tampered_artifact_fails(self, sim_evidence_bundle, tmp_path):
        bundle = sim_evidence_bundle()
        approval = _write_approval(tmp_path, bundle)
        # Tamper with a bundled artifact after approval.
        (bundle / "source_run" / "safety_summary.json").write_text('{"tampered": true}')
        result = verify_approval(bundle, approval)
        assert not result.checksum_matches
        assert not result.approval_valid

    def test_tampered_manifest_hash_fails(self, sim_evidence_bundle, tmp_path):
        bundle = sim_evidence_bundle()
        approval = _write_approval(tmp_path, bundle)
        (bundle / "bundle_manifest.json").write_text('{"changed": true}')
        result = verify_approval(bundle, approval)
        assert not result.approval_valid

    def test_reruns_required_checks_after_artifact_removed(self, sim_evidence_bundle, tmp_path):
        # Approve, then remove a required artifact from the bundle. verify must
        # fail because it re-runs the required checks (not just hash matching).
        bundle = sim_evidence_bundle()
        approval = _write_approval(tmp_path, bundle)
        (bundle / "source_run" / "safety_quality_report.json").unlink()
        result = verify_approval(bundle, approval)
        assert not result.approval_valid
        assert any(c.name == "approval_required_checks_still_pass" and not c.passed
                   for c in result.checks)

    def test_rejects_forged_manifest_missing_artifact_bindings(self, sim_evidence_bundle, tmp_path):
        # A hand-crafted manifest with empty artifacts must not verify even if
        # its own (few) hashes are internally consistent.
        bundle = sim_evidence_bundle()
        manifest = approve_bundle(_cfg(bundle))
        manifest["artifacts"] = {}
        approval = tmp_path / "approval_manifest.json"
        approval.write_text(json.dumps(manifest))
        result = verify_approval(bundle, approval)
        assert not result.approval_valid
        assert any(c.name == "required_artifacts_bound" and not c.passed
                   for c in result.checks)

    def test_overclaim_approval_fails(self, sim_evidence_bundle, tmp_path):
        bundle = sim_evidence_bundle()
        manifest = approve_bundle(_cfg(bundle))
        manifest["claims"]["proves_physical_safety"] = True
        approval = tmp_path / "approval_manifest.json"
        approval.write_text(json.dumps(manifest))
        result = verify_approval(bundle, approval)
        assert not result.approval_valid
