# test_cli_approval.py — CLI tests for approval + release readiness (v0.9.3).

import json

from lerobot_coreai import cli

_ATTEST = [
    "--i-understand-this-does-not-prove-physical-safety",
    "--i-understand-this-does-not-authorize-unrestricted-real-world-actuation",
]


def _approve(bundle, out, extra=None):
    return cli.main(["approve-bundle", "--bundle-dir", str(bundle),
                     "--operator", "Kevin Saltarelli", "--output-dir", str(out)]
                    + _ATTEST + (extra or []))


class TestApprovalRequest:
    def test_writes_files_rc0(self, sim_evidence_bundle, tmp_path):
        bundle = sim_evidence_bundle()
        out = tmp_path / "approvals"
        rc = cli.main(["approval-request", "--bundle-dir", str(bundle),
                       "--output-dir", str(out)])
        assert rc == 0
        assert (out / "approval_request.json").is_file()
        assert (out / "approval_checklist.md").is_file()

    def test_invalid_bundle_rc1(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert cli.main(["approval-request", "--bundle-dir", str(empty)]) == 1

    def test_json(self, sim_evidence_bundle, capsys):
        bundle = sim_evidence_bundle()
        rc = cli.main(["approval-request", "--bundle-dir", str(bundle), "--json"])
        assert rc == 0
        assert json.loads(capsys.readouterr().out)["ok"] is True


class TestApproveBundle:
    def test_writes_manifest_rc0(self, sim_evidence_bundle, tmp_path):
        bundle = sim_evidence_bundle()
        out = tmp_path / "approvals"
        assert _approve(bundle, out) == 0
        assert (out / "approval_manifest.json").is_file()
        manifest = json.loads((out / "approval_manifest.json").read_text())
        assert manifest["approved"] is True

    def test_missing_attestation_rc1(self, sim_evidence_bundle, tmp_path):
        bundle = sim_evidence_bundle()
        rc = cli.main(["approve-bundle", "--bundle-dir", str(bundle),
                       "--operator", "K", "--output-dir", str(tmp_path / "a")])
        assert rc == 1

    def test_failed_gate_rc1(self, sim_evidence_bundle, tmp_path):
        bundle = sim_evidence_bundle(passed_quality=False)
        assert _approve(bundle, tmp_path / "a") == 1

    def test_allow_missing_regression(self, sim_evidence_bundle, tmp_path):
        bundle = sim_evidence_bundle(with_regression=False)
        rc = _approve(bundle, tmp_path / "a",
                      extra=["--allow-missing-regression", "--allow-warnings"])
        assert rc == 0


class TestVerifyApproval:
    def test_valid_rc0(self, sim_evidence_bundle, tmp_path):
        bundle = sim_evidence_bundle()
        out = tmp_path / "a"
        _approve(bundle, out)
        rc = cli.main(["verify-approval", "--bundle-dir", str(bundle),
                       "--approval", str(out / "approval_manifest.json")])
        assert rc == 0

    def test_tampered_rc1(self, sim_evidence_bundle, tmp_path):
        bundle = sim_evidence_bundle()
        out = tmp_path / "a"
        _approve(bundle, out)
        (bundle / "source_run" / "safety_summary.json").write_text('{"x":1}')
        rc = cli.main(["verify-approval", "--bundle-dir", str(bundle),
                       "--approval", str(out / "approval_manifest.json")])
        assert rc == 1

    def test_json(self, sim_evidence_bundle, tmp_path, capsys):
        bundle = sim_evidence_bundle()
        out = tmp_path / "a"
        _approve(bundle, out)
        capsys.readouterr()
        rc = cli.main(["verify-approval", "--bundle-dir", str(bundle),
                       "--approval", str(out / "approval_manifest.json"), "--json"])
        assert rc == 0
        assert json.loads(capsys.readouterr().out)["ok"] is True


class TestReleaseReadiness:
    def test_ready_rc0(self, sim_evidence_bundle, tmp_path):
        bundle = sim_evidence_bundle()
        out = tmp_path / "a"
        _approve(bundle, out)
        rc = cli.main(["release-readiness", "--bundle-dir", str(bundle),
                       "--approval", str(out / "approval_manifest.json"),
                       "--output-dir", str(tmp_path / "readiness")])
        assert rc == 0
        assert (tmp_path / "readiness" / "release_readiness_report.json").is_file()
        assert (tmp_path / "readiness" / "release_readiness_report.md").is_file()

    def test_missing_regression_rc1_by_default(self, sim_evidence_bundle, tmp_path):
        bundle = sim_evidence_bundle(with_regression=False)
        out = tmp_path / "a"
        assert _approve(bundle, out,
                        extra=["--allow-missing-regression", "--allow-warnings"]) == 0
        approval = out / "approval_manifest.json"
        # Default readiness blocks the waived-missing regression.
        assert cli.main(["release-readiness", "--bundle-dir", str(bundle),
                         "--approval", str(approval)]) == 1
        # With the explicit readiness flag it becomes ready.
        assert cli.main(["release-readiness", "--bundle-dir", str(bundle),
                         "--approval", str(approval),
                         "--allow-missing-regression"]) == 0

    def test_not_ready_missing_approval_rc1(self, sim_evidence_bundle, tmp_path):
        bundle = sim_evidence_bundle()
        rc = cli.main(["release-readiness", "--bundle-dir", str(bundle),
                       "--approval", str(tmp_path / "nope.json")])
        assert rc == 1

    def test_json(self, sim_evidence_bundle, tmp_path, capsys):
        bundle = sim_evidence_bundle()
        out = tmp_path / "a"
        _approve(bundle, out)
        capsys.readouterr()
        rc = cli.main(["release-readiness", "--bundle-dir", str(bundle),
                       "--approval", str(out / "approval_manifest.json"), "--json"])
        assert rc == 0
        assert json.loads(capsys.readouterr().out)["ready"] is True


def test_full_pre_v1_workflow(sim_evidence_bundle, tmp_path):
    # request -> approve -> verify -> release-readiness end to end.
    bundle = sim_evidence_bundle()
    approvals = tmp_path / "approvals"
    assert cli.main(["approval-request", "--bundle-dir", str(bundle),
                     "--output-dir", str(approvals)]) == 0
    assert _approve(bundle, approvals) == 0
    approval = approvals / "approval_manifest.json"
    assert cli.main(["verify-approval", "--bundle-dir", str(bundle),
                     "--approval", str(approval)]) == 0
    assert cli.main(["release-readiness", "--bundle-dir", str(bundle),
                     "--approval", str(approval), "--output-dir", str(tmp_path / "r")]) == 0
