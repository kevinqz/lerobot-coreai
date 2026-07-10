# test_release_governance.py — release channel governance (v1.2.1).

import json

from lerobot_coreai.release_governance import (
    CHANNELS, ReleasePolicy, default_policy, evaluate_release,
)


def _bundle(tmp_path, *, overclaim=False, secret=False, real_session=False,
            external_http=False, reports=None):
    d = tmp_path / "bundle"
    (d / "reports").mkdir(parents=True)
    reports = reports if reports is not None else [
        "lerobot_compatibility_report.json", "lerobot_bridge_report.json",
        "lerobot_feature_mapping.json", "lerobot_eval_v2_report.json",
        "obs_bridge_report.json"]
    for r in reports:
        payload = {"ok": True, "claims": {"proves_physical_safety": overclaim}}
        if secret:
            payload["auth"] = {"token": "supersecretvalue123"}
        if external_http:
            payload["robot_adapter"] = "external-http"
        (d / "reports" / r).write_text(json.dumps(payload))
    if real_session:
        (d / "real_report.json").write_text(json.dumps({"ok": True}))
    return d


def test_all_channels_have_default_policy():
    for ch in CHANNELS:
        assert default_policy(ch).channel == ch


def test_internal_channel_passes_clean_bundle(tmp_path):
    report = evaluate_release(_bundle(tmp_path), artifact_type="bridge_benchmark",
                              policy=default_policy("internal"))
    assert report["ok"] is True


def test_public_demo_requires_signature(tmp_path):
    report = evaluate_release(_bundle(tmp_path), artifact_type="bridge_benchmark",
                              policy=default_policy("public-demo"))
    assert report["ok"] is False
    names = {c["name"]: c["passed"] for c in report["checks"]}
    assert names["signature_valid"] is False


def test_public_demo_rejects_real_session_artifacts(tmp_path):
    policy = ReleasePolicy("public-demo", allow_real_session_artifacts=False,
                           require_signature=False)
    report = evaluate_release(_bundle(tmp_path, real_session=True),
                              artifact_type="bridge_benchmark", policy=policy)
    names = {c["name"]: c["passed"] for c in report["checks"]}
    assert names["no_real_session_artifacts"] is False
    assert report["ok"] is False


def test_public_demo_rejects_external_http(tmp_path):
    policy = ReleasePolicy("public-demo", allow_external_http_artifacts=False,
                           require_signature=False)
    report = evaluate_release(_bundle(tmp_path, external_http=True),
                              artifact_type="bridge_benchmark", policy=policy)
    names = {c["name"]: c["passed"] for c in report["checks"]}
    assert names["no_external_http_artifacts"] is False


def test_overclaim_blocks_release(tmp_path):
    report = evaluate_release(_bundle(tmp_path, overclaim=True),
                              artifact_type="bridge_benchmark",
                              policy=default_policy("internal"))
    names = {c["name"]: c["passed"] for c in report["checks"]}
    assert names["no_overclaims"] is False
    assert report["ok"] is False


def test_raw_secret_blocks_release(tmp_path):
    report = evaluate_release(_bundle(tmp_path, secret=True),
                              artifact_type="bridge_benchmark",
                              policy=default_policy("internal"))
    names = {c["name"]: c["passed"] for c in report["checks"]}
    assert names["no_raw_secrets"] is False


def test_redacted_and_fingerprint_not_flagged_as_secret(tmp_path):
    d = tmp_path / "b"
    (d / "reports").mkdir(parents=True)
    (d / "reports" / "r.json").write_text(json.dumps({
        "auth": {"token_sha256_prefix": "sha256:abcd", "token_env": "MY_ENV",
                 "operator": "<redacted>"}}))
    report = evaluate_release(d, artifact_type="x", policy=default_policy("internal"))
    names = {c["name"]: c["passed"] for c in report["checks"]}
    assert names["no_raw_secrets"] is True


def test_guarded_real_evidence_requires_full_chain(tmp_path):
    d = _bundle(tmp_path, reports=["lerobot_compatibility_report.json"])
    report = evaluate_release(d, artifact_type="bridge_benchmark",
                              policy=ReleasePolicy("guarded-real-evidence",
                                                   require_guarded_real_evidence=True,
                                                   require_signature=False))
    names = {c["name"]: c["passed"] for c in report["checks"]}
    assert names["guarded_real_evidence_complete"] is False


def test_report_claims_honest(tmp_path):
    report = evaluate_release(_bundle(tmp_path), artifact_type="bridge_benchmark",
                              policy=default_policy("internal"))
    assert report["claims"]["proves_physical_safety"] is False
    assert report["claims"]["authorizes_robot_actuation"] is False
