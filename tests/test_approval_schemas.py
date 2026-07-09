# test_approval_schemas.py — schema validation for approval/readiness (v0.9.3).

import json
from importlib.resources import files

import jsonschema
import pytest


def _schema(name):
    return json.loads(files("lerobot_coreai.schemas").joinpath(name).read_text())


def _valid_approval():
    return {
        "schema_version": "lerobot-coreai.operator_approval.v0",
        "lerobot_coreai_version": "0.9.3", "approval_id": "approval_x",
        "created_at": "2026-07-09T00:00:00Z", "approved": True,
        "approved_by": "K", "approval_scope": "sim_to_guarded_real_readiness",
        "expires_at": "2026-08-08T00:00:00Z",
        "bundle": {"manifest_sha256": "sha256:abc", "checksums_sha256": "sha256:def"},
        "artifacts": {}, "checks": [],
        "operator_attestation": {"text": "…", "accepted": True},
        "claims": {
            "proves_operator_reviewed_evidence": True,
            "proves_physical_safety": False, "proves_real_world_safety": False,
            "authorizes_unrestricted_real_world_actuation": False,
        },
    }


def test_valid_approval_passes():
    jsonschema.validate(_valid_approval(), _schema("operator-approval.schema.json"))


def test_approval_bad_scope_fails():
    a = _valid_approval()
    a["approval_scope"] = "real_full_send"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(a, _schema("operator-approval.schema.json"))


def test_approval_physical_overclaim_fails():
    a = _valid_approval()
    a["claims"]["proves_physical_safety"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(a, _schema("operator-approval.schema.json"))


def test_approval_actuation_overclaim_fails():
    a = _valid_approval()
    a["claims"]["authorizes_unrestricted_real_world_actuation"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(a, _schema("operator-approval.schema.json"))


def test_approval_missing_attestation_fails():
    a = _valid_approval()
    del a["operator_attestation"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(a, _schema("operator-approval.schema.json"))


def test_approved_manifest_requires_accepted_attestation():
    # v0.9.4: approved=true but attestation.accepted=false must fail schema.
    a = _valid_approval()
    a["operator_attestation"]["accepted"] = False
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(a, _schema("operator-approval.schema.json"))


def test_approved_manifest_requires_operator():
    a = _valid_approval()
    a["approved_by"] = None
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(a, _schema("operator-approval.schema.json"))


def test_bundle_requires_manifest_hash():
    a = _valid_approval()
    a["bundle"] = {}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(a, _schema("operator-approval.schema.json"))


def test_unapproved_draft_allows_unaccepted_attestation():
    # A draft (approved=false) is allowed to have accepted=false.
    a = _valid_approval()
    a["approved"] = False
    a["approved_by"] = None
    a["operator_attestation"]["accepted"] = False
    a["claims"]["proves_operator_reviewed_evidence"] = False
    jsonschema.validate(a, _schema("operator-approval.schema.json"))


def _valid_readiness():
    return {
        "schema_version": "lerobot-coreai.release_readiness_report.v0",
        "ready": True, "readiness_scope": "sim_to_guarded_real_readiness",
        "bundle": {}, "approval": {}, "evidence": {}, "blocking_failures": [], "warnings": [],
        "claims": {
            "proves_release_readiness_for_scope": True,
            "proves_physical_safety": False, "proves_real_world_safety": False,
            "authorizes_unrestricted_real_world_actuation": False,
        },
    }


def test_valid_readiness_passes():
    jsonschema.validate(_valid_readiness(), _schema("release-readiness-report.schema.json"))


def test_readiness_physical_overclaim_fails():
    r = _valid_readiness()
    r["claims"]["proves_physical_safety"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(r, _schema("release-readiness-report.schema.json"))


def test_readiness_actuation_overclaim_fails():
    r = _valid_readiness()
    r["claims"]["authorizes_unrestricted_real_world_actuation"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(r, _schema("release-readiness-report.schema.json"))
