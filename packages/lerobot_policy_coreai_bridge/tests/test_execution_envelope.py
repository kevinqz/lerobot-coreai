# test_execution_envelope.py — v1.3.19 session lifecycle + FailureEvidence v2.
#
# Session state machine (begin/end misuse rejection) and independently verifiable
# failure bundles. No runner, no hardware, no egress.

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("lerobot")

from lerobot_policy_coreai_bridge.configuration_coreai_bridge import (  # noqa: E402
    CoreAIBridgeConfig,
)
from lerobot_policy_coreai_bridge.modeling_coreai_bridge import (  # noqa: E402
    CoreAIBridgePolicy, PluginBindingError,
)
from lerobot_policy_coreai_bridge.rollout_evidence import (  # noqa: E402
    write_failure_evidence, write_matrix_manifest,
)


def _policy():
    return CoreAIBridgePolicy(CoreAIBridgeConfig(runtime_binding_mode="in_memory"))


# --- session state machine (A / P1.8 / P1.9) ---

def test_end_without_begin_fails():
    with pytest.raises(PluginBindingError):
        _policy().end_evidence_session()


def test_double_begin_fails():
    p = _policy()
    p.begin_evidence_session("run-1")
    with pytest.raises(PluginBindingError):     # would silently overwrite (P1.9)
        p.begin_evidence_session("run-2")


def test_double_end_fails():
    p = _policy()
    p.begin_evidence_session("run-1")
    p.end_evidence_session()
    with pytest.raises(PluginBindingError):
        p.end_evidence_session()


def test_begin_requires_run_id():
    with pytest.raises(PluginBindingError):
        _policy().begin_evidence_session("")


def test_begin_end_emits_bracketed_stream():
    p = _policy()
    p.begin_evidence_session("run-1")
    p.end_evidence_session()
    assert [e["event"] for e in p.queue_events] == [
        "execution.started", "execution.completed"]
    assert all(e["execution_id"] == "run-1" for e in p.queue_events)


# --- FailureEvidence v2 (J / P1.13): schema-valid, independently verifiable ---

def test_failure_bundle_written_and_verifies(tmp_path):
    from lerobot_coreai.rollout_verify import verify_official_rollout_evidence
    case = "runner_negotiate-b1"
    write_failure_evidence(
        str(tmp_path / case), case=case, failed_stage="runner_negotiate",
        exception_type="PluginBindingError", message="runner refused",
        target="local", batch_size=1, mode="single_only",
        execution_id="exec-1", environment={"target": "local"})
    # no matrix required: a single failure bundle must verify on its own.
    res = verify_official_rollout_evidence(str(tmp_path), require_complete_matrix=False)
    assert res.ok, {k: v for k, v in res.checks.items() if v != "passed"}


def test_failure_bundle_tamper_detected(tmp_path):
    from lerobot_coreai.rollout_verify import verify_official_rollout_evidence
    case = "validation-b2"
    write_failure_evidence(
        str(tmp_path / case), case=case, failed_stage="validation",
        exception_type="ValueError", message="bad chunk", target="local",
        batch_size=2, mode="native_batch", status="failed")
    rp = tmp_path / case / "failure_report.json"
    d = json.loads(rp.read_text())
    d["failed_stage"] = "rollout"               # not reflected in the manifest digest
    rp.write_text(json.dumps(d))
    assert not verify_official_rollout_evidence(
        str(tmp_path), require_complete_matrix=False).ok


def test_failure_bundle_is_matrix_representable(tmp_path):
    from lerobot_coreai.rollout_verify import verify_official_rollout_evidence
    case = "runner_negotiate-b1"
    root = write_failure_evidence(
        str(tmp_path / case), case=case, failed_stage="runner_negotiate",
        exception_type="PluginBindingError", message="runner refused", target="local")
    write_matrix_manifest(str(tmp_path), "local",
                          {case: {"passed": False, "bundle_root_sha256": root}})
    res = verify_official_rollout_evidence(str(tmp_path), require_complete_matrix=False)
    assert res.ok, {k: v for k, v in res.checks.items() if v != "passed"}
    # the matrix must bind the failure case as NOT passed.
    assert res.checks.get(f"matrix_binding:{case}") == "passed"


def test_failure_bundle_rejects_promoted_claim(tmp_path):
    # the schema pins claims to false; a hand-forged true claim must not validate.
    from lerobot_coreai.rollout_evidence_schema import FAILURE_REPORT_SCHEMA
    import jsonschema
    bad = {"schema_version": "lerobot-coreai.official_rollout_failure.v2",
           "case": "c", "target": "local", "failed_stage": "rollout",
           "exception_type": "E", "message": "m",
           "claims": {"official_rollout_pipeline_smoke_passed": True,
                      "official_eval_certified": False, "authenticity_verified": False,
                      "proves_task_success": False, "proves_physical_safety": False}}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, FAILURE_REPORT_SCHEMA)
