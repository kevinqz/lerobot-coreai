# test_official_eval_certificate.py — OfficialEvalCertificate v1 (v1.3.27 core).
# The critical gate: direct rollout() evidence can NEVER set official_eval_certified;
# only the real lerobot-eval CLI argv + all checks + complete cases + clean exit does.

from lerobot_coreai.official_eval_certificate import (
    REQUIRED_CASES, build_official_eval_certificate,
    verify_official_eval_certificate,
)

_H = "sha256:" + "a" * 64
_ALL_TRUE = {"official_cli_entrypoint_used": True,
             "third_party_plugin_registration_used": True,
             "all_required_cases_passed": True, "outputs_schema_valid": True,
             "evidence_replay_passed": True}


def _execution(argv, exit_code=0):
    return {"argv": argv, "command_sha256": _H, "resolved_config_sha256": _H,
            "output_tree_sha256": _H, "exit_code": exit_code}


def _scope(cases=REQUIRED_CASES):
    return {"lerobot_version": "0.6.0", "lerobot_distribution_sha256": _H,
            "environment": "coreai_cert_env", "cases": list(cases)}


def _inputs():
    return {"artifact_root_sha256": _H, "feature_contract_sha256": _H,
            "dataset_metadata_sha256": _H, "processor_parity_sha256": _H}


_CLI_ARGV = ["/usr/bin/lerobot-eval", "--policy.type=coreai_bridge",
             "--env.type=coreai_cert_env", "--eval.batch_size=4"]


def test_official_cli_run_certifies():
    cert = build_official_eval_certificate(
        scope=_scope(), inputs=_inputs(), execution=_execution(_CLI_ARGV),
        checks=_ALL_TRUE)
    assert cert["claims"]["official_eval_certified"] is True
    ok, errs = verify_official_eval_certificate(cert)
    assert ok, errs


def test_python_m_entrypoint_certifies():
    argv = ["python", "-m", "lerobot.scripts.lerobot_eval", "--env.type=coreai_cert_env"]
    cert = build_official_eval_certificate(
        scope=_scope(), inputs=_inputs(), execution=_execution(argv), checks=_ALL_TRUE)
    assert cert["claims"]["official_eval_certified"] is True


def test_direct_rollout_evidence_cannot_certify():
    # argv is NOT lerobot-eval (e.g. a wrapper / direct pytest rollout) -> never certifies.
    cert = build_official_eval_certificate(
        scope=_scope(), inputs=_inputs(),
        execution=_execution(["pytest", "test_e2e_official_rollout.py"]),
        checks=_ALL_TRUE)
    assert cert["checks"]["official_cli_entrypoint_used"] is False
    assert cert["claims"]["official_eval_certified"] is False


def test_wrapper_pretending_to_be_lerobot_eval_fails():
    # a script literally named to look like the CLI but not the console script / module.
    cert = build_official_eval_certificate(
        scope=_scope(), inputs=_inputs(),
        execution=_execution(["./fake-lerobot-eval-wrapper.sh"]), checks=_ALL_TRUE)
    assert cert["claims"]["official_eval_certified"] is False


def test_nonzero_exit_blocks_certification():
    cert = build_official_eval_certificate(
        scope=_scope(), inputs=_inputs(), execution=_execution(_CLI_ARGV, exit_code=1),
        checks=_ALL_TRUE)
    assert cert["claims"]["official_eval_certified"] is False


def test_missing_case_blocks_certification():
    cert = build_official_eval_certificate(
        scope=_scope(cases=("single-b1", "native-b2")), inputs=_inputs(),
        execution=_execution(_CLI_ARGV), checks=_ALL_TRUE)
    assert cert["claims"]["official_eval_certified"] is False


def test_any_missing_check_blocks_certification():
    for k in ("third_party_plugin_registration_used", "all_required_cases_passed",
              "outputs_schema_valid", "evidence_replay_passed"):
        checks = {**_ALL_TRUE, k: False}
        cert = build_official_eval_certificate(
            scope=_scope(), inputs=_inputs(), execution=_execution(_CLI_ARGV),
            checks=checks)
        assert cert["claims"]["official_eval_certified"] is False, k


def test_output_tamper_detected():
    cert = build_official_eval_certificate(
        scope=_scope(), inputs=_inputs(), execution=_execution(_CLI_ARGV),
        checks=_ALL_TRUE)
    ok, errs = verify_official_eval_certificate(cert, output_tree_sha256="sha256:" + "b" * 64)
    assert not ok and any("output tamper" in e for e in errs)


def test_forged_claim_detected():
    cert = build_official_eval_certificate(
        scope=_scope(), inputs=_inputs(),
        execution=_execution(["pytest"]), checks=_ALL_TRUE)   # can't certify
    cert["claims"]["official_eval_certified"] = True          # forge it
    ok, errs = verify_official_eval_certificate(cert)
    assert not ok and any("gate" in e for e in errs)


def test_task_success_and_safety_always_false():
    cert = build_official_eval_certificate(
        scope=_scope(), inputs=_inputs(), execution=_execution(_CLI_ARGV),
        checks=_ALL_TRUE)
    assert cert["claims"]["proves_general_task_success"] is False
    assert cert["claims"]["proves_physical_safety"] is False
