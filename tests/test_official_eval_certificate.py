# test_official_eval_certificate.py — OfficialEvalCertificate v1 (v1.3.27 core;
# promotion authority closed in v1.3.26.8, P0.1). official_eval_certified can be set
# ONLY by promoting a VerifiedOfficialEvalExecutionReceipt from a REAL lerobot-eval
# subprocess; a hand-built execution dict of booleans can never certify.

import pytest

from lerobot_coreai.authority import (
    AuthorityError, verify_official_eval_execution_receipt,
)
from lerobot_coreai.official_eval_certificate import (
    _ROOT_KEYS, REQUIRED_CASES, build_diagnostic_official_eval_report,
    promote_official_eval_certificate, verify_official_eval_certificate,
)

_H = "sha256:" + "a" * 64
_ALL_TRUE = {"official_cli_entrypoint_used": True,
             "third_party_plugin_registration_used": True,
             "all_required_cases_passed": True, "outputs_schema_valid": True,
             "evidence_replay_passed": True}
_CLI_ARGV = ["/usr/local/bin/lerobot-eval", "--policy.type=coreai_bridge",
             "--env.type=coreai_cert_env", "--eval.batch_size=4"]


def _execution(argv, exit_code=0):
    return {"argv": argv, "command_sha256": _H, "resolved_config_sha256": _H,
            "output_tree_sha256": _H, "exit_code": exit_code}


def _scope(cases=REQUIRED_CASES):
    return {"lerobot_version": "0.6.0", "lerobot_distribution_sha256": _H,
            "environment": "coreai_cert_env", "cases": list(cases)}


def _inputs():
    return {k: _H for k in _ROOT_KEYS}      # full evidence-graph root set, non-null


def _receipt(**overrides):
    r = {"real_subprocess": True, "fake_executor": False,
         "resolution_method": "console_script",
         "executable_realpath": "/usr/local/bin/lerobot-eval", "argv": list(_CLI_ARGV),
         "lerobot_distribution_sha256": _H, "coreai_env_instantiated": True,
         "cases": list(REQUIRED_CASES), "exit_code": 0, "command_sha256": _H,
         "resolved_config_sha256": _H, "output_tree_sha256": _H,
         "schema_report": {"outputs_schema_valid": True, "output_manifest_sha256": _H},
         "replay_report": {"evidence_replay_passed": True, "replay_root_sha256": _H},
         "verified_cases_root_sha256": _H}
    r.update(overrides)
    return r


# --- diagnostic builder NEVER certifies (P0.1) ---

def test_diagnostic_report_never_certifies_even_with_all_true():
    cert = build_diagnostic_official_eval_report(
        scope=_scope(), inputs=_inputs(), execution=_execution(_CLI_ARGV),
        checks=_ALL_TRUE)
    assert cert["evidence_grade"] == "diagnostic"
    assert cert["claims"]["official_eval_certified"] is False
    ok, errs = verify_official_eval_certificate(cert)
    assert ok, errs


# --- promotion authority (the only true-claim path) ---

def test_promote_from_real_receipt_certifies():
    cert = promote_official_eval_certificate(
        receipt=verify_official_eval_execution_receipt(_receipt()), inputs=_inputs())
    assert cert["evidence_grade"] == "certificate"
    assert cert["claims"]["official_eval_certified"] is True
    ok, errs = verify_official_eval_certificate(cert)
    assert ok, errs


def test_python_m_entrypoint_receipt_certifies():
    argv = ["python", "-m", "lerobot.scripts.lerobot_eval", "--env.type=coreai_cert_env"]
    receipt = verify_official_eval_execution_receipt(
        _receipt(argv=argv, resolution_method="python_-m",
                 executable_realpath="/usr/local/lib/python3.12/site-packages/lerobot"))
    cert = promote_official_eval_certificate(receipt=receipt, inputs=_inputs())
    assert cert["claims"]["official_eval_certified"] is True


def test_promote_rejects_plain_dict_or_bool():
    # a hand-built execution dict / bool is refused at the type boundary.
    with pytest.raises(TypeError):
        promote_official_eval_certificate(receipt={"official_eval_certified": True},
                                          inputs=_inputs())
    with pytest.raises(TypeError):
        promote_official_eval_certificate(receipt=True, inputs=_inputs())


# --- receipt verifier refuses non-real / forged runs ---

def test_tmp_shim_executable_refused():
    # /tmp/lerobot-eval passes the basename check but is not an installed executable.
    with pytest.raises(AuthorityError):
        verify_official_eval_execution_receipt(
            _receipt(argv=["/tmp/lerobot-eval", "--env.type=coreai_cert_env"],
                     executable_realpath="/tmp/lerobot-eval"))


def test_fake_executor_refused():
    with pytest.raises(AuthorityError):
        verify_official_eval_execution_receipt(_receipt(fake_executor=True))


def test_non_official_argv_refused():
    with pytest.raises(AuthorityError):
        verify_official_eval_execution_receipt(
            _receipt(argv=["pytest", "test_e2e_official_rollout.py"],
                     executable_realpath="/usr/local/bin/pytest"))


def test_wrapper_pretending_to_be_lerobot_eval_refused():
    with pytest.raises(AuthorityError):
        verify_official_eval_execution_receipt(
            _receipt(argv=["./fake-lerobot-eval-wrapper.sh"],
                     executable_realpath="/usr/local/bin/fake-lerobot-eval-wrapper.sh"))


def test_env_not_instantiated_refused():
    with pytest.raises(AuthorityError):
        verify_official_eval_execution_receipt(_receipt(coreai_env_instantiated=False))


def test_nonzero_exit_refused():
    with pytest.raises(AuthorityError):
        verify_official_eval_execution_receipt(_receipt(exit_code=1))


def test_incomplete_case_matrix_refused():
    with pytest.raises(AuthorityError):
        verify_official_eval_execution_receipt(_receipt(cases=["single-b1", "native-b2"]))


# --- verifier-side integrity ---

def test_output_tamper_detected():
    cert = promote_official_eval_certificate(
        receipt=verify_official_eval_execution_receipt(_receipt()), inputs=_inputs())
    ok, errs = verify_official_eval_certificate(cert, output_tree_sha256="sha256:" + "b" * 64)
    assert not ok and any("output tamper" in e for e in errs)


def test_forged_diagnostic_claim_detected():
    cert = build_diagnostic_official_eval_report(
        scope=_scope(), inputs=_inputs(), execution=_execution(["pytest"]),
        checks=_ALL_TRUE)
    cert["claims"]["official_eval_certified"] = True          # forge it
    ok, errs = verify_official_eval_certificate(cert)
    assert not ok and any("forged" in e or "diagnostic" in e for e in errs)


def test_forged_certificate_claim_detected():
    # a certificate-grade cert whose gate is broken (argv stripped) must fail.
    cert = promote_official_eval_certificate(
        receipt=verify_official_eval_execution_receipt(_receipt()), inputs=_inputs())
    cert["execution"]["argv"] = ["pytest"]      # break the entrypoint gate
    ok, errs = verify_official_eval_certificate(cert)
    assert not ok


def test_task_success_and_safety_always_false():
    cert = promote_official_eval_certificate(
        receipt=verify_official_eval_execution_receipt(_receipt()), inputs=_inputs())
    assert cert["claims"]["proves_general_task_success"] is False
    assert cert["claims"]["proves_physical_safety"] is False


# --- P0.3: checks derived from receipt reports, not hardcoded ---

def test_failed_schema_report_receipt_refused():
    with pytest.raises(AuthorityError):
        verify_official_eval_execution_receipt(
            _receipt(schema_report={"outputs_schema_valid": False,
                                    "output_manifest_sha256": _H}))


def test_failed_replay_report_receipt_refused():
    with pytest.raises(AuthorityError):
        verify_official_eval_execution_receipt(
            _receipt(replay_report={"evidence_replay_passed": False,
                                    "replay_root_sha256": _H}))


# --- P0.4/WS5: full evidence-graph root set, non-null ---

def test_incomplete_root_graph_cannot_certify():
    receipt = verify_official_eval_execution_receipt(_receipt())
    inputs = _inputs(); inputs["model_conversion_sha256"] = None    # a null root
    cert = promote_official_eval_certificate(receipt=receipt, inputs=inputs)
    assert cert["claims"]["official_eval_certified"] is False        # root graph incomplete


def test_full_root_graph_certifies_and_verifies():
    cert = promote_official_eval_certificate(
        receipt=verify_official_eval_execution_receipt(_receipt()), inputs=_inputs())
    assert cert["claims"]["official_eval_certified"] is True
    assert all(cert["inputs"][k] for k in _ROOT_KEYS)
    ok, errs = verify_official_eval_certificate(cert)
    assert ok, errs


def test_certified_with_forged_null_root_detected_by_verifier():
    cert = promote_official_eval_certificate(
        receipt=verify_official_eval_execution_receipt(_receipt()), inputs=_inputs())
    cert["inputs"]["rollout_matrix_sha256"] = None      # strip a bound root post-hoc
    ok, errs = verify_official_eval_certificate(cert)
    assert not ok and any("root graph" in e for e in errs)
