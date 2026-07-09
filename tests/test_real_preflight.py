# test_real_preflight.py — guarded real-mode preflight gates (v1.0.0).

import json
from importlib.resources import files

import jsonschema

from lerobot_coreai.real_preflight import RealPreflightConfig, evaluate_real_preflight


def _cfg(sc, mode="guarded", **over):
    base = dict(
        mode=mode, policy_path="kevinqz/EVO1-SO100-CoreAI",
        runner_url="http://127.0.0.1:8710", robot_adapter="mock",
        robot_type=sc["robot_type"], safety_profile=sc["profile"],
        readiness_report=sc["readiness"], approval=sc["approval"],
        bundle_dir=sc["bundle_dir"], operator="Kevin Saltarelli",
        max_steps=10, fps=2.0, attest_real_hardware=True,
        attest_physical_estop=True, attest_workspace_clear=True,
    )
    base.update(over)
    return RealPreflightConfig(**base)


def _failed(result):
    return {c.name for c in result.checks if c.severity == "required" and not c.passed}


def test_valid_scenario_preflight_passes(real_ready_scenario):
    sc = real_ready_scenario()
    result = evaluate_real_preflight(_cfg(sc))
    assert result.ok, _failed(result)


def test_preflight_report_validates_schema(real_ready_scenario):
    sc = real_ready_scenario()
    result = evaluate_real_preflight(_cfg(sc))
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "real-preflight.schema.json").read_text())
    jsonschema.validate(result.report, schema)
    assert result.report["actions_sent_to_robot"] == 0


def test_refuses_readiness_not_ready(real_ready_scenario):
    sc = real_ready_scenario(ready=False)
    result = evaluate_real_preflight(_cfg(sc))
    assert not result.ok
    assert "readiness_report_ready_true" in _failed(result)


def test_refuses_missing_readiness(real_ready_scenario, tmp_path):
    sc = real_ready_scenario()
    result = evaluate_real_preflight(_cfg(sc, readiness_report=tmp_path / "nope.json"))
    assert not result.ok
    assert "readiness_report_exists" in _failed(result)


def test_refuses_missing_approval(real_ready_scenario, tmp_path):
    sc = real_ready_scenario()
    result = evaluate_real_preflight(_cfg(sc, approval=tmp_path / "nope.json"))
    assert not result.ok
    assert "approval_exists" in _failed(result)


def test_refuses_tampered_bundle(real_ready_scenario):
    sc = real_ready_scenario()
    (sc["bundle_dir"] / "source_run" / "sim_report.json").write_text('{"x":1}')
    result = evaluate_real_preflight(_cfg(sc))
    assert not result.ok
    assert "bundle_verifies" in _failed(result) or "approval_valid" in _failed(result)


def test_refuses_robot_type_mismatch(real_ready_scenario):
    sc = real_ready_scenario(robot_type="so100")
    result = evaluate_real_preflight(_cfg(sc, robot_type="so101"))
    assert not result.ok
    assert "safety_profile_robot_type_matches" in _failed(result)


def test_refuses_unknown_adapter(real_ready_scenario):
    sc = real_ready_scenario()
    result = evaluate_real_preflight(_cfg(sc, robot_adapter="so100"))
    assert not result.ok
    assert "robot_adapter_known" in _failed(result)


class TestReadinessValidation:
    def test_refuses_fake_readiness_json(self, real_ready_scenario):
        # A bare {"ready": true} is not a valid readiness report.
        sc = real_ready_scenario()
        sc["readiness"].write_text(json.dumps({"ready": True}))
        result = evaluate_real_preflight(_cfg(sc))
        assert not result.ok
        assert "readiness_schema_valid" in _failed(result)

    def test_refuses_unrestricted_actuation_overclaim(self, real_ready_scenario):
        sc = real_ready_scenario()
        rr = json.loads(sc["readiness"].read_text())
        rr["claims"]["authorizes_unrestricted_real_world_actuation"] = True
        sc["readiness"].write_text(json.dumps(rr))
        result = evaluate_real_preflight(_cfg(sc))
        assert not result.ok
        # schema pins this claim to false, and the explicit no-overclaim check fails.
        assert ("readiness_no_overclaim" in _failed(result)
                or "readiness_schema_valid" in _failed(result))

    def test_refuses_bundle_path_mismatch(self, real_ready_scenario, tmp_path):
        sc = real_ready_scenario()
        rr = json.loads(sc["readiness"].read_text())
        rr["bundle"]["path"] = str(tmp_path / "some-other-bundle")
        sc["readiness"].write_text(json.dumps(rr))
        result = evaluate_real_preflight(_cfg(sc))
        assert not result.ok
        assert "readiness_bundle_path_matches" in _failed(result)

    def test_refuses_regression_not_passed(self, real_ready_scenario):
        sc = real_ready_scenario()
        rr = json.loads(sc["readiness"].read_text())
        rr["evidence"]["safety_regression_passed"] = False
        sc["readiness"].write_text(json.dumps(rr))
        result = evaluate_real_preflight(_cfg(sc))
        assert not result.ok
        assert "readiness_evidence_safety_regression_passed" in _failed(result)


class TestProfileIntendedForReal:
    def test_sim_only_profile_refused(self, real_ready_scenario):
        sc = real_ready_scenario()
        prof = json.loads(sc["profile"].read_text())
        prof["intended_modes"] = ["sim", "shadow"]
        sc["profile"].write_text(json.dumps(prof))
        result = evaluate_real_preflight(_cfg(sc))
        assert not result.ok
        assert "safety_profile_intended_for_real" in _failed(result)

    def test_missing_intended_modes_refused(self, real_ready_scenario):
        sc = real_ready_scenario()
        prof = json.loads(sc["profile"].read_text())
        prof["intended_modes"] = []
        sc["profile"].write_text(json.dumps(prof))
        result = evaluate_real_preflight(_cfg(sc))
        assert not result.ok
        assert "safety_profile_intended_for_real" in _failed(result)

    def test_guarded_real_profile_passes(self, real_ready_scenario):
        sc = real_ready_scenario()  # fixture uses so100-real-guarded intended modes
        assert evaluate_real_preflight(_cfg(sc)).ok


class TestGuardedRequirements:
    def test_refuses_missing_operator(self, real_ready_scenario):
        sc = real_ready_scenario()
        assert not evaluate_real_preflight(_cfg(sc, operator=None)).ok

    def test_refuses_missing_max_steps(self, real_ready_scenario):
        sc = real_ready_scenario()
        r = evaluate_real_preflight(_cfg(sc, max_steps=None))
        assert not r.ok
        assert "max_steps_present" in _failed(r)

    def test_refuses_high_fps(self, real_ready_scenario):
        sc = real_ready_scenario()
        r = evaluate_real_preflight(_cfg(sc, fps=100.0))
        assert not r.ok
        assert "fps_bounded" in _failed(r)

    def test_refuses_missing_attestations(self, real_ready_scenario):
        sc = real_ready_scenario()
        for flag in ("attest_real_hardware", "attest_physical_estop",
                     "attest_workspace_clear"):
            r = evaluate_real_preflight(_cfg(sc, **{flag: False}))
            assert not r.ok

    def test_external_adapter_not_contacted_without_attestations(self, real_ready_scenario):
        # Guarded + external-http + missing attestations: the adapter preflight is
        # skipped (no request to the real controller) and preflight fails.
        sc = real_ready_scenario()
        result = evaluate_real_preflight(_cfg(
            sc, robot_adapter="external-http", robot_endpoint="http://127.0.0.1:9",
            attest_real_hardware=False))
        assert not result.ok
        pf = next(c for c in result.checks if c.name == "robot_adapter_preflight_passes")
        assert not pf.passed
        assert "skipped" in pf.message

    def test_preflight_mode_skips_guarded_requirements(self, real_ready_scenario):
        # In preflight mode, operator/attestations aren't required.
        sc = real_ready_scenario()
        r = evaluate_real_preflight(_cfg(
            sc, mode="preflight", operator=None, max_steps=None,
            attest_real_hardware=False, attest_physical_estop=False,
            attest_workspace_clear=False))
        assert r.ok, _failed(r)
