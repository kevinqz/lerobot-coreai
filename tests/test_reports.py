# test_reports.py — tests for rollout report builders.

import pytest
from lerobot_coreai.reports import build_success_report, build_failure_report


class TestSuccessReport:
    def test_report_has_schema_version(self):
        r = build_success_report(
            policy_path="test", source_repo_id="s", policy_type="act",
            model_id="act-so100", robot_type="so100", runner_url="http://x",
            runner_timing={}, parity_passed=True, fixture_source="f.json",
            observation_keys=[], action=[[0.0]*7]*16, files={},
        )
        assert r["schema_version"] == "lerobot-coreai.rollout_report.v0"

    def test_report_has_version(self):
        from lerobot_coreai import __version__
        r = build_success_report(
            policy_path="test", source_repo_id="s", policy_type="act",
            model_id="act-so100", robot_type="so100", runner_url="http://x",
            runner_timing={}, parity_passed=True, fixture_source="f.json",
            observation_keys=[], action=[[0.0]*7]*16, files={},
        )
        assert r["lerobot_coreai_version"] == __version__

    def test_report_actions_sent_zero(self):
        r = build_success_report(
            policy_path="test", source_repo_id="s", policy_type="act",
            model_id="act-so100", robot_type="so100", runner_url="http://x",
            runner_timing={}, parity_passed=True, fixture_source="f.json",
            observation_keys=[], action=[[0.0]*7]*16, files={},
        )
        assert r["robot"]["actions_sent"] == 0

    def test_report_no_physical_actuation(self):
        r = build_success_report(
            policy_path="test", source_repo_id="s", policy_type="act",
            model_id="act-so100", robot_type="so100", runner_url="http://x",
            runner_timing={}, parity_passed=True, fixture_source="f.json",
            observation_keys=[], action=[[0.0]*7]*16, files={},
        )
        assert r["safety"]["physical_actuation_possible"] is False
        assert r["safety"]["motor_commands_available"] is False


class TestFailureReport:
    def test_failure_report_has_error(self):
        r = build_failure_report(
            policy_path="test", robot_type="so100", mode="dry_run",
            error_type="FixtureError", error_message="missing", stage="fixture.load",
        )
        assert r["ok"] is False
        assert len(r["errors"]) == 1
        assert r["errors"][0]["type"] == "FixtureError"

    def test_failure_report_actions_sent_zero(self):
        r = build_failure_report(
            policy_path="test", robot_type="so100", mode="dry_run",
            error_type="FixtureError", error_message="missing", stage="fixture.load",
        )
        assert r["robot"]["actions_sent"] == 0
        assert r["safety"]["physical_actuation_possible"] is False
