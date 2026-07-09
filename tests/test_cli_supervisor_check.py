# test_cli_supervisor_check.py — CLI tests for supervisor-check + sim flags (v0.9.0).

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from lerobot_coreai import cli
from lerobot_coreai.sim import SimResult


def _write_actions(tmp_path, actions):
    path = tmp_path / "actions.jsonl"
    with open(path, "w") as f:
        for i, a in enumerate(actions):
            f.write(json.dumps({"step": i, "ok": a is not None, "action": a}) + "\n")
    return path


class TestSupervisorCheckCli:
    def test_valid_actions_rc0(self, tmp_path):
        actions = _write_actions(tmp_path, [[[0.0] * 7] * 16, [[0.1] * 7] * 16])
        rc = cli.main(["supervisor-check", "--actions", str(actions),
                       "--safety.profile-name", "default-sim-safe",
                       "--output-dir", str(tmp_path / "check")])
        assert rc == 0
        assert (tmp_path / "check" / "safety_summary.json").is_file()

    def test_blocked_action_rc1_with_fail_on_block(self, tmp_path):
        actions = _write_actions(tmp_path, [[[float("nan")] * 7] * 16])
        rc = cli.main(["supervisor-check", "--actions", str(actions),
                       "--safety.profile-name", "default-sim-safe",
                       "--fail-on-block"])
        assert rc == 1

    def test_blocked_action_rc0_without_fail_flag(self, tmp_path):
        actions = _write_actions(tmp_path, [[[float("nan")] * 7] * 16])
        rc = cli.main(["supervisor-check", "--actions", str(actions),
                       "--safety.profile-name", "default-sim-safe"])
        assert rc == 0

    def test_json_output(self, tmp_path, capsys):
        actions = _write_actions(tmp_path, [[[0.0] * 7] * 16])
        rc = cli.main(["supervisor-check", "--actions", str(actions),
                       "--safety.profile-name", "default-sim-safe", "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["actions_supervised"] == 1
        assert payload["blocked"] == 0

    def test_missing_actions_file_rc1(self, tmp_path):
        rc = cli.main(["supervisor-check", "--actions", str(tmp_path / "nope.jsonl"),
                       "--safety.profile-name", "default-sim-safe"])
        assert rc == 1

    def test_invalid_json_line_is_critical_finding(self, tmp_path, capsys):
        # A malformed actions line must be a fail-closed critical finding,
        # not a silent skip.
        path = tmp_path / "actions.jsonl"
        path.write_text('{"step":0,"action":[[0.0]]}\nnot json at all\n')
        rc = cli.main(["supervisor-check", "--actions", str(path),
                       "--safety.profile-name", "default-sim-safe",
                       "--fail-on-block", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["actions_supervised"] == 2
        assert payload["blocked"] >= 1          # the bad line counts as blocked
        assert rc == 1

    def test_summary_includes_would_block_fields(self, tmp_path):
        # supervisor-check runs in enforce, so unsafe actions block; the written
        # summary carries the new critical_findings field.
        actions = _write_actions(tmp_path, [[[float("nan")] * 7] * 16])
        cli.main(["supervisor-check", "--actions", str(actions),
                  "--safety.profile-name", "default-sim-safe",
                  "--output-dir", str(tmp_path / "check")])
        summary = json.loads((tmp_path / "check" / "safety_summary.json").read_text())
        assert "would_block_actions" in summary
        assert "critical_findings" in summary
        assert summary["critical_findings"] >= 1
        assert summary["passed"] is False

    def test_invalid_profile_rc1(self, tmp_path):
        actions = _write_actions(tmp_path, [[[0.0] * 7] * 16])
        bad = tmp_path / "bad.json"
        bad.write_text('{"schema_version": "wrong", "name": "x", "mode": "fail_closed"}')
        rc = cli.main(["supervisor-check", "--actions", str(actions),
                       "--safety.profile", str(bad)])
        assert rc == 1


class TestSimSupervisorFlags:
    def _mock_result(self, tmp_path):
        mock = MagicMock(spec=SimResult)
        mock.ok = True
        mock.output_dir = tmp_path / "run"
        mock.report_path = tmp_path / "run" / "sim_report.json"
        mock.trace_path = tmp_path / "run" / "sim_trace.jsonl"
        mock.actions_path = tmp_path / "run" / "actions.jsonl"
        mock.observations_path = tmp_path / "run" / "observations.jsonl"
        mock.episodes_path = tmp_path / "run" / "episodes.jsonl"
        mock.report = {"mode": "sim", "ok": True, "metrics": {}, "safety": {}, "files": {}}
        return mock

    def test_sim_accepts_supervisor_flags(self, tmp_path):
        mock_result = self._mock_result(tmp_path)
        with patch("lerobot_coreai.cli.run_sim_mode", return_value=mock_result) as run:
            cli.main([
                "sim", "--policy.path", "test", "--env.type", "fake",
                "--output-dir", str(tmp_path / "run"), "--confirm-sim-egress",
                "--supervisor.mode", "enforce",
                "--safety.profile-name", "so100-sim-default",
            ])
        cfg = run.call_args[0][0]
        assert cfg.supervisor_mode == "enforce"
        assert cfg.safety_profile_name == "so100-sim-default"

    def test_sim_supervisor_mode_default_enforce(self, tmp_path):
        mock_result = self._mock_result(tmp_path)
        with patch("lerobot_coreai.cli.run_sim_mode", return_value=mock_result) as run:
            cli.main([
                "sim", "--policy.path", "test", "--env.type", "fake",
                "--output-dir", str(tmp_path / "run"), "--confirm-sim-egress",
            ])
        cfg = run.call_args[0][0]
        assert cfg.supervisor_mode == "enforce"
