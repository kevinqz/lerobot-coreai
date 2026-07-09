# test_cli_profile_toolkit.py — CLI tests for the profile toolkit (v0.9.1).

import json

from lerobot_coreai import cli


def _write_actions(tmp_path, n=100, val=0.4, shape=(16, 7)):
    path = tmp_path / "actions.jsonl"
    with open(path, "w") as f:
        for i in range(n):
            act = [[val] * shape[1] for _ in range(shape[0])]
            f.write(json.dumps({"step": i, "ok": True, "action": act}) + "\n")
    return path


class TestList:
    def test_list_json(self, capsys):
        assert cli.main(["profile-list", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert "so100-sim-default" in payload["profiles"]


class TestShow:
    def test_show_builtin_json(self, capsys):
        assert cli.main(["profile-show", "--profile-name", "so100-sim-default", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["robot_type"] == "so100"
        assert payload["profile_type"] == "software_bounds"


class TestValidate:
    def test_validate_builtin_rc0(self):
        assert cli.main(["profile-validate", "--profile-name", "so100-sim-default"]) == 0

    def test_validate_invalid_rc1(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text('{"schema_version":"wrong","name":"x","mode":"fail_closed","profile_type":"software_bounds"}')
        assert cli.main(["profile-validate", "--profile", str(bad)]) == 1

    def test_validate_json(self, tmp_path, capsys):
        assert cli.main(["profile-validate", "--profile-name", "pusht-sim-default", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is True

    def test_all_builtins_validate_rc0(self):
        for name in ("default-sim-safe", "generic-7dof-sim-default",
                     "so100-sim-default", "so101-sim-default", "pusht-sim-default"):
            assert cli.main(["profile-validate", "--profile-name", name]) == 0

    def test_overclaiming_limitation_fails_validate(self, tmp_path):
        bad = tmp_path / "overclaim.json"
        bad.write_text(json.dumps({
            "schema_version": "lerobot-coreai.safety_profile.v0",
            "name": "overclaimer", "profile_type": "software_bounds",
            "mode": "fail_closed", "max_abs_action": 1.0,
            "limitations": ["proves physical safety"],
        }))
        assert cli.main(["profile-validate", "--profile", str(bad)]) == 1

    def test_no_limitations_fails_validate(self, tmp_path):
        bare = tmp_path / "bare.json"
        bare.write_text(json.dumps({
            "schema_version": "lerobot-coreai.safety_profile.v0",
            "name": "bare", "profile_type": "software_bounds",
            "mode": "fail_closed", "max_abs_action": 1.0,
        }))
        assert cli.main(["profile-validate", "--profile", str(bare)]) == 1


class TestRecommend:
    def test_recommend_from_actions_json(self, tmp_path, capsys):
        actions = _write_actions(tmp_path)
        assert cli.main(["profile-recommend", "--actions", str(actions), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["recommended_profile"] == "so100-sim-default"

    def test_recommend_robot_type_json(self, capsys):
        assert cli.main(["profile-recommend", "--robot-type", "so101", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["recommended_profile"] == "so101-sim-default"


class TestCalibrate:
    def test_calibrate_writes_outputs(self, tmp_path):
        actions = _write_actions(tmp_path, n=100, val=0.4)
        out_profile = tmp_path / "cal.json"
        out_dir = tmp_path / "caldir"
        rc = cli.main(["profile-calibrate", "--actions", str(actions),
                       "--base-profile-name", "so100-sim-default",
                       "--output-profile", str(out_profile),
                       "--output-dir", str(out_dir)])
        assert rc == 0
        assert out_profile.is_file()
        assert (out_dir / "profile_calibration_report.json").is_file()
        assert (out_dir / "profile_calibration_report.md").is_file()
        # The calibrated profile is itself valid.
        assert cli.main(["profile-validate", "--profile", str(out_profile)]) == 0

    def test_calibrate_missing_actions_rc1(self, tmp_path):
        assert cli.main(["profile-calibrate", "--actions", str(tmp_path / "nope.jsonl")]) == 1

    def test_calibrate_insufficient_samples_rc1(self, tmp_path):
        actions = _write_actions(tmp_path, n=3)
        rc = cli.main(["profile-calibrate", "--actions", str(actions),
                       "--min-samples", "10"])
        assert rc == 1

    def test_calibrate_json(self, tmp_path, capsys):
        actions = _write_actions(tmp_path, n=50, val=0.3)
        rc = cli.main(["profile-calibrate", "--actions", str(actions), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["samples"] == 50
        assert "recommended_bounds" in payload


class TestCompare:
    def test_compare_writes_report(self, tmp_path, capsys):
        actions = _write_actions(tmp_path, n=20, val=0.4)
        out_dir = tmp_path / "cmp"
        rc = cli.main(["profile-compare",
                       "--profile-a-name", "default-sim-safe",
                       "--profile-b-name", "pusht-sim-default",
                       "--actions", str(actions),
                       "--output-dir", str(out_dir), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["actions_supervised"] == 20
        assert (out_dir / "profile_comparison_report.json").is_file()

    def test_compare_missing_actions_rc1(self, tmp_path):
        rc = cli.main(["profile-compare", "--profile-a-name", "default-sim-safe",
                       "--profile-b-name", "default-sim-safe",
                       "--actions", str(tmp_path / "nope.jsonl")])
        assert rc == 1
