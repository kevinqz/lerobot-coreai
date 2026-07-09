# test_cli_safety_regression.py — CLI tests for safety-regression (v0.9.2).

import json

from lerobot_coreai import cli


def _summary(**over):
    base = {
        "actions_supervised": 100, "actions_blocked": 0, "actions_modified": 0,
        "would_block_actions": 0, "critical_failures": 0, "critical_findings": 0,
        "passed": True, "profile": "so100-sim-default",
    }
    base.update(over)
    return base


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return p


class TestSafetyRegression:
    def test_clean_rc0(self, tmp_path):
        b = _write(tmp_path, "base.json", _summary())
        c = _write(tmp_path, "cand.json", _summary())
        assert cli.main(["safety-regression", "--baseline", str(b), "--candidate", str(c),
                         "--fail-on-regression"]) == 0

    def test_regression_report_only_rc0(self, tmp_path):
        b = _write(tmp_path, "base.json", _summary())
        c = _write(tmp_path, "cand.json", _summary(actions_blocked=3, passed=False))
        assert cli.main(["safety-regression", "--baseline", str(b), "--candidate", str(c)]) == 0

    def test_regression_fail_flag_rc1(self, tmp_path):
        b = _write(tmp_path, "base.json", _summary())
        c = _write(tmp_path, "cand.json", _summary(actions_blocked=3, passed=False))
        assert cli.main(["safety-regression", "--baseline", str(b), "--candidate", str(c),
                         "--fail-on-regression"]) == 1

    def test_run_dir_inputs(self, tmp_path):
        base = tmp_path / "base"; base.mkdir()
        cand = tmp_path / "cand"; cand.mkdir()
        _write(base, "safety_summary.json", _summary())
        _write(cand, "safety_summary.json", _summary(critical_findings=2, passed=False))
        assert cli.main(["safety-regression", "--baseline-run-dir", str(base),
                         "--candidate-run-dir", str(cand), "--fail-on-regression"]) == 1

    def test_bundle_dir_inputs(self, tmp_path):
        for side, s in (("base", _summary()), ("cand", _summary())):
            src = tmp_path / side / "source_run"
            src.mkdir(parents=True)
            _write(src, "safety_summary.json", s)
        assert cli.main(["safety-regression",
                         "--baseline-bundle-dir", str(tmp_path / "base"),
                         "--candidate-bundle-dir", str(tmp_path / "cand"),
                         "--fail-on-regression"]) == 0

    def test_json_and_report(self, tmp_path, capsys):
        b = _write(tmp_path, "base.json", _summary())
        c = _write(tmp_path, "cand.json", _summary(actions_blocked=1, passed=False))
        out = tmp_path / "reg"
        rc = cli.main(["safety-regression", "--baseline", str(b), "--candidate", str(c),
                       "--output-dir", str(out), "--json"])
        assert rc == 0  # no --fail-on-regression
        payload = json.loads(capsys.readouterr().out)
        assert payload["passed"] is False
        assert (out / "safety_regression_report.json").is_file()

    def test_missing_input_rc1(self, tmp_path):
        b = _write(tmp_path, "base.json", _summary())
        assert cli.main(["safety-regression", "--baseline", str(b),
                         "--candidate", str(tmp_path / "nope.json")]) == 1
