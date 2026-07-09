# test_cli_safety_gate.py — CLI tests for safety-gate (v0.9.2).

import json

from lerobot_coreai import cli


def _summary(**over):
    base = {
        "actions_supervised": 100, "actions_allowed": 100, "actions_blocked": 0,
        "actions_modified": 0, "critical_failures": 0, "would_block_actions": 0,
        "critical_findings": 0, "top_reasons": {}, "passed": True,
    }
    base.update(over)
    return base


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return p


class TestSafetyGate:
    def test_clean_rc0(self, tmp_path):
        s = _write(tmp_path, "safety_summary.json", _summary())
        assert cli.main(["safety-gate", "--safety-summary", str(s),
                         "--fail-on-safety-quality"]) == 0

    def test_failed_report_only_rc0(self, tmp_path):
        s = _write(tmp_path, "safety_summary.json",
                   _summary(actions_blocked=2, critical_findings=2, passed=False))
        assert cli.main(["safety-gate", "--safety-summary", str(s)]) == 0

    def test_failed_with_fail_flag_rc1(self, tmp_path):
        s = _write(tmp_path, "safety_summary.json",
                   _summary(actions_blocked=2, critical_findings=2, passed=False))
        assert cli.main(["safety-gate", "--safety-summary", str(s),
                         "--fail-on-safety-quality"]) == 1

    def test_run_dir_resolution(self, tmp_path):
        run = tmp_path / "run"
        run.mkdir()
        _write(run, "safety_summary.json", _summary())
        assert cli.main(["safety-gate", "--run-dir", str(run),
                         "--fail-on-safety-quality"]) == 0

    def test_bundle_dir_resolution(self, tmp_path):
        src = tmp_path / "bundle" / "source_run"
        src.mkdir(parents=True)
        _write(src, "safety_summary.json", _summary(actions_blocked=1, passed=False))
        assert cli.main(["safety-gate", "--bundle-dir", str(tmp_path / "bundle"),
                         "--fail-on-safety-quality"]) == 1

    def test_writes_report(self, tmp_path):
        s = _write(tmp_path, "safety_summary.json", _summary(actions_blocked=1, passed=False))
        out = tmp_path / "gate"
        cli.main(["safety-gate", "--safety-summary", str(s), "--output-dir", str(out)])
        assert (out / "safety_quality_report.json").is_file()
        assert (out / "safety_quality_report.md").is_file()

    def test_json_output(self, tmp_path, capsys):
        s = _write(tmp_path, "safety_summary.json", _summary())
        assert cli.main(["safety-gate", "--safety-summary", str(s), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["passed"] is True

    def test_missing_input_rc1(self, tmp_path):
        assert cli.main(["safety-gate", "--safety-summary", str(tmp_path / "nope.json"),
                         "--fail-on-safety-quality"]) == 1

    def test_malformed_summary_rc1(self, tmp_path):
        # A summary that only claims passed=true must not pass the gate.
        s = tmp_path / "safety_summary.json"
        s.write_text('{"passed": true}')
        assert cli.main(["safety-gate", "--safety-summary", str(s),
                         "--fail-on-safety-quality"]) == 1

    def test_zero_action_summary_rc1(self, tmp_path):
        s = _write(tmp_path, "safety_summary.json", _summary(actions_supervised=0))
        assert cli.main(["safety-gate", "--safety-summary", str(s),
                         "--fail-on-safety-quality"]) == 1

    def test_allow_summary_failed(self, tmp_path):
        # summary.passed=false but no blocks/findings; --allow-summary-failed passes.
        s = _write(tmp_path, "safety_summary.json", _summary(passed=False))
        assert cli.main(["safety-gate", "--safety-summary", str(s),
                         "--allow-summary-failed", "--fail-on-safety-quality"]) == 0
