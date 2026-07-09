# test_supervisor_check_safety_quality.py — supervisor-check + safety gates (v0.9.2).

import json

from lerobot_coreai import cli


def _write_actions(tmp_path, actions):
    path = tmp_path / "actions.jsonl"
    with open(path, "w") as f:
        for i, a in enumerate(actions):
            f.write(json.dumps({"step": i, "ok": a is not None, "action": a}) + "\n")
    return path


def test_supervisor_check_writes_safety_quality_report(tmp_path):
    actions = _write_actions(tmp_path, [[[0.0] * 7] * 16, [[0.1] * 7] * 16])
    out = tmp_path / "check"
    rc = cli.main(["supervisor-check", "--actions", str(actions),
                   "--safety.profile-name", "default-sim-safe",
                   "--output-dir", str(out),
                   "--safety.max-actions-blocked", "0"])
    assert rc == 0
    assert (out / "safety_quality_report.json").is_file()
    # profile_fit still written (v0.9.1 behavior preserved).
    assert (out / "profile_fit.json").is_file()


def test_supervisor_check_rc1_on_safety_gate_fail(tmp_path):
    # NaN action → blocked → gate fails with fail flag.
    actions = _write_actions(tmp_path, [[[float("nan")] * 7] * 16])
    out = tmp_path / "check"
    rc = cli.main(["supervisor-check", "--actions", str(actions),
                   "--safety.profile-name", "default-sim-safe",
                   "--output-dir", str(out),
                   "--safety.fail-on-safety-quality"])
    assert rc == 1


def test_supervisor_check_gate_report_only_rc0(tmp_path):
    actions = _write_actions(tmp_path, [[[float("nan")] * 7] * 16])
    out = tmp_path / "check"
    rc = cli.main(["supervisor-check", "--actions", str(actions),
                   "--safety.profile-name", "default-sim-safe",
                   "--output-dir", str(out),
                   "--safety.max-actions-blocked", "0"])  # no fail flag
    assert rc == 0
    assert (out / "safety_quality_report.json").is_file()
