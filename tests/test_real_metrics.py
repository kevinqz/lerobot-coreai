# test_real_metrics.py — real-session metrics + redaction (v1.0.5).

import json
from unittest.mock import MagicMock, patch

from lerobot_coreai.real_metrics import RealMetricsCollector, build_real_metrics_report
from lerobot_coreai.real_mode import RealModeConfig, run_real_mode


def test_collector_summary():
    c = RealMetricsCollector(fps=10.0)
    c.add(observation_ms=1.0, policy_ms=2.0, egress_ms=3.0, loop_ms=50.0)
    c.add(observation_ms=1.0, policy_ms=2.0, egress_ms=3.0, loop_ms=300.0)  # > 100ms deadline
    s = c.summary(wall_seconds=1.0)
    assert s["steps"] == 2
    assert s["missed_deadline_count"] == 1  # 300ms > 100ms budget
    assert s["loop_ms"]["max"] == 300.0
    assert s["effective_fps"] == 2.0


def test_metrics_report_schema_version():
    c = RealMetricsCollector(fps=2.0)
    c.add(observation_ms=1, policy_ms=1, egress_ms=1, loop_ms=1)
    r = build_real_metrics_report(c, wall_seconds=0.5)
    assert r["schema_version"] == "lerobot-coreai.real_metrics.v0"
    assert len(r["per_step"]) == 1


def _mock_policy(valid_manifest_dict):
    from lerobot_coreai.manifest import LeRobotCoreAIManifest
    m = MagicMock()
    m.predict_action.return_value = {"action": [[0.0] * 7] * 16,
                                     "metadata": {"timing": {"total_ms": 5.0}}}
    m.manifest = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
    m.policy_type = "evo1"
    m.robot_type = "so100"
    m.policy_repo_id = "kevinqz/EVO1-SO100-CoreAI"
    return m


def _cfg(sc, tmp_path, **over):
    base = dict(
        mode="guarded", policy_path="kevinqz/EVO1-SO100-CoreAI",
        runner_url="http://127.0.0.1:8710", robot_adapter="mock",
        robot_type=sc["robot_type"], safety_profile=sc["profile"],
        readiness_report=sc["readiness"], approval=sc["approval"],
        bundle_dir=sc["bundle_dir"], output_dir=tmp_path / "out", operator="Kevin",
        max_steps=3, fps=10.0, attest_real_hardware=True,
        attest_physical_estop=True, attest_workspace_clear=True)
    base.update(over)
    return RealModeConfig(**base)


def test_session_writes_metrics(real_ready_scenario, tmp_path, valid_manifest_dict):
    sc = real_ready_scenario()
    with patch("lerobot_coreai.real_mode.CoreAIPolicy.from_pretrained",
               return_value=_mock_policy(valid_manifest_dict)):
        run_real_mode(_cfg(sc, tmp_path))
    out = tmp_path / "out"
    assert (out / "real_metrics.json").is_file()
    assert (out / "real_metrics.md").is_file()
    assert (out / "real_metrics.csv").is_file()
    m = json.loads((out / "real_metrics.json").read_text())
    assert m["summary"]["steps"] == 3


def test_redaction_removes_runner_url_and_operator(real_ready_scenario, tmp_path,
                                                   valid_manifest_dict):
    sc = real_ready_scenario()
    with patch("lerobot_coreai.real_mode.CoreAIPolicy.from_pretrained",
               return_value=_mock_policy(valid_manifest_dict)):
        run_real_mode(_cfg(sc, tmp_path, redact_runner_url=True, redact_operator=True,
                           redact_paths=True))
    session = json.loads((tmp_path / "out" / "real_session.json").read_text())
    assert session["runner_url"] == "<redacted>"
    assert session["operator"] == "<redacted>"
    # paths reduced to basenames (no directory separators).
    assert "/" not in session["approval"]
    report = json.loads((tmp_path / "out" / "real_report.json").read_text())
    assert report["operator"] == "<redacted>"


def test_no_redaction_by_default(real_ready_scenario, tmp_path, valid_manifest_dict):
    sc = real_ready_scenario()
    with patch("lerobot_coreai.real_mode.CoreAIPolicy.from_pretrained",
               return_value=_mock_policy(valid_manifest_dict)):
        run_real_mode(_cfg(sc, tmp_path))
    session = json.loads((tmp_path / "out" / "real_session.json").read_text())
    assert session["runner_url"] == "http://127.0.0.1:8710"
    assert session["operator"] == "Kevin"
