# test_cli_bridge_benchmark.py — CLI for package/verify bridge benchmark (v1.1.7).

import json

from lerobot_coreai import cli


def _reports(tmp_path):
    compat = tmp_path / "compat" / "lerobot_compatibility_report.json"
    compat.parent.mkdir(parents=True)
    compat.write_text(json.dumps({
        "schema_version": "lerobot-coreai.lerobot_compat.v0", "ok": True,
        "policy_path": "kevinqz/EVO1-SO100-CoreAI",
        "claims": {"native_upstream_registry": False, "supports_training": False,
                   "supports_physical_safety": False}}))
    ev2 = tmp_path / "eval-v2"
    ev2.mkdir()
    (ev2 / "lerobot_eval_v2_report.json").write_text(json.dumps({
        "schema_version": "lerobot-coreai.lerobot_eval_v2.v0", "ok": True,
        "dataset_repo_id": "lerobot/pusht",
        "claims": {"proves_task_success": False, "proves_physical_safety": False}}))
    (ev2 / "lerobot_feature_mapping.json").write_text(json.dumps({
        "schema_version": "lerobot-coreai.lerobot_feature_mapping.v0", "passed": True,
        "features": {}}))
    return compat, ev2


def test_cli_package_and_verify(tmp_path):
    compat, ev2 = _reports(tmp_path)
    out = tmp_path / "pack"
    rc = cli.main(["package-bridge-benchmark", "--compat-report", str(compat),
                   "--eval-v2-dir", str(ev2), "--output-dir", str(out)])
    assert rc == 0
    assert (out / "benchmark_manifest.json").is_file()

    rc2 = cli.main(["verify-bridge-benchmark", "--bundle-dir", str(out), "--json"])
    assert rc2 == 0


def test_cli_verify_detects_tamper(tmp_path):
    compat, ev2 = _reports(tmp_path)
    out = tmp_path / "pack"
    cli.main(["package-bridge-benchmark", "--compat-report", str(compat),
              "--eval-v2-dir", str(ev2), "--output-dir", str(out)])
    victim = out / "reports" / "eval_v2_report.json"
    victim.write_text(victim.read_text().replace("true", "false"))
    rc = cli.main(["verify-bridge-benchmark", "--bundle-dir", str(out)])
    assert rc == 1


def test_cli_package_refuses_overclaim(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "x",
                               "claims": {"authorizes_robot_actuation": True}}))
    rc = cli.main(["package-bridge-benchmark", "--bridge-report", str(bad),
                   "--output-dir", str(tmp_path / "pack")])
    assert rc == 1
