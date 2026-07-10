# test_cli_policy_card.py — policy-card CLI incl. artifact-index mode (v1.2.3).

import hashlib
import json

from lerobot_coreai import cli


def _bundle(tmp_path, name="evo1-pusht-bridge-benchmark"):
    d = tmp_path / name
    (d / "reports").mkdir(parents=True)
    fm = {
        "reports/lerobot_compatibility_report.json": {
            "ok": True, "lerobot_version": "0.6.0", "policy_path": "kevinqz/EVO1-SO100-CoreAI",
            "claims": {"native_upstream_registry": False, "supports_training": False,
                       "supports_physical_safety": False}},
        "reports/lerobot_bridge_report.json": {
            "ok": True, "policy_type": "coreai_bridge",
            "claims": {"native_upstream_registry": False, "proves_physical_safety": False}},
    }
    for rel, data in fm.items():
        (d / rel).write_text(json.dumps(data))
    checks = {rel: hashlib.sha256((d / rel).read_bytes()).hexdigest() for rel in fm}
    (d / "benchmark_manifest.json").write_text(json.dumps({
        "schema_version": "lerobot-coreai.bridge_benchmark_pack.v0",
        "bundle_type": "bridge_benchmark", "policy_path": "kevinqz/EVO1-SO100-CoreAI",
        "dataset_repo_id": "lerobot/pusht",
        "reports": {k.split("/")[-1].split(".")[0]: k for k in fm},
        "claims": {"proves_task_success": False, "proves_physical_safety": False,
                   "authorizes_robot_actuation": False}}))
    (d / "checksums.json").write_text(json.dumps(checks))
    return d


def test_cli_policy_card_from_bundle(tmp_path):
    d = _bundle(tmp_path)
    out = tmp_path / "README.md"
    rep = tmp_path / "policy_card_report.json"
    rc = cli.main(["policy-card", "--benchmark-bundle", str(d),
                   "--output", str(out), "--output-report", str(rep)])
    assert rc == 0
    assert out.is_file() and "Policy Card" in out.read_text()
    report = json.loads(rep.read_text())
    assert report["ok"] is True
    assert (tmp_path / "policy_card_report.md").is_file()


def test_cli_policy_card_from_index(tmp_path):
    idx = tmp_path / "idx"
    d = _bundle(tmp_path)
    cli.main(["artifact-index", "init", "--index-dir", str(idx)])
    cli.main(["artifact-index", "add", "--index-dir", str(idx), "--artifact-dir", str(d),
              "--artifact-type", "bridge_benchmark", "--release-channel", "public-demo"])
    entries = json.loads((idx / "index.json").read_text())["entries"]
    artifact_id = json.loads((idx / entries[0]).read_text())["artifact_id"]
    rc = cli.main(["policy-card", "--artifact-index", str(idx),
                   "--artifact-id", artifact_id, "--json"])
    assert rc == 0


def test_cli_policy_card_overclaim_rc1(tmp_path):
    d = _bundle(tmp_path)
    r = d / "reports" / "lerobot_bridge_report.json"
    r.write_text(json.dumps({"ok": True, "claims": {"proves_physical_safety": True}}))
    (d / "checksums.json").write_text(json.dumps({
        "reports/lerobot_compatibility_report.json":
        hashlib.sha256((d / "reports" / "lerobot_compatibility_report.json").read_bytes()).hexdigest(),
        "reports/lerobot_bridge_report.json": hashlib.sha256(r.read_bytes()).hexdigest()}))
    rc = cli.main(["policy-card", "--benchmark-bundle", str(d)])
    assert rc == 1
