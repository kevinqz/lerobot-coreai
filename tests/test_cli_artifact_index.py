# test_cli_artifact_index.py — artifact-index CLI (v1.2.2).

import hashlib
import json

from lerobot_coreai import cli


def _bundle(tmp_path, name="evo1-pusht-bridge-benchmark"):
    d = tmp_path / name
    (d / "reports").mkdir(parents=True)
    (d / "reports" / "compat.json").write_text(json.dumps({"ok": True, "claims": {}}))
    (d / "benchmark_manifest.json").write_text(json.dumps({
        "schema_version": "lerobot-coreai.bridge_benchmark_pack.v0",
        "bundle_type": "bridge_benchmark",
        "policy_path": "kevinqz/EVO1-SO100-CoreAI", "dataset_repo_id": "lerobot/pusht",
        "reports": {"compat": "reports/compat.json"},
        "claims": {"proves_task_success": False, "proves_physical_safety": False,
                   "authorizes_robot_actuation": False}}))
    (d / "checksums.json").write_text(json.dumps({
        "reports/compat.json":
        hashlib.sha256((d / "reports" / "compat.json").read_bytes()).hexdigest()}))
    return d


def test_cli_init_add_list_find_verify(tmp_path, capsys):
    idx = tmp_path / "idx"
    assert cli.main(["artifact-index", "init", "--index-dir", str(idx)]) == 0
    d = _bundle(tmp_path)
    assert cli.main(["artifact-index", "add", "--index-dir", str(idx),
                     "--artifact-dir", str(d), "--artifact-type", "bridge_benchmark",
                     "--release-channel", "public-demo"]) == 0
    assert cli.main(["artifact-index", "list", "--index-dir", str(idx), "--json"]) == 0
    out = capsys.readouterr().out
    assert "kevinqz/EVO1-SO100-CoreAI" in out
    assert cli.main(["artifact-index", "find", "--index-dir", str(idx),
                     "--artifact-type", "bridge_benchmark", "--json"]) == 0
    assert cli.main(["artifact-index", "verify", "--index-dir", str(idx)]) == 0


def test_cli_add_overclaim_rc1(tmp_path):
    idx = tmp_path / "idx"
    cli.main(["artifact-index", "init", "--index-dir", str(idx)])
    d = _bundle(tmp_path)
    # inject an overclaim
    r = d / "reports" / "compat.json"
    r.write_text(json.dumps({"ok": True, "claims": {"proves_physical_safety": True}}))
    (d / "checksums.json").write_text(json.dumps({
        "reports/compat.json": hashlib.sha256(r.read_bytes()).hexdigest()}))
    rc = cli.main(["artifact-index", "add", "--index-dir", str(idx),
                   "--artifact-dir", str(d), "--artifact-type", "bridge_benchmark"])
    assert rc == 1


def test_cli_verify_detects_tamper_rc1(tmp_path):
    idx = tmp_path / "idx"
    cli.main(["artifact-index", "init", "--index-dir", str(idx)])
    d = _bundle(tmp_path)
    cli.main(["artifact-index", "add", "--index-dir", str(idx),
              "--artifact-dir", str(d), "--artifact-type", "bridge_benchmark"])
    (d / "reports" / "compat.json").write_text(json.dumps({"ok": False}))
    assert cli.main(["artifact-index", "verify", "--index-dir", str(idx)]) == 1
