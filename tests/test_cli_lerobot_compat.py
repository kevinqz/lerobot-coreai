# test_cli_lerobot_compat.py — CLI for lerobot-compat-check (v1.1.2).

import json

from lerobot_coreai import cli


def _has_lerobot():
    try:
        import importlib.metadata as md
        md.version("lerobot")
        return True
    except Exception:
        return False


def test_cli_compat_check_writes_report(tmp_path):
    out = tmp_path / "compat"
    rc = cli.main(["lerobot-compat-check", "--output-dir", str(out)])
    # Non-strict: base package alone is compatible.
    assert rc == 0
    report = json.loads((out / "lerobot_compatibility_report.json").read_text())
    assert report["claims"]["native_upstream_registry"] is False
    assert (out / "lerobot_compatibility_report.md").is_file()


def test_cli_compat_check_json_rc(tmp_path):
    rc = cli.main(["lerobot-compat-check", "--json"])
    assert rc == 0


def test_cli_compat_check_strict_rc_without_lerobot():
    rc = cli.main(["lerobot-compat-check", "--strict", "--json"])
    if not _has_lerobot():
        assert rc == 1
    else:
        assert rc in (0, 1)
