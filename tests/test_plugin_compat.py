# test_plugin_compat.py — official-plugin compatibility profile (v1.3.1).

import json
from importlib.resources import files

import jsonschema

from lerobot_coreai.plugin_compat import (
    PLUGIN_COMPAT_SCHEMA_VERSION, evaluate_plugin_compat,
)


def test_report_schema_valid_and_honest():
    report = evaluate_plugin_compat()
    assert report["schema_version"] == PLUGIN_COMPAT_SCHEMA_VERSION
    assert report["profile"] == "official_plugin"
    c = report["claims"]
    assert c["upstream_native"] is False
    assert c["supports_training"] is False
    assert c["official_eval_certified"] is False
    assert c["proves_physical_safety"] is False
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "lerobot-plugin-compatibility-report.schema.json").read_text())
    jsonschema.validate(report, schema)


def test_levels_present_for_all_rungs():
    report = evaluate_plugin_compat()
    for k in ("plugin_discovery", "config_registry", "policy_class_contract",
              "policy_factory", "processor_pipeline", "official_eval"):
        assert k in report["levels"]


def test_official_eval_never_certified():
    # Even when the plugin is installed, official_eval stays not certified here.
    report = evaluate_plugin_compat()
    assert report["levels"]["official_eval"] in ("not_tested",)
    assert report["claims"]["official_eval_certified"] is False


def test_not_installed_reports_not_tested(monkeypatch):
    import lerobot_coreai.plugin_compat as pc
    monkeypatch.setattr(pc, "_plugin_installed", lambda: False)
    report = evaluate_plugin_compat()
    assert report["plugin_installed"] is False
    assert all(v == "not_tested" for v in report["levels"].values())
