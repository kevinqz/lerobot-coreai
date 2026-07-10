# test_lerobot_contracts.py — leveled LeRobot compatibility contract (v1.2.4).

import importlib
import json
import sys
from importlib.resources import files

import jsonschema

from lerobot_coreai.lerobot_contracts import (
    LEROBOT_COMPAT_V1_SCHEMA_VERSION, evaluate_compatibility_contract,
)


def test_report_schema_valid():
    report = evaluate_compatibility_contract(strict=False)
    assert report["schema_version"] == LEROBOT_COMPAT_V1_SCHEMA_VERSION
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "lerobot-compatibility-report-v1.schema.json").read_text())
    jsonschema.validate(report, schema)


def test_claims_never_overstate_official_support():
    report = evaluate_compatibility_contract(strict=False)
    c = report["claims"]
    assert c["official_plugin_compatible"] is False
    assert c["official_eval_compatible"] is False
    assert c["official_rollout_compatible"] is False
    assert c["native_upstream_registry"] is False
    assert c["supports_training"] is False
    assert c["proves_physical_safety"] is False


def test_action_semantics_reported_failed():
    # select_action is chunk-passthrough today; must not be reported as aligned.
    report = evaluate_compatibility_contract(strict=False)
    assert report["levels"]["action_semantics"] == "failed"
    assert report["levels"]["action_tensor_contract"] == "failed"
    assert report["levels"]["action_method_name"] == "passed"


def test_official_levels_reported_failed_or_unsupported():
    report = evaluate_compatibility_contract(strict=False)
    lv = report["levels"]
    assert lv["official_plugin_discovery"] == "failed"
    assert lv["official_config_registry"] == "failed"
    assert lv["official_policy_factory"] == "failed"
    assert lv["official_processor_pipeline"] == "failed"
    assert lv["official_eval"] == "failed"
    assert lv["official_rollout_sync"] == "not_supported"
    assert lv["guarded_real_separate_runtime"] == "separate_runtime"


def test_targets_present():
    report = evaluate_compatibility_contract(strict=False)
    assert report["targets"]["stable"]["version"] == "0.6.0"
    assert report["targets"]["stable"]["required"] is True
    assert report["targets"]["development"]["required"] is False


def test_bridge_reported_as_duck_typed_local():
    report = evaluate_compatibility_contract(strict=False)
    assert report["detections"]["bridge_kind"] == "duck_typed_local_runtime_only"


def test_base_import_does_not_pull_torch_or_lerobot():
    import lerobot_coreai.lerobot_contracts as mod
    importlib.reload(mod)
    assert not hasattr(mod, "torch")
    assert not hasattr(mod, "lerobot")


def test_ok_is_consistent_non_strict():
    # Non-strict report is internally consistent (does not require official support).
    report = evaluate_compatibility_contract(strict=False)
    assert report["ok"] is True
