# test_policy_card.py — policy card generator (v1.2.3).

import hashlib
import json
from importlib.resources import files

import jsonschema
import pytest

from lerobot_coreai.policy_card import (
    POLICY_CARD_REPORT_SCHEMA_VERSION, PolicyCardError, PolicyCardInputs,
    generate_policy_card,
)
from lerobot_coreai.policy_card_templates import MANDATORY_NON_CLAIMS


def _bundle(tmp_path, *, overclaim=False):
    d = tmp_path / "bundle"
    (d / "reports").mkdir(parents=True)
    files_map = {
        "reports/lerobot_compatibility_report.json": {
            "ok": True, "lerobot_version": "0.6.0", "policy_path": "kevinqz/EVO1-SO100-CoreAI",
            "claims": {"compatible_with_lerobot_0_6_x_shape": True,
                       "native_upstream_registry": False, "supports_training": False,
                       "supports_physical_safety": False}},
        "reports/lerobot_bridge_report.json": {
            "ok": True, "policy_type": "coreai_bridge", "robot_type": "so100",
            "policy_path": "kevinqz/EVO1-SO100-CoreAI",
            "claims": {"native_upstream_registry": False, "proves_physical_safety": overclaim}},
        "reports/eval_v2_report.json": {
            "ok": True, "dataset_repo_id": "lerobot/pusht", "strict": True,
            "frames_evaluated": 0, "feature_mapping": {"passed": True,
            "unknown_dataset_features": []},
            "claims": {"proves_task_success": False, "proves_physical_safety": False}},
        "reports/obs_bridge_report.json": {
            "ok": True, "dropped_keys": [],
            "claims": {"proves_task_success": False, "proves_physical_safety": False}},
    }
    for rel, data in files_map.items():
        (d / rel).write_text(json.dumps(data))
    checks = {rel: hashlib.sha256((d / rel).read_bytes()).hexdigest()
              for rel in files_map}
    (d / "benchmark_manifest.json").write_text(json.dumps({
        "schema_version": "lerobot-coreai.bridge_benchmark_pack.v0",
        "bundle_type": "bridge_benchmark", "policy_path": "kevinqz/EVO1-SO100-CoreAI",
        "dataset_repo_id": "lerobot/pusht",
        "reports": {k.split("/")[-1].split(".")[0]: k for k in files_map},
        "claims": {"proves_task_success": False, "proves_physical_safety": False,
                   "authorizes_robot_actuation": False}}))
    (d / "checksums.json").write_text(json.dumps(checks))
    return d


def test_card_generated_and_deterministic(tmp_path):
    card1, report1 = generate_policy_card(PolicyCardInputs(benchmark_bundle=_bundle(tmp_path)))
    card2, _ = generate_policy_card(PolicyCardInputs(benchmark_bundle=_bundle(tmp_path / "b2")))
    assert card1 == card2  # deterministic — no timestamps/randomness
    assert report1["schema_version"] == POLICY_CARD_REPORT_SCHEMA_VERSION
    assert report1["ok"] is True


def test_card_includes_mandatory_non_claims(tmp_path):
    card, _ = generate_policy_card(PolicyCardInputs(benchmark_bundle=_bundle(tmp_path)))
    for nc in MANDATORY_NON_CLAIMS:
        assert nc in card


def test_report_schema_valid_and_honest(tmp_path):
    _card, report = generate_policy_card(PolicyCardInputs(benchmark_bundle=_bundle(tmp_path)))
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "policy-card-report.schema.json").read_text())
    jsonschema.validate(report, schema)
    for k in ("proves_physical_safety", "authorizes_robot_actuation",
              "supports_training", "native_upstream_registry"):
        assert report["claims"][k] is False


def test_card_includes_key_evidence(tmp_path):
    card, _ = generate_policy_card(PolicyCardInputs(benchmark_bundle=_bundle(tmp_path)))
    assert "kevinqz/EVO1-SO100-CoreAI" in card
    assert "coreai_bridge" in card
    assert "lerobot/pusht" in card
    assert "0.6.0" in card


def test_overclaim_source_fails_closed(tmp_path):
    with pytest.raises(PolicyCardError):
        generate_policy_card(PolicyCardInputs(benchmark_bundle=_bundle(tmp_path, overclaim=True)))


def test_tampered_bundle_fails_closed(tmp_path):
    d = _bundle(tmp_path)
    victim = d / "reports" / "obs_bridge_report.json"
    victim.write_text(victim.read_text().replace("true", "false"))
    with pytest.raises(PolicyCardError):
        generate_policy_card(PolicyCardInputs(benchmark_bundle=d))


def test_no_sources_fails(tmp_path):
    with pytest.raises(PolicyCardError):
        generate_policy_card(PolicyCardInputs())


def test_direct_report_mode(tmp_path):
    compat = tmp_path / "compat.json"
    compat.write_text(json.dumps({
        "ok": True, "lerobot_version": "0.6.0", "policy_path": "p",
        "claims": {"native_upstream_registry": False, "supports_training": False,
                   "supports_physical_safety": False}}))
    card, report = generate_policy_card(PolicyCardInputs(compat_report=compat))
    assert report["source_mode"] == "direct"
    assert "0.6.0" in card
