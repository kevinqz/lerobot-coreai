# test_action_contract.py — action + batch contract parsing (v1.2.5).

import json
from importlib.resources import files

import jsonschema
import pytest

from lerobot_coreai.action_contract import (
    ACTION_CONTRACT_SCHEMA_VERSION, ActionContract, BatchContract,
    build_action_contract_report, parse_action_contract_from_manifest,
    parse_batch_contract_from_manifest,
)
from lerobot_coreai.manifest import LeRobotCoreAIManifest


def test_infers_chunk_from_2d_action_shape(valid_manifest_dict):
    m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
    contract = parse_action_contract_from_manifest(m)
    # valid_manifest action feature is [16, 7] → chunk horizon 16, dim 7.
    assert contract.representation == "chunk"
    assert contract.horizon == 16
    assert contract.action_dim == 7
    assert contract.is_chunked() is True


def test_infers_single_from_1d_shape():
    contract = parse_action_contract_from_manifest(
        {"action_features": {"action": {"dtype": "float32", "shape": [7]}}})
    assert contract.representation == "single"
    assert contract.horizon == 1
    assert contract.action_dim == 7


def test_explicit_contract_wins():
    contract = parse_action_contract_from_manifest({
        "action_features": {"action": {"shape": [16, 7]}},
        "action_contract": {"representation": "single", "horizon": 1, "action_dim": 7}})
    assert contract.representation == "single"


def test_unknown_shape_defaults_single():
    contract = parse_action_contract_from_manifest({})
    assert contract.representation == "single"
    assert contract.action_dim is None


def test_batch_contract_default_and_explicit():
    assert parse_batch_contract_from_manifest({}).supports_batch is False
    bc = parse_batch_contract_from_manifest(
        {"batch_contract": {"supports_batch": True, "max_batch_size": 4,
                            "fallback": "reject"}})
    assert bc.supports_batch is True and bc.max_batch_size == 4 and bc.fallback == "reject"


def test_report_schema_valid_and_honest():
    report = build_action_contract_report(
        ActionContract(representation="chunk", horizon=16, action_dim=7),
        BatchContract())
    assert report["schema_version"] == ACTION_CONTRACT_SCHEMA_VERSION
    assert report["claims"]["supports_training"] is False
    assert report["claims"]["proves_physical_safety"] is False
    assert report["claims"]["matches_lerobot_select_action_semantics"] is True
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "action-contract.schema.json").read_text())
    jsonschema.validate(report, schema)


# --- fail-closed on semantically invalid contracts (v1.3.5) ---

def test_single_with_horizon_gt_1_fails():
    with pytest.raises(ValueError):
        parse_action_contract_from_manifest(
            {"contracts": {"action": {"representation": "single", "horizon": 16,
                                      "action_dim": 7}}})


def test_single_with_horizon_1_is_ok():
    c = parse_action_contract_from_manifest(
        {"contracts": {"action": {"representation": "single", "horizon": 1,
                                  "action_dim": 7}}})
    assert c.representation == "single" and c.horizon == 1


def test_unknown_representation_fails():
    with pytest.raises(ValueError):
        parse_action_contract_from_manifest(
            {"contracts": {"action": {"representation": "bogus", "horizon": 1}}})


def test_chunk_with_horizon_0_fails():
    with pytest.raises(ValueError):
        parse_action_contract_from_manifest(
            {"contracts": {"action": {"representation": "chunk", "horizon": 0,
                                      "action_dim": 7}}})
