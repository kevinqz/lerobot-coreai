# test_manifest_contracts.py — v1 contracts block on the manifest (v1.2.8).

import json
from importlib.resources import files

import jsonschema

from lerobot_coreai.action_contract import (
    parse_action_contract_from_manifest, parse_batch_contract_from_manifest,
)
from lerobot_coreai.manifest import LeRobotCoreAIManifest
from lerobot_coreai.processor_contract import parse_processor_contract_from_manifest


def _with_contracts(valid_manifest_dict):
    d = dict(valid_manifest_dict)
    d["contracts"] = {
        "action": {"representation": "chunk", "horizon": 8, "action_dim": 7,
                   "selection_semantics": "next_action"},
        "batch": {"runner_supports_batch": False, "max_batch_size": 1,
                  "fallback": "split_and_stack"},
        "processor": {"observation_input": {"expects": "raw_lerobot_observation"},
                      "action_output": {"returns": "postprocessed_action"}},
    }
    return d


def test_manifest_with_contracts_is_schema_valid(valid_manifest_dict):
    d = _with_contracts(valid_manifest_dict)
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "lerobot-coreai.schema.json").read_text())
    jsonschema.validate(d, schema)  # additionalProperties:false must now allow contracts


def test_manifest_exposes_contracts(valid_manifest_dict):
    m = LeRobotCoreAIManifest.from_dict(_with_contracts(valid_manifest_dict))
    assert m.contracts["action"]["horizon"] == 8


def test_v0_manifest_has_empty_contracts(valid_manifest_dict):
    m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
    assert m.contracts == {}


def test_action_contract_read_from_v1_block(valid_manifest_dict):
    m = LeRobotCoreAIManifest.from_dict(_with_contracts(valid_manifest_dict))
    c = parse_action_contract_from_manifest(m)
    # v1 contracts.action wins over shape inference (which would give horizon 16).
    assert c.horizon == 8
    assert c.select_action_semantics == "next_action"


def test_batch_contract_read_from_v1_block(valid_manifest_dict):
    m = LeRobotCoreAIManifest.from_dict(_with_contracts(valid_manifest_dict))
    bc = parse_batch_contract_from_manifest(m)
    assert bc.supports_batch is False and bc.fallback == "split_and_stack"


def test_processor_contract_read_from_v1_block(valid_manifest_dict):
    m = LeRobotCoreAIManifest.from_dict(_with_contracts(valid_manifest_dict))
    pc = parse_processor_contract_from_manifest(m, strict=True)
    assert pc.is_ambiguous() is False
    assert pc.expects == "raw_lerobot_observation"


def test_v0_manifest_falls_back_to_inference(valid_manifest_dict):
    m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
    c = parse_action_contract_from_manifest(m)
    assert c.representation == "chunk"  # inferred from [16,7]
    assert c.horizon == 16
