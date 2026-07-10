# test_processor_contract.py — processor ownership contract (v1.2.6).

import json
from importlib.resources import files

import jsonschema
import pytest

from lerobot_coreai.processor_contract import (
    PROCESSOR_CONTRACT_SCHEMA_VERSION, ProcessorContractError,
    build_processor_contract_report, parse_processor_contract_from_manifest,
)

_FULL = {
    "processor_contract": {
        "observation_input": {"expects": "raw_lerobot_observation",
                              "image_layout": "CHW", "image_range": [0.0, 1.0]},
        "action_output": {"returns": "postprocessed_action", "action_order": ["x", "y"]},
        "stats": {"dataset_stats_sha256": "sha256:abc"},
    }
}


def test_full_contract_not_ambiguous():
    c = parse_processor_contract_from_manifest(_FULL)
    assert c.is_ambiguous() is False
    assert c.expects == "raw_lerobot_observation"
    assert c.returns == "postprocessed_action"


def test_missing_contract_is_ambiguous():
    c = parse_processor_contract_from_manifest({})
    assert c.is_ambiguous() is True


def test_strict_fails_on_ambiguous():
    with pytest.raises(ProcessorContractError):
        parse_processor_contract_from_manifest({}, strict=True)


def test_strict_passes_on_full():
    c = parse_processor_contract_from_manifest(_FULL, strict=True)
    assert c.is_ambiguous() is False


def test_unknown_mode_is_ambiguous():
    bad = {"processor_contract": {"observation_input": {"expects": "weird"},
                                  "action_output": {"returns": "postprocessed_action"}}}
    assert parse_processor_contract_from_manifest(bad).is_ambiguous() is True


def test_report_schema_valid_and_honest():
    report = build_processor_contract_report(
        parse_processor_contract_from_manifest(_FULL), strict=True)
    assert report["schema_version"] == PROCESSOR_CONTRACT_SCHEMA_VERSION
    assert report["ambiguous"] is False
    assert report["claims"]["proves_physical_safety"] is False
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "processor-contract-report.schema.json").read_text())
    jsonschema.validate(report, schema)
