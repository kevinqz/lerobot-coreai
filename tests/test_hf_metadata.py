# test_hf_metadata.py — honest HF-style CoreAI metadata (v1.1.6).

import json
from importlib.resources import files

import jsonschema
import pytest

from lerobot_coreai.errors import CoreAIPolicyError
from lerobot_coreai.hf_metadata import (
    HF_METADATA_SCHEMA_VERSION, build_hf_metadata, validate_hf_metadata,
)


def test_metadata_is_honest_and_valid():
    m = build_hf_metadata(policy_path="kevinqz/EVO1-SO100-CoreAI", robot_type="so100")
    assert m["schema_version"] == HF_METADATA_SCHEMA_VERSION
    assert m["bridge"]["native_registry"] is False
    assert m["bridge"]["upstream_native"] is False
    assert m["bridge"]["training"] is False
    assert m["safety"]["physical_safety_proof"] is False
    assert m["safety"]["unrestricted_actuation"] is False
    validate_hf_metadata(m)  # does not raise


def test_metadata_schema_valid():
    m = build_hf_metadata()
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "hf-metadata.schema.json").read_text())
    jsonschema.validate(m, schema)


def test_validator_rejects_native_registry_overclaim():
    m = build_hf_metadata()
    m["bridge"]["native_registry"] = True
    with pytest.raises(CoreAIPolicyError):
        validate_hf_metadata(m)


def test_validator_rejects_training_overclaim():
    m = build_hf_metadata()
    m["bridge"]["training"] = True
    with pytest.raises(CoreAIPolicyError):
        validate_hf_metadata(m)


def test_validator_rejects_physical_safety_overclaim():
    m = build_hf_metadata()
    m["safety"]["physical_safety_proof"] = True
    with pytest.raises(CoreAIPolicyError):
        validate_hf_metadata(m)


def test_validator_rejects_missing_section():
    with pytest.raises(CoreAIPolicyError):
        validate_hf_metadata({"library_name": "lerobot-coreai", "runtime": "coreai"})
