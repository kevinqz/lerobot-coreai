# artifact_schemas.py — strict JSON schemas for the plugin artifact (v1.3.7).
#
# Embedded as dicts (not package-data files) so they load with no packaging config
# and are trivially unit-testable. All use additionalProperties=false: an unknown
# key is a validation failure, so the schema is real evidence, not a loose hint.

from __future__ import annotations

_SHA256 = {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}

PLUGIN_ARTIFACT_INVENTORY_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "files", "artifact_root_sha256"],
    "properties": {
        "schema_version": {"const": "lerobot-coreai.plugin_inventory.v1"},
        "artifact_root_sha256": _SHA256,
        "files": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "role", "sha256", "size_bytes"],
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "role": {"type": "string", "minLength": 1},
                    "sha256": _SHA256,
                    "size_bytes": {"type": "integer", "minimum": 0},
                },
            },
        },
    },
}

PROCESSOR_CONTRACT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "observation_input", "action_output"],
    "properties": {
        "schema_version": {"const": "coreai-processor-contract.v2"},
        "observation_input": {
            "type": "object",
            "additionalProperties": False,
            "required": ["owner", "expects"],
            "properties": {
                "owner": {"type": "string"},
                "expects": {"type": "string"},
            },
        },
        "action_output": {
            "type": "object",
            "additionalProperties": False,
            "required": ["owner", "returns"],
            "properties": {
                "owner": {"type": "string"},
                "returns": {"type": "string"},
            },
        },
    },
}

PLUGIN_ARTIFACT_MANIFEST_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "policy_type", "files", "versions", "runner",
                 "action_contract", "source_coreai_artifact_reference", "claims"],
    "properties": {
        "schema_version": {"const": "lerobot-coreai.plugin_artifact.v1"},
        "policy_type": {"const": "coreai_bridge"},
        "files": {
            "type": "object",
            "additionalProperties": False,
            "required": ["config", "preprocessor", "postprocessor", "coreai_manifest",
                         "inventory"],
            "properties": {
                "config": {"type": "string"},
                "preprocessor": {"type": "string"},
                "postprocessor": {"type": "string"},
                "coreai_manifest": {"type": "string"},
                "inventory": {"type": "string"},
            },
        },
        "versions": {
            "type": "object",
            "additionalProperties": False,
            "required": ["lerobot_coreai", "lerobot_policy_coreai_bridge", "lerobot"],
            "properties": {
                "lerobot_coreai": {"type": "string"},
                "lerobot_policy_coreai_bridge": {"type": "string"},
                "lerobot": {"type": ["string", "null"]},
            },
        },
        "runner": {
            "type": "object",
            "additionalProperties": False,
            "required": ["runner_url_env", "minimum_runner_protocol"],
            "properties": {
                "runner_url_env": {"type": "string"},
                "minimum_runner_protocol": {"type": "string"},
            },
        },
        "action_contract": {"type": "object"},
        "source_coreai_artifact_reference": {
            "type": "object",
            "additionalProperties": False,
            "required": ["mode", "manifest_sha256"],
            "properties": {
                "mode": {"enum": ["embedded", "external"]},
                "manifest_sha256": _SHA256,
                "repo": {"type": ["string", "null"]},
                "revision": {"type": ["string", "null"]},
            },
        },
        "claims": {
            "type": "object",
            "required": ["official_eval_certified", "upstream_native",
                         "supports_training", "proves_task_success",
                         "proves_physical_safety"],
        },
    },
}

PLUGIN_ARTIFACT_VERIFICATION_REPORT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "checks", "claims"],
    "properties": {
        "schema_version": {"const": "lerobot-coreai.plugin_artifact_verification.v1"},
        "artifact_root_sha256": {"type": ["string", "null"]},
        "checks": {"type": "object"},
        "claims": {
            "type": "object",
            "additionalProperties": False,
            "required": ["integrity_verified", "authenticity_verified",
                         "processor_contract_verified", "factory_b1_certified",
                         "official_eval_certified", "proves_physical_safety"],
            "properties": {
                "integrity_verified": {"type": "boolean"},
                "authenticity_verified": {"type": "boolean"},
                "processor_contract_verified": {"type": "boolean"},
                "factory_b1_certified": {"type": "boolean"},
                "official_eval_certified": {"type": "boolean"},
                "proves_physical_safety": {"type": "boolean"},
            },
        },
    },
}
