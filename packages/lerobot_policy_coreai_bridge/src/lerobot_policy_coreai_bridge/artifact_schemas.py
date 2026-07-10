# artifact_schemas.py — strict JSON schemas for the plugin artifact (v1.3.7/1.3.8).
#
# Embedded as dicts (not package-data files) so they load with no packaging config
# and are trivially unit-testable. All use additionalProperties=false: an unknown
# key is a validation failure, so the schema is real evidence, not a loose hint.

from __future__ import annotations

_SHA256 = {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}

# Closed set of inventory roles (v1.3.8): exactly one file per role.
ARTIFACT_ROLES = ("policy_config", "policy_preprocessor", "policy_postprocessor",
                  "coreai_manifest", "plugin_manifest", "readme")

PLUGIN_ARTIFACT_INVENTORY_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "files", "artifact_root_sha256",
                 "artifact_root_algorithm"],
    "properties": {
        "schema_version": {"const": "lerobot-coreai.plugin_inventory.v1"},
        "artifact_root_sha256": _SHA256,
        "artifact_root_algorithm": {"const": "canonical-json-sha256.v1"},
        "files": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "role", "sha256", "size_bytes"],
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "role": {"enum": list(ARTIFACT_ROLES)},
                    "sha256": _SHA256,
                    "size_bytes": {"type": "integer", "minimum": 0},
                },
            },
        },
    },
}

# v1.3.8: schema_version is OPTIONAL (a CoreAI manifest's contracts.processor block
# need not carry it), but if present it is pinned. additionalProperties=false makes
# an unknown/deformed contract fail even when the four required strings are present.
PROCESSOR_CONTRACT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["observation_input", "action_output"],
    "properties": {
        "schema_version": {"const": "coreai-processor-contract.v2"},
        "observation_input": {
            "type": "object",
            "additionalProperties": False,
            "required": ["owner", "expects"],
            "properties": {"owner": {"type": "string"}, "expects": {"type": "string"}},
        },
        "action_output": {
            "type": "object",
            "additionalProperties": False,
            "required": ["owner", "returns"],
            "properties": {"owner": {"type": "string"}, "returns": {"type": "string"}},
        },
    },
}

# Strict action contract (v1.3.8): representation enum, horizon>=1, single => 1.
ACTION_CONTRACT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["representation", "horizon"],
    "properties": {
        "representation": {"enum": ["single", "chunk"]},
        "horizon": {"type": "integer", "minimum": 1},
        "action_dim": {"type": ["integer", "null"], "minimum": 1},
        "select_action_semantics": {"type": "string"},
        "predict_action_chunk_semantics": {"type": "string"},
        "queue_owner": {"type": "string"},
        "reset_clears_queue": {"type": "boolean"},
        "temporal_ensembling": {"type": "boolean"},
    },
    "if": {"properties": {"representation": {"const": "single"}}},
    "then": {"properties": {"horizon": {"const": 1}}},
}

# Closed claims (v1.3.8): forbidden claims pinned false; factory-compat null|false.
CLAIMS_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["official_plugin_factory_compatible", "official_eval_certified",
                 "upstream_native", "supports_training", "proves_task_success",
                 "proves_physical_safety"],
    "properties": {
        "official_plugin_factory_compatible": {"enum": [None, False]},
        "official_eval_certified": {"const": False},
        "upstream_native": {"const": False},
        "supports_training": {"const": False},
        "proves_task_success": {"const": False},
        "proves_physical_safety": {"const": False},
    },
}

_SOURCE_REF_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["mode", "embedded_manifest_sha256"],
    "properties": {
        "mode": {"enum": ["embedded", "external"]},
        "embedded_manifest_sha256": _SHA256,
        "source_repo": {"type": ["string", "null"]},
        "requested_ref": {"type": ["string", "null"]},
        "resolved_commit_sha": {"type": ["string", "null"],
                                "pattern": r"^[0-9a-f]{40}$"},
    },
}

BATCH_CONTRACT_V3_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "native_batch", "client_split", "fallback", "queue"],
    "properties": {
        "schema_version": {"const": "coreai-batch-contract.v3"},
        "native_batch": {
            "type": "object", "additionalProperties": False,
            "required": ["supported", "max_batch_size", "required_slot_isolation"],
            "properties": {
                "supported": {"type": "boolean"},
                "max_batch_size": {"type": "integer", "minimum": 1},
                # runtime only certifies independent native slots (v1.3.11).
                "required_slot_isolation": {"const": "independent"},
            }},
        "client_split": {
            "type": "object", "additionalProperties": False,
            "required": ["supported", "max_batch_size", "allowed_state_scopes"],
            "properties": {
                "supported": {"type": "boolean"},
                "max_batch_size": {"type": "integer", "minimum": 1},
                "allowed_state_scopes": {
                    "type": "array", "minItems": 1,
                    "items": {"enum": ["stateless", "request_scoped"]}},
            }},
        "fallback": {"enum": ["split_and_stack", "reject"]},
        "queue": {
            "type": "object", "additionalProperties": False,
            "required": ["layout", "commit_semantics"],
            "properties": {
                "layout": {"const": "time_major_batched"},
                "commit_semantics": {"const": "atomic_queue_commit"}}},
        "observation_stage": {"type": "string"},
    },
}

PLUGIN_ARTIFACT_MANIFEST_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "policy_type", "files", "versions", "runner",
                 "action_contract", "batch_contract", "batch_contract_sha256",
                 "processor_stage_contract", "source_coreai_artifact_reference",
                 "claims"],
    "properties": {
        "schema_version": {"const": "lerobot-coreai.plugin_artifact.v1"},
        "policy_type": {"const": "coreai_bridge"},
        "files": {
            "type": "object",
            "additionalProperties": False,
            "required": ["config", "preprocessor", "postprocessor", "coreai_manifest",
                         "inventory"],
            "properties": {
                "config": {"type": "string"}, "preprocessor": {"type": "string"},
                "postprocessor": {"type": "string"},
                "coreai_manifest": {"type": "string"}, "inventory": {"type": "string"},
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
        "action_contract": ACTION_CONTRACT_SCHEMA,
        "batch_contract": BATCH_CONTRACT_V3_SCHEMA,
        "batch_contract_sha256": _SHA256,
        "processor_stage_contract": {
            "type": "object",
            "additionalProperties": False,
            "required": ["observation_input_stage", "action_output_stage"],
            "properties": {
                "observation_input_stage": {"type": "string"},
                "action_output_stage": {"type": "string"},
            }},
        "source_coreai_artifact_reference": _SOURCE_REF_SCHEMA,
        "claims": CLAIMS_SCHEMA,
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
        "semantics": {"type": "object"},
        "claims": {
            "type": "object",
            "additionalProperties": False,
            "required": ["integrity_verified", "authenticity_verified",
                         "processor_contract_verified",
                         "semantic_consistency_verified",
                         "semantic_completeness_verified",
                         "factory_b1_certified", "official_eval_certified",
                         "proves_physical_safety"],
            "properties": {
                "integrity_verified": {"type": "boolean"},
                "authenticity_verified": {"type": "boolean"},
                "processor_contract_verified": {"type": "boolean"},
                "semantic_consistency_verified": {"type": "boolean"},
                "semantic_completeness_verified": {"type": "boolean"},
                "factory_b1_certified": {"type": "boolean"},
                "official_eval_certified": {"type": "boolean"},
                "proves_physical_safety": {"type": "boolean"},
            },
        },
    },
}
