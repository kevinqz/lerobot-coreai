# runtime_support.py — the RuntimeSupportProfile (v1.3.21).
#
# The universal contract schemas (BatchContract v3, stage vocabulary) describe a
# GENERAL language. This profile declares the SUBSET the certified lerobot-coreai
# 1.3.x runtime actually supports — separating "expressible" from "supported" (P1.14).
# A native `shared`/`unknown` slot isolation or a `session_scoped`/`global` split
# scope is expressible in the contract but NOT in this profile. Pure Python.

from __future__ import annotations

from .stages import ACTION_STAGES, OBSERVATION_STAGES

RUNTIME_SUPPORT_SCHEMA_VERSION = "lerobot-coreai.runtime-support.v1"

# The frozen support profile of the 1.3.x runtime.
RUNTIME_SUPPORT_PROFILE = {
    "schema_version": RUNTIME_SUPPORT_SCHEMA_VERSION,
    "batch": {
        "native_slot_isolation": ["independent"],
        "split_state_scopes": ["stateless", "request_scoped"],
    },
    "observation_stages": list(OBSERVATION_STAGES),
    "action_stages": list(ACTION_STAGES),
}

RUNTIME_SUPPORT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "batch", "observation_stages", "action_stages"],
    "properties": {
        "schema_version": {"const": RUNTIME_SUPPORT_SCHEMA_VERSION},
        "batch": {
            "type": "object", "additionalProperties": False,
            "required": ["native_slot_isolation", "split_state_scopes"],
            "properties": {
                # certified runtime: native ONLY independent; split ONLY stateless /
                # request_scoped (session_scoped / global are out of profile).
                "native_slot_isolation": {
                    "type": "array", "items": {"enum": ["independent"]},
                    "uniqueItems": True},
                "split_state_scopes": {
                    "type": "array",
                    "items": {"enum": ["stateless", "request_scoped"]},
                    "uniqueItems": True}}},
        "observation_stages": {"type": "array", "items": {"enum": list(OBSERVATION_STAGES)}},
        "action_stages": {"type": "array", "items": {"enum": list(ACTION_STAGES)}},
    },
}


def runtime_support_profile() -> dict:
    """The canonical, frozen RuntimeSupportProfile for this runtime."""
    return {**RUNTIME_SUPPORT_PROFILE,
            "batch": {k: list(v) for k, v in RUNTIME_SUPPORT_PROFILE["batch"].items()},
            "observation_stages": list(RUNTIME_SUPPORT_PROFILE["observation_stages"]),
            "action_stages": list(RUNTIME_SUPPORT_PROFILE["action_stages"])}
