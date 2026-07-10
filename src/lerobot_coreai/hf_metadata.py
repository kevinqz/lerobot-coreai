# hf_metadata.py — honest Hugging Face-style metadata for CoreAI artifacts (v1.1.6).
#
# Makes CoreAI artifacts discoverable as LeRobot-shaped *runtime* artifacts
# without overclaiming. The metadata is honest by construction and validated:
# native registry, upstream-native integration, training, physical-safety proof,
# and unrestricted actuation are all always false.

from __future__ import annotations

from typing import Any

from . import __version__
from .errors import CoreAIPolicyError
from .lerobot_bridge import LEROBOT_MAX_EXCLUSIVE, LEROBOT_MIN

HF_METADATA_SCHEMA_VERSION = "lerobot-coreai.hf_metadata.v0"

# Fields that must be present and must be exactly False.
_FALSE_INVARIANTS = {
    ("bridge", "training"),
    ("bridge", "native_registry"),
    ("bridge", "upstream_native"),
    ("safety", "physical_safety_proof"),
    ("safety", "unrestricted_actuation"),
}


def _compat_range() -> str:
    return f"{LEROBOT_MIN[0]}.{LEROBOT_MIN[1]}.x"


def build_hf_metadata(*, policy_path: str | None = None,
                      robot_type: str | None = None) -> dict[str, Any]:
    """Build honest HF-style metadata for a CoreAI artifact."""
    return {
        "schema_version": HF_METADATA_SCHEMA_VERSION,
        "library_name": "lerobot-coreai",
        "base_library": "lerobot",
        "lerobot_compatibility": _compat_range(),
        "lerobot_coreai_version": __version__,
        "runtime": "coreai",
        "policy_path": policy_path,
        "robot_type": robot_type,
        "bridge": {
            "select_action": True,
            "training": False,
            "native_registry": False,
            "upstream_native": False,
        },
        "safety": {
            "physical_safety_proof": False,
            "unrestricted_actuation": False,
        },
    }


def validate_hf_metadata(metadata: dict[str, Any]) -> None:
    """Validate HF metadata, failing closed on any overclaim.

    Raises CoreAIPolicyError if a required section is missing or any honesty
    invariant (native registry / upstream-native / training / physical-safety
    proof / unrestricted actuation) is not exactly False.
    """
    if metadata.get("library_name") != "lerobot-coreai":
        raise CoreAIPolicyError("hf metadata: library_name must be 'lerobot-coreai'.")
    if metadata.get("runtime") != "coreai":
        raise CoreAIPolicyError("hf metadata: runtime must be 'coreai'.")
    for section, key in _FALSE_INVARIANTS:
        sec = metadata.get(section)
        if not isinstance(sec, dict) or key not in sec:
            raise CoreAIPolicyError(
                f"hf metadata: missing {section}.{key}.")
        if sec[key] is not False:
            raise CoreAIPolicyError(
                f"hf metadata: {section}.{key} must be False (overclaim refused).")


def build_hf_metadata_markdown(metadata: dict[str, Any]) -> str:
    b = metadata.get("bridge", {})
    return (
        "# lerobot-coreai HF Metadata\n\n"
        f"- library: {metadata.get('library_name')} "
        f"(base: {metadata.get('base_library')})\n"
        f"- LeRobot compatibility: {metadata.get('lerobot_compatibility')}\n"
        f"- runtime: {metadata.get('runtime')}\n"
        f"- bridge.select_action: {b.get('select_action')}\n"
        f"- bridge.training: {b.get('training')}\n"
        f"- bridge.native_registry: {b.get('native_registry')}\n"
        f"- bridge.upstream_native: {b.get('upstream_native')}\n\n"
        "Local runtime bridge — not upstream-native LeRobot. Train with LeRobot, "
        "run with CoreAI. Proves nothing about physical safety.\n"
    )
