# catalog.py — client for the coreai-catalog LeRobot compatibility view.
#
# Queries dist/lerobot-coreai.json from the coreai-catalog GitHub Pages. This is the
# LeRobot-specific index of all CoreAI artifacts that are compatible with LeRobot workflows.

from __future__ import annotations

from typing import Any

import requests

# The catalog publishes a LeRobot-specific dist file (spec §18.3).
# Falls back to the full search-index.json if lerobot-coreai.json is not yet published.
_CATALOG_LEROBOT_URL = (
    "https://raw.githubusercontent.com/kevinqz/coreai-catalog/main/dist/lerobot-coreai.json"
)
_CATALOG_SEARCH_URL = (
    "https://raw.githubusercontent.com/kevinqz/coreai-catalog/main/dist/search-index.json"
)

_cache: dict[str, Any] = {}


def list_lerobot_policies() -> list[dict[str, Any]]:
    """List all LeRobot-compatible CoreAI policies from the catalog.

    Returns a list of policy dicts with keys:
        repo_id, catalog_model_id, policy_type, robot_type, runtime, status, default_mode

    Falls back to filtering search-index.json by bundle_kind=action + family=LeRobot
    if the LeRobot-specific dist file is not yet published.
    """
    # Try the LeRobot-specific index first.
    try:
        resp = requests.get(_CATALOG_LEROBOT_URL, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("policies", [])
    except requests.RequestException:
        pass

    # Fallback: filter the full search index.
    try:
        resp = requests.get(_CATALOG_SEARCH_URL, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("models", [])
            # Filter to LeRobot fabric action entries.
            return [
                {
                    "repo_id": m.get("id", ""),
                    "catalog_model_id": m.get("id", ""),
                    "policy_type": _infer_policy_type(m.get("id", "")),
                    "robot_type": _infer_robot_type(m.get("id", "")),
                    "runtime": "coreai",
                    "status": "indexed",
                    "default_mode": "dry_run",
                }
                for m in models
                if m.get("bundle_kind") == "action" and m.get("family") == "LeRobot"
            ]
    except requests.RequestException:
        pass

    return []


def _infer_policy_type(model_id: str) -> str:
    """Infer the policy type from a model ID (best-effort)."""
    ml = model_id.lower()
    for t in ("pi0fast", "pi05", "pi0", "smolvla", "vqbet", "diffusion", "evo1", "act", "fastwam", "bitvla"):
        if t in ml:
            return t
    return "unknown"


def _infer_robot_type(model_id: str) -> str:
    """Infer the robot type from a model ID (best-effort)."""
    ml = model_id.lower()
    for r in ("so100", "so101", "aloha", "libero"):
        if r in ml:
            return r
    return "unknown"
