# external_http_contract.py — capability contract for external-http controllers (v1.0.3).
#
# Before a guarded real session may egress through an operator-run external-http
# controller, the controller's /preflight response must declare a rigid
# capability contract, and it must be coherent with the requested robot type,
# safety profile action shape, and fps. Fail-closed: an underspecified or
# mismatched controller is refused.

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

EXTERNAL_HTTP_SCHEMA_VERSION = "lerobot-coreai.external_http.v0"


def _schema() -> dict[str, Any]:
    return json.loads(files("lerobot_coreai.schemas").joinpath(
        "external-http-preflight.schema.json").read_text())


def validate_controller_preflight(
    preflight: dict[str, Any], *, robot_type: str,
    profile_action_shape: list[int] | None, requested_fps: float | None,
) -> list[tuple[str, bool, str]]:
    """Validate an external-http controller's /preflight capabilities.

    Returns a list of (check_name, passed, message). Fail-closed: a malformed
    response fails every downstream check.
    """
    checks: list[tuple[str, bool, str]] = []

    # Schema.
    try:
        import jsonschema
        jsonschema.validate(preflight, _schema())
        checks.append(("external_controller_schema_valid", True, ""))
    except Exception as e:
        checks.append(("external_controller_schema_valid", False,
                       getattr(e, "message", str(e))))
        # If the schema is invalid we can't trust any field; stop here.
        return checks

    ct = preflight.get("robot_type")
    checks.append(("external_controller_robot_type_matches", ct == robot_type,
                   "" if ct == robot_type else f"controller {ct} != {robot_type}"))

    ca = preflight.get("action_shape")
    shape_ok = profile_action_shape is None or ca == profile_action_shape
    checks.append(("external_controller_action_shape_matches_profile", shape_ok,
                   "" if shape_ok else f"controller {ca} != profile {profile_action_shape}"))

    cmax = preflight.get("max_fps")
    fps_ok = (requested_fps is None or cmax is None or requested_fps <= cmax)
    checks.append(("external_controller_max_fps_allows_requested_fps", fps_ok,
                   "" if fps_ok else f"requested {requested_fps} > controller max {cmax}"))

    checks.append(("external_controller_supports_stop",
                   preflight.get("supports_stop") is True, ""))
    checks.append(("external_controller_supports_ready",
                   preflight.get("supports_ready") is True, ""))
    return checks
