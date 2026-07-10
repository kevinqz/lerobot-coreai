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
EXTERNAL_HTTP_SAFETY_STATE_SCHEMA_VERSION = "lerobot-coreai.external_http_safety_state.v0"

# Only these safety-state values are safe to proceed on. Anything else —
# including "unknown" — fails closed.
_ESTOP_OK = "armed"
_WORKSPACE_OK = "clear"


def _schema() -> dict[str, Any]:
    return json.loads(files("lerobot_coreai.schemas").joinpath(
        "external-http-preflight.schema.json").read_text())


def _safety_state_schema() -> dict[str, Any]:
    return json.loads(files("lerobot_coreai.schemas").joinpath(
        "external-http-safety-state.schema.json").read_text())


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
    # v1.1.1: observation, safety-state, and physical e-stop are required for a
    # non-mock real controller. A controller that can't report its safety state
    # or doesn't require a physical e-stop must never receive guarded actions.
    checks.append(("external_controller_supports_observation",
                   preflight.get("supports_observation") is True, ""))
    checks.append(("external_controller_supports_safety_state",
                   preflight.get("supports_safety_state") is True, ""))
    checks.append(("external_controller_physical_estop_required",
                   preflight.get("physical_estop_required") is True, ""))
    return checks


def validate_controller_safety_state(
    safety_state: dict[str, Any], *, robot_type: str,
) -> list[tuple[str, bool, str]]:
    """Validate an external-http controller's /safety-state before guarded egress.

    Returns a list of (check_name, passed, message). Fail-closed: a malformed
    response, an unknown e-stop/workspace value, any fault, or a not-ready
    controller all block. Never sends an action.
    """
    checks: list[tuple[str, bool, str]] = []
    try:
        import jsonschema
        jsonschema.validate(safety_state, _safety_state_schema())
        checks.append(("external_controller_safety_state_schema_valid", True, ""))
    except Exception as e:
        checks.append(("external_controller_safety_state_schema_valid", False,
                       getattr(e, "message", str(e))))
        return checks  # can't trust any field

    ready = safety_state.get("ready") is True
    checks.append(("external_controller_safety_state_ready", ready, ""))

    st = safety_state.get("robot_type")
    checks.append(("external_controller_safety_state_robot_type_matches",
                   st == robot_type,
                   "" if st == robot_type else f"controller {st} != {robot_type}"))

    checks.append(("external_controller_connected",
                   safety_state.get("controller_connected") is True, ""))

    estop = safety_state.get("physical_estop_state")
    checks.append(("external_controller_estop_armed", estop == _ESTOP_OK,
                   "" if estop == _ESTOP_OK else f"physical_estop_state={estop!r}"))

    workspace = safety_state.get("workspace_state")
    checks.append(("external_controller_workspace_clear", workspace == _WORKSPACE_OK,
                   "" if workspace == _WORKSPACE_OK else f"workspace_state={workspace!r}"))

    faults = safety_state.get("faults")
    faults_empty = isinstance(faults, list) and len(faults) == 0
    checks.append(("external_controller_faults_empty", faults_empty,
                   "" if faults_empty else f"faults={faults!r}"))
    return checks
