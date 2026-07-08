# safety.py — safety mode enforcement for rollout (v0.3).

from __future__ import annotations

from .errors import SafetyError


def ensure_mode_supported_for_v03(
    mode: str,
    *,
    confirm_real_robot_actuation: bool = False,
) -> None:
    """Check if the rollout mode is supported in v0.3.

    v0.3 only supports dry_run. shadow, sim, and real are blocked.

    Args:
        mode: The rollout mode (dry_run, shadow, sim, real).
        confirm_real_robot_actuation: Whether the confirmation flag was passed.

    Raises:
        SafetyError: If the mode is not supported in v0.3.
    """
    if mode == "dry_run":
        return  # Supported.

    if mode == "real":
        raise SafetyError(
            f"mode='real' is not implemented in lerobot-coreai v0.3.0. "
            f"Real mode is planned for v1.0.0.\n"
            f"No robot commands were sent."
        )

    # shadow, sim, or anything else
    raise SafetyError(
        f"mode='{mode}' is not implemented in lerobot-coreai v0.3.0. "
        f"Only dry_run is supported.\n"
        f"No robot commands were sent."
    )


def assert_no_physical_actuation_available() -> None:
    """v0.3 has no code path for sending robot commands.

    This function exists as a documented invariant — it always returns None
    because there is no actuation code in the codebase. The test suite checks
    for banned hardware tokens to enforce this.
    """
    return None
