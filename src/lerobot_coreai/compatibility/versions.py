# versions.py — LeRobot version compatibility checks (spec §26).
#
# lerobot-coreai 0.4.x supports LeRobot 0.6.x public APIs. If the installed LeRobot version
# is outside the supported range, we warn clearly and allow metadata-only commands, but block
# rollout/eval unless --allow-unsupported-lerobot is passed.

from __future__ import annotations

from typing import Literal

from ..errors import VersionMismatchError

# LeRobot major.minor versions this lerobot-coreai release supports (spec §26).
# Format: set of "<major>.<minor>" strings.
SUPPORTED_LEROBOT_VERSIONS = {"0.6"}

# Baseline version (the one the current artifacts were exported against).
BASELINE_LEROBOT_VERSION = "0.6.0"

# Latest verified version (tested against upstream main).
RECOMMENDED_LEROBOT_VERSION = "0.6.1"


def get_installed_lerobot_version() -> str | None:
    """Return the installed LeRobot version, or None if LeRobot is not installed."""
    try:
        import lerobot  # type: ignore[import-not-found]
        return getattr(lerobot, "__version__", None)
    except ImportError:
        return None


def check_lerobot_compatibility(
    *,
    required_version: str | None = None,
    allow_unsupported: bool = False,
) -> tuple[Literal["supported", "unsupported", "missing"], str]:
    """Check if the installed LeRobot version is compatible.

    Args:
        required_version: The version the artifact was exported against (from the manifest).
            If None, checks against SUPPORTED_LEROBOT_VERSIONS.
        allow_unsupported: If True, returns "unsupported" instead of raising.

    Returns:
        A tuple of (status, message) where status is:
        - 'supported': LeRobot is installed and compatible.
        - 'unsupported': LeRobot is installed but outside the supported range.
        - 'missing': LeRobot is not installed (metadata-only commands still work).

    Raises:
        VersionMismatchError: If LeRobot is unsupported and allow_unsupported is False.
    """
    installed = get_installed_lerobot_version()

    if installed is None:
        return (
            "missing",
            f"LeRobot is not installed. Metadata-only commands work without LeRobot. "
            f"LeRobotDataset eval requires Python 3.12+ and "
            f"`pip install \"lerobot-coreai[lerobot]\"`.",
        )

    # Extract major.minor from the installed version.
    parts = installed.split(".")
    if len(parts) < 2:
        major_minor = installed
    else:
        major_minor = f"{parts[0]}.{parts[1]}"

    if major_minor in SUPPORTED_LEROBOT_VERSIONS:
        return (
            "supported",
            f"LeRobot {installed} is compatible (supported: {', '.join(sorted(SUPPORTED_LEROBOT_VERSIONS))}.x).",
        )

    msg = (
        f"LeRobot {installed} is outside the supported range "
        f"({', '.join(sorted(SUPPORTED_LEROBOT_VERSIONS))}.x). "
        f"This may cause import or API errors. "
        f"Metadata-only commands work; eval/rollout require --allow-unsupported-lerobot."
    )

    if not allow_unsupported:
        raise VersionMismatchError(msg, installed=installed, required=required_version)

    return ("unsupported", msg)
