# lerobot_compat.py — tested LeRobot 0.6.x compatibility certificate (v1.1.2).
#
# Turns the "compatible with LeRobot 0.6.x shape" claim into a verifiable
# artifact. Runs a set of environment/shape checks and emits a compatibility
# report. Importing this module does NOT import torch or lerobot; the LeRobot
# probes are lazy and best-effort. Honest by construction: native-registry,
# training, and physical-safety claims are always false.

from __future__ import annotations

import sys
from typing import Any

from . import __version__
from .lerobot_bridge import LEROBOT_MAX_EXCLUSIVE, LEROBOT_MIN, probe_lerobot

LEROBOT_COMPAT_SCHEMA_VERSION = "lerobot-coreai.lerobot_compat.v0"

# LeRobot 0.6.x needs Python 3.12+; the base package needs 3.10+.
_LEROBOT_MIN_PYTHON = (3, 12)


def _check(name: str, passed: bool, severity: str = "required",
           detail: str = "") -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity,
            "detail": detail}


def _dataset_importable() -> bool:
    """True if the LeRobotDataset module is discoverable.

    Uses ``find_spec`` rather than importing — LeRobot 0.6.x's dataset module
    pulls heavy optional deps (av/torchcodec) at import time that need not be
    present to confirm shape compatibility. LeRobot 0.6.x moved this out of the
    old ``lerobot.common`` namespace to ``lerobot.datasets``.
    """
    import importlib.util
    for path in ("lerobot.datasets.lerobot_dataset",
                 "lerobot.common.datasets.lerobot_dataset"):
        try:
            if importlib.util.find_spec(path) is not None:
                return True
        except Exception:
            continue
    return False


def evaluate_lerobot_compat(*, strict: bool = False) -> dict[str, Any]:
    """Evaluate LeRobot compatibility and return a report dict.

    When ``strict`` is True, a missing LeRobot install (or a version out of
    range) is a required failure. Otherwise LeRobot-dependent checks are
    informational so the certificate can still be produced on a base install.
    Never imports torch/lerobot unless they are already installed.
    """
    lr = probe_lerobot()
    py = sys.version_info[:2]
    checks: list[dict[str, Any]] = []

    checks.append(_check(
        "python_version_compatible", py >= (3, 10), detail=f"python {py[0]}.{py[1]}"))
    checks.append(_check(
        "python_supports_lerobot", py >= _LEROBOT_MIN_PYTHON,
        severity="info", detail=f"lerobot needs >= 3.12; running {py[0]}.{py[1]}"))

    # LeRobot-dependent checks. Required only in strict mode (i.e. the compat CI
    # job, which installs the [lerobot] extra).
    lr_sev = "required" if strict else "info"
    checks.append(_check("lerobot_installed", lr["available"], severity=lr_sev,
                         detail="" if lr["available"] else "[lerobot] extra not installed"))

    if lr["available"]:
        checks.append(_check(
            "lerobot_version_in_range", bool(lr["in_range"]),
            detail=f"lerobot {lr['version']} (want "
                   f">={'.'.join(map(str, LEROBOT_MIN))},"
                   f"<{'.'.join(map(str, LEROBOT_MAX_EXCLUSIVE))})"))
        checks.append(_check(
            "pretrained_policy_importable", lr["pretrained_policy_import"]))
        checks.append(_check(
            "lerobot_dataset_importable", _dataset_importable()))
    elif strict:
        # Strict mode requires these; record them as failed rather than absent.
        checks.append(_check("lerobot_version_in_range", False,
                             detail="lerobot not installed"))
        checks.append(_check("pretrained_policy_importable", False,
                             detail="lerobot not installed"))
        checks.append(_check("lerobot_dataset_importable", False,
                             detail="lerobot not installed"))

    # Bridge shape checks — always available (base package, no torch/lerobot).
    bridge_ok, bridge_detail, honest = _bridge_shape_checks()
    checks.append(_check("coreai_bridge_importable", bridge_ok, detail=bridge_detail))
    checks.append(_check("native_registry_claim_false", honest["native_registry_false"]))
    checks.append(_check("training_claim_false", honest["training_false"]))
    checks.append(_check("physical_safety_claim_false", True,
                         detail="no compatibility artifact claims physical safety"))

    ok = all(c["passed"] for c in checks if c["severity"] == "required")
    return build_compat_report(ok, checks, lr, py)


def _bridge_shape_checks() -> tuple[bool, str, dict[str, bool]]:
    """Verify the bridge imports and its honest-claim invariants — no policy load."""
    honest = {"native_registry_false": False, "training_false": False}
    try:
        from .lerobot_config import BRIDGE_POLICY_TYPE, CoreAIBridgeConfig
        from .lerobot_policy import CoreAILeRobotPolicyBridge
        cfg = CoreAIBridgeConfig()
        honest["native_registry_false"] = cfg.native_registry is False
        honest["training_false"] = cfg.training_supported is False
        # Deliberately not "coreai" — no upstream registry entry.
        shape_ok = (CoreAILeRobotPolicyBridge.policy_type == BRIDGE_POLICY_TYPE
                    == "coreai_bridge")
        return shape_ok, f"policy_type={BRIDGE_POLICY_TYPE!r}", honest
    except Exception as e:  # pragma: no cover - import failure is the failure
        return False, f"{type(e).__name__}: {e}", honest


def build_compat_report(ok: bool, checks: list[dict[str, Any]],
                        lerobot: dict[str, Any],
                        python_version: tuple[int, int]) -> dict[str, Any]:
    return {
        "schema_version": LEROBOT_COMPAT_SCHEMA_VERSION,
        "lerobot_coreai_version": __version__,
        "python_version": f"{python_version[0]}.{python_version[1]}",
        "lerobot_version": lerobot.get("version"),
        "ok": ok,
        "checks": checks,
        "claims": {
            "compatible_with_lerobot_0_6_x_shape": ok and bool(lerobot.get("in_range")),
            "native_upstream_registry": False,
            "supports_training": False,
            "supports_physical_safety": False,
        },
    }


def build_compat_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# LeRobot Compatibility Certificate",
        "",
        f"- OK: {report.get('ok')}",
        f"- lerobot-coreai: {report.get('lerobot_coreai_version')}",
        f"- Python: {report.get('python_version')}",
        f"- LeRobot: {report.get('lerobot_version')}",
        "",
        "## Checks",
    ]
    for c in report.get("checks", []):
        mark = "✅" if c["passed"] else "❌"
        detail = f" — {c['detail']}" if c.get("detail") else ""
        lines.append(f"- {mark} `{c['name']}` ({c['severity']}){detail}")
    lines += [
        "",
        "Tested compatibility with the LeRobot 0.6.x **shape** only. This is a "
        "**local, runtime-only** bridge — `policy_type=\"coreai\"` is not "
        "registered upstream, training remains LeRobot's job, and nothing here "
        "proves physical safety.",
        "",
    ]
    return "\n".join(lines)
