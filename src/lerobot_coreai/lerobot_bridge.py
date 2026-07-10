# lerobot_bridge.py — local LeRobot bridge for CoreAI policies (v1.1.0).
#
# Public entry point: `load_coreai_policy_for_lerobot(...)` returns a
# CoreAILeRobotPolicyBridge — a LeRobot-shaped, runtime-only wrapper around a
# CoreAIPolicy. This is a LOCAL bridge, not upstream-native LeRobot integration:
# it registers nothing in LeRobot's registry/factory, monkeypatches nothing
# globally, and imports neither torch nor lerobot at module load. The optional
# LeRobot import/version probe below is lazy and best-effort.

from __future__ import annotations

from typing import Any

from . import __version__
from .errors import CoreAIPolicyError
from .lerobot_policy import CoreAILeRobotPolicyBridge
from .policy import CoreAIPolicy

BRIDGE_REPORT_SCHEMA_VERSION = "lerobot-coreai.lerobot_bridge.v0"

# The LeRobot version range this bridge targets.
LEROBOT_MIN = (0, 6, 0)
LEROBOT_MAX_EXCLUSIVE = (0, 7, 0)


def load_coreai_policy_for_lerobot(
    policy_path: str,
    *,
    runner_url: str | None = None,
    validate_runner: bool = False,
    **kwargs: Any,
) -> CoreAILeRobotPolicyBridge:
    """Load a CoreAI policy and return it wrapped in a LeRobot-shaped bridge.

    Args:
        policy_path: HF repo id of the CoreAI artifact.
        runner_url: coreai-runner URL/socket. Required for actual inference.
        validate_runner: If True, validate the runner is reachable at load time.
        **kwargs: Forwarded to :meth:`CoreAIPolicy.from_pretrained`.

    Returns:
        A :class:`CoreAILeRobotPolicyBridge`. This is a local bridge — not an
        upstream-native LeRobot policy.
    """
    coreai_policy = CoreAIPolicy.from_pretrained(
        policy_path, runner_url=runner_url, validate_runner=validate_runner,
        return_metadata=True, **kwargs)
    return CoreAILeRobotPolicyBridge(coreai_policy)


def _parse_version(v: str) -> tuple[int, int, int]:
    parts = []
    for chunk in v.split(".")[:3]:
        num = ""
        for ch in chunk:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num) if num else 0)
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


def probe_lerobot() -> dict[str, Any]:
    """Best-effort probe of the optional LeRobot install.

    Never raises. Returns availability, version, and whether the version is in
    the supported range. Imports lerobot lazily so the core package stays
    torch/lerobot-free.
    """
    result: dict[str, Any] = {
        "available": False, "version": None, "in_range": None,
        "pretrained_policy_import": False,
    }
    try:
        import importlib.metadata as _md
        version = _md.version("lerobot")
    except Exception:
        return result
    result["available"] = True
    result["version"] = version
    try:
        parsed = _parse_version(version)
        result["in_range"] = LEROBOT_MIN <= parsed < LEROBOT_MAX_EXCLUSIVE
    except Exception:
        result["in_range"] = None
    # Confirm the PreTrainedPolicy import path exists (we do NOT subclass it).
    try:  # pragma: no cover - only runs when lerobot is installed
        __import__("lerobot.common.policies.pretrained")
        result["pretrained_policy_import"] = True
    except Exception:
        try:  # pragma: no cover
            __import__("lerobot.policies.pretrained")
            result["pretrained_policy_import"] = True
        except Exception:
            result["pretrained_policy_import"] = False
    return result


def _check(name: str, passed: bool, severity: str = "required",
           detail: str = "") -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity,
            "detail": detail}


def evaluate_bridge_check(
    policy_path: str,
    *,
    runner_url: str | None = None,
    dataset_repo_id: str | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Run the bridge checks and return a report dict. Never sends a robot action.

    All inference here is dataset/mock only — this is a compatibility probe, not
    a robot session.
    """
    checks: list[dict[str, Any]] = []
    bridge: CoreAILeRobotPolicyBridge | None = None
    lerobot = probe_lerobot()

    # 1. Policy loads.
    try:
        bridge = load_coreai_policy_for_lerobot(
            policy_path, runner_url=runner_url,
            validate_runner=bool(runner_url))
        checks.append(_check("coreai_policy_loads", True))
    except Exception as e:
        checks.append(_check("coreai_policy_loads", False, detail=f"{type(e).__name__}: {e}"))

    # 2. Runner reachability (only if a runner_url was supplied — validate_runner
    #    above already enforced it, so a loaded bridge implies reachable).
    if runner_url:
        checks.append(_check("runner_reachable", bridge is not None,
                             detail="validated at load" if bridge else "load failed"))

    # 3. Bridge shape checks (no network needed).
    if bridge is not None:
        checks.append(_check("select_action_callable", callable(bridge.select_action)))
        checks.append(_check("predict_action_metadata_available",
                             callable(bridge.predict_action)))
        # train(True) must fail.
        train_guarded = False
        try:
            bridge.train(True)
        except CoreAIPolicyError:
            train_guarded = True
        except Exception:
            train_guarded = False
        checks.append(_check("no_training_claim", train_guarded,
                             detail="train(True) raises" if train_guarded
                             else "train(True) did not raise"))
        # eval/to are safe no-ops returning the bridge.
        safe_noops = (bridge.eval() is bridge and bridge.to("cuda") is bridge)
        checks.append(_check("eval_to_safe_noops", safe_noops))
        # Honest metadata.
        md = bridge.metadata()
        checks.append(_check("no_native_registry_claim",
                             md.get("native_registry") is False))
        checks.append(_check("no_training_supported_claim",
                             md.get("training_supported") is False))

    # 4. Optional LeRobot version check (informational unless installed).
    if lerobot["available"]:
        checks.append(_check("lerobot_version_in_range", bool(lerobot["in_range"]),
                             severity="required" if lerobot["in_range"] is not None
                             else "info",
                             detail=f"lerobot {lerobot['version']}"))
    else:
        checks.append(_check("lerobot_installed", False, severity="info",
                             detail="[lerobot] extra not installed — bridge still usable"))

    # 5. Optional dataset item -> batch smoke (only if requested AND lerobot present).
    if dataset_repo_id:
        if lerobot["available"] and bridge is not None:
            ok, detail = _dataset_smoke(bridge, dataset_repo_id, max_frames or 1)
            checks.append(_check("dataset_item_to_batch", ok, severity="info", detail=detail))
        else:
            checks.append(_check("dataset_item_to_batch", False, severity="info",
                                 detail="skipped: lerobot not installed or policy load failed"))

    ok = all(c["passed"] for c in checks if c["severity"] == "required")
    return build_bridge_report(policy_path, ok, checks, lerobot, runner_url)


def _dataset_smoke(bridge: CoreAILeRobotPolicyBridge, dataset_repo_id: str,
                   max_frames: int) -> tuple[bool, str]:  # pragma: no cover - needs lerobot+net
    """Best-effort: load one dataset frame and confirm it shapes into a batch."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except Exception:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
        except Exception as e:
            return False, f"LeRobotDataset import failed: {e}"
    try:
        ds = LeRobotDataset(dataset_repo_id)
        item = ds[0]
        return bool(item), f"loaded 1/{max_frames} frame(s) from {dataset_repo_id}"
    except Exception as e:
        return False, f"dataset load failed: {type(e).__name__}: {e}"


def build_bridge_report(policy_path: str, ok: bool, checks: list[dict[str, Any]],
                        lerobot: dict[str, Any],
                        runner_url: str | None) -> dict[str, Any]:
    return {
        "schema_version": BRIDGE_REPORT_SCHEMA_VERSION,
        "lerobot_coreai_version": __version__,
        "ok": ok,
        "policy_path": policy_path,
        "runner_url": runner_url,
        "lerobot": lerobot,
        "checks": checks,
        "claims": {
            "provides_local_lerobot_bridge": True,
            "native_upstream_policy_registry": False,
            "supports_training": False,
            "proves_physical_safety": False,
        },
    }


def build_bridge_markdown(report: dict[str, Any]) -> str:
    lr = report.get("lerobot", {})
    lines = [
        "# LeRobot Bridge Check",
        "",
        f"- OK: {report.get('ok')}",
        f"- Policy: {report.get('policy_path')}",
        f"- lerobot-coreai: {report.get('lerobot_coreai_version')}",
        f"- LeRobot installed: {lr.get('available')} "
        f"(version {lr.get('version')}, in range {lr.get('in_range')})",
        "",
        "## Checks",
    ]
    for c in report.get("checks", []):
        mark = "✅" if c["passed"] else "❌"
        detail = f" — {c['detail']}" if c.get("detail") else ""
        lines.append(f"- {mark} `{c['name']}` ({c['severity']}){detail}")
    lines += [
        "",
        "This is a **local** LeRobot bridge, not upstream-native integration "
        "(`policy_type=\"coreai\"` is not registered upstream). Training is not "
        "supported — train with LeRobot, run with CoreAI. Proves nothing about "
        "physical safety.",
        "",
    ]
    return "\n".join(lines)
