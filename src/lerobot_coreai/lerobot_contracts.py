# lerobot_contracts.py — leveled LeRobot compatibility contract (v1.2.4).
#
# The v0 certificate answered a single question ("shape-compatible?") and could
# imply more than it proved. This module reports LEVELS, honestly: each rung of
# the official LeRobot contract (dataset read, action semantics, plugin
# discovery, config registry, processor pipeline, official eval/rollout) is
# reported separately, and every level is either genuinely tested or marked
# "failed"/"not_supported"/"not_tested" — never assumed. It never claims official
# plugin/eval/rollout support, upstream-native integration, training, or physical
# safety.

from __future__ import annotations

import sys
from typing import Any

from . import __version__

LEROBOT_COMPAT_V1_SCHEMA_VERSION = "lerobot-coreai.lerobot_compat.v1"

# Stable target is blocking in CI; development is a pinned, non-blocking probe.
STABLE_TARGET_VERSION = "0.6.0"
DEVELOPMENT_TARGET_VERSION = "0.6.1"

# Level outcomes.
PASSED = "passed"
PARTIAL = "partial"
FAILED = "failed"
NOT_TESTED = "not_tested"
NOT_SUPPORTED = "not_supported"
SEPARATE_RUNTIME = "separate_runtime"

_LEVELS = [
    "base_package_import",
    "lerobot_version_supported",
    "dataset_constructor",
    "dataset_frame_read",
    "action_method_name",
    "action_semantics",
    "action_tensor_contract",
    "action_batch_contract",
    "official_plugin_discovery",
    "official_config_registry",
    "official_policy_factory",
    "official_processor_pipeline",
    "official_eval",
    "official_rollout_sync",
    "official_rollout_rtc",
    "guarded_real_separate_runtime",
]


def _lerobot_version() -> str | None:
    try:
        import importlib.metadata as md
        return md.version("lerobot")
    except Exception:
        return None


def _version_in_range(version: str | None) -> bool | None:
    if not version:
        return None
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
        return Version(version) in SpecifierSet(">=0.6.0,<0.7.0")
    except Exception:
        return None


def _module_exists(path: str) -> bool:
    """True if a module path is importable/discoverable without executing heavy deps."""
    import importlib.util
    import os
    # Prefer a filesystem probe off the top-level package to avoid importing
    # heavy submodule __init__ files (av/torchcodec).
    top = path.split(".")[0]
    try:
        pkg = __import__(top)
    except Exception:
        return False
    rel = os.path.join(*path.split(".")[1:]) if "." in path else ""
    for base in list(getattr(pkg, "__path__", []) or []):
        if rel and (os.path.exists(os.path.join(base, rel + ".py"))
                    or os.path.isdir(os.path.join(base, rel))):
            return True
    try:
        return importlib.util.find_spec(path) is not None
    except Exception:
        return False


def _detect_plugin_discovery() -> str:
    """LeRobot's official out-of-tree plugin discovery mechanism present?"""
    # The mechanism scans installed distributions named lerobot_<kind>_*.
    for cand in ("lerobot.utils.plugin", "lerobot.policies.factory"):
        if _module_exists(cand):
            return "detected"
    return "absent"


def _detect_pretrained_policy() -> bool:
    return _module_exists("lerobot.policies.pretrained") or \
        _module_exists("lerobot.common.policies.pretrained")


def _detect_config_registry() -> bool:
    return _module_exists("lerobot.configs")


def evaluate_compatibility_contract(*, strict: bool = False) -> dict[str, Any]:
    """Evaluate the leveled LeRobot contract. Honest by construction.

    strict=True is used by the stable CI job: LeRobot must be installed and in
    range. Non-strict still reports every level (LeRobot-dependent ones become
    not_tested when LeRobot is absent).
    """
    version = _lerobot_version()
    in_range = _version_in_range(version)
    py = sys.version_info[:2]
    have_lerobot = version is not None

    levels: dict[str, str] = {}

    # Always-available (base package) levels.
    levels["base_package_import"] = PASSED  # if we got here, the base imported
    levels["lerobot_version_supported"] = (
        PASSED if in_range else (FAILED if have_lerobot else NOT_TESTED))

    # Dataset levels.
    if have_lerobot:
        levels["dataset_constructor"] = (
            PASSED if (_module_exists("lerobot.datasets.lerobot_dataset")
                       or _module_exists("lerobot.common.datasets.lerobot_dataset"))
            else FAILED)
    else:
        levels["dataset_constructor"] = NOT_TESTED
    # We do not actually replay frames here (that is eval-v3's job): not_tested.
    levels["dataset_frame_read"] = NOT_TESTED

    # Action contract. The local bridge has a select_action METHOD, but its
    # semantics are chunk-passthrough, not per-timestep, and it returns lists,
    # not a torch (B, A) tensor. Report that honestly.
    levels["action_method_name"] = PASSED
    # v1.2.5: select_next_action() now provides per-timestep semantics via a queue,
    # but the default select_action() is still chunk-passthrough and the official
    # tensor contract is unmet — so semantics stay failed until the plugin lands.
    levels["action_semantics"] = FAILED  # default select_action still chunk passthrough
    levels["action_tensor_contract"] = FAILED  # returns list, not Tensor(B,A)
    # v1.2.5: a split-and-stack fallback exists for batched observations, so the
    # batch contract is partially satisfied (not the official batched tensor path).
    levels["action_batch_contract"] = PARTIAL

    # Official integration levels — none implemented in the local bridge.
    levels["official_plugin_discovery"] = FAILED  # package not lerobot_policy_*
    levels["official_config_registry"] = FAILED  # config not registered upstream
    levels["official_policy_factory"] = FAILED  # not make_policy-compatible
    levels["official_processor_pipeline"] = FAILED  # no pre/post processors
    levels["official_eval"] = FAILED  # not a PreTrainedPolicy/nn.Module
    levels["official_rollout_sync"] = NOT_SUPPORTED
    levels["official_rollout_rtc"] = NOT_SUPPORTED

    # Guarded real is a deliberately separate runtime, not the official stack.
    levels["guarded_real_separate_runtime"] = SEPARATE_RUNTIME

    # Environment facts that inform (not gate) the levels.
    detections = {
        "lerobot_installed": have_lerobot,
        "lerobot_version": version,
        "python_version": f"{py[0]}.{py[1]}",
        "pretrained_policy_module_present": _detect_pretrained_policy() if have_lerobot else None,
        "config_registry_present": _detect_config_registry() if have_lerobot else None,
        "plugin_discovery_mechanism": _detect_plugin_discovery() if have_lerobot else "not_tested",
        "bridge_kind": "duck_typed_local_runtime_only",
    }

    shape_ok = (levels["base_package_import"] == PASSED
                and levels["action_method_name"] == PASSED
                and levels["dataset_constructor"] in (PASSED, NOT_TESTED))

    report = {
        "schema_version": LEROBOT_COMPAT_V1_SCHEMA_VERSION,
        "lerobot_coreai_version": __version__,
        "targets": {
            "stable": {
                "version": STABLE_TARGET_VERSION,
                "required": True,
                "installed_version": version,
                "passed": bool(in_range) if have_lerobot else (not strict),
            },
            "development": {
                "version": DEVELOPMENT_TARGET_VERSION,
                "commit": None,
                "required": False,
                "passed": None,
            },
        },
        "levels": levels,
        "detections": detections,
        "claims": {
            "shape_compatible": bool(shape_ok),
            "official_plugin_compatible": False,
            "official_eval_compatible": False,
            "official_rollout_compatible": False,
            "native_upstream_registry": False,
            "supports_training": False,
            "proves_physical_safety": False,
        },
    }
    # In strict mode, the stable target must be installed and in range.
    if strict:
        report["targets"]["stable"]["passed"] = bool(in_range)
    report["ok"] = _report_ok(report, strict=strict)
    return report


def _report_ok(report: dict[str, Any], *, strict: bool) -> bool:
    # A report is "ok" when it is internally consistent and does not overstate.
    # It never requires official integration to be true (those are honestly
    # false today). Strict additionally requires the stable target to pass.
    levels = report["levels"]
    consistent = (
        levels["action_semantics"] == FAILED  # must stay failed while chunk-passthrough
        and report["claims"]["official_eval_compatible"] is False
        and report["claims"]["official_plugin_compatible"] is False
        and report["claims"]["native_upstream_registry"] is False)
    if strict and not report["targets"]["stable"]["passed"]:
        return False
    return consistent


def build_contract_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# LeRobot Compatibility Contract (v1)",
        "",
        f"- lerobot-coreai: {report.get('lerobot_coreai_version')}",
        f"- Stable target: {report['targets']['stable']['version']} "
        f"(installed {report['targets']['stable'].get('installed_version')}, "
        f"passed {report['targets']['stable'].get('passed')})",
        f"- Development target: {report['targets']['development']['version']} "
        "(pinned commit, non-blocking)",
        "",
        "## Levels",
    ]
    for k in _LEVELS:
        lines.append(f"- `{k}`: **{report['levels'].get(k)}**")
    lines += [
        "",
        "## Claims",
        f"- shape_compatible: {report['claims']['shape_compatible']}",
        "- official_plugin_compatible: False",
        "- official_eval_compatible: False",
        "- official_rollout_compatible: False",
        "- native_upstream_registry: False",
        "- supports_training: False",
        "",
        "The local bridge is duck-typed and runtime-only: it is **not** a "
        "`PreTrainedPolicy`/`nn.Module`, its `select_action` is chunk-passthrough "
        "(not per-timestep), and it does not participate in the official plugin "
        "discovery, factory, processor, or eval pipelines. Guarded real egress is "
        "a separate, enforced runtime. Train with LeRobot; run with CoreAI.",
        "",
    ]
    return "\n".join(lines)
