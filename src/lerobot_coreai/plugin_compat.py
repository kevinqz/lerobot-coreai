# plugin_compat.py — compatibility profile for the official plugin (v1.3.1).
#
# The base contract (lerobot_contracts) describes the base package's LOCAL bridge
# — its official_* levels stay failed on purpose. The optional companion package
# `lerobot_policy_coreai_bridge` is a genuine out-of-tree plugin, so it gets its
# OWN profile. A level is "passed" only when actually exercisable here; end-to-end
# official eval stays "not_tested" until a live eval proves it. Never claims
# upstream-native, training, or physical safety. No torch/lerobot import at base
# module load — all probes are lazy.

from __future__ import annotations

from typing import Any

from . import __version__

PLUGIN_COMPAT_SCHEMA_VERSION = "lerobot-coreai.plugin_compat.v0"

PASSED = "passed"
PARTIAL = "partial"
FAILED = "failed"
NOT_TESTED = "not_tested"
NOT_APPLICABLE = "not_applicable"


def _plugin_installed() -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec("lerobot_policy_coreai_bridge") is not None
    except Exception:
        return False


def _config_registered() -> bool:  # pragma: no cover - needs lerobot + plugin
    try:
        import lerobot_policy_coreai_bridge  # noqa: F401  (self-registers)
        from lerobot.configs.policies import PreTrainedConfig
        return "coreai_bridge" in PreTrainedConfig.get_known_choices()
    except Exception:
        return False


def _policy_class_ok() -> bool:  # pragma: no cover - needs lerobot + plugin
    try:
        import torch.nn as nn
        from lerobot.policies.pretrained import PreTrainedPolicy
        from lerobot_policy_coreai_bridge import CoreAIBridgePolicy
        return (issubclass(CoreAIBridgePolicy, PreTrainedPolicy)
                and issubclass(CoreAIBridgePolicy, nn.Module)
                and CoreAIBridgePolicy.name == "coreai_bridge")
    except Exception:
        return False


def evaluate_plugin_compat() -> dict[str, Any]:
    """Report the official-plugin compatibility profile. Honest by construction."""
    installed = _plugin_installed()
    levels: dict[str, str] = {}
    if not installed:
        for k in ("plugin_discovery", "config_registry", "policy_class_contract",
                  "policy_factory", "processor_pipeline", "official_eval"):
            levels[k] = NOT_TESTED
    else:
        registered = _config_registered()
        levels["plugin_discovery"] = PASSED  # installed as lerobot_policy_*
        levels["config_registry"] = PASSED if registered else FAILED
        levels["policy_class_contract"] = PASSED if _policy_class_ok() else FAILED
        # These require a real factory/from_pretrained/eval run; the plugin can be
        # constructed but end-to-end factory/eval is exercised in integration
        # tests, not asserted here.
        levels["policy_factory"] = PARTIAL if registered else FAILED
        levels["processor_pipeline"] = PARTIAL
        levels["official_eval"] = NOT_TESTED

    return {
        "schema_version": PLUGIN_COMPAT_SCHEMA_VERSION,
        "lerobot_coreai_version": __version__,
        "profile": "official_plugin",
        "plugin_installed": installed,
        "levels": levels,
        "claims": {
            "official_plugin_registered": levels.get("config_registry") == PASSED,
            "upstream_native": False,
            "supports_training": False,
            "official_eval_certified": False,
            "proves_physical_safety": False,
        },
    }


def build_plugin_compat_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Official Plugin Compatibility Profile",
        "",
        f"- Plugin installed: {report.get('plugin_installed')}",
        "",
        "## Levels",
    ]
    for k, v in report["levels"].items():
        lines.append(f"- `{k}`: **{v}**")
    lines += [
        "",
        "The base package's local bridge keeps its own (failed) official levels; "
        "this profile is for the companion `lerobot_policy_coreai_bridge`. "
        "`official_eval_certified` stays false until a live end-to-end eval proves "
        "it. Not upstream-native; no training; proves no physical safety.",
        "",
    ]
    return "\n".join(lines)
