# lerobot_registry.py — local, opt-in LeRobot policy registry adapter (v1.1.3).
#
# Gives registry-style ergonomics for CoreAI-backed policies WITHOUT pretending
# LeRobot upstream knows about them. The default `CoreAILeRobotRegistry` is a
# process-local mapping — it never touches LeRobot's factory. The optional
# `local_lerobot_registry_patch()` context manager installs a reversible,
# process-local wrapper over `lerobot.policies.factory.get_policy_class` for the
# duration of the `with` block only, and restores the original on exit. There is
# NO global monkeypatch at import time, and `policy_type="coreai"` is never used
# (only "coreai_bridge") so nothing implies upstream registration.

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable

from .errors import CoreAIPolicyError
from .lerobot_bridge import load_coreai_policy_for_lerobot
from .lerobot_config import BRIDGE_POLICY_TYPE
from .lerobot_policy import CoreAILeRobotPolicyBridge

# Reserved: implies upstream LeRobot registration, which does not exist.
_RESERVED = "coreai"


class CoreAILeRobotRegistry:
    """A process-local registry of CoreAI bridge loaders.

    This is a local adapter, not the upstream LeRobot registry/factory.
    """

    def __init__(self) -> None:
        self._loaders: dict[str, Callable[..., CoreAILeRobotPolicyBridge]] = {}

    def register(self, policy_type: str = BRIDGE_POLICY_TYPE,
                 loader: Callable[..., CoreAILeRobotPolicyBridge] | None = None) -> None:
        """Register a bridge loader under a local policy type.

        Refuses ``"coreai"`` — that name would imply upstream registration.
        """
        if policy_type == _RESERVED:
            raise CoreAIPolicyError(
                "Refusing to register policy_type='coreai': that implies upstream "
                "LeRobot registration, which does not exist. Use 'coreai_bridge'.")
        self._loaders[policy_type] = loader or load_coreai_policy_for_lerobot

    def is_registered(self, policy_type: str) -> bool:
        return policy_type in self._loaders

    def registered(self) -> list[str]:
        return sorted(self._loaders)

    def load(self, policy_type: str, *, policy_path: str,
             runner_url: str | None = None, **kwargs: Any) -> CoreAILeRobotPolicyBridge:
        """Load a bridge by local policy type. Fail-closed on unknown types."""
        loader = self._loaders.get(policy_type)
        if loader is None:
            raise CoreAIPolicyError(
                f"policy_type {policy_type!r} is not registered locally. "
                f"Registered: {self.registered() or '[]'}.")
        return loader(policy_path=policy_path, runner_url=runner_url, **kwargs)


def _import_factory():
    """Return lerobot's policy factory module, or None if not installed."""
    try:
        import lerobot.policies.factory as factory  # type: ignore
        return factory
    except Exception:
        return None


@contextmanager
def local_lerobot_registry_patch(registry: CoreAILeRobotRegistry | None = None):
    """Temporarily teach LeRobot's factory about the local ``coreai_bridge`` type.

    Process-local and reversible: inside the ``with`` block,
    ``lerobot.policies.factory.get_policy_class("coreai_bridge")`` resolves to the
    bridge class; all other names delegate to the original. On exit the original
    ``get_policy_class`` is restored exactly. If LeRobot isn't installed this is a
    no-op over the factory (the local registry still works). This is a local,
    experimental adapter — NOT upstream-native registration.
    """
    reg = registry or CoreAILeRobotRegistry()
    if not reg.is_registered(BRIDGE_POLICY_TYPE):
        reg.register(BRIDGE_POLICY_TYPE)

    factory = _import_factory()
    original = None
    if factory is not None:
        original = factory.get_policy_class

        def _patched(name):
            if name == BRIDGE_POLICY_TYPE:
                return CoreAILeRobotPolicyBridge
            return original(name)

        factory.get_policy_class = _patched
    try:
        yield reg
    finally:
        if factory is not None and original is not None:
            factory.get_policy_class = original


def evaluate_registry_check(
    policy_type: str = BRIDGE_POLICY_TYPE, *,
    policy_path: str | None = None, runner_url: str | None = None,
) -> dict[str, Any]:
    """Run the local-registry checks and return a report dict. Sends no action."""
    from . import __version__
    checks: list[dict[str, Any]] = []

    def _c(name, passed, detail="", severity="required"):
        checks.append({"name": name, "passed": bool(passed), "severity": severity,
                       "detail": detail})

    # 'coreai' must be refused.
    reg = CoreAILeRobotRegistry()
    try:
        reg.register("coreai")
        _c("reserved_coreai_refused", False, "registry accepted 'coreai'")
    except CoreAIPolicyError:
        _c("reserved_coreai_refused", True)

    reg.register(policy_type)
    _c("registry_registers", reg.is_registered(policy_type))

    # Default: upstream factory is untouched.
    factory = _import_factory()
    if factory is not None:
        before = factory.get_policy_class
        _c("upstream_factory_unchanged_by_default", factory.get_policy_class is before)
        # Context manager resolves the bridge and restores afterwards.
        with local_lerobot_registry_patch(reg):
            inside_ok = factory.get_policy_class(BRIDGE_POLICY_TYPE) is CoreAILeRobotPolicyBridge
        _c("context_manager_resolves_bridge", inside_ok)
        _c("context_manager_restores_factory", factory.get_policy_class is before)
    else:
        _c("lerobot_installed", False, "[lerobot] not installed — local registry only",
           severity="info")

    # Optional: actually load a bridge (best-effort; needs a reachable runner).
    if policy_path:
        try:
            bridge = reg.load(policy_type, policy_path=policy_path,
                              runner_url=runner_url, validate_runner=bool(runner_url))
            _c("registry_load_returns_bridge",
               isinstance(bridge, CoreAILeRobotPolicyBridge), severity="info")
        except Exception as e:
            _c("registry_load_returns_bridge", False,
               f"{type(e).__name__}: {e}", severity="info")

    ok = all(c["passed"] for c in checks if c["severity"] == "required")
    return {
        "schema_version": "lerobot-coreai.lerobot_registry.v0",
        "lerobot_coreai_version": __version__,
        "ok": ok,
        "policy_type": policy_type,
        "checks": checks,
        "claims": {
            "provides_local_registry_adapter": True,
            "native_upstream_registry": False,
            "supports_training": False,
            "proves_physical_safety": False,
        },
    }


def build_registry_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# LeRobot Local Registry Check",
        "",
        f"- OK: {report.get('ok')}",
        f"- policy_type: {report.get('policy_type')}",
        "",
        "## Checks",
    ]
    for c in report.get("checks", []):
        mark = "✅" if c["passed"] else "❌"
        detail = f" — {c['detail']}" if c.get("detail") else ""
        lines.append(f"- {mark} `{c['name']}` ({c['severity']}){detail}")
    lines += [
        "",
        "Local, opt-in registry adapter — **not** upstream LeRobot registration. "
        "The upstream factory is untouched by default and any in-context patch is "
        "reversed on exit. No training; proves nothing about physical safety.",
        "",
    ]
    return "\n".join(lines)
