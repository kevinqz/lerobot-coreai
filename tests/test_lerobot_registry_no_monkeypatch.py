# test_lerobot_registry_no_monkeypatch.py — the local registry must not touch
# LeRobot's factory except inside the opt-in context manager (v1.1.3).

import importlib

import pytest

from lerobot_coreai.lerobot_config import BRIDGE_POLICY_TYPE
from lerobot_coreai.lerobot_policy import CoreAILeRobotPolicyBridge
from lerobot_coreai.lerobot_registry import local_lerobot_registry_patch


def _factory():
    try:
        import lerobot.policies.factory as f  # type: ignore
        return f
    except Exception:
        return None


def test_import_does_not_patch_factory():
    # Importing the registry module must not alter lerobot's factory.
    import lerobot_coreai.lerobot_registry as reg_mod
    importlib.reload(reg_mod)
    factory = _factory()
    if factory is None:
        pytest.skip("lerobot not installed")
    # Importing the base registry module must not blanket-patch the factory:
    # a genuinely-unregistered type must still be rejected. (Note: the official
    # companion plugin, if installed, legitimately registers "coreai_bridge" via
    # register_subclass — that is not this module's doing, so we probe a name
    # that no package registers.)
    with pytest.raises(Exception):
        factory.get_policy_class("definitely-not-a-registered-policy-xyz")


@pytest.mark.skipif(_factory() is None, reason="lerobot not installed")
def test_context_manager_patches_and_restores():
    factory = _factory()
    before = factory.get_policy_class
    with local_lerobot_registry_patch():
        assert factory.get_policy_class is not before  # patched inside
        assert factory.get_policy_class(BRIDGE_POLICY_TYPE) is CoreAILeRobotPolicyBridge
        # Non-bridge names still delegate to the original factory.
        with pytest.raises(Exception):
            factory.get_policy_class("definitely-not-a-policy")
    assert factory.get_policy_class is before  # restored on exit


@pytest.mark.skipif(_factory() is None, reason="lerobot not installed")
def test_context_manager_restores_even_on_error():
    factory = _factory()
    before = factory.get_policy_class
    with pytest.raises(RuntimeError):
        with local_lerobot_registry_patch():
            raise RuntimeError("boom")
    assert factory.get_policy_class is before


def test_context_manager_no_lerobot_is_noop():
    # Without lerobot the context manager still yields a working local registry.
    if _factory() is not None:
        pytest.skip("lerobot installed")
    with local_lerobot_registry_patch() as reg:
        assert reg.is_registered(BRIDGE_POLICY_TYPE)
