# lerobot_policy_coreai_bridge — official out-of-tree LeRobot policy plugin.
#
# LeRobot discovers installed distributions named lerobot_policy_* and imports
# them so they self-register. Importing this package registers the
# "coreai_bridge" PreTrainedConfig subclass and exposes the policy + processor
# factory. Runtime-only: it does not train. `policy_type="coreai"` is NOT
# registered — only "coreai_bridge".

__version__ = "1.3.28"

from .configuration_coreai_bridge import POLICY_TYPE, CoreAIBridgeConfig
from .modeling_coreai_bridge import CoreAIBridgePolicy
from .processor_coreai_bridge import (
    make_coreai_bridge_pre_post_processors,
    save_coreai_bridge_processors,
)
# side-effect import: registers the gymnasium env + the "coreai_cert_env" EnvConfig
# choice so the official `lerobot-eval --env.type=coreai_cert_env` path resolves when
# lerobot auto-imports this plugin (v1.3.27.1).
from . import coreai_cert_env  # noqa: F401

__all__ = [
    "POLICY_TYPE",
    "CoreAIBridgeConfig",
    "CoreAIBridgePolicy",
    "make_coreai_bridge_pre_post_processors",
    "save_coreai_bridge_processors",
    "build_plugin_artifact",
    "verify_plugin_artifact",
]


def build_plugin_artifact(*args, **kwargs):
    """Lazy proxy to artifact.build_plugin_artifact (imports lerobot on call)."""
    from .artifact import build_plugin_artifact as _impl
    return _impl(*args, **kwargs)


def verify_plugin_artifact(*args, **kwargs):
    """Lazy proxy to artifact.verify_plugin_artifact."""
    from .artifact import verify_plugin_artifact as _impl
    return _impl(*args, **kwargs)
