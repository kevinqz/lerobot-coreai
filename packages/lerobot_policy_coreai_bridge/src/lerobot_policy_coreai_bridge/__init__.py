# lerobot_policy_coreai_bridge — official out-of-tree LeRobot policy plugin.
#
# LeRobot discovers installed distributions named lerobot_policy_* and imports
# them so they self-register. Importing this package registers the
# "coreai_bridge" PreTrainedConfig subclass and exposes the policy + processor
# factory. Runtime-only: it does not train. `policy_type="coreai"` is NOT
# registered — only "coreai_bridge".

from .configuration_coreai_bridge import POLICY_TYPE, CoreAIBridgeConfig
from .modeling_coreai_bridge import CoreAIBridgePolicy
from .processor_coreai_bridge import make_coreai_bridge_pre_post_processors

__version__ = "1.3.2"

__all__ = [
    "POLICY_TYPE",
    "CoreAIBridgeConfig",
    "CoreAIBridgePolicy",
    "make_coreai_bridge_pre_post_processors",
]
