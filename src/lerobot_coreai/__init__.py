# lerobot-coreai — Apple CoreAI runtime backend for LeRobot policies.
#
# Canonical sentence: Same LeRobot workflow. CoreAI runtime.

__version__ = "0.1.0"

from .policy import CoreAIPolicy
from .config import CoreAIRuntimeConfig, CoreAIPolicyConfig
from .manifest import LeRobotCoreAIManifest, load_manifest
from .errors import CoreAIPolicyError, ManifestError, VersionMismatchError

__all__ = [
    "__version__",
    "CoreAIPolicy",
    "CoreAIRuntimeConfig",
    "CoreAIPolicyConfig",
    "LeRobotCoreAIManifest",
    "load_manifest",
    "CoreAIPolicyError",
    "ManifestError",
    "VersionMismatchError",
]
