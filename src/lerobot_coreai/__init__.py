# lerobot-coreai — Apple CoreAI runtime backend for LeRobot policies.
#
# Canonical sentence: Same LeRobot workflow. CoreAI runtime.

__version__ = "1.2.3"

from .policy import CoreAIPolicy
from .config import CoreAIRuntimeConfig, CoreAIPolicyConfig
from .manifest import LeRobotCoreAIManifest, load_manifest
from .runner import RunnerClient
from .types import ActionPredictRequest, ActionPredictResponse, RunnerHealth, RunnerCapabilities
from .errors import (
    CoreAIPolicyError,
    ManifestError,
    VersionMismatchError,
    RunnerError,
    RunnerNotReachableError,
    RunnerCapabilityError,
    RunnerProtocolError,
    RunnerRequestError,
    RunnerExecutionError,
    RunnerTimeoutError,
    ObservationValidationError,
    ActionValidationError,
    FixtureError,
)

__all__ = [
    "__version__",
    "CoreAIPolicy",
    "CoreAIRuntimeConfig",
    "CoreAIPolicyConfig",
    "LeRobotCoreAIManifest",
    "load_manifest",
    "RunnerClient",
    "ActionPredictRequest",
    "ActionPredictResponse",
    "RunnerHealth",
    "RunnerCapabilities",
    "CoreAIPolicyError",
    "ManifestError",
    "VersionMismatchError",
    "RunnerError",
    "RunnerNotReachableError",
    "RunnerCapabilityError",
    "RunnerProtocolError",
    "RunnerRequestError",
    "RunnerExecutionError",
    "RunnerTimeoutError",
    "ObservationValidationError",
    "ActionValidationError",
    "FixtureError",
]
