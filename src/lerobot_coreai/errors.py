# errors.py — exception hierarchy for lerobot-coreai.
#
# All lerobot-coreai errors derive from CoreAIPolicyError so callers can catch them broadly.
# v0.2 adds runner and validation errors for the runtime backend.

from __future__ import annotations


class CoreAIPolicyError(Exception):
    """Base error for all lerobot-coreai failures."""


class ManifestError(CoreAIPolicyError):
    """The lerobot-coreai.json manifest is missing, malformed, or fails schema validation."""

    def __init__(self, detail: str, *, repo_id: str | None = None):
        self.repo_id = repo_id
        super().__init__(detail)


class VersionMismatchError(CoreAIPolicyError):
    """The installed LeRobot version is outside the supported range (spec §26)."""

    def __init__(self, detail: str, *, installed: str | None = None, required: str | None = None):
        self.installed = installed
        self.required = required
        super().__init__(detail)


class DownloadError(CoreAIPolicyError):
    """Failed to download the manifest or artifact from Hugging Face."""


class SafetyError(CoreAIPolicyError):
    """A safety gate was violated (e.g. real mode without confirmation)."""


# --- v0.2: Runner errors ---

class RunnerError(CoreAIPolicyError):
    """Base error for all coreai-runner communication failures."""


class RunnerNotReachableError(RunnerError):
    """coreai-runner is not reachable at the configured URL."""


class RunnerCapabilityError(RunnerError):
    """coreai-runner does not support a required capability (e.g. runtime_kind=action)."""


class RunnerProtocolError(RunnerError):
    """coreai-runner returned an invalid or unexpected response (bad JSON, missing fields)."""


class RunnerRequestError(RunnerError):
    """coreai-runner rejected the request (HTTP 400 — bad model_id, bad payload)."""


class RunnerExecutionError(RunnerError):
    """coreai-runner failed to execute inference (HTTP 500 — model crash, OOM)."""


class RunnerTimeoutError(RunnerError):
    """coreai-runner did not respond within the timeout."""


# --- v0.2: Validation errors ---

class ObservationValidationError(CoreAIPolicyError):
    """The observation batch does not match the manifest's feature contract."""


class ActionValidationError(CoreAIPolicyError):
    """The action output from the runner does not match the manifest's feature contract."""
