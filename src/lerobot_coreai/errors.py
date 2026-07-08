# errors.py — exception hierarchy for lerobot-coreai.
#
# All lerobot-coreai errors derive from CoreAIPolicyError so callers can catch them broadly.

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


class RunnerNotReachableError(CoreAIPolicyError):
    """coreai-runner is not reachable at the configured URL."""


class DownloadError(CoreAIPolicyError):
    """Failed to download the manifest or artifact from Hugging Face."""


class SafetyError(CoreAIPolicyError):
    """A safety gate was violated (e.g. real mode without confirmation)."""
