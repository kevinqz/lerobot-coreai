# policy.py — the LeRobot-compatible CoreAI policy wrapper (spec §10).
#
# This is the primary user-facing API:
#
#     from lerobot_coreai import CoreAIPolicy
#     policy = CoreAIPolicy.from_pretrained("kevinqz/EVO1-SO100-CoreAI")
#     out = policy.select_action(batch)
#     action = out["action"]
#
# No user-facing CoreAI graph names leak into the default Python API (spec §11.3).
# CoreAIPolicy is inference-only — train with LeRobot.

from __future__ import annotations

from typing import Any

from .config import CoreAIRuntimeConfig, CoreAIPolicyConfig
from .errors import CoreAIPolicyError, ManifestError, RunnerNotReachableError
from .manifest import LeRobotCoreAIManifest, load_manifest


class CoreAIPolicy:
    """LeRobot-compatible policy wrapper backed by Apple CoreAI.

    Loads a CoreAI artifact's manifest from Hugging Face, validates it, and provides
    the same select_action(batch) interface as a LeRobot policy. Inference is routed
    through coreai-runner (the Swift binary that executes .aimodel graphs).

    MVP v0.1: metadata loading only. select_action requires a reachable runner
    (added in v0.2).
    """

    def __init__(
        self,
        manifest: LeRobotCoreAIManifest,
        *,
        runtime: CoreAIRuntimeConfig | None = None,
    ):
        self._manifest = manifest
        self._runtime = runtime or CoreAIRuntimeConfig()
        self._config = CoreAIPolicyConfig(
            path=manifest.policy_repo_id,
            policy_type=manifest.policy_type,
            robot_type=manifest.robot_type,
            observation_features={
                name: {"dtype": f.dtype, "shape": f.shape, "required": f.required}
                for name, f in manifest.observation_features.items()
            },
            action_features={
                name: {"dtype": f.dtype, "shape": f.shape}
                for name, f in manifest.action_features.items()
            },
            runtime=self._runtime,
        )
        self._runner_client: Any = None  # lazy-initialized in v0.2

    # MARK: - Loading

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        runner_url: str | None = None,
        download: str = "auto",
        trust_remote_code: bool = False,
        revision: str = "main",
        **kwargs: Any,
    ) -> "CoreAIPolicy":
        """Load a CoreAI-backed LeRobot policy from a Hugging Face artifact.

        Downloads and validates ``lerobot-coreai.json`` from the repo. The runner is
        not contacted at load time — use :meth:`select_action` or ``rollout`` to
        drive inference through coreai-runner.

        Args:
            repo_id: HF repo id of the CoreAI artifact (e.g. 'kevinqz/EVO1-SO100-CoreAI').
            runner_url: Runner URL or Unix socket path. Defaults to the standard socket.
            download: Download mode ('auto', 'never', 'force'). MVP v0.1 only fetches
                the manifest, not the full artifact.
            trust_remote_code: Unused (CoreAI policies are inference-only; no remote code).
            revision: HF revision (default 'main').

        Returns:
            A CoreAIPolicy instance with metadata loaded.

        Raises:
            ManifestError: If lerobot-coreai.json is missing or invalid.
            DownloadError: If the download fails.
        """
        manifest = load_manifest(repo_id, revision=revision)

        runtime = CoreAIRuntimeConfig(
            runner_url=runner_url or CoreAIRuntimeConfig.runner_url,
            download=download,  # type: ignore[arg-type]
        )

        return cls(manifest, runtime=runtime)

    # MARK: - Inference

    def select_action(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Run the policy on a LeRobot-shaped batch and return an action.

        Input (spec §11.1)::

            batch = {
                "observation.images.wrist": wrist_image,
                "observation.state": state,
                "task": "pick up the cube",
            }

        Output (spec §11.2)::

            {"action": action_chunk}

        MVP v0.1: this raises NotImplementedError because runner integration
        is added in v0.2. Metadata-only commands (inspect, doctor) are available now.
        """
        raise NotImplementedError(
            "select_action requires coreai-runner integration (v0.2). "
            "For metadata-only access, use 'lerobot-coreai inspect'."
        )

    # MARK: - Lifecycle

    def reset(self) -> None:
        """Reset the policy's internal state (e.g. KV cache, action queue)."""
        # MVP v0.1: no-op. v0.2 will reset the runner session.
        pass

    def eval(self) -> "CoreAIPolicy":
        """Set the policy to evaluation mode (inference-only). Always returns self."""
        return self

    def train(self) -> "CoreAIPolicy":
        """CoreAI policies are inference-only. Train with LeRobot."""
        raise NotImplementedError(
            "CoreAIPolicy only supports inference. Train with LeRobot."
        )

    def to(self, device: str) -> "CoreAIPolicy":
        """CoreAI policies always run on the CoreAI device.

        Args:
            device: Must be 'coreai' or 'auto'.

        Raises:
            ValueError: If device is not 'coreai' or 'auto'.
        """
        if device not in {"coreai", "auto"}:
            raise ValueError("CoreAIPolicy only supports device='coreai' or 'auto'.")
        return self

    # MARK: - Accessors

    @property
    def config(self) -> CoreAIPolicyConfig:
        """The policy configuration derived from the manifest."""
        return self._config

    @property
    def manifest(self) -> LeRobotCoreAIManifest:
        """The raw manifest."""
        return self._manifest

    @property
    def repo_id(self) -> str:
        return self._manifest.policy_repo_id

    @property
    def policy_type(self) -> str:
        return self._manifest.policy_type

    @property
    def robot_type(self) -> str:
        return self._manifest.robot_type

    @property
    def parity_passed(self) -> bool:
        return self._manifest.parity_passed

    def __repr__(self) -> str:
        return (
            f"CoreAIPolicy(repo_id={self.repo_id!r}, "
            f"type={self.policy_type!r}, robot={self.robot_type!r}, "
            f"parity={'passed' if self.parity_passed else 'unknown'})"
        )
