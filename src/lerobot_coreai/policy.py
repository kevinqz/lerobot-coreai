# policy.py — the LeRobot-compatible CoreAI policy wrapper (spec §10).
#
# v0.1: metadata loading only.
# v0.2: select_action() with runner-backed action inference.
#
# No user-facing CoreAI graph names leak into the default Python API (spec §11.3).
# CoreAIPolicy is inference-only — train with LeRobot.

from __future__ import annotations

from typing import Any

from .config import CoreAIRuntimeConfig, CoreAIPolicyConfig
from .errors import (
    CoreAIPolicyError,
    ManifestError,
    ObservationValidationError,
    RunnerNotReachableError,
)
from .manifest import LeRobotCoreAIManifest, load_manifest
from .runner import RunnerClient
from .types import ActionPredictRequest
from .validation import validate_action_output, validate_observation_batch


class CoreAIPolicy:
    """LeRobot-compatible policy wrapper backed by Apple CoreAI.

    Loads a CoreAI artifact's manifest from Hugging Face, validates it, and provides
    the same select_action(batch) interface as a LeRobot policy. Inference is routed
    through coreai-runner (the Swift binary that executes .aimodel graphs).

    v0.2: select_action() works with a running coreai-runner. Metadata API still works
    without a runner.
    """

    def __init__(
        self,
        manifest: LeRobotCoreAIManifest,
        *,
        runtime: CoreAIRuntimeConfig | None = None,
        runner_client: RunnerClient | None = None,
        validate_io: bool = True,
        strict_observation_keys: bool = False,
        return_metadata: bool = False,
    ):
        self._manifest = manifest
        self._runtime = runtime or CoreAIRuntimeConfig()
        self._runner_client = runner_client
        self._validate_io = validate_io
        self._strict_observation_keys = strict_observation_keys
        self._return_metadata = return_metadata
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
        # v1.2.5: per-timestep action queue for select_next_action().
        from .action_contract import parse_action_contract_from_manifest
        from .action_queue import ActionQueue
        self._action_contract = parse_action_contract_from_manifest(manifest)
        self._action_queue = ActionQueue()

    # MARK: - Loading

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        runner_url: str | None = None,
        endpoint: str | None = None,
        download: str = "auto",
        trust_remote_code: bool = False,
        revision: str = "main",
        validate_runner: bool = False,
        validate_io: bool = True,
        strict_observation_keys: bool = False,
        return_metadata: bool = False,
        **kwargs: Any,
    ) -> "CoreAIPolicy":
        """Load a CoreAI-backed LeRobot policy from a Hugging Face artifact.

        Downloads and validates ``lerobot-coreai.json`` from the repo. By default
        (validate_runner=False) no runner connection is needed — metadata commands
        work offline. select_action() will require a runner.

        Args:
            repo_id: HF repo id of the CoreAI artifact (e.g. 'kevinqz/EVO1-SO100-CoreAI').
            runner_url: Runner URL or Unix socket path. Defaults to the standard socket.
            endpoint: Remote runner endpoint (e.g. 'http://mac-studio.local:8710').
            download: Download mode ('auto', 'never', 'force').
            trust_remote_code: Unused (CoreAI policies are inference-only).
            revision: HF revision (default 'main').
            validate_runner: If True, check runner health/capabilities at load time.
            validate_io: If True, validate observation/action against manifest.
            strict_observation_keys: If True, reject unknown observation keys.
            return_metadata: If True, include metadata in select_action response.

        Returns:
            A CoreAIPolicy instance with metadata loaded.

        Raises:
            ManifestError: If lerobot-coreai.json is missing or invalid.
            DownloadError: If the download fails.
        """
        manifest = load_manifest(repo_id, revision=revision)

        runtime = CoreAIRuntimeConfig(
            runner_url=runner_url or (endpoint or CoreAIRuntimeConfig.runner_url),
            download=download,  # type: ignore[arg-type]
        )

        # Create runner client if a URL is provided or validate_runner is True.
        runner_client: RunnerClient | None = None
        url = endpoint or runner_url
        if not url and validate_runner:
            # validate_runner=True without explicit URL: use the default runtime URL.
            url = runtime.runner_url
        if url:
            runner_client = RunnerClient(url, timeout_s=runtime.timeout_s)

        policy = cls(
            manifest,
            runtime=runtime,
            runner_client=runner_client,
            validate_io=validate_io,
            strict_observation_keys=strict_observation_keys,
            return_metadata=return_metadata,
        )

        if validate_runner:
            policy._validate_runner()

        return policy

    # MARK: - Inference

    def predict_action(
        self,
        batch: dict[str, Any],
        *,
        return_metadata: bool | None = None,
        runner_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the policy on a LeRobot-shaped batch and return a dict with the action.

        This is the richer API — always returns a dict::

            {"action": action_chunk}

        With return_metadata=True::

            {"action": action_chunk, "metadata": {...}}

        Guarantees:
        - Returns a dict with "action" key on success.
        - No physical robot actuation happens.

        Raises:
            RunnerNotReachableError: If no runner client is configured or runner is down.
            ObservationValidationError: If the batch doesn't match the manifest.
            ActionValidationError: If the runner returns an invalid action.
        """
        self._ensure_runner()

        # Validate observation against manifest.
        if self._validate_io:
            validate_observation_batch(
                batch, self._manifest,
                strict_observation_keys=self._strict_observation_keys,
            )

        # Build runner request. runner_options (v1.3.4) carry the negotiated
        # observation encoding / protocol version / observation sha256.
        request = ActionPredictRequest(
            model_id=self._manifest.model_id,
            observation=batch,
            options=dict(runner_options) if runner_options else {},
        )

        # Call runner.
        assert self._runner_client is not None  # _ensure_runner guarantees this
        response = self._runner_client.predict_action(request)

        # Validate action output.
        if self._validate_io:
            validate_action_output(response.action, self._manifest)

        want_metadata = return_metadata if return_metadata is not None else self._return_metadata
        if want_metadata:
            return {
                "action": response.action,
                "metadata": {
                    "runtime": "coreai",
                    "model_id": self._manifest.model_id,
                    "policy_type": self._manifest.policy_type,
                    "robot_type": self._manifest.robot_type,
                    "timing": response.timing,
                },
            }
        return {"action": response.action}

    def select_action(
        self,
        batch: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Return the raw action as produced by the runner (LEGACY semantics).

        NOTE (v1.2.5): for a chunked policy this returns the whole action chunk
        ``[H, A]``, NOT a single per-timestep action. This is the historical
        CoreAIPolicy behavior and is preserved for backward compatibility. For
        LeRobot-correct per-timestep semantics use :meth:`select_next_action`
        (which pops one action from an internal queue). See
        :meth:`predict_action_chunk`.
        """
        result = self.predict_action(batch, return_metadata=False)
        return result["action"]

    def predict_action_chunk(self, batch: dict[str, Any], *,
                             runner_options: dict[str, Any] | None = None,
                             **kwargs: Any) -> Any:
        """Return the full action chunk ``[H, A]`` for ``batch``.

        Explicit name for the chunk semantics that :meth:`select_action` has
        historically had. The next-action queue is filled from this.
        """
        return self.predict_action(batch, return_metadata=False,
                                   runner_options=runner_options)["action"]

    def select_next_action(self, batch: dict[str, Any], **kwargs: Any) -> Any:
        """Return ONE per-timestep action ``[A]`` (LeRobot-correct semantics).

        Owns an internal queue: when empty it predicts a fresh chunk via
        :meth:`predict_action_chunk` and loads it; each call pops the next action.
        For a single-action policy the queue simply holds one row per prediction.
        """
        if self._action_queue.empty:
            chunk = self.predict_action_chunk(batch, **kwargs)
            self._action_queue.load_chunk(chunk)
        return self._action_queue.pop_next()

    # MARK: - Lifecycle

    def reset(self) -> None:
        """Reset the policy's internal state.

        v1.2.5: clears the local per-timestep action queue and asks the runner to
        reset its session when supported (best-effort).
        """
        self._action_queue.reset()
        client = self._runner_client
        reset_fn = getattr(client, "reset_session", None) if client else None
        if callable(reset_fn):
            try:  # pragma: no cover - depends on runner capability
                reset_fn()
            except Exception:
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

    # MARK: - Runner management

    def _ensure_runner(self) -> None:
        """Ensure a runner client exists and is reachable. Raises if not."""
        if self._runner_client is None:
            raise RunnerNotReachableError(
                "No coreai-runner configured. Pass runner_url to from_pretrained() "
                "or construct a RunnerClient directly.\n"
                "No robot commands were sent."
            )

    def _validate_runner(self) -> None:
        """Validate that the runner is alive and supports action inference.

        Calls health() and supports_action(). Raises on any failure.
        """
        self._ensure_runner()
        assert self._runner_client is not None
        self._runner_client.health()
        self._runner_client.supports_action()

    @property
    def runner(self) -> RunnerClient | None:
        """The runner client, or None if not configured."""
        return self._runner_client

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
