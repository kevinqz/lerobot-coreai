# processor_coreai_bridge.py — official serializable processors for the CoreAI bridge (v1.3.6).
#
# v1.3.5 used a custom _IdentityProcessor callable, which the official LeRobot
# factory cannot load from policy_preprocessor.json / policy_postprocessor.json.
# v1.3.6 returns real lerobot.processor.PolicyProcessorPipeline instances with the
# official transition converters, so make_pre_post_processors(pretrained_path=...)
# reconstructs them and the composition pre -> policy -> post is genuinely official.
#
# The pipelines are STEP-EMPTY (identity in the sense of "no extra normalization"):
# the CoreAI runner owns observation preprocessing and action postprocessing. That
# ownership must be declared in the manifest processor contract; packaging/binding
# fails closed otherwise (a runner that expects LeRobot to normalize would get raw
# values). No hardware, no egress.

from __future__ import annotations

from typing import Any

from lerobot.policies.factory import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.processor import (
    PolicyProcessorPipeline,
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)

PREPROCESSOR_FILENAME = f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
POSTPROCESSOR_FILENAME = f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"


class ProcessorOwnershipError(RuntimeError):
    """Raised when the manifest does not declare CoreAI processor ownership."""


def _processor_contract(manifest: Any) -> dict:
    if isinstance(manifest, dict):
        contracts = manifest.get("contracts")
    else:
        contracts = getattr(manifest, "contracts", None)
    if not isinstance(contracts, dict):
        return {}
    proc = contracts.get("processor")
    return proc if isinstance(proc, dict) else {}


def manifest_declares_coreai_ownership(manifest: Any) -> bool:
    """True iff the manifest declares coreai_runner owns pre- AND post-processing."""
    proc = _processor_contract(manifest)
    obs = proc.get("observation_input") if isinstance(proc, dict) else None
    act = proc.get("action_output") if isinstance(proc, dict) else None
    if not isinstance(obs, dict) or not isinstance(act, dict):
        return False
    return obs.get("owner") == "coreai_runner" and act.get("owner") == "coreai_runner"


def require_coreai_processor_ownership(manifest: Any) -> None:
    """Fail closed unless the manifest explicitly cedes processing to the runner."""
    if not manifest_declares_coreai_ownership(manifest):
        raise ProcessorOwnershipError(
            "identity processors require the manifest to declare "
            "contracts.processor.observation_input.owner == 'coreai_runner' AND "
            "contracts.processor.action_output.owner == 'coreai_runner'. "
            "Ambiguous or absent ownership is refused (the runner would receive "
            "un-normalized observations).")


def build_coreai_bridge_processors() -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Return (pre, post) as real, step-empty PolicyProcessorPipeline instances."""
    pre = PolicyProcessorPipeline(
        steps=[], name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline(
        steps=[], name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action)
    return pre, post


def make_coreai_bridge_pre_post_processors(config: Any, dataset_stats: Any = None):
    """Return (preprocessor, postprocessor) for the CoreAI bridge policy.

    Real PolicyProcessorPipeline instances (step-empty). Signature matches the
    official ``make_<name>_pre_post_processors(config, dataset_stats=None)``.
    """
    return build_coreai_bridge_processors()


def save_coreai_bridge_processors(output_dir: str, *, manifest: Any = None) -> tuple[str, str]:
    """Save policy_preprocessor.json / policy_postprocessor.json to ``output_dir``.

    If ``manifest`` is given, CoreAI processor ownership is required (fail-closed).
    Returns the two written filenames.
    """
    if manifest is not None:
        require_coreai_processor_ownership(manifest)
    pre, post = build_coreai_bridge_processors()
    pre.save_pretrained(output_dir, config_filename=PREPROCESSOR_FILENAME)
    post.save_pretrained(output_dir, config_filename=POSTPROCESSOR_FILENAME)
    return PREPROCESSOR_FILENAME, POSTPROCESSOR_FILENAME
