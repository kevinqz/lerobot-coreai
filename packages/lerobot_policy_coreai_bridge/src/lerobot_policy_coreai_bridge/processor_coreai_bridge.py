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
    DeviceProcessorStep,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)

# The ONLY steps the bridge preprocessor may contain (v1.3.27.2): env/device PLUMBING
# — key renaming (env obs keys → canonical policy keys) and tensor device/dtype
# placement. These do NOT normalize or otherwise transform observation semantics; the
# CoreAI runner still owns the raw-observation → action inference. Their presence is
# what makes the bridge drivable by the official `lerobot-eval` CLI (which injects
# `device_processor` + `rename_observations_processor` overrides). Any SEMANTIC step
# (normalize/latent/etc.) remains forbidden — that would silently pre-process data the
# runner is contracted to own.
_ALLOWED_PREPROCESSOR_STEP_NAMES = ("rename_observations_processor", "device_processor")

PREPROCESSOR_FILENAME = f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
POSTPROCESSOR_FILENAME = f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"

# Processor contract v2 (v1.3.7): step-empty (identity) pipelines are permitted
# ONLY when the CoreAI runner owns BOTH ends with these exact semantics. Any other
# semantics (e.g. "policy_preprocessed_observation", "normalized_action",
# "latent_action") would require concrete processor steps — a step-empty pipeline
# there would silently ship un-normalized data to the runner.
PROCESSOR_CONTRACT_SCHEMA_V2 = "coreai-processor-contract.v2"
IDENTITY_OBS_EXPECTS = "raw_lerobot_observation"
IDENTITY_ACTION_RETURNS = "postprocessed_environment_action"
_RUNNER = "coreai_runner"


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
    """True iff the manifest cedes BOTH ends to the runner with identity semantics.

    v1.3.7: the exact ``expects``/``returns`` values are required, not just the
    owner. Owner-correct + wrong expects/returns is NOT identity-eligible.
    """
    proc = _processor_contract(manifest)
    obs = proc.get("observation_input") if isinstance(proc, dict) else None
    act = proc.get("action_output") if isinstance(proc, dict) else None
    if not isinstance(obs, dict) or not isinstance(act, dict):
        return False
    return (obs.get("owner") == _RUNNER and act.get("owner") == _RUNNER
            and obs.get("expects") == IDENTITY_OBS_EXPECTS
            and act.get("returns") == IDENTITY_ACTION_RETURNS)


def require_coreai_processor_ownership(manifest: Any) -> None:
    """Fail closed unless the manifest explicitly cedes processing to the runner."""
    if not manifest_declares_coreai_ownership(manifest):
        raise ProcessorOwnershipError(
            "identity (step-empty) processors require the manifest to declare "
            "contracts.processor with owner 'coreai_runner' on BOTH ends AND "
            f"observation_input.expects == {IDENTITY_OBS_EXPECTS!r} AND "
            f"action_output.returns == {IDENTITY_ACTION_RETURNS!r}. "
            "Any other semantics require concrete processor steps; ambiguous or "
            "absent ownership is refused (the runner would receive un-normalized "
            "observations).")


def build_coreai_bridge_processors() -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Return (pre, post). The pre-processor carries ONLY env/device plumbing steps
    (rename → device); the CoreAI runner still owns all observation/action semantics, so
    there is NO normalization or other semantic step. This shape is what the official
    ``lerobot-eval`` CLI can drive (it overrides those two steps). The post-processor
    stays step-empty (the runner returns the environment-ready action)."""
    pre = PolicyProcessorPipeline(
        steps=[RenameObservationsProcessorStep(rename_map={}),
               DeviceProcessorStep(device="cpu")],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline(
        steps=[], name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action)
    return pre, post


def make_coreai_bridge_pre_post_processors(config: Any, dataset_stats: Any = None):
    """Fail-closed (v1.3.7): building processors from scratch is not supported.

    The official factory only uses this create-path when no ``pretrained_path`` is
    set. A CoreAI bridge is runtime-only and always ships a canonical artifact, so
    identity processors must come from a verified artifact (whose manifest declares
    CoreAI ownership) via ``make_pre_post_processors(pretrained_path=...)``, NOT be
    silently synthesized here without any ownership evidence.
    """
    raise ProcessorOwnershipError(
        "coreai_bridge processors cannot be built from scratch without artifact "
        "evidence. Package a canonical artifact "
        "(`lerobot-coreai package-lerobot-plugin-artifact`) and load it via "
        "make_pre_post_processors(pretrained_path=...), which reconstructs the "
        "serialized PolicyProcessorPipeline files.")


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
