# processor_contract.py — declare who owns pre/post processing (v1.2.6).
#
# A CoreAI-vs-PyTorch comparison is only valid if both sides agree on WHERE
# normalization/denormalization happens. LeRobot runs
# observation -> preprocessor -> policy -> postprocessor -> action. The CoreAI
# artifact must declare whether its runner expects raw or preprocessed
# observations and whether it returns normalized or postprocessed actions. When
# ownership is ambiguous, --strict-processors fails closed. No hardware, no egress.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PROCESSOR_CONTRACT_SCHEMA_VERSION = "lerobot-coreai.processor_contract.v0"

_VALID_EXPECTS = {"raw_lerobot_observation", "policy_preprocessed_observation"}
_VALID_RETURNS = {"postprocessed_action", "normalized_action"}


@dataclass
class ProcessorContract:
    expects: str = "unknown"          # what observation the runner expects
    returns: str = "unknown"          # what action the runner returns
    image_layout: str | None = None   # CHW / HWC
    image_dtype: str | None = None
    image_range: list[float] | None = None
    state_normalization: str | None = None
    action_units: str | None = None
    action_order: list[str] = field(default_factory=list)
    dataset_stats_sha256: str | None = None

    def is_ambiguous(self) -> bool:
        return self.expects not in _VALID_EXPECTS or self.returns not in _VALID_RETURNS

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_input": {
                "expects": self.expects,
                "image_layout": self.image_layout,
                "image_dtype": self.image_dtype,
                "image_range": self.image_range,
                "state_normalization": self.state_normalization,
            },
            "action_output": {
                "returns": self.returns,
                "action_units": self.action_units,
                "action_order": self.action_order,
            },
            "stats": {"dataset_stats_sha256": self.dataset_stats_sha256},
        }


class ProcessorContractError(Exception):
    """Raised when a processor contract is ambiguous under strict mode."""


def parse_processor_contract_from_manifest(manifest, *, strict: bool = False) -> ProcessorContract:
    """Parse a processor contract; fail closed on ambiguity under strict mode."""
    block = None
    if isinstance(manifest, dict):
        block = manifest.get("processor_contract")
    else:
        block = getattr(manifest, "processor_contract", None)

    if not isinstance(block, dict):
        contract = ProcessorContract()
    else:
        obs = block.get("observation_input", {}) or {}
        act = block.get("action_output", {}) or {}
        stats = block.get("stats", {}) or {}
        contract = ProcessorContract(
            expects=obs.get("expects", "unknown"),
            returns=act.get("returns", "unknown"),
            image_layout=obs.get("image_layout"),
            image_dtype=obs.get("image_dtype"),
            image_range=obs.get("image_range"),
            state_normalization=obs.get("state_normalization"),
            action_units=act.get("action_units"),
            action_order=act.get("action_order", []),
            dataset_stats_sha256=stats.get("dataset_stats_sha256"))

    if strict and contract.is_ambiguous():
        raise ProcessorContractError(
            "processor contract is ambiguous: declare observation_input.expects "
            f"({sorted(_VALID_EXPECTS)}) and action_output.returns "
            f"({sorted(_VALID_RETURNS)}) to run --strict-processors.")
    return contract


def build_processor_contract_report(contract: ProcessorContract, *,
                                    strict: bool) -> dict[str, Any]:
    return {
        "schema_version": PROCESSOR_CONTRACT_SCHEMA_VERSION,
        "ambiguous": contract.is_ambiguous(),
        "strict": strict,
        "contract": contract.to_dict(),
        "claims": {
            "proves_processor_ownership_declared": not contract.is_ambiguous(),
            "proves_task_success": False,
            "proves_physical_safety": False,
        },
    }
