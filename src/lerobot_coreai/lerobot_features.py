# lerobot_features.py — dataset ↔ policy feature mapping (v1.1.4).
#
# Pure, LeRobot-free feature-mapping logic: given the features a LeRobotDataset
# exposes and the observation features a CoreAI policy manifest expects, produce
# an auditable mapping and a pass/fail verdict. `strict` turns missing required
# keys, shape mismatches, and (optionally) unknown features into failures. This
# proves the observation *mapping* is coherent for the sample — never task
# success or physical safety.

from __future__ import annotations

from typing import Any

FEATURE_MAPPING_SCHEMA_VERSION = "lerobot-coreai.lerobot_feature_mapping.v0"


def _shape_compatible(dataset_shape, policy_shape) -> bool:
    if policy_shape is None or dataset_shape is None:
        return True  # unconstrained on one side
    return list(dataset_shape) == list(policy_shape)


def build_feature_mapping(
    *,
    dataset_features: dict[str, Any],
    policy_obs_features: dict[str, dict[str, Any]],
    task_in_config: bool = False,
    strict: bool = False,
    fail_on_unknown: bool = False,
) -> dict[str, Any]:
    """Map dataset features onto policy-expected observation features.

    Args:
        dataset_features: name -> shape (list|None) available from the dataset.
        policy_obs_features: name -> {"shape": [...]|None, "required": bool}.
        task_in_config: True if `task` is supplied by config rather than the dataset.
        strict: missing required key / shape mismatch become failures.
        fail_on_unknown: in strict mode, an unknown dataset feature also fails
            (otherwise it is a warning).

    Returns a report dict with per-feature entries and a `passed` verdict.
    """
    features: dict[str, Any] = {}
    problems: list[str] = []
    warnings: list[str] = []

    for name, spec in policy_obs_features.items():
        required = bool(spec.get("required", True))
        pshape = spec.get("shape")
        present = name in dataset_features
        provided_by_config = (name == "task" and task_in_config)

        entry: dict[str, Any] = {
            "policy_expected": True,
            "required": required,
            "dataset_present": present,
        }
        if provided_by_config:
            entry["provided_by_config"] = True

        shape_ok = True
        if present:
            shape_ok = _shape_compatible(dataset_features.get(name), pshape)
            entry["shape_compatible"] = shape_ok

        satisfied = present or provided_by_config or not required
        if not satisfied:
            problems.append(f"missing required feature {name!r}")
        if present and not shape_ok:
            problems.append(
                f"shape mismatch for {name!r}: dataset {dataset_features.get(name)} "
                f"!= policy {pshape}")
        features[name] = entry

    # Dataset features the policy does not expect.
    unknown = [n for n in dataset_features if n not in policy_obs_features
               and n != "task"]
    for n in unknown:
        features.setdefault(n, {"policy_expected": False, "dataset_present": True})
        msg = f"unknown dataset feature {n!r} (not expected by policy)"
        if strict and fail_on_unknown:
            problems.append(msg)
        else:
            warnings.append(msg)

    # In non-strict mode, requirement/shape issues are warnings, not failures.
    passed = True
    if strict:
        passed = not problems
    else:
        warnings = warnings + problems
        problems = []

    return {
        "schema_version": FEATURE_MAPPING_SCHEMA_VERSION,
        "strict": strict,
        "features": features,
        "unknown_dataset_features": unknown,
        "problems": problems,
        "warnings": warnings,
        "passed": passed,
    }


def manifest_obs_features(manifest) -> dict[str, dict[str, Any]]:
    """Extract policy observation features from a manifest into the mapping shape."""
    out: dict[str, dict[str, Any]] = {}
    for name, f in manifest.observation_features.items():
        out[name] = {
            "shape": list(f.shape) if getattr(f, "shape", None) is not None else None,
            "required": bool(getattr(f, "required", True)),
        }
    return out
