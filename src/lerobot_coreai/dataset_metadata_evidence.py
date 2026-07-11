# dataset_metadata_evidence.py — DatasetMetadataEvidence v1 (v1.3.25).
#
# Bind policy semantics to the EXACT LeRobot dataset metadata tree that defines
# robot_type, fps, features, names, shapes, tasks and statistics. Captures a
# duck-typed metadata object (the real LeRobotDatasetMetadata, or any object exposing
# the same properties) + a deterministic metadata-tree root. dataset_metadata_verified
# is scoped to the exact root/revision; dataset_content_verified stays false (content
# is a separate proof). Pure Python; no lerobot import required to VERIFY.

from __future__ import annotations

from typing import Any

from .dataset_metadata_hash import (
    METADATA_TREE_HASH_ALGORITHM, compute_metadata_tree, metadata_tree_sha256,
)

DATASET_METADATA_EVIDENCE_SCHEMA_VERSION = "lerobot-coreai.dataset-metadata-evidence.v1"
_SHA256 = {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}
_METADATA_SOURCE_MODES = ("local_fixture", "local_dataset", "hub_snapshot")

DATASET_METADATA_EVIDENCE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "repo_id", "revision", "root_kind",
                 "metadata_tree_algorithm", "metadata_tree_sha256", "files",
                 "robot_type", "fps", "features", "camera_keys", "names", "shapes",
                 "task_count", "episode_count", "claims"],
    "properties": {
        "schema_version": {"const": DATASET_METADATA_EVIDENCE_SCHEMA_VERSION},
        "repo_id": {"type": "string", "minLength": 1},
        "revision": {"type": ["string", "null"]},
        "codebase_version": {"type": ["string", "null"]},
        "root_kind": {"enum": list(_METADATA_SOURCE_MODES)},
        "resolved_commit": {"type": ["string", "null"]},
        # v1.3.26.3: certificate grade requires the REAL loader identity (no
        # duck-typed stand-in); diagnostic grade may omit it.
        "evidence_grade": {"enum": ["diagnostic", "certificate"]},
        "loader_identity": {"anyOf": [{"type": "null"}, {
            "type": "object", "additionalProperties": False,
            "required": ["module", "class_name", "lerobot_version"],
            "properties": {"module": {"type": "string"}, "class_name": {"type": "string"},
                           "lerobot_version": {"type": ["string", "null"]},
                           "source_commit": {"type": ["string", "null"]},
                           "wheel_digest": {"type": ["string", "null"]}}}]},
        "metadata_tree_algorithm": {"const": METADATA_TREE_HASH_ALGORITHM},
        "metadata_tree_sha256": _SHA256,
        "files": {"type": "object", "additionalProperties": _SHA256},
        "robot_type": {"type": ["string", "null"]},
        "fps": {"type": "integer", "minimum": 1},
        "features": {"type": "object"},
        "camera_keys": {"type": "array", "items": {"type": "string"}},
        "names": {"type": "object"},
        "shapes": {"type": "object"},
        "task_count": {"type": "integer", "minimum": 0},
        "episode_count": {"type": "integer", "minimum": 0},
        "claims": {
            "type": "object", "additionalProperties": False,
            "required": ["dataset_metadata_verified", "dataset_content_verified",
                         "proves_task_success"],
            "properties": {"dataset_metadata_verified": {"type": "boolean"},
                           "dataset_content_verified": {"const": False},
                           "proves_task_success": {"const": False}}},
    },
}


def _feature_dict(features: Any) -> dict:
    """Normalize a features mapping (feature -> {dtype, shape, names}) to plain JSON."""
    out: dict[str, dict] = {}
    for k, v in dict(features or {}).items():
        dtype = v.get("dtype") if isinstance(v, dict) else getattr(v, "dtype", None)
        shape = v.get("shape") if isinstance(v, dict) else getattr(v, "shape", None)
        names = v.get("names") if isinstance(v, dict) else getattr(v, "names", None)
        out[k] = {"dtype": dtype,
                  "shape": list(shape) if shape is not None else None,
                  "names": list(names) if names is not None else None}
    return out


def _derive_loader_identity(meta: Any) -> dict:
    """The concrete loader class identity + its LeRobot distribution version."""
    cls = type(meta)
    try:
        from importlib.metadata import version
        lv = version("lerobot")
    except Exception:  # noqa: BLE001
        lv = None
    return {"module": cls.__module__, "class_name": cls.__qualname__,
            "lerobot_version": lv, "source_commit": None, "wheel_digest": None}


def capture_dataset_metadata_evidence(
    meta: Any, *, root: str, repo_id: str, revision: str | None = None,
    root_kind: str = "local_fixture", resolved_commit: str | None = None,
    evidence_grade: str = "diagnostic", loader_identity: dict | None = None,
) -> dict:
    """Build DatasetMetadataEvidence from a metadata object + its on-disk tree.

    ``meta`` is duck-typed for DIAGNOSTIC grade. In CERTIFICATE grade the real loader
    identity is recorded (module/class/version) and must come from the official
    ``LeRobotDatasetMetadata`` — a stand-in/adapter is refused (v1.3.26.3)."""
    files, tree_sha = compute_metadata_tree(root)
    fps = int(getattr(meta, "fps"))
    if fps < 1:
        raise ValueError(f"fps must be positive, got {fps}")
    if evidence_grade == "certificate":
        loader_identity = loader_identity or _derive_loader_identity(meta)
        if not (loader_identity.get("module", "").startswith("lerobot.")
                and loader_identity.get("class_name") == "LeRobotDatasetMetadata"):
            raise ValueError(
                "certificate-grade metadata evidence requires the official "
                f"LeRobotDatasetMetadata loader; got {loader_identity}")
    return {
        "evidence_grade": evidence_grade, "loader_identity": loader_identity,
        "schema_version": DATASET_METADATA_EVIDENCE_SCHEMA_VERSION,
        "repo_id": repo_id, "revision": revision,
        "codebase_version": getattr(meta, "_version", None) and str(getattr(meta, "_version")),
        "root_kind": root_kind, "resolved_commit": resolved_commit,
        "metadata_tree_algorithm": METADATA_TREE_HASH_ALGORITHM,
        "metadata_tree_sha256": tree_sha, "files": files,
        "robot_type": getattr(meta, "robot_type", None), "fps": fps,
        "features": _feature_dict(getattr(meta, "features", {})),
        "camera_keys": list(getattr(meta, "camera_keys", []) or []),
        "names": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                  for k, v in dict(getattr(meta, "names", {}) or {}).items()},
        "shapes": {k: list(v) if isinstance(v, (list, tuple)) else v
                   for k, v in dict(getattr(meta, "shapes", {}) or {}).items()},
        "task_count": int(getattr(meta, "total_tasks", 0) or 0),
        "episode_count": int(getattr(meta, "total_episodes", 0) or 0),
        "claims": {"dataset_metadata_verified": True,
                   "dataset_content_verified": False, "proves_task_success": False},
    }


def verify_dataset_metadata_evidence(evidence: dict, root: str) -> tuple[bool, list]:
    """Offline: schema-valid, and the recorded tree hash recomputes from ``root``.

    Certificate grade for a hub snapshot additionally requires a resolved_commit."""
    import jsonschema
    errors: list[str] = []
    try:
        jsonschema.validate(evidence, DATASET_METADATA_EVIDENCE_SCHEMA)
    except Exception as exc:  # noqa: BLE001
        return False, [f"schema: {exc}"]
    try:
        files, tree_sha = compute_metadata_tree(root)
    except Exception as exc:  # noqa: BLE001
        return False, [f"recompute: {exc}"]
    if files != evidence["files"]:
        errors.append("metadata file set/digests differ from disk")
    if tree_sha != evidence["metadata_tree_sha256"]:
        errors.append("metadata_tree_sha256 mismatch")
    if evidence["root_kind"] == "hub_snapshot" and not evidence.get("resolved_commit"):
        errors.append("hub_snapshot without a resolved_commit is not certificate-grade")
    # v1.3.26.3: certificate grade requires the official loader identity.
    if evidence.get("evidence_grade") == "certificate":
        li = evidence.get("loader_identity") or {}
        if not (str(li.get("module", "")).startswith("lerobot.")
                and li.get("class_name") == "LeRobotDatasetMetadata"):
            errors.append("certificate grade requires the official LeRobotDatasetMetadata "
                          "loader identity")
    return (not errors), errors
