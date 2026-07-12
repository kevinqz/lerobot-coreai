# episode_staging.py — mobile-recording bridge (RFC-0700 §22 / RFC-0900 §22, LR11).
#
# The Apple app records an append-only `org.lerobot.episode-staging.v1` session (manifest
# + events). Canonical conversion to an upstream LeRobotDataset runs on Mac through
# lerobot-coreai (RFC-0700 §22.3). This module is the base-package half: it VALIDATES a
# staged episode (schema shape, monotonic per-source time, required-feature coverage,
# freshness, session/artifact identity binding) and produces a deterministic conversion
# PLAN (episode features → LeRobotDataset feature keys + fps + episode boundary). The
# actual `LeRobotDataset` write needs upstream lerobot and runs in the rollout jobs; it
# is deliberately NOT done here. Pure Python; no torch/lerobot.

from __future__ import annotations

EPISODE_STAGING_SCHEMA = "org.lerobot.episode-staging.v1"

_MANIFEST_REQUIRED = ("schema", "session_id", "created_monotonic_ns", "features",
                      "events_ref", "checksums_ref")
_EVENT_REQUIRED = ("session_id", "monotonic_ns", "sequence", "source", "schema_version")


def validate_episode_manifest(manifest: dict) -> tuple[bool, list]:
    """Validate the staging manifest's shape + identity (fail-closed)."""
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return False, ["manifest must be a dict"]
    if manifest.get("schema") != EPISODE_STAGING_SCHEMA:
        errors.append(f"schema must be {EPISODE_STAGING_SCHEMA!r}")
    for k in _MANIFEST_REQUIRED:
        if k not in manifest:
            errors.append(f"missing manifest field {k!r}")
    feats = manifest.get("features")
    if not isinstance(feats, dict) or not feats:
        errors.append("features must be a non-empty object")
    else:
        for name, spec in feats.items():
            if not isinstance(spec, dict) or "dtype" not in spec:
                errors.append(f"feature {name!r} needs a dtype")
    if not isinstance(manifest.get("created_monotonic_ns"), int):
        errors.append("created_monotonic_ns must be an integer (monotonic ns)")
    return (not errors), errors


def validate_episode_events(manifest: dict, events: list) -> tuple[bool, list]:
    """Validate the event stream against the manifest: identity binding, per-source
    monotonic time + sequence, required-feature coverage, and freshness (max_age_ns).
    A staged episode that violates any of these MUST NOT convert (fail-closed)."""
    errors: list[str] = []
    sid = manifest.get("session_id")
    feats = manifest.get("features") or {}
    if not isinstance(events, list) or not events:
        return False, ["events must be a non-empty list"]
    last_seq: dict = {}
    last_ns: dict = {}
    seen_features: set = set()
    last_ns_by_feature: dict = {}
    for i, ev in enumerate(events):
        for k in _EVENT_REQUIRED:
            if k not in ev:
                errors.append(f"event[{i}] missing {k!r}")
                continue
        if ev.get("session_id") != sid:
            errors.append(f"event[{i}] session_id != manifest session (identity break)")
        src = ev.get("source")
        ns, seq = ev.get("monotonic_ns"), ev.get("sequence")
        if isinstance(ns, int) and src in last_ns and ns < last_ns[src]:
            errors.append(f"event[{i}] non-monotonic time for source {src!r}")
        if isinstance(seq, int) and src in last_seq and seq <= last_seq[src]:
            errors.append(f"event[{i}] non-increasing sequence for source {src!r}")
        if isinstance(ns, int):
            last_ns[src] = ns
        if isinstance(seq, int):
            last_seq[src] = seq
        feat = ev.get("feature")
        if feat is not None:
            seen_features.add(feat)
            # freshness: consecutive samples of a feature must be within its max_age_ns.
            max_age = (feats.get(feat) or {}).get("max_age_ns")
            if isinstance(max_age, int) and feat in last_ns_by_feature and \
                    isinstance(ns, int) and (ns - last_ns_by_feature[feat]) > max_age:
                errors.append(f"feature {feat!r} exceeded max_age_ns between samples")
            if isinstance(ns, int):
                last_ns_by_feature[feat] = ns
    # every manifest-declared feature must actually appear in the stream.
    missing = [f for f in feats if f not in seen_features]
    if missing:
        errors.append(f"declared features never recorded: {missing}")
    return (not errors), errors


# episode-staging feature name → canonical LeRobotDataset key mapping.
def _lerobot_key(name: str) -> str:
    if name in ("state", "observation.state", "agent_pos"):
        return "observation.state"
    if name == "action":
        return "action"
    if name.startswith("image") or name.startswith("observation.images"):
        # normalize "image.front" / "front" → observation.images.<cam>
        cam = name.split(".")[-1]
        return f"observation.images.{cam}"
    if name == "task":
        return "task"
    return name


def build_lerobot_conversion_plan(manifest: dict) -> dict:
    """Deterministic plan mapping the staged episode to LeRobotDataset feature keys + fps
    + a single-episode boundary. This is the validated conversion SPEC; the actual
    `LeRobotDataset.create/add_frame/save_episode` write runs in the rollout jobs (needs
    upstream lerobot) and is intentionally not performed here."""
    ok, reasons = validate_episode_manifest(manifest)
    if not ok:
        raise ValueError(f"cannot plan conversion for an invalid manifest: {reasons}")
    feats = manifest["features"]
    features_map = {name: _lerobot_key(name) for name in feats}
    fps = manifest.get("fps", 10)
    return {
        "target": "LeRobotDataset",
        "session_id": manifest["session_id"],
        "fps": fps,
        "features_map": features_map,
        "lerobot_features": {features_map[n]: {"dtype": feats[n]["dtype"],
                                               "shape": feats[n].get("shape")}
                             for n in feats},
        "episodes": 1,
        "requires_upstream_lerobot": True,   # actual write happens in the rollout job
    }
