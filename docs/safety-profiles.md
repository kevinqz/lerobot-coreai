# Safety Profiles

A **safety profile** is a conservative *software* contract for action bounds. It
is **not** a certified hardware safety envelope and does **not** prove physical
robot safety. It only bounds what the [runtime supervisor](safety-supervisor.md)
will let egress.

## Fields

| Field | Meaning |
|-------|---------|
| `name` | Profile identifier (required) |
| `robot_type` | Expected robot type; enforced if `require_robot_type_match` |
| `action_shape` | Expected action shape (e.g. `[16, 7]`) |
| `min_action` / `max_action` | Lower/upper bound (scalar or per-element list) |
| `max_abs_action` | Maximum absolute value of any element |
| `max_delta` | Maximum per-element change from the previous action |
| `max_l2_norm` | Maximum L2 norm of the action |
| `allow_nan` / `allow_inf` | Permit non-finite values (default `false`) |
| `allow_shape_change` | Permit a shape different from `action_shape` |
| `clip_to_bounds` | Clip out-of-bound actions instead of blocking |
| `block_on_clip` | Block whenever clipping would occur |
| `require_finite` | Require finite values |
| `require_known_shape` | Block when the shape cannot be inferred |
| `require_robot_type_match` | Require `context.robot_type == robot_type` |
| `mode` | Fixed to `fail_closed` |

## Metadata (v0.9.1)

Profiles are explainable, auditable artifacts. Beyond the bound fields above,
each profile carries `profile_version`, `profile_type` (`software_bounds`),
`intended_modes` / `intended_envs` / `intended_policy_types`, optional
`calibrated_from` / `calibration_method` / `calibration_date`, and a
`limitations` list. Every profile's limitations must acknowledge that it does
not prove physical robot safety.

## Built-in profiles

- **`default-sim-safe`** — minimal guard: finite required, `max_abs 1.0`, no
  shape/robot-type requirement. Safe, non-restrictive default.
- **`generic-7dof-sim-default`** — single `[7]` action, bounded abs/delta/L2.
- **`so100-sim-default`** — SO-100 family: `[16, 7]`, bounded delta `0.35` / L2 `6.0`.
- **`so101-sim-default`** — SO-101 family, same shape/bounds as SO-100.
- **`pusht-sim-default`** — PushT-style envs: `[2]` action, `intended_envs`
  `["PushT-v0", "pusht"]`.

Inspect them with `lerobot-coreai profile-list` / `profile-show`, validate with
`profile-validate`, pick one with [profile-recommend](profile-recommendation.md),
and fit one to your data with [profile-calibrate](profile-calibration.md).

> Delta bounds and shape changes (v0.9.1): if a profile sets `max_delta`, a step
> whose action length changed from the previous one is **unverifiable** and is
> blocked (fail-closed) — `max_delta` guarantees bounded motion, which cannot be
> proven across a shape change. Omit `max_delta` (as `generic-7dof-sim-default`
> does) if you intend `allow_shape_change` to permit varying shapes.

Load by name (`--safety.profile-name so100-sim-default`) or by path
(`--safety.profile path/to/profile.json`).

## Example

```json
{
  "schema_version": "lerobot-coreai.safety_profile.v0",
  "name": "so100-sim-default",
  "profile_version": "0.1.0",
  "profile_type": "software_bounds",
  "robot_type": "so100",
  "action_shape": [16, 7],
  "max_abs_action": 1.0,
  "max_delta": 0.35,
  "max_l2_norm": 6.0,
  "allow_nan": false,
  "allow_inf": false,
  "allow_shape_change": false,
  "clip_to_bounds": false,
  "block_on_clip": false,
  "require_finite": true,
  "require_known_shape": true,
  "require_robot_type_match": true,
  "mode": "fail_closed",
  "intended_modes": ["sim", "shadow"],
  "limitations": [
    "Software action-bound profile for simulator/shadow workflows.",
    "Does not prove SO-100 physical safety."
  ]
}
```

Profiles are validated against `schemas/safety-profile.schema.json`. `mode` must
be `fail_closed` and `profile_type` must be `software_bounds`.
