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

## Built-in profiles

- **`default-sim-safe`** — minimal guard: finite values required, no shape,
  robot-type, or numeric bounds. A safe, non-restrictive default for sim.
- **`so100-sim-default`** — conservative SO-100-family example: `16x7` action,
  normalized `[-1, 1]`, bounded delta and L2 norm, clipping enabled.

Load by name (`--safety.profile-name so100-sim-default`) or by path
(`--safety.profile path/to/profile.json`).

## Example

```json
{
  "schema_version": "lerobot-coreai.safety_profile.v0",
  "name": "so100-sim-default",
  "robot_type": "so100",
  "action_shape": [16, 7],
  "min_action": -1.0,
  "max_action": 1.0,
  "max_abs_action": 1.0,
  "max_delta": 0.5,
  "max_l2_norm": 8.0,
  "allow_nan": false,
  "allow_inf": false,
  "allow_shape_change": false,
  "clip_to_bounds": true,
  "block_on_clip": false,
  "mode": "fail_closed"
}
```

Profiles are validated against `schemas/safety-profile.schema.json`. `mode` must
be `fail_closed`.
