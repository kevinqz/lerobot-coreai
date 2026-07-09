# No-Actuation Contract

This document is the formal contract for shadow mode (v0.7). It defines exactly what
shadow mode guarantees and what it does not.

## The contract

**Shadow mode generates actions, then blocks them.**

Every action produced by the policy is:
1. **Validated** against the manifest (shape, NaN, Inf)
2. **Logged** to `actions.jsonl` and `blocked_actions.jsonl`
3. **Blocked** by `ActionBlocker` — no egress path exists

There is no code path in `shadow.py`, `observation_sources.py`, or `action_blocker.py`
that forwards an action to a robot, motor, simulator, or actuator.

## What this means

### Shadow mode IS

- A **runtime action generation** proof: the artifact runs in a loop and produces valid actions.
- An **action validation/logging** proof: every action is validated and audited.
- A **no-actuation** proof: `actions_sent` is always 0, enforced by schema.

### Shadow mode IS NOT

- **Not real mode.** Shadow mode does not actuate a physical robot.
- **Not sim mode.** Shadow mode does not forward actions to a simulator.
- **Not a task success proof.** Shadow mode does not close a control loop.
- **Not a physical safety proof.** Shadow mode does not test real-world behavior.

## Enforcement mechanisms

### 1. `ActionBlocker` (runtime)

The `ActionBlocker` class is the sole egress gate. Its `send()` method unconditionally
raises `SafetyError("Action egress is disabled in shadow mode. No robot commands were sent.")`.

```python
blocker = ActionBlocker(mode="shadow")
blocked = blocker.block(action)  # OK — records action, returns BlockedAction(sent=False)
blocker.send(action)             # ALWAYS raises SafetyError
```

### 2. Schema invariants (report)

`shadow-report.schema.json` enforces 10 `const` invariants that make it impossible for a
valid report to claim any actuation occurred:

- `mode` = `"shadow"`
- `metrics.actions_sent` = `0`
- `safety.actions_sent` = `0`
- `safety.action_egress` = `"blocked"`
- `safety.physical_actuation_possible` = `false`
- `safety.motor_commands_available` = `false`
- `safety.actuation_device_connected` = `false`
- `safety.robot_connected` = `false`
- `claims.proves_task_success` = `false`
- `claims.proves_robot_safety` = `false`
- `claims.proves_real_world_safety` = `false`

These hold even in **failure reports**.

### 3. Static source analysis (tests)

`test_no_hardware_actuation.py` scans all `.py` files under `src/` for forbidden hardware
tokens and actuation egress patterns:

```
import serial, dynamixel, feetech
.send_action(, serial.Serial, dynamixel_sdk, motor_bus, teleop
write_position, write_goal_position, robot.connect(
```

`test_shadow_no_actuation.py` verifies that the shadow, observation_sources, and
action_blocker modules contain no hardware imports.

## The line of value

```
v0.6 proves the artifact.
v0.7 proves the artifact can run in a loop without action egress.
v0.8 proves actions can affect a simulator.
v0.9 proves unsafe actions can be supervised.
v1.0 allows guarded robot actuation.
```

**The v0.7 win is not movement. The win is auditable runtime action generation under a
hard no-actuation contract.**
