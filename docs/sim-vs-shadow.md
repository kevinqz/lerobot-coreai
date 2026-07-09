# Sim vs Shadow

Sim mode (v0.8) and shadow mode (v0.7) both run a CoreAI-backed LeRobot policy in a
loop and log actions. They differ in **where actions go**.

## At a glance

| | Shadow (v0.7) | Sim (v0.8) |
|---|---|---|
| Observations | streamed/replayed | from a SimEnvironment |
| Actions generated | yes | yes |
| Action egress | **blocked** (logged only) | **sent to simulator** (`env.step()`) |
| `actions_sent_to_robot` | 0 | 0 |
| Confirmation required | no | `--confirm-sim-egress` |
| Egress object | `ActionBlocker` | `SimEgress` |

## What each proves

| | Shadow | Sim |
|---|---|---|
| Runtime action generation | ✅ | ✅ |
| Actions affect a simulator | ❌ | ✅ |
| Sim task success (when env reports it) | ❌ | ✅ (conditional) |
| Real-world task success | ❌ | ❌ |
| Physical robot safety | ❌ | ❌ |
| Real-world safety | ❌ | ❌ |

## Pipeline difference

```
shadow:  observation → policy → action → ActionBlocker → logs
                                            (actions_sent = 0)

sim:     env.reset() → observation → policy → action → SimEgress → env.step()
                                                          (actions_sent_to_simulator > 0)
```

In shadow, `ActionBlocker.send()` always raises — no action leaves the process.
In sim, `SimEgress.send_to_simulator()` forwards the action to the environment's
`step()`, and `SimEgress.send_to_robot()` always raises.

## When to use which

- **Shadow**: you want to verify the policy runs and produces sane actions
  against real observations, without driving anything. Actions are audited but
  never applied.
- **Sim**: you want to see the policy drive a simulated environment, observe
  reward/success feedback, and measure episode-level behavior. Actions still
  never touch a robot.

Neither mode is real mode. Neither proves physical safety.
