# Profile Recommendation

v0.9.1 recommends a built-in [safety profile](safety-profiles.md) from a policy
manifest and/or the dominant action shape in an actions log. The recommendation
is a **heuristic** and does not prove physical safety.

## Recommend

```bash
# From a policy (reads robot_type from the manifest):
lerobot-coreai profile-recommend --policy.path kevinqz/EVO1-SO100-CoreAI

# From an actions log (uses the dominant action shape):
lerobot-coreai profile-recommend --actions runs/pusht/actions.jsonl

# From explicit signals:
lerobot-coreai profile-recommend --robot-type so101 --env.id PushT-v0
```

## Heuristic

| Signal | Recommendation | Confidence |
|--------|----------------|-----------|
| `robot_type == so100` | `so100-sim-default` | high |
| `robot_type == so101` | `so101-sim-default` | high |
| `env_id` contains `pusht` | `pusht-sim-default` | medium |
| dominant shape `[2]` | `pusht-sim-default` | medium |
| dominant shape `[16, 7]` | `so100-sim-default` | medium |
| dominant shape `[7]` | `generic-7dof-sim-default` | medium |
| none of the above | `default-sim-safe` | low |

Robot type is the strongest signal and overrides a conflicting shape. Every
recommendation includes a warning that it is heuristic and does not prove
physical safety.

## Output

```json
{
  "recommended_profile": "so100-sim-default",
  "confidence": "high",
  "reasons": ["policy manifest robot_type=so100"],
  "warnings": ["Recommendation is heuristic and does not prove physical safety."]
}
```
