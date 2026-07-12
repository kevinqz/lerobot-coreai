# Apple-app conformance fixtures (RFC-0700 §19.1 / RFC-0900, LR12)

Published by `lerobot-coreai` and consumed by `LeRobotCoreAIKit` in `lerobot-coreai-apple`.
They pin the semantics the Swift app MUST reproduce, so it binds LeRobot behavior rather
than redefining it:

- `policy-manifest.json` — `org.huggingface.lerobot.policy.v1` binding (feature contract,
  action-chunk horizon, processor ownership).
- `skill-plan.json` — a valid `org.lerobot.robot-brain.skill-plan.v1` (planner output).
- `action-queue-cases.json` — reference `ActionQueue` transitions (in-order drain, empty,
  deadline-expiry invalidation) — see `lerobot_coreai.app_conformance.ReferenceActionQueue`.

The planner→policy handoff (`skill_plan_to_policy_tasks`) proves the language model emits
goals/skills only — never motor commands.
