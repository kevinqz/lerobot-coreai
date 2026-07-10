# LeRobot Local Registry Adapter (v1.1.3)

> **Local, opt-in, reversible.** Registry-style ergonomics for CoreAI-backed
> policies **without** pretending LeRobot upstream knows about them. There is no
> global monkeypatch at import, `policy_type="coreai"` is never used (only
> `"coreai_bridge"`), and any patch of LeRobot's factory happens only inside an
> explicit `with` block and is reversed on exit.

## Local registry

```python
from lerobot_coreai.lerobot_registry import CoreAILeRobotRegistry

registry = CoreAILeRobotRegistry()
registry.register("coreai_bridge")           # refuses "coreai" — that implies upstream
policy = registry.load(
    "coreai_bridge",
    policy_path="kevinqz/EVO1-SO100-CoreAI",
    runner_url="http://127.0.0.1:8710",
)                                            # -> CoreAILeRobotPolicyBridge
```

This registry is entirely process-local; it never touches LeRobot's factory.

## Opt-in, reversible factory patch

```python
from lerobot_coreai.lerobot_registry import local_lerobot_registry_patch
from lerobot.policies.factory import get_policy_class

with local_lerobot_registry_patch():
    get_policy_class("coreai_bridge")        # -> CoreAILeRobotPolicyBridge
    get_policy_class("act")                  # still delegates to LeRobot

# outside the block, get_policy_class is the original again
```

Inside the block, `lerobot.policies.factory.get_policy_class("coreai_bridge")`
resolves to the bridge class and every other name delegates to the original.
On exit — including on exception — the original function is restored exactly. If
LeRobot isn't installed, the context manager is a no-op over the factory and the
local registry still works.

## Check CLI

```bash
lerobot-coreai lerobot-registry-check \
  --policy.type coreai_bridge \
  --policy.path kevinqz/EVO1-SO100-CoreAI \
  --runner.url http://127.0.0.1:8710 \
  --output-dir reports/registry-check
```

Verifies: `"coreai"` is refused, the type registers, the upstream factory is
unchanged by default, the context manager resolves the bridge and restores the
factory, and (best-effort) a load returns a bridge. Writes
`lerobot_registry_report.json/md` with `native_upstream_registry=false`.

## What this is not

- Not upstream LeRobot registration (`policy_type="coreai"` does not exist upstream).
- Not a global/default monkeypatch.
- Not a training path. Proves nothing about physical safety.
