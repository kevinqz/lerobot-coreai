# External HTTP Controller Contract (v1.1.1)

> **No controller contract, no external real egress.** Before a guarded real
> session may egress through an operator-run `external-http` controller, the
> controller must (1) live on a loopback endpoint, (2) declare a rigid
> capability contract at `GET /preflight`, and (3) report a ready, safe
> `GET /safety-state`. Any gap fails closed. This proves nothing about physical
> safety — it hardens the software boundary.

The `external-http` adapter is the only built-in path to real hardware, and it
is loopback-only and operator-controlled. There is **no** native SO-100/SO-101
adapter and no serial/motor/dynamixel/feetech imports.

## Endpoint rules

`--robot.endpoint` must be an explicit loopback `http://` URL **with a port**:

| Endpoint | Result |
|----------|--------|
| `http://127.0.0.1:8765` | ✅ |
| `http://localhost:8765` | ✅ |
| `http://[::1]:8765` | ✅ |
| `http://192.168.0.10:8765` | ❌ remote host |
| `https://127.0.0.1:8765` | ❌ non-http scheme |
| `file:///tmp/x` | ❌ non-http scheme |
| `http://127.0.0.1` | ❌ missing explicit port |
| `http://0.0.0.0:8765` | ❌ not loopback |
| `http://2130706433:8765` | ❌ obfuscated numeric host |

## `GET /preflight` — capability contract

```json
{
  "ok": true,
  "controller_schema_version": "lerobot-coreai.external_http.v0",
  "adapter": "external-http",
  "controller_name": "so100-local-controller",
  "robot_type": "so100",
  "action_shape": [16, 7],
  "observation_keys": ["observation.state", "task"],
  "supports_stop": true,
  "supports_ready": true,
  "supports_observation": true,
  "supports_safety_state": true,
  "physical_estop_required": true,
  "max_fps": 5.0
}
```

Enforced invariants: `ok` true, schema version exact, `robot_type` matches
`--robot.type`, `action_shape` matches the safety-profile/policy shape when
known, `max_fps` ≥ requested `--fps`, and `supports_stop` / `supports_ready` /
`supports_observation` / `supports_safety_state` / `physical_estop_required` all
true.

## `GET /safety-state` — pre-egress gate (guarded mode)

```json
{
  "ok": true,
  "ready": true,
  "robot_type": "so100",
  "controller_connected": true,
  "physical_estop_state": "armed",
  "workspace_state": "clear",
  "motors_powered": true,
  "faults": []
}
```

For guarded egress: `ready` true, `robot_type` matches, `controller_connected`
true, `physical_estop_state == "armed"`, `workspace_state == "clear"`, and
`faults == []`. Any `unknown`, `triggered`, `not_clear`, or non-empty `faults`
fails closed.

## Local auth token

Pass the token by **env-var name** so it never appears in `argv`:

```bash
export LEROBOT_COREAI_ROBOT_TOKEN="$(openssl rand -hex 32)"

lerobot-coreai real \
  --mode guarded \
  --robot.adapter external-http \
  --robot.endpoint http://127.0.0.1:8765 \
  --robot.auth-token-env LEROBOT_COREAI_ROBOT_TOKEN \
  ...
```

`--robot.auth-token-env NAME` fails closed if `NAME` is unset. Every request
carries `Authorization: Bearer <token>` and `X-LeRobot-CoreAI-Token`, plus
`X-LeRobot-CoreAI-Session` / `X-LeRobot-CoreAI-Approval` when known and
`X-LeRobot-CoreAI-Intent` for `preflight` / `safety-state` probes. The **raw
token is never written** to any report or trace — only a `sha256:` prefix is
recorded, under `external_http.auth.token_sha256_prefix` in the preflight report.

## Minimal fake controller (for local testing)

```python
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class Controller(BaseHTTPRequestHandler):
    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/preflight":
            return self._json({
                "ok": True,
                "controller_schema_version": "lerobot-coreai.external_http.v0",
                "adapter": "external-http", "controller_name": "fake-so100-controller",
                "robot_type": "so100", "action_shape": [16, 7],
                "observation_keys": ["observation.state", "task"],
                "supports_stop": True, "supports_ready": True,
                "supports_observation": True, "supports_safety_state": True,
                "physical_estop_required": True, "max_fps": 5.0,
            })
        if self.path == "/ready":
            return self._json({"ready": True})
        if self.path == "/safety-state":
            return self._json({
                "ok": True, "ready": True, "robot_type": "so100",
                "controller_connected": True, "physical_estop_state": "armed",
                "workspace_state": "clear", "motors_powered": True, "faults": [],
            })
        if self.path == "/observation":
            return self._json({
                "observation.state": [0.0] * 7,
                "task": "mock external-http guarded session",
            })

    def do_POST(self):
        if self.path in ("/connect", "/disconnect", "/stop"):
            return self._json({"ok": True})
        if self.path == "/action":
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode())
            return self._json({"sent": True, "received_action": payload["action"]})

HTTPServer(("127.0.0.1", 8765), Controller).serve_forever()
```

This is a **testing stand-in**, not a real robot controller. Operating real
hardware is the operator's responsibility, behind all real-mode gates.
