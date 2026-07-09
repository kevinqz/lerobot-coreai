# Robot Adapters

A **robot adapter** is the only component that can touch a robot. It is invoked
exclusively by the [`RealEgressGuard`](real-mode-safety.md), only in guarded real
mode, only after every gate passes. There is no bundled motor/serial driver and
no hidden fallback to any robot API.

## Protocol

```python
class RobotAdapter(Protocol):
    name: str
    robot_type: str
    def preflight(self) -> dict: ...
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_ready(self) -> bool: ...
    def get_observation(self) -> dict: ...
    def send_action(self, action) -> dict: ...
    def stop(self) -> None: ...
```

## Built-in: `mock`

`MockRobotAdapter` touches **no hardware**. It records actions and reports
`safe_mock=True`, letting you exercise the entire gated flow safely. The deadman
may be disabled only for the mock.

```bash
--robot.adapter mock --robot.type so100
```

## Optional: `external-http`

Delegates egress to an **operator-controlled** HTTP controller you run and own.
This is real egress and runs behind every gate.

```bash
--robot.adapter external-http --robot.endpoint http://127.0.0.1:8765
```

Expected endpoints:

```
GET  /preflight      POST /connect     POST /disconnect
GET  /ready          GET  /observation POST /action        POST /stop
```

**Loopback-only (v1.0.0):** the endpoint must be `127.0.0.1` / `localhost` — a
remote host is refused. Run the controller on the same machine. In guarded mode,
the external `/preflight` is not even contacted until the operator attestations
are present.

## Native SO-100 / SO-101

Not built in. `lerobot-coreai` does not ship a native motor-bus driver. Point an
`external-http` adapter at a controller you operate — behind readiness,
approval, safety profile, supervisor enforce, and operator attestations.

Unknown adapter names fail closed:

```
Unknown or unimplemented robot adapter: 'so100'. Available: mock, external-http.
```
