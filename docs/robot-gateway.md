# Robot Gateway reference (RFC-0700 §19.1 / RFC-0900 §13 — LR10)

The **gateway** is the final *software* authority before hardware egress — **distinct**
from the CoreAI Runner (inference) and the Server (LAN). `robot_gateway.py` is the Python
reference implementation of the `org.lerobot.robot-gateway.v1` protocol (defined in
`coreai-interop`).

`ReferenceRobotGateway` accepts an action chunk **only** when every gate passes, and every
gate **fails closed**:

- authenticated app + matching **session** id;
- matching **robot identity** and **policy artifact root**;
- **monotonic sequence** (a replayed or out-of-order chunk is refused);
- unexpired **deadline**;
- **bounded, finite** actions (`|a| ≤ max_abs_action`, no NaN/Inf, optional action-dim);
- **watchdog**: too long since the last accepted message → stop.

Egress is an **injected** `RobotEgress`. The default `DryRunEgress` **sends nothing** —
dry-run is the safe default. Only a `guarded`-mode session with a real
(upstream-LeRobot `Robot`) egress actuates, and even then behind every gate above. The
session **receipt** records accepted/rejected/executed and always sets
`proves_physical_safety = false`.

This layer proves only that the configured software checks ran. It is **never** a
mechanical or physical-safety guarantee; hardware e-stop and controller limits remain
independent (RFC-0900 §14).
