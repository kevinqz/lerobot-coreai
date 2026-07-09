# Real Mode Safety Model

> No verified readiness, no real egress.
> No approval, no real egress.
> No safety profile, no real egress.
> No supervisor enforce, no real egress.
> No operator attestation, no real egress.
> No bounded session, no real egress.
> No adapter preflight, no real egress.

## Guard layers

Every generated action passes through, in order:

1. **Release readiness** — `ready=true` required (preflight gate).
2. **Operator approval** — valid, unexpired, bound to the bundle (preflight gate).
3. **Safety profile** — present, valid, robot-type match (preflight gate).
4. **Runtime supervisor** — hardcoded `enforce`; evaluates every action.
5. **RealEgressGuard** — the single egress path; fail-closed on session state,
   deadman, e-stop, rate limit, adapter readiness, and supervisor verdict.
6. **Deadman switch** — software liveness (see below).
7. **Rate limiter** — bounds egress to `fps`.
8. **Adapter preflight** — the adapter must self-report ready.

## Fail-closed

Any of these blocks the action before it can reach the adapter:

- session not running · e-stop triggered · deadman unhealthy · rate exceeded ·
  adapter not ready · supervisor blocked · supervisor internal error

Any exception in the loop stops **and** disconnects the adapter. A blocked
action ends the guarded session — real mode does not keep generating actions
after a block.

## No-bypass guarantee

The only `.send_action(` call sites are the adapter definitions and the
`RealEgressGuard`. A test enforces that no other module calls an adapter's
`send_action`, and that a blocked supervisor action never reaches the adapter.

## Emergency stop

The **software deadman is not a substitute for a physical emergency stop.** The
guarded-mode attestations require the operator to confirm a physical e-stop is
ready and the workspace is clear. The deadman may be disabled **only** for the
`mock` adapter; for any real adapter it cannot be disabled.

## Honest claims

Real mode never claims physical safety. The strongest true statement is:

> This run executed a bounded guarded real session under verified software
> readiness, operator approval, and enforced safety supervision.

The statement that must never appear:

> This proves the robot is physically safe.
